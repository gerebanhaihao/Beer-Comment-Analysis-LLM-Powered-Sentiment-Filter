"""Local CSV drop-folder ingestion for Quark exports.

This is intentionally file-based. It does not connect to Quark or any other
platform. A user can place downloaded CSV files in an inbox directory and
move or copy validated files into the pipeline's data directory.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beer_sentiment.io.csv_io import (
    detect_text_columns,
    detect_time_column,
    read_csv_rows,
)


@dataclass
class IngestResult:
    source: str
    destination: str = ""
    status: str = ""
    rows: int = 0
    encoding: str = ""
    sha256: str = ""
    message: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"files": {}}
    return data if isinstance(data, dict) and isinstance(data.get("files"), dict) else {"files": {}}


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def recent_csv_files(data_dir: str | Path, limit: int = 2) -> list[Path]:
    """Return the newest imported CSVs, falling back to file modification time."""
    data_path = Path(data_dir)
    if not data_path.is_dir() or limit <= 0:
        return []
    manifest = _load_manifest(data_path / ".ingest_manifest.json")
    imported_at: dict[str, float] = {}
    for item in manifest["files"].values():
        if not isinstance(item, dict):
            continue
        name = item.get("stored_name")
        stamp = item.get("ingested_at")
        if not isinstance(name, str) or not isinstance(stamp, str):
            continue
        try:
            imported_at[name] = max(
                imported_at.get(name, 0), dt.datetime.fromisoformat(stamp).timestamp()
            )
        except ValueError:
            continue
    paths = [
        path
        for path in data_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    ]
    paths.sort(
        key=lambda path: (
            path.name in imported_at,
            imported_at.get(path.name, path.stat().st_mtime),
            path.name,
        ),
        reverse=True,
    )
    return paths[:limit]


def _unique_destination(source: Path, data_dir: Path, digest: str) -> Path:
    destination = data_dir / source.name
    if not destination.exists():
        return destination
    if _sha256(destination) == digest:
        return destination
    return data_dir / f"{source.stem}__{digest[:8]}{source.suffix}"


def ingest_directory(
    input_dir: str | Path,
    data_dir: str | Path,
    *,
    move: bool = False,
    encoding: str = "auto",
) -> list[IngestResult]:
    """Validate one CSV or CSVs in a folder, then copy/move to ``data_dir``."""
    input_path = Path(input_dir)
    data_path = Path(data_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"源数据目录或文件不存在：{input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"源数据文件不是 CSV：{input_path}")
        if input_path.parent.resolve() == data_path.resolve():
            raise ValueError("源数据文件已在待导入目录中")
        csv_paths = [input_path]
    elif input_path.is_dir():
        csv_paths = sorted(path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".csv")
    else:
        raise ValueError(f"源数据路径无效：{input_path}")
    if input_path.resolve() == data_path.resolve():
        raise ValueError("源数据目录不能与待导入目录相同")
    if not csv_paths:
        return []
    data_path.mkdir(parents=True, exist_ok=True)

    manifest_path = data_path / ".ingest_manifest.json"
    manifest = _load_manifest(manifest_path)
    files = manifest.setdefault("files", {})
    results: list[IngestResult] = []

    for source in csv_paths:
        digest = _sha256(source)
        if digest in files:
            results.append(
                IngestResult(
                    source=str(source),
                    status="duplicate",
                    sha256=digest,
                    message="文件内容已接入，跳过",
                )
            )
            continue

        try:
            rows, detected_encoding = read_csv_rows(source, encoding)
            if not rows:
                results.append(
                    IngestResult(
                        source=str(source),
                        status="rejected",
                        sha256=digest,
                        message="CSV 为空",
                    )
                )
                continue
            headers = list(rows[0].keys())
            text_columns = detect_text_columns(headers)
            if not text_columns:
                results.append(
                    IngestResult(
                        source=str(source),
                        status="rejected",
                        sha256=digest,
                        message="缺少标题/正文/OCR 等文本列",
                    )
                )
                continue

            destination = _unique_destination(source, data_path, digest)
            if destination.exists() and _sha256(destination) == digest:
                status = "duplicate"
            elif move:
                shutil.move(str(source), str(destination))
                status = "moved"
            else:
                shutil.copy2(source, destination)
                status = "copied"

            files[digest] = {
                "stored_name": destination.name,
                "rows": len(rows),
                "encoding": detected_encoding,
                "has_time_column": bool(detect_time_column(headers)),
                "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"),
            }
            results.append(
                IngestResult(
                    source=str(source),
                    destination=str(destination),
                    status=status,
                    rows=len(rows),
                    encoding=detected_encoding,
                    sha256=digest,
                    message="已接入",
                )
            )
        except (OSError, ValueError, UnicodeError) as exc:
            results.append(
                IngestResult(
                    source=str(source),
                    status="rejected",
                    sha256=digest,
                    message=str(exc),
                )
            )

    _save_manifest(manifest_path, manifest)
    return results
