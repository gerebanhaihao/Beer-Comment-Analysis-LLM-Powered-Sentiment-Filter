import csv
from pathlib import Path

from beer_sentiment.io.ingest import ingest_directory, recent_csv_files


def _write_csv(path: Path, title: str = "百威避雷"):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["数据范围", "发帖时间", "标题", "正文"])
        writer.writerow(["百威（本品）-产品", "2026-08-24 09:20:00", title, "太难喝"])


def test_ingest_copies_and_skips_duplicate(tmp_path):
    incoming = tmp_path / "incoming"
    data = tmp_path / "data"
    incoming.mkdir()
    source = incoming / "quark.csv"
    _write_csv(source)

    first = ingest_directory(incoming, data)
    assert first[0].status == "copied"
    assert (data / "quark.csv").exists()
    assert source.exists()

    second = ingest_directory(incoming, data)
    assert second[0].status == "duplicate"


def test_ingest_moves_when_requested(tmp_path):
    incoming = tmp_path / "incoming"
    data = tmp_path / "data"
    incoming.mkdir()
    source = incoming / "quark.csv"
    _write_csv(source)

    result = ingest_directory(incoming, data, move=True)
    assert result[0].status == "moved"
    assert not source.exists()
    assert (data / "quark.csv").exists()


def test_ingest_single_csv_into_data_directory(tmp_path):
    source_dir = tmp_path / "source"
    data = tmp_path / "data"
    source_dir.mkdir()
    source = source_dir / "chosen.csv"
    ignored = source_dir / "other.csv"
    _write_csv(source)
    _write_csv(ignored, title="青岛避雷")

    results = ingest_directory(source, data)

    assert len(results) == 1
    assert results[0].status == "copied"
    assert (data / "chosen.csv").exists()
    assert not (data / "other.csv").exists()


def test_ingest_empty_source_does_not_create_destination(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    data = tmp_path / "data"

    assert ingest_directory(source_dir, data) == []
    assert not data.exists()


def test_recent_csv_files_prefers_last_imported_files(tmp_path):
    source_dir = tmp_path / "source"
    data = tmp_path / "data"
    source_dir.mkdir()
    files = [source_dir / f"quark__2026-08-31 11{i}000.csv" for i in range(3)]
    for source in files:
        _write_csv(source, title=source.stem)
        ingest_directory(source, data)

    selected = recent_csv_files(data)
    assert len(selected) == 2
    assert {path.name for path in selected} == {files[1].name, files[2].name}
