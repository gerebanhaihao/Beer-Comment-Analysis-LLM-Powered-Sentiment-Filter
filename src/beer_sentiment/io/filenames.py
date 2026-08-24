"""Output filename conventions."""

from __future__ import annotations

from pathlib import Path


def session_cn(session: str) -> str:
    return "上午" if session.lower() == "morning" else "下午"


def review_filename(session: str) -> str:
    return f"待筛选_{session_cn(session)}.csv"


def output_filename(source_file: str, output_class: str, session: str) -> str:
    prefix = f"{output_class}{session_cn(session)}"
    stem = Path(source_file).stem if source_file else ""
    if "__" in stem:
        suffix = stem.split("__", 1)[1]
        return f"{prefix}__{suffix}.xlsx"
    return f"{prefix}.xlsx"
