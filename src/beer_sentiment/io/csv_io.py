"""CSV reading, column detection, and time-window helpers."""

from __future__ import annotations

import csv
import datetime as dt
import re
from collections import Counter
from pathlib import Path
from typing import Any

from beer_sentiment.rules.normalize import clean_text


ENCODINGS = ["utf-8-sig", "gbk", "utf-8"]

TIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y年%m月%d日 %H:%M:%S",
    "%Y年%m月%d日 %H:%M",
    "%Y年%m月%d日%H:%M",
]


def parse_time(value: Any) -> dt.datetime | None:
    text = clean_text(value)
    if not text:
        return None
    text = re.sub(r"\.\d{1,6}$", "", text)
    text = re.sub(r"(Z|[+-]\d{2}:?\d{2})$", "", text)
    text = text.replace("T", " ")
    for fmt in TIME_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def find_column(headers: list[str], candidates: list[str]) -> str | None:
    for header in headers:
        if header is None:
            continue
        lowered = clean_text(header).lower()
        if any(candidate in lowered for candidate in candidates):
            return header
    return None


def detect_time_column(headers: list[str]) -> str | None:
    return find_column(headers, ["发帖时间", "发布时间", "时间", "日期", "date", "time"])


def detect_data_column(headers: list[str]) -> str | None:
    return find_column(headers, ["数据范围", "数据", "分类"])


def detect_text_columns(headers: list[str]) -> list[str]:
    exact = ["正文", "封面OCR", "内容OCR", "标题"]
    found = [header for header in headers if header in exact]
    if found:
        return found
    candidates = ["内容", "正文", "文本", "帖子", "评论", "text", "content", "message"]
    return [header for header in headers if find_column([header], candidates)]


def read_csv_rows(path: str | Path, encoding: str = "auto") -> tuple[list[dict[str, Any]], str]:
    encodings = ENCODINGS if encoding == "auto" else [encoding]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            with Path(path).open("r", encoding=enc, newline="") as handle:
                rows = list(csv.DictReader(handle))
            return rows, enc
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue
    raise ValueError(f"无法识别文件编码 {path}: {last_error}")


def compute_window(
    session: str,
    today: dt.date,
    time_config: dict[str, Any] | None = None,
) -> tuple[dt.datetime, dt.datetime]:
    cfg = time_config or {}

    def clock(key: str, default: str) -> dt.time:
        return dt.time.fromisoformat(cfg.get(key, default))

    if session == "morning":
        morning_start = clock("morning_start", "17:30")
        morning_end = clock("morning_end", "10:00")
        if today.weekday() == 0:
            offset = int(cfg.get("monday_morning_offset_days", 3))
            start_day = today - dt.timedelta(days=offset)
        else:
            start_day = today - dt.timedelta(days=1)
        start = dt.datetime.combine(start_day, morning_start)
        end = dt.datetime.combine(today, morning_end)
    else:
        start = dt.datetime.combine(today, clock("afternoon_start", "10:00"))
        end = dt.datetime.combine(today, clock("afternoon_end", "17:30"))
    return start, end


def resolve_session(value: str | None) -> str:
    if value:
        return value.lower()
    now = dt.datetime.now()
    return "morning" if now.hour < 10 else "afternoon"


def infer_output_class(data_value: Any, filename: str) -> str:
    data_text = clean_text(data_value)
    if data_text.startswith("啤酒行业") or ("行业" in data_text and "竞品" not in data_text):
        return "行业"
    if "行业" in filename or "industry" in filename.lower():
        return "行业"
    return "品牌"


def infer_file_output_class(
    rows: list[dict[str, Any]],
    data_col: str,
    filename: str,
) -> str:
    if not rows:
        return infer_output_class("", filename)
    counts: Counter[str] = Counter()
    order: list[str] = []
    for row in rows:
        output = infer_output_class(row.get(data_col, ""), filename)
        if output not in counts:
            order.append(output)
        counts[output] += 1
    if not counts:
        return "品牌"
    return max(order, key=lambda item: (counts[item], -order.index(item)))
