"""End-to-end run: CSV -> time window -> Stage 1 -> Stage 2 -> colored Excel."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from beer_sentiment.config import AppConfig
from beer_sentiment.io.csv_io import (
    compute_window,
    detect_data_column,
    detect_text_columns,
    detect_time_column,
    read_csv_rows,
)
from beer_sentiment.io.excel import write_colored_excel
from beer_sentiment.io.filenames import output_filename
from beer_sentiment.llm.base import Judge
from beer_sentiment.models import Label, RunSummary
from beer_sentiment.pipeline.stage1 import Stage1Pipeline
from beer_sentiment.pipeline.stage2 import Stage2Pipeline


def run_file(
    input_path: str | Path,
    output_dir: str | Path,
    session: str,
    date: str | None,
    judge: Judge,
    config: AppConfig,
) -> RunSummary:
    path = Path(input_path)
    rows, _ = read_csv_rows(path, "auto")
    if not rows:
        raise ValueError(f"CSV 为空：{path}")
    headers = list(rows[0].keys())

    time_col = detect_time_column(headers)
    if not time_col:
        raise ValueError(f"无法识别时间列：{path}")
    data_col = detect_data_column(headers) or headers[0]
    text_cols = detect_text_columns(headers)
    if not text_cols:
        raise ValueError(f"无法识别 正文/封面OCR/内容OCR/标题 列：{path}")

    today = dt.date.fromisoformat(date) if date else dt.date.today()
    start, end = compute_window(session, today, config.time)
    preparation = Stage1Pipeline(config).prepare(
        rows,
        time_col,
        data_col,
        text_cols,
        path.name,
        start,
        end,
    )
    judged, low_confidence = Stage2Pipeline(config).judge(preparation.prepared_rows, judge)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / output_filename(
        preparation.source_file,
        preparation.output_class,
        session,
    )
    labels = [row.result.label for row in judged]
    write_colored_excel(
        out_file,
        [row.prepared.row for row in judged],
        labels,
        headers,
        config,
    )

    return RunSummary(
        source_file=preparation.source_file,
        output_path=str(out_file),
        output_class=preparation.output_class,
        total_rows=len(judged),
        candidates=sum(1 for row in judged if row.prepared.stage1.is_candidate),
        blue_rows=sum(1 for row in judged if row.result.label == Label.BLUE),
        yellow_rows=sum(1 for row in judged if row.result.label == Label.YELLOW),
        low_confidence_rows=low_confidence,
        total_latency_ms=sum(row.result.latency_ms for row in judged),
        total_cost_usd=sum(row.result.cost_usd for row in judged),
    )


def run_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    session: str,
    date: str | None,
    judge: Judge,
    config: AppConfig,
    name_contains: list[str] | None = None,
) -> tuple[list[RunSummary], list]:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_path}")
    csv_paths = sorted(input_path.rglob("*.csv"))
    if name_contains:
        csv_paths = [
            path
            for path in csv_paths
            if any(keyword in path.name for keyword in name_contains)
        ]
    if not csv_paths:
        raise ValueError(f"输入目录没有 CSV：{input_path}")

    summaries = []
    all_low: list = []
    for path in csv_paths:
        summary = run_file(path, output_dir, session, date, judge, config)
        summaries.append(summary)
        all_low.extend(summary.low_confidence_rows)
    return summaries, all_low
