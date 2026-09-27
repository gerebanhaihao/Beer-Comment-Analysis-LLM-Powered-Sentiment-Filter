"""End-to-end run: CSV -> time window -> Stage 1 -> Stage 2 -> colored Excel."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
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
from beer_sentiment.models import JudgedRow, Label, RunSummary
from beer_sentiment.pipeline.stage1 import Stage1Pipeline
from beer_sentiment.pipeline.stage2 import Stage2Pipeline

ReviewCallback = Callable[[str, list[JudgedRow]], dict[int, Label] | None]


def run_file(
    input_path: str | Path,
    output_dir: str | Path,
    session: str,
    date: str | None,
    judge: Judge,
    config: AppConfig,
    all_time: bool = False,
    review_callback: ReviewCallback | None = None,
) -> RunSummary:
    path = Path(input_path)
    rows, _ = read_csv_rows(path, "auto")
    if not rows:
        raise ValueError(f"CSV 为空：{path}")
    headers = list(rows[0].keys())

    time_col = detect_time_column(headers)
    if not time_col and not all_time:
        raise ValueError(f"无法识别时间列：{path}（或使用 --all-time 跳过时间过滤）")
    data_col = detect_data_column(headers) or headers[0]
    text_cols = detect_text_columns(headers)
    if not text_cols:
        raise ValueError(f"无法识别 正文/封面OCR/内容OCR/标题 列：{path}")

    if all_time:
        start = end = None
    else:
        today = (
            dt.date.fromisoformat(date)
            if date
            else dt.datetime.now(dt.timezone.utc).astimezone().date()
        )
        start, end = compute_window(session, today, config.time)
    preparation = Stage1Pipeline(config).prepare(
        rows,
        time_col or "",
        data_col,
        text_cols,
        path.name,
        start,
        end,
    )
    judged, low_confidence = Stage2Pipeline(config).judge(preparation.prepared_rows, judge)
    if review_callback is not None:
        decisions = review_callback(path.name, judged)
        if decisions is None:
            raise RuntimeError(f"已取消人工复核，未生成文件：{path.name}")
        valid_rows = {row.prepared.original_row_number for row in judged}
        if not set(decisions).issubset(valid_rows):
            raise ValueError("人工复核结果包含无效行号")
        for row in judged:
            line_number = row.prepared.original_row_number
            if line_number in decisions:
                row.result.label = Label.parse(decisions[line_number])
                row.low_confidence = False
        low_confidence = [row for row in judged if row.low_confidence]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / output_filename(
        preparation.source_file,
        preparation.output_class,
        session,
    )
    labels = [Label.NONE if row.low_confidence else row.result.label for row in judged]
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
        blue_rows=labels.count(Label.BLUE),
        yellow_rows=labels.count(Label.YELLOW),
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
    all_time: bool = False,
    selected_files: list[str | Path] | None = None,
    review_callback: ReviewCallback | None = None,
    summary_callback: Callable[[RunSummary], None] | None = None,
) -> tuple[list[RunSummary], list]:
    input_path = Path(input_dir)
    if selected_files is not None:
        csv_paths = list(dict.fromkeys(Path(path) for path in selected_files))
        if any(
            not path.is_file() or path.suffix.lower() != ".csv" for path in csv_paths
        ):
            raise ValueError("选中的文件不存在或不是 CSV")
    else:
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
        raise ValueError(
            "未选择 CSV 文件"
            if selected_files is not None
            else f"输入目录没有 CSV：{input_path}"
        )

    summaries = []
    all_low: list = []
    for path in csv_paths:
        summary = run_file(
            path,
            output_dir,
            session,
            date,
            judge,
            config,
            all_time=all_time,
            review_callback=review_callback,
        )
        summaries.append(summary)
        all_low.extend(summary.low_confidence_rows)
        if summary_callback is not None:
            summary_callback(summary)
    return summaries, all_low
