"""Command line interface for the beer sentiment pipeline."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from beer_sentiment.config import AppConfig, load_config
from beer_sentiment.eval.benchmark import load_benchmark
from beer_sentiment.eval.metrics import evaluate
from beer_sentiment.eval.report import render_compare, save_run
from beer_sentiment.io.csv_io import (
    compute_window,
    detect_data_column,
    detect_text_columns,
    detect_time_column,
    infer_output_class,
    parse_time,
    read_csv_rows,
    resolve_session,
)
from beer_sentiment.io.excel import write_colored_excel
from beer_sentiment.io.filenames import output_filename, review_filename
from beer_sentiment.llm.base import Judge
from beer_sentiment.llm.mock import MockJudge
from beer_sentiment.llm.openai_compat import OpenAICompatJudge
from beer_sentiment.models import Label
from beer_sentiment.pipeline.run import run_directory
from beer_sentiment.rules.classify import Stage1Classifier
from beer_sentiment.rules.normalize import clean_text


REVIEW_HEADERS = ["颜色", "判断说明", "命中关键词", "合并文本", "输出", "来源文件", "原行号"]
REVIEW_COLUMNS = set(REVIEW_HEADERS)

COLOR_ALIASES = {
    "蓝": "蓝",
    "蓝色": "蓝",
    "本品": "蓝",
    "own": "蓝",
    "blue": "蓝",
    "黄": "黄",
    "黄色": "黄",
    "竞品": "黄",
    "行业": "黄",
    "yellow": "黄",
    "待确认": "待确认",
    "不确定": "待确认",
    "存疑": "待确认",
    "unsure": "待确认",
}
OUTPUT_ALIASES = {
    "品牌": "品牌",
    "brand": "品牌",
    "brands": "品牌",
    "行业": "行业",
    "industry": "行业",
}


def normalize_color(value) -> str:
    key = clean_text(value).lower().replace(" ", "")
    return COLOR_ALIASES.get(key, "不标")


def normalize_output(value) -> str:
    key = clean_text(value).lower().replace(" ", "")
    return OUTPUT_ALIASES.get(key, "")


def build_judge(name: str, config: AppConfig) -> Judge:
    model_config = config.model_config(name)
    kind = model_config.get("type", "openai_compatible")
    if kind == "mock":
        return MockJudge(config)
    if kind == "openai_compatible":
        return OpenAICompatJudge(name, model_config, config)
    raise ValueError(f"未知模型类型：{kind}")


def cmd_prepare(args, config: AppConfig) -> None:
    session = resolve_session(args.session)
    today = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    start, end = compute_window(session, today, config.time)
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"输入目录不存在：{input_dir.resolve()}")
        sys.exit(1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.rglob("*.csv"))
    if args.name_contains:
        csv_paths = [
            path
            for path in csv_paths
            if any(keyword in path.name for keyword in args.name_contains)
        ]
    if not csv_paths:
        print(f"输入目录没有 CSV：{input_dir.resolve()}")
        sys.exit(1)

    classifier = Stage1Classifier(config)
    original_headers: list[str] = []
    rows = []
    for path in csv_paths:
        source_rows, _ = read_csv_rows(path, args.encoding)
        if not source_rows:
            continue
        source_headers = list(source_rows[0].keys())
        time_col = args.time_column or detect_time_column(source_headers)
        data_col = args.data_column or detect_data_column(source_headers) or source_headers[0]
        text_cols = args.text_columns or detect_text_columns(source_headers)
        if not time_col:
            print(f"无法在 {path.name} 中识别时间列，请用 --time-column 指定")
            sys.exit(1)
        if not text_cols:
            print(f"无法在 {path.name} 中识别 正文/封面OCR/内容OCR/标题 列，请用 --text-column 指定")
            sys.exit(1)

        for header in source_headers:
            if header not in original_headers:
                original_headers.append(header)

        for row_index, source_row in enumerate(source_rows, start=2):
            parsed = parse_time(source_row.get(time_col, ""))
            if parsed is None or not (start <= parsed <= end):
                continue
            texts = [clean_text(source_row.get(header, "")) for header in text_cols]
            combined = "\n".join(text for text in texts if text)
            data_value = source_row.get(data_col, "")
            stage1 = classifier.classify(combined)
            row = {header: source_row.get(header, "") for header in source_headers}
            row.update(
                {
                    "颜色": "",
                    "判断说明": "粗筛候选，待人工判断" if stage1.is_candidate else "粗筛未命中",
                    "命中关键词": "、".join(
                        stage1.matched_keywords + stage1.matched_association_keywords
                    ),
                    "合并文本": combined,
                    "输出": infer_output_class(data_value, path.name),
                    "来源文件": path.name,
                    "原行号": row_index,
                }
            )
            rows.append(row)

    review_headers = REVIEW_HEADERS + original_headers
    out_path = output_dir / review_filename(session)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"场次：{'上午' if session == 'morning' else '下午'}")
    print(f"时间窗：{start:%Y-%m-%d %H:%M} 至 {end:%Y-%m-%d %H:%M}")
    print(f"输入 CSV：{len(csv_paths)} 个，进入待筛选：{len(rows)} 行")
    print(f"待筛选文件：{out_path.resolve()}")


def cmd_build(args, config: AppConfig) -> None:
    review_path = Path(args.review_csv)
    rows, _ = read_csv_rows(review_path, "auto")
    if not rows:
        print(f"待筛选 CSV 为空：{review_path}")
        sys.exit(1)
    all_headers = list(rows[0].keys())
    headers = [header for header in all_headers if header not in REVIEW_COLUMNS]
    session = args.session.lower()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = {}
    output_counts = {}
    uncertain = []
    for row_index, row in enumerate(rows, start=2):
        color = normalize_color(row.get("颜色", ""))
        if color == "待确认":
            uncertain.append((row_index, row.get("判断说明", "")))
            continue
        output = normalize_output(row.get("输出", ""))
        if not output:
            data_value = row.get(headers[0], "") if headers else ""
            filename = row.get("来源文件", "")
            output = infer_output_class(data_value, filename)
        source_file = row.get("来源文件", "")
        grouped.setdefault(source_file, []).append((color, row_index, row))
        output_counts.setdefault(source_file, {})
        output_counts[source_file][output] = output_counts[source_file].get(output, 0) + 1

    if uncertain:
        print("以下行标记为待确认，请先人工判断后再生成：")
        for row_index, note in uncertain:
            print(f"  行 {row_index}: {note}")
        sys.exit(1)

    for source_file, item_rows in grouped.items():
        output_name = max(output_counts[source_file], key=output_counts[source_file].get)
        out_path = output_dir / output_filename(source_file, output_name, session)
        labels = [Label.parse(color) for color, _, _ in item_rows]
        write_colored_excel(
            out_path,
            [row for _, _, row in item_rows],
            labels,
            headers,
            config,
        )
        print(f"已生成：{out_path.resolve()}（{len(item_rows)} 行）")


def cmd_run(args, config: AppConfig) -> None:
    model_name = args.model or config.default_model
    judge = build_judge(model_name, config)
    session = resolve_session(args.session)
    summaries, low_confidence = run_directory(
        args.input_dir,
        args.output_dir,
        session,
        args.date,
        judge,
        config,
        args.name_contains,
    )
    print(f"模型：{judge.name}")
    for summary in summaries:
        print(
            f"已生成：{summary.output_path}（{summary.total_rows} 行，"
            f"蓝 {summary.blue_rows} / 黄 {summary.yellow_rows}，候选 {summary.candidates}）"
        )
    if low_confidence:
        print("\n以下行置信度低于阈值，未自动标色，请人工复核：")
        for row in low_confidence:
            print(f"  {row.prepared.source_file} 第 {row.prepared.original_row_number} 行")
            print(f"    {row.prepared.combined_text[:120]}")
            print(
                f"    模型判定：{row.result.label.value}（置信度 {row.result.confidence:.2f}），"
                f"{row.result.reason}"
            )


def cmd_eval(args, config: AppConfig) -> None:
    samples = load_benchmark(args.benchmark)
    model_names = (
        [name.strip() for name in args.models.split(",") if name.strip()]
        if args.models
        else [config.default_model]
    )
    entries = []
    for name in model_names:
        judge = build_judge(name, config)
        metrics = evaluate(samples, judge)
        paths = save_run(args.artifacts_dir, name, metrics, args.benchmark, config.digest())
        entries.append((name, metrics))
        print(
            f"模型 {name}：样本 {metrics.total}，准确率 {metrics.accuracy:.4f}，"
            f"宏F1 {metrics.macro_f1:.4f}，负面召回 {metrics.negative_recall:.4f}，"
            f"负面精确率 {metrics.negative_precision:.4f}，误报率 {metrics.false_positive_rate:.4f}"
        )
        print(f"  运行 JSON：{paths['json']}")
        print(f"  评测报告：{paths['markdown']}")
        if metrics.errors:
            print(f"  Bad Case {len(metrics.errors)} 条：")
            for error in metrics.errors[:10]:
                print(
                    f"    {error['id']} gold={error['gold']} pred={error['pred']} "
                    f"note={error['note']}"
                )
    if len(entries) > 1:
        compare_path = Path(args.artifacts_dir) / "reports" / "model_compare.md"
        compare_path.parent.mkdir(parents=True, exist_ok=True)
        compare_path.write_text(render_compare(entries), encoding="utf-8")
        print(f"模型对比报告：{compare_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beer-sentiment",
        description="啤酒行业负面舆情双阶段筛选与评测管线",
    )
    parser.add_argument("--config-dir", default=None, help="配置目录，默认使用仓库 config/")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare", help="按时间窗生成待人工筛选 CSV")
    prepare_parser.add_argument("--input-dir", default="original")
    prepare_parser.add_argument("--name-contains", action="append")
    prepare_parser.add_argument("--output-dir", default=".")
    prepare_parser.add_argument("--session", choices=["morning", "afternoon"])
    prepare_parser.add_argument("--date")
    prepare_parser.add_argument("--time-column")
    prepare_parser.add_argument("--data-column")
    prepare_parser.add_argument("--text-column", action="append", dest="text_columns")
    prepare_parser.add_argument("--encoding", default="auto")

    build_parser_cmd = sub.add_parser("build", help="按人工筛选结果生成着色 Excel")
    build_parser_cmd.add_argument("--review-csv", required=True)
    build_parser_cmd.add_argument("--session", choices=["morning", "afternoon"], required=True)
    build_parser_cmd.add_argument("--output-dir", default="object")

    run_parser = sub.add_parser("run", help="端到端运行：粗筛 + 模型判定 + 着色 Excel")
    run_parser.add_argument("--input-dir", default="original")
    run_parser.add_argument("--output-dir", default="object")
    run_parser.add_argument("--session", choices=["morning", "afternoon"])
    run_parser.add_argument("--date")
    run_parser.add_argument("--model", default=None)
    run_parser.add_argument("--name-contains", action="append")

    eval_parser = sub.add_parser("eval", help="在 Benchmark 上评测模型并生成报告")
    eval_parser.add_argument("--benchmark", default="benchmark/beer_sentiment_benchmark.jsonl")
    eval_parser.add_argument(
        "--models",
        default=None,
        help="逗号分隔的模型名，例如 mock,deepseek-v4",
    )
    eval_parser.add_argument("--artifacts-dir", default="artifacts")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config_dir)
    if args.command == "prepare":
        cmd_prepare(args, config)
    elif args.command == "build":
        cmd_build(args, config)
    elif args.command == "run":
        cmd_run(args, config)
    elif args.command == "eval":
        cmd_eval(args, config)
    else:
        raise AssertionError(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
