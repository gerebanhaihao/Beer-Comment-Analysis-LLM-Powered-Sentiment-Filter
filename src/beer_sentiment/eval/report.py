"""Experiment run artifacts and Markdown reports."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from beer_sentiment.eval.metrics import EvalMetrics


def render_confusion(confusion: dict[str, dict[str, int]]) -> str:
    labels = ["blue", "yellow", "none"]
    lines = [
        "| 真实 \\ 预测 | blue | yellow | none |",
        "| --- | --- | --- | --- |",
    ]
    for truth in labels:
        cells = " | ".join(str(confusion[truth][pred]) for pred in labels)
        lines.append(f"| {truth} | {cells} |")
    return "\n".join(lines)


def render_markdown(
    metrics: EvalMetrics,
    model_name: str,
    benchmark_path: str | Path,
) -> str:
    lines = [
        f"# Benchmark 评测：{model_name}",
        "",
        f"- 数据集：`{benchmark_path}`",
        f"- 样本数：{metrics.total}",
        "",
        "## 指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 准确率 | {metrics.accuracy:.4f} |",
        f"| 宏平均 F1 | {metrics.macro_f1:.4f} |",
        f"| 负面检测精确率 | {metrics.negative_precision:.4f} |",
        f"| 负面检测召回率 | {metrics.negative_recall:.4f} |",
        f"| 负面检测 F1 | {metrics.negative_f1:.4f} |",
        f"| 误报率 | {metrics.false_positive_rate:.4f} |",
        f"| 漏报率 | {metrics.false_negative_rate:.4f} |",
        f"| 平均延迟 (ms) | {metrics.avg_latency_ms:.2f} |",
        f"| 总成本 (USD) | {metrics.total_cost_usd:.6f} |",
        "",
        "## 混淆矩阵",
        "",
        render_confusion(metrics.confusion),
        "",
        "## Bad Case",
        "",
    ]
    if metrics.errors:
        for error in metrics.errors:
            lines.append(
                f"- `{error['id']}`：gold=`{error['gold']}`，pred=`{error['pred']}`，"
                f"置信度 `{error['confidence']}`，备注：{error['note']}"
            )
    else:
        lines.append("无")
    return "\n".join(lines)


def save_run(
    artifacts_dir: str | Path,
    model_name: str,
    metrics: EvalMetrics,
    benchmark_path: str | Path,
    config_digest: str,
    timestamp: str | None = None,
) -> dict[str, Path]:
    run_dir = Path(artifacts_dir) / "runs"
    report_dir = Path(artifacts_dir) / "reports"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9_-]", "-", model_name)
    json_path = run_dir / f"{stamp}_{safe_model}.json"
    md_path = report_dir / f"{stamp}_{safe_model}.md"
    payload = {
        "model": model_name,
        "benchmark": str(benchmark_path),
        "config_digest": config_digest,
        "timestamp": stamp,
        "metrics": metrics.to_dict(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(metrics, model_name, benchmark_path),
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": md_path}


def render_compare(entries: list[tuple[str, EvalMetrics]]) -> str:
    lines = [
        "# 模型对比",
        "",
        "| 模型 | 准确率 | 宏F1 | 负面召回 | 负面精确 | 误报率 | 平均延迟ms | 成本USD |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, metrics in entries:
        lines.append(
            f"| {name} | {metrics.accuracy:.4f} | {metrics.macro_f1:.4f} | "
            f"{metrics.negative_recall:.4f} | {metrics.negative_precision:.4f} | "
            f"{metrics.false_positive_rate:.4f} | {metrics.avg_latency_ms:.2f} | "
            f"{metrics.total_cost_usd:.6f} |"
        )
    return "\n".join(lines)
