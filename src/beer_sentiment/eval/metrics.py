"""Classification metrics and confusion matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from beer_sentiment.llm.base import Judge
from beer_sentiment.models import JudgeResult, Label
from beer_sentiment.eval.benchmark import BenchmarkSample


LABELS = ["blue", "yellow", "none"]


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvalMetrics:
    total: int
    accuracy: float
    macro_f1: float
    negative_precision: float
    negative_recall: float
    negative_f1: float
    false_positive_rate: float
    false_negative_rate: float
    per_class: dict[str, ClassMetrics]
    confusion: dict[str, dict[str, int]]
    avg_latency_ms: float
    total_cost_usd: float
    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "negative_precision": round(self.negative_precision, 4),
            "negative_recall": round(self.negative_recall, 4),
            "negative_f1": round(self.negative_f1, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "per_class": {
                label: {
                    "precision": round(metric.precision, 4),
                    "recall": round(metric.recall, 4),
                    "f1": round(metric.f1, 4),
                    "support": metric.support,
                }
                for label, metric in self.per_class.items()
            },
            "confusion": self.confusion,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "error_count": len(self.errors),
            "errors": self.errors,
        }


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(
    gold_labels: list[Label],
    pred_labels: list[Label],
    results: list[JudgeResult] | None = None,
) -> EvalMetrics:
    confusion = {truth: {pred: 0 for pred in LABELS} for truth in LABELS}
    for gold, pred in zip(gold_labels, pred_labels):
        confusion[gold.value][pred.value] += 1

    total = len(gold_labels)
    correct = 0
    per_class: dict[str, ClassMetrics] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[truth][label] for truth in LABELS if truth != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            support=support,
        )
        correct += tp

    accuracy = correct / total if total else 0.0
    scored = [per_class[label].f1 for label in LABELS if per_class[label].support > 0]
    macro_f1 = sum(scored) / len(scored) if scored else 0.0

    tp = sum(
        confusion[truth][pred]
        for truth in LABELS
        for pred in LABELS
        if truth != "none" and pred != "none"
    )
    fp = sum(confusion["none"][pred] for pred in LABELS if pred != "none")
    fn = sum(confusion[truth]["none"] for truth in LABELS if truth != "none")
    tn = confusion["none"]["none"]
    negative_precision = tp / (tp + fp) if tp + fp else 0.0
    negative_recall = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0

    latency_ms = [result.latency_ms for result in (results or [])]
    cost_usd = [result.cost_usd for result in (results or [])]
    avg_latency = sum(latency_ms) / len(latency_ms) if latency_ms else 0.0

    return EvalMetrics(
        total=total,
        accuracy=accuracy,
        macro_f1=macro_f1,
        negative_precision=negative_precision,
        negative_recall=negative_recall,
        negative_f1=_f1(negative_precision, negative_recall),
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        per_class=per_class,
        confusion=confusion,
        avg_latency_ms=avg_latency,
        total_cost_usd=sum(cost_usd),
        errors=[],
    )


def evaluate(
    samples: list[BenchmarkSample],
    judge: Judge,
) -> EvalMetrics:
    gold_labels = [sample.label for sample in samples]
    pred_labels: list[Label] = []
    results: list[JudgeResult] = []
    errors: list[dict[str, Any]] = []
    for sample in samples:
        result = judge.judge(sample.combined_text)
        results.append(result)
        pred_labels.append(result.label)
        if result.label != sample.label:
            errors.append(
                {
                    "id": sample.id,
                    "gold": sample.label.value,
                    "pred": result.label.value,
                    "confidence": round(result.confidence, 4),
                    "text": sample.combined_text,
                    "note": sample.note,
                }
            )
    metrics = compute_metrics(gold_labels, pred_labels, results)
    metrics.errors = errors
    return metrics
