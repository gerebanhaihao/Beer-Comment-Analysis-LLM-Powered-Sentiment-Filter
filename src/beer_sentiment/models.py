"""Shared data models for the beer sentiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Label(str, Enum):
    """Final sentiment label applied to an Excel row."""

    BLUE = "blue"
    YELLOW = "yellow"
    NONE = "none"

    @classmethod
    def parse(cls, value: Any) -> "Label":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.NONE
        text = str(value).strip().lower().replace(" ", "")
        if text in {"blue", "蓝", "蓝色", "本品"}:
            return cls.BLUE
        if text in {"yellow", "黄", "黄色", "竞品", "行业"}:
            return cls.YELLOW
        if text in {"", "none", "不标", "无", "null"}:
            return cls.NONE
        raise ValueError(f"无法识别的标签: {value!r}")


class Category(str, Enum):
    """Benchmark sample category."""

    OWN = "own"
    COMPETITOR = "competitor"
    INDUSTRY = "industry"
    NONE = "none"


@dataclass
class JudgeResult:
    """Structured result returned by a judge model."""

    label: Label
    confidence: float
    reason: str
    brands: list[str] = field(default_factory=list)
    model: str = "unknown"
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "brands": self.brands,
            "reason": self.reason,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class Stage1Result:
    """Rule-based coarse filter output."""

    is_candidate: bool
    score: int = 0
    matched_keywords: list[str] = field(default_factory=list)
    matched_association_keywords: list[str] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)
    hint_label: Label | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_candidate": self.is_candidate,
            "score": self.score,
            "matched_keywords": self.matched_keywords,
            "matched_association_keywords": self.matched_association_keywords,
            "brands": self.brands,
            "hint_label": self.hint_label.value if self.hint_label else None,
        }


@dataclass
class PreparedRow:
    """A time-window row prepared for Stage 2 judgment."""

    row: dict[str, Any]
    source_file: str
    original_row_number: int
    combined_text: str
    stage1: Stage1Result


@dataclass
class JudgedRow:
    """A prepared row after Stage 2 judgment."""

    prepared: PreparedRow
    result: JudgeResult
    low_confidence: bool = False


@dataclass
class RunSummary:
    """Summary of one source file processed end to end."""

    source_file: str
    output_path: str
    output_class: str
    total_rows: int
    candidates: int
    blue_rows: int
    yellow_rows: int
    low_confidence_rows: list["JudgedRow"] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
