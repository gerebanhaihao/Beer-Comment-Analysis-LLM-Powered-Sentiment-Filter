"""Load the human-labeled benchmark dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from beer_sentiment.models import Category, Label


@dataclass
class BenchmarkSample:
    """One human-labeled benchmark row."""

    id: str
    title: str
    text: str
    ocr_text: str
    label: Label
    category: Category
    brands: list[str]
    note: str

    @property
    def combined_text(self) -> str:
        parts = [self.title, self.text, self.ocr_text]
        return "\n".join(part for part in parts if part)


def load_benchmark(path: str | Path) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                samples.append(
                    BenchmarkSample(
                        id=str(data.get("id") or f"line_{line_no}"),
                        title=str(data.get("title") or ""),
                        text=str(data.get("text") or ""),
                        ocr_text=str(data.get("ocr_text") or ""),
                        label=Label.parse(data.get("label")),
                        category=Category(data.get("category") or "none"),
                        brands=list(data.get("brands") or []),
                        note=str(data.get("note") or ""),
                    )
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Benchmark 第 {line_no} 行解析失败：{exc}") from exc
    if not samples:
        raise ValueError(f"Benchmark 为空：{path}")
    return samples
