"""Local-only smoke tests for the private, real benchmark."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from beer_sentiment.config import PROJECT_ROOT
from beer_sentiment.eval.benchmark import load_benchmark


def _real_benchmark_path() -> Path:
    configured = os.getenv("BEER_SENTIMENT_REAL_BENCHMARK")
    if not configured:
        return PROJECT_ROOT / "benchmark" / "beer_sentiment_benchmark_real.jsonl"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


REAL_BENCHMARK = _real_benchmark_path()

pytestmark = pytest.mark.skipif(
    not REAL_BENCHMARK.is_file(),
    reason="本地真实 Benchmark 不存在；GitHub/CI 仅测试仓库内的模拟数据",
)


def test_real_benchmark_loads_and_has_valid_records():
    samples = load_benchmark(REAL_BENCHMARK)

    assert samples
    assert len({sample.id for sample in samples}) == len(samples)
    assert all(sample.combined_text for sample in samples)
    assert {sample.label.value for sample in samples} <= {"blue", "yellow", "none"}
    assert {sample.category.value for sample in samples} <= {
        "own",
        "competitor",
        "industry",
        "none",
    }
