import json

from beer_sentiment.config import PROJECT_ROOT
from beer_sentiment.eval.benchmark import load_benchmark
from beer_sentiment.eval.metrics import evaluate
from beer_sentiment.eval.report import save_run


BENCHMARK = PROJECT_ROOT / "benchmark" / "beer_sentiment_benchmark.jsonl"


def test_load_benchmark():
    samples = load_benchmark(BENCHMARK)
    assert len(samples) >= 20


def test_evaluate_mock(config, mock_judge):
    samples = load_benchmark(BENCHMARK)
    metrics = evaluate(samples, mock_judge)
    assert metrics.total == len(samples)
    assert metrics.accuracy > 0.7
    assert isinstance(metrics.errors, list)


def test_save_run(tmp_path, config, mock_judge):
    samples = load_benchmark(BENCHMARK)
    metrics = evaluate(samples, mock_judge)
    paths = save_run(
        tmp_path,
        "mock",
        metrics,
        BENCHMARK,
        config.digest(),
        timestamp="20260824_120000",
    )
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["metrics"]["total"] == len(samples)
