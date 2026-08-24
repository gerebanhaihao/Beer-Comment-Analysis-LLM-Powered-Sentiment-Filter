from beer_sentiment.cli import main
from beer_sentiment.config import PROJECT_ROOT


def test_cli_eval_mock(tmp_path, config):
    benchmark = PROJECT_ROOT / "benchmark" / "beer_sentiment_benchmark.jsonl"
    main(
        [
            "--config-dir",
            str(config.config_dir),
            "eval",
            "--model",
            "mock",
            "--benchmark",
            str(benchmark),
            "--artifacts-dir",
            str(tmp_path),
        ]
    )
    assert list(tmp_path.rglob("*.json"))
