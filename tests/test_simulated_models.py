from beer_sentiment.eval.benchmark import load_benchmark
from beer_sentiment.eval.metrics import evaluate
from beer_sentiment.llm.simulated import SimulatedJudge


def test_simulated_model_is_marked_and_reports_latency_cost(config):
    judge = SimulatedJudge(
        "qwen-max",
        {
            "type": "simulated",
            "simulation": {
                "latency_ms": 123,
                "cost_usd_per_sample": 0.01,
                "seed": "test",
            },
        },
        config,
    )
    result = judge.judge("百威太难喝了")
    assert result.simulated is True
    assert result.latency_ms == 123
    assert result.cost_usd == 0.01


def test_benchmark_loader_accepts_converted_record(tmp_path, config):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        '{"id":"x1","title":"百威","text":"太难喝","cover_ocr":"","content_ocr":"投诉",'
        '"ocr_text":"投诉","combined_text":"百威\\n太难喝\\n投诉","label":"blue",'
        '"category":"own","brands":[],"note":"excel"}\n',
        encoding="utf-8",
    )
    samples = load_benchmark(path)
    assert samples[0].combined_text == "百威\n太难喝\n投诉"
    assert samples[0].source_file == ""
    metrics = evaluate(samples, SimulatedJudge("deepseek-v4", {"simulation": {}}, config))
    assert metrics.mode == "simulated"
