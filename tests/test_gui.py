from beer_sentiment.gui import MODELS, command_args, low_confidence_rows
from beer_sentiment.models import JudgedRow, JudgeResult, Label, PreparedRow, Stage1Result


def values(**updates):
    result = {
        "root": ".",
        "input": "data",
        "source": "source.csv",
        "data": "data",
        "output": "output",
        "benchmark": "benchmark/test.jsonl",
        "session": "上午",
        "date": "2026-09-26",
        "model": "DeepSeek",
        "all_time": True,
        "move": False,
    }
    result.update(updates)
    return result


def test_desktop_models_use_separate_api_key_environment_variables():
    assert set(MODELS) == {"deepseek", "qwen", "kimi"}
    assert len({item["env"] for item in MODELS.values()}) == 3


def test_desktop_models_are_real_openai_compatible_providers(config):
    for name in MODELS:
        provider = config.model_config(name)
        assert provider["type"] == "openai_compatible"
        assert provider["api_key_env"] == MODELS[name]["env"]


def test_run_arguments_select_real_model_and_disable_unbundled_rag():
    args = command_args("run", values(model="Qwen"))
    assert args[args.index("--model") + 1] == "qwen"
    assert "--all-time" in args
    assert "--no-rag" in args


def test_run_arguments_include_only_selected_files():
    args = command_args(
        "run",
        values(run_files=["data/quark__2026-08-31 114223.csv", "data/quark__2026-08-31 113949.csv"]),
    )
    assert args.count("--input-file") == 2
    assert args[args.index("--input-file") + 1] == "data/quark__2026-08-31 114223.csv"


def test_eval_arguments_disable_unbundled_rag():
    args = command_args("eval", values(model="Kimi"))
    assert args[args.index("--models") + 1] == "kimi"
    assert "--no-rag" in args


def test_ingest_arguments_can_move_files():
    args = command_args("ingest", values(source="source.csv", move=True))
    assert args[-1] == "--move"
    assert "--data-dir" in args
    assert args[args.index("--input-dir") + 1] == "source.csv"
    assert args[args.index("--data-dir") + 1] == "data"


def test_ingest_arguments_support_multiple_selected_files():
    args = command_args(
        "ingest", values(source_files=["first.csv", "second.csv"])
    )
    assert "--input-dir" not in args
    assert args.count("--input-file") == 2
    assert args[args.index("--data-dir") + 1] == "data"


def test_review_includes_low_confidence_colored_and_uncolored_rows():
    rows = [
        JudgedRow(
            prepared=PreparedRow({}, "source.csv", line, "text", Stage1Result(True)),
            result=JudgeResult(label, 0.2, "uncertain"),
            low_confidence=low,
        )
        for line, label, low in (
            (2, Label.BLUE, True),
            (3, Label.YELLOW, True),
            (4, Label.NONE, True),
            (5, Label.YELLOW, False),
        )
    ]
    assert [row.prepared.original_row_number for row in low_confidence_rows(rows)] == [2, 3, 4]
