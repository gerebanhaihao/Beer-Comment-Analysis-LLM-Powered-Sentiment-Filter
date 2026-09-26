from beer_sentiment.gui import MODELS, command_args


def values(**updates):
    result = {
        "root": ".",
        "input": "data",
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


def test_eval_arguments_disable_unbundled_rag():
    args = command_args("eval", values(model="Kimi"))
    assert args[args.index("--models") + 1] == "kimi"
    assert "--no-rag" in args


def test_ingest_arguments_can_move_files():
    args = command_args("ingest", values(input="incoming", move=True))
    assert args[-1] == "--move"
    assert "--data-dir" in args
