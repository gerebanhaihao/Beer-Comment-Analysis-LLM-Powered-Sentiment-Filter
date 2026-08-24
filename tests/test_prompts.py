from beer_sentiment.llm.prompts import build_messages


def test_build_messages(config):
    messages = build_messages("百威太难喝", config)
    assert messages[0]["role"] == "system"
    assert "百威" in messages[0]["content"]
    assert "百威太难喝" in messages[1]["content"]
