from beer_sentiment.llm.parsing import parse_judge_json
from beer_sentiment.models import Label


def test_parse_judge_json():
    result = parse_judge_json(
        '```json\n{"label": "blue", "confidence": 0.9, "brands": ["百威"], "reason": "明确负面"}\n```'
    )
    assert result["label"] == Label.BLUE
    assert result["confidence"] == 0.9
    assert result["brands"] == ["百威"]


def test_parse_clamps_confidence():
    result = parse_judge_json('{"label": "none", "confidence": 2.5}')
    assert result["confidence"] == 1.0
