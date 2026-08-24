import pytest

from beer_sentiment.eval.metrics import compute_metrics
from beer_sentiment.models import Label


def test_compute_metrics():
    gold = [Label.BLUE, Label.BLUE, Label.YELLOW, Label.NONE, Label.NONE]
    pred = [Label.BLUE, Label.YELLOW, Label.YELLOW, Label.NONE, Label.NONE]
    metrics = compute_metrics(gold, pred)
    assert metrics.total == 5
    assert metrics.accuracy == pytest.approx(0.8)
    assert metrics.negative_precision == 1.0
    assert metrics.negative_recall == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.false_negative_rate == 0.0


def test_false_positive_and_miss():
    gold = [Label.NONE, Label.BLUE, Label.YELLOW, Label.YELLOW]
    pred = [Label.BLUE, Label.BLUE, Label.YELLOW, Label.NONE]
    metrics = compute_metrics(gold, pred)
    assert metrics.false_positive_rate == pytest.approx(1.0)
    assert metrics.false_negative_rate == pytest.approx(1 / 3)
