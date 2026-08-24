"""Judge protocol shared by all model backends."""

from __future__ import annotations

from typing import Protocol

from beer_sentiment.models import JudgeResult


class Judge(Protocol):
    """A semantic judge that labels one combined text row."""

    name: str

    def judge(self, sample: str, context: str = "") -> JudgeResult:
        ...
