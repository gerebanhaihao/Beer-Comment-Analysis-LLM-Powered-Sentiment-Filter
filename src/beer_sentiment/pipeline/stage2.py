"""Stage 2: semantic judgment with low-confidence handling."""

from __future__ import annotations

from beer_sentiment.config import AppConfig
from beer_sentiment.llm.base import Judge
from beer_sentiment.models import JudgeResult, JudgedRow, Label, PreparedRow


class Stage2Pipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def judge(
        self,
        prepared_rows: list[PreparedRow],
        judge: Judge,
    ) -> tuple[list[JudgedRow], list[JudgedRow]]:
        threshold = float(self.config.stage2.get("low_confidence_threshold", 0.6))
        use_fallback = bool(self.config.stage2.get("fallback_to_stage1", False))
        judged: list[JudgedRow] = []
        low_confidence: list[JudgedRow] = []

        for prepared in prepared_rows:
            if not prepared.stage1.is_candidate:
                result = JudgeResult(
                    label=Label.NONE,
                    confidence=1.0,
                    reason="粗筛未命中，直接跳过",
                    model=judge.name,
                )
                judged.append(JudgedRow(prepared=prepared, result=result))
                continue

            result = judge.judge(prepared.combined_text)
            is_low = result.confidence < threshold
            if is_low and use_fallback and prepared.stage1.hint_label is not None:
                result = JudgeResult(
                    label=prepared.stage1.hint_label,
                    confidence=max(result.confidence, 0.5),
                    brands=result.brands,
                    reason=f"{result.reason}；低置信度回退至规则提示",
                    model=result.model,
                    latency_ms=result.latency_ms,
                    cost_usd=result.cost_usd,
                )
                is_low = False

            judged_row = JudgedRow(
                prepared=prepared,
                result=result,
                low_confidence=is_low,
            )
            judged.append(judged_row)
            if is_low:
                low_confidence.append(judged_row)

        return judged, low_confidence
