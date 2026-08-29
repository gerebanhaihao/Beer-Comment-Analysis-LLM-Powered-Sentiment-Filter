"""Stage 2: semantic judgment with low-confidence handling."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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
        max_workers = int(self.config.stage2.get("max_workers", 1))
        judged: list[JudgedRow] = []
        low_confidence: list[JudgedRow] = []

        def _judge_one(prepared: PreparedRow) -> JudgeResult:
            return judge.judge(prepared.combined_text)

        candidates = [row for row in prepared_rows if row.stage1.is_candidate]
        if len(candidates) > 1 and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                candidate_results = list(executor.map(_judge_one, candidates))
        else:
            candidate_results = [_judge_one(row) for row in candidates]
        result_by_id = {
            id(prepared): result for prepared, result in zip(candidates, candidate_results)
        }

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

            result = result_by_id[id(prepared)]
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
