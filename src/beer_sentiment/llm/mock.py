"""Deterministic mock judge that encodes the skill rules as heuristics."""

from __future__ import annotations

from beer_sentiment.config import AppConfig
from beer_sentiment.llm.base import Judge
from beer_sentiment.models import JudgeResult, Label
from beer_sentiment.rules.classify import Stage1Classifier
from beer_sentiment.rules.normalize import normalize_ocr_noise


class MockJudge(Judge):
    """Rule-based stand-in used for tests, CI, and demos without an API key."""

    name = "mock"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._classifier = Stage1Classifier(config)

    def judge(self, sample: str, context: str = "") -> JudgeResult:
        normalized = normalize_ocr_noise(sample)
        stage1 = self._classifier.classify(normalized)
        label, confidence, reason = self._decide(normalized, stage1)
        return JudgeResult(
            label=label,
            confidence=confidence,
            reason=reason,
            brands=stage1.brands,
            model=self.name,
        )

    def _has_strong_phrase(self, text: str) -> bool:
        phrases = self.config.stage1.get("strong_negative_phrases", [])
        return any(phrase in text for phrase in phrases)

    def _is_industry_context(self, text: str) -> bool:
        return any(token in text for token in ("行业", "整体", "大盘", "市场"))

    def _decide(self, normalized: str, stage1):
        if any(
            token in normalized
            for token in ("教你", "教大家", "教学", "新国标", "别被骗", "识破", "科普")
        ):
            return Label.NONE, 0.85, "教学/辟谣类内容，命中关键词不标"
        if any(
            token in normalized
            for token in ("我们", "我家", "自家", "原产地酿造", "精酿活啤酒")
        ) and ("勾兑" in normalized or "工业" in normalized):
            return Label.NONE, 0.8, "商家对比宣传，不标"
        if any(token in normalized for token in ("过度解读", "理性讨论", "纯属")):
            return Label.NONE, 0.8, "个人体验/理性讨论，不标"
        if any(token in normalized for token in ("仿冒", "侵权", "浮雕商标")):
            return Label.NONE, 0.85, "第三方仿冒/侵权，不标本品负面"
        if "回到过去" in normalized or ("过去" in normalized and "工业啤酒" in normalized):
            return Label.NONE, 0.8, "回忆过去，不标"

        if self._has_strong_phrase(normalized):
            own = [brand for brand in stage1.brands if brand in self.config.own_brands]
            competitor = [
                brand
                for brand in stage1.brands
                if brand in self.config.competitor_brands
            ]
            if own and not self._is_industry_context(normalized):
                return Label.BLUE, 0.9, "明确本品负面"
            if competitor:
                return Label.YELLOW, 0.88, "明确竞品负面"
            if self._is_industry_context(normalized):
                return Label.YELLOW, 0.8, "行业整体负面"
            return Label.YELLOW, 0.7, "明确负面但未指向本品"

        if stage1.is_candidate:
            if "吗" in normalized or normalized.rstrip("。 ").endswith("？"):
                return Label.NONE, 0.5, "疑问句未明确表达负面"
            return Label.YELLOW, 0.55, "命中关键词但语义待确认"

        return Label.NONE, 1.0, "粗筛未命中"
