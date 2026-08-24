"""Stage 1: keyword/association based candidate filtering."""

from __future__ import annotations

from beer_sentiment.config import AppConfig
from beer_sentiment.models import Label, Stage1Result
from beer_sentiment.rules.normalize import extract_brands, normalize_ocr_noise


class Stage1Classifier:
    """Coarse filter that decides which rows need semantic judgment."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @property
    def strong_phrases(self) -> list[str]:
        return self.config.stage1.get("strong_negative_phrases", [])

    def classify(self, combined_text: str) -> Stage1Result:
        normalized = normalize_ocr_noise(combined_text)
        matched_negative = [
            keyword for keyword in self.config.negative_keywords if keyword in normalized
        ]
        matched_association = [
            keyword
            for keyword in self.config.association_keywords
            if keyword in normalized
        ]
        brands = extract_brands(normalized, self.config)
        min_score = int(self.config.stage1.get("candidate_min_score", 1))
        score = len(matched_negative) * 2 + len(matched_association)
        is_candidate = score >= min_score
        hint_label = (
            self._hint_label(normalized, brands, matched_negative)
            if is_candidate
            else None
        )
        return Stage1Result(
            is_candidate=is_candidate,
            score=score,
            matched_keywords=matched_negative,
            matched_association_keywords=matched_association,
            brands=brands,
            hint_label=hint_label,
        )

    def _is_industry_context(self, text: str) -> bool:
        return any(token in text for token in ("行业", "整体", "大盘", "市场"))

    def _hint_label(
        self,
        normalized: str,
        brands: list[str],
        matched_negative: list[str],
    ) -> Label | None:
        strong_hit = any(phrase in normalized for phrase in self.strong_phrases)
        own_hit = any(brand in self.config.own_brands for brand in brands)
        competitor_hit = any(brand in self.config.competitor_brands for brand in brands)
        if own_hit and strong_hit and not self._is_industry_context(normalized):
            return Label.BLUE
        if competitor_hit and strong_hit:
            return Label.YELLOW
        if matched_negative:
            return Label.YELLOW
        return None
