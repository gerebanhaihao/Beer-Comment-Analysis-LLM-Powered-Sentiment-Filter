"""Text cleaning and fuzzy brand matching."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterator

from beer_sentiment.config import AppConfig

try:
    from pypinyin import lazy_pinyin
except ImportError:  # pragma: no cover - optional dependency fallback
    lazy_pinyin = None


ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\ufeff"

OCR_NOISE_FIXES = {
    "狗兑": "勾兑",
    "勾对": "勾兑",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    for char in ZERO_WIDTH_CHARS:
        text = text.replace(char, "")
    return text.strip()


def normalize_ocr_noise(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", clean_text(text)).replace("　", " ")
    for wrong, right in OCR_NOISE_FIXES.items():
        text = text.replace(wrong, right)
    return text


def _compact(text: str) -> str:
    """Normalize text for matching while keeping Chinese and alphanumerics."""
    normalized = normalize_ocr_noise(text).lower()
    return re.sub(r"[\s\W_]+", "", normalized)


def _pinyin(text: str) -> str:
    if lazy_pinyin is None or not text:
        return ""
    return "".join(lazy_pinyin(text)).lower()


def _edit_distance(left: str, right: str, limit: int | None = None) -> int:
    """Levenshtein distance with an optional early-stop limit."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_char in enumerate(right, start=1):
        current = [right_index]
        row_min = current[0]
        for left_index, left_char in enumerate(left, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                current[-1] + 1,
                previous[left_index] + 1,
                previous[left_index - 1] + cost,
            )
            current.append(value)
            row_min = min(row_min, value)
        if limit is not None and row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _fuzzy_contains(text: str, term: str, max_distance: int, min_length: int) -> bool:
    if not term or len(term) < min_length:
        return False
    if term in text:
        return True
    if max_distance <= 0:
        return False
    if len(set(term) - set(text)) > max_distance:
        return False

    min_window = max(min_length, len(term) - max_distance)
    max_window = min(len(text), len(term) + max_distance)
    for window_length in range(min_window, max_window + 1):
        for start in range(0, len(text) - window_length + 1):
            window = text[start : start + window_length]
            if _edit_distance(window, term, max_distance) <= max_distance:
                return True
    return False


def _as_aliases(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [clean_text(item) for item in value if clean_text(item)]
    value = clean_text(value)
    return [value] if value else []


def _brand_terms(config: AppConfig) -> Iterator[tuple[str, str]]:
    for brand in config.all_brands():
        terms = [brand, *_as_aliases(config.brand_aliases.get(brand, []))]
        pinyin = _pinyin(brand)
        if pinyin:
            terms.append(pinyin)
        for term in terms:
            if term:
                yield brand, term


def _matches(term: str, compact_text: str, config: AppConfig) -> bool:
    normalized_term = _compact(term)
    if not normalized_term:
        return False
    if normalized_term in compact_text:
        return True
    matching = config.matching or {}
    if not bool(matching.get("fuzzy_enabled", True)):
        return False
    max_distance = int(matching.get("max_edit_distance", 1))
    min_length = int(matching.get("min_fuzzy_length", 2))
    return _fuzzy_contains(compact_text, normalized_term, max_distance, min_length)


def extract_brands(text: str, config: AppConfig) -> list[str]:
    """Return canonical brands found by alias, pinyin or fuzzy matching."""
    compact_text = _compact(text)
    brands: list[str] = []
    for brand, term in _brand_terms(config):
        if brand not in brands and _matches(term, compact_text, config):
            brands.append(brand)
    return brands
