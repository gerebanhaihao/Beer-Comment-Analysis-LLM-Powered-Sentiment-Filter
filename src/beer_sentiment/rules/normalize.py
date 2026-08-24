"""Text cleaning, OCR noise normalization, and brand extraction."""

from __future__ import annotations

from typing import Any

from beer_sentiment.config import AppConfig


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
    text = clean_text(text).replace("　", " ")
    for wrong, right in OCR_NOISE_FIXES.items():
        text = text.replace(wrong, right)
    return text


def extract_brands(text: str, config: AppConfig) -> list[str]:
    normalized = normalize_ocr_noise(text).lower()
    found: list[str] = []
    for brand in config.all_brands():
        if brand in found:
            continue
        keys = [brand, *config.brand_aliases.get(brand, [])]
        if any(key.lower() in normalized for key in keys):
            found.append(brand)
    return found
