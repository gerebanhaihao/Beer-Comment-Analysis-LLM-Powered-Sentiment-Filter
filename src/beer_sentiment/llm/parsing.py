"""Parse and validate structured JSON output from judge models."""

from __future__ import annotations

import json
import re
from typing import Any

from beer_sentiment.models import Label


def parse_judge_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("模型输出不是 JSON 对象")
    label = Label.parse(data.get("label"))
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    brands = data.get("brands") or []
    if isinstance(brands, str):
        brands = [brands]
    reason = str(data.get("reason") or "")
    return {
        "label": label,
        "confidence": confidence,
        "brands": [str(brand) for brand in brands],
        "reason": reason,
    }
