"""Deterministic offline model simulations for Benchmark comparisons.

These adapters intentionally do not call a remote API. They make the model
selection and reporting flow runnable before API credentials are available.
Every result is marked ``simulated=True`` and the report labels it as such.
When credentials become available, replace the model config type with
``openai_compatible`` instead of treating these numbers as real-model scores.
"""

from __future__ import annotations

import hashlib
from typing import Any

from beer_sentiment.config import AppConfig
from beer_sentiment.llm.base import Judge
from beer_sentiment.llm.mock import MockJudge
from beer_sentiment.models import JudgeResult, Label
from beer_sentiment.rules.normalize import normalize_ocr_noise


class SimulatedJudge(Judge):
    """A deterministic, profile-driven approximation of a remote LLM."""

    def __init__(self, name: str, model_config: dict[str, Any], config: AppConfig) -> None:
        self.name = name
        self.model_config = model_config
        self.config = config
        self.profile = model_config.get("simulation", {})
        self._baseline = MockJudge(config)

    def judge(self, sample: str, context: str = "") -> JudgeResult:
        baseline = self._baseline.judge(sample, context=context)
        normalized = normalize_ocr_noise(sample)
        label = baseline.label
        reason = baseline.reason

        # Model-specific treatment of ambiguous keyword hits. This mirrors a
        # common production difference: conservative models abstain, while
        # aggressive models trade precision for recall.
        if baseline.confidence < float(self.profile.get("ambiguity_threshold", 0.6)):
            policy = self.profile.get("ambiguity_policy", "balanced")
            if policy == "aggressive" and label == Label.NONE:
                label = Label.YELLOW
                reason = "模拟模型将弱负面候选判为行业/竞品负面"
            elif policy == "conservative":
                label = Label.NONE
                reason = "模拟模型对弱负面候选选择保守不标"

        # Deterministic synthetic error rate. It is deliberately hash-based so
        # repeated evaluations produce identical results and can be compared.
        noise_rate = max(0.0, min(1.0, float(self.profile.get("noise_rate", 0.0))))
        seed = str(self.profile.get("seed", self.name))
        digest = hashlib.sha256(f"{seed}\n{normalized}".encode("utf-8")).hexdigest()
        trigger = int(digest[:8], 16) / 0xFFFFFFFF
        if trigger < noise_rate:
            if label == Label.NONE:
                label = Label.YELLOW
                reason = "模拟模型噪声：产生一次可复现的误报"
            else:
                label = Label.NONE
                reason = "模拟模型噪声：产生一次可复现的漏报"

        confidence_scale = float(self.profile.get("confidence_scale", 1.0))
        confidence = max(0.0, min(1.0, baseline.confidence * confidence_scale))
        return JudgeResult(
            label=label,
            confidence=confidence,
            reason=f"[模拟] {reason}",
            brands=baseline.brands,
            model=self.name,
            latency_ms=float(self.profile.get("latency_ms", 0.0)),
            cost_usd=float(self.profile.get("cost_usd_per_sample", 0.0)),
            simulated=True,
        )
