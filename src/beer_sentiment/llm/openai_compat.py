"""OpenAI-compatible chat judge used by DeepSeek/Qwen/Kimi style endpoints."""

from __future__ import annotations

import os
import time
from typing import Any

from beer_sentiment.config import AppConfig
from beer_sentiment.llm.base import Judge
from beer_sentiment.llm.parsing import parse_judge_json
from beer_sentiment.llm.prompts import build_messages
from beer_sentiment.models import JudgeResult


def create_client(model_config: dict[str, Any]) -> Any:
    """按模型配置创建 OpenAI 兼容客户端（供判定与重排序共用）。"""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 依赖，请安装：pip install 'beer-sentiment[llm]'") from exc
    key_env = model_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(key_env)
    if not api_key:
        raise RuntimeError(f"缺少环境变量 {key_env}")
    base_url = model_config.get("base_url") or os.getenv("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


class OpenAICompatJudge(Judge):
    """Judge backed by any OpenAI-compatible chat completions API."""

    def __init__(
        self,
        name: str,
        model_config: dict[str, Any],
        app_config: AppConfig,
        prompt_path: str | None = None,
    ) -> None:
        self.name = name
        self.model_config = model_config
        self.config = app_config
        self.prompt_path = prompt_path
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            self._client = create_client(self.model_config)
        return self._client

    def judge(self, sample: str, context: str = "") -> JudgeResult:
        messages = build_messages(
            sample,
            self.config,
            context=context,
            prompt_path=self.prompt_path,
        )
        temperature = float(self.config.stage2.get("temperature", 0))
        retries = int(self.config.stage2.get("max_retries", 2))
        last_error: Exception | None = None
        for _ in range(retries + 1):
            started = time.perf_counter()
            try:
                response = self._get_client().chat.completions.create(
                    model=self.model_config.get("model", self.name),
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                latency_ms = (time.perf_counter() - started) * 1000
                content = response.choices[0].message.content or ""
                parsed = parse_judge_json(content)
                return JudgeResult(
                    label=parsed["label"],
                    confidence=parsed["confidence"],
                    brands=parsed["brands"],
                    reason=parsed["reason"],
                    model=self.name,
                    latency_ms=latency_ms,
                    cost_usd=self._estimate_cost(getattr(response, "usage", None)),
                    raw=content,
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"{self.name} 判定失败: {last_error}") from last_error

    def _estimate_cost(self, usage) -> float:
        if usage is None:
            return 0.0
        input_price = float(self.model_config.get("input_price_per_million", 0))
        output_price = float(self.model_config.get("output_price_per_million", 0))
        return (
            usage.prompt_tokens * input_price + usage.completion_tokens * output_price
        ) / 1_000_000
