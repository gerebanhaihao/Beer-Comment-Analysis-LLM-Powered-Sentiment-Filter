"""Prompt template loading and rendering."""

from __future__ import annotations

from pathlib import Path

from beer_sentiment.config import PROJECT_ROOT, AppConfig


DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "judge_v1.txt"


def load_prompt(path: str | Path | None = None) -> str:
    prompt_path = Path(path) if path else DEFAULT_PROMPT_PATH
    return prompt_path.read_text(encoding="utf-8")


def render_system_prompt(config: AppConfig, template: str) -> str:
    replacements = {
        "{{own_brands}}": "、".join(config.own_brands),
        "{{competitor_brands}}": "、".join(config.competitor_brands),
        "{{negative_keywords}}": "、".join(config.negative_keywords),
        "{{association_keywords}}": "、".join(config.association_keywords),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_messages(
    sample: str,
    config: AppConfig,
    context: str = "",
    prompt_path: str | Path | None = None,
) -> list[dict[str, str]]:
    system = render_system_prompt(config, load_prompt(prompt_path))
    user = "请判断以下帖子是否属于啤酒负面舆情。\n\n"
    if context:
        user += f"参考上下文：\n{context}\n\n"
    user += f"帖子内容：\n{sample}\n\n只输出 JSON 对象。"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
