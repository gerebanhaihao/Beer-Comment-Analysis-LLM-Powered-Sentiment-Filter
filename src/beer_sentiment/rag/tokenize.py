"""共享分词器：中文按字符 n-gram 切分，英文/数字按连续串切分。"""

from __future__ import annotations

import re

from beer_sentiment.rules.normalize import clean_text, normalize_ocr_noise

_RUN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+")


def tokenize(text: str, ngram: int = 2) -> list[str]:
    """把文本切成 n-gram token，供 BM25 与稠密向量共同使用。

    中文按字符 bigram（对无空格分词场景友好），英文与数字按连续串。
    """
    normalized = normalize_ocr_noise(clean_text(text)).lower()
    tokens: list[str] = []
    for run in _RUN_PATTERN.findall(normalized):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[i : i + ngram] for i in range(len(run) - ngram + 1))
        # 保留完整英文/数字词，增强召回
        if run.isascii():
            tokens.append(run)
    return tokens
