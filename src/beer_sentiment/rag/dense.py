"""稠密召回：字符 n-gram 特征哈希（Feature Hashing）稠密向量 + 余弦相似度。

无外部模型依赖：把 token 哈希进固定维度向量，再 L2 归一化，
对中文短文本的语义邻近检索足够稳定，且可离线确定性复现。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def embed(tokens: list[str], dim: int = 256) -> list[float]:
    """token 序列 -> 归一化稠密向量（亚线性词频 + L2 归一化）。"""
    vector = [0.0] * dim
    if not tokens:
        return vector
    counter = Counter(tokens)
    for token, freq in counter.items():
        weight = 1.0 + math.log(freq)
        vector[_hash_token(token, dim)] += weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class DenseIndex:
    """预计算知识库条目向量，检索时与查询向量算余弦相似度。"""

    def __init__(self, docs: list[list[str]], dim: int = 256) -> None:
        self.dim = dim
        self.vectors = [embed(doc, dim) for doc in docs]

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        query = embed(query_tokens, self.dim)
        scored = [
            (index, cosine(query, vector))
            for index, vector in enumerate(self.vectors)
        ]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
