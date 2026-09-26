"""Semantic dense retrieval backed by a sentence embedding model."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=4)
def load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("RAG 需要 sentence-transformers；请安装 beer-sentiment[rag]") from exc
    return SentenceTransformer(model_name)


def _vectors(values: Any) -> list[list[float]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [[float(value) for value in vector] for vector in values]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding 维度不一致")
    return sum(x * y for x, y in zip(a, b))


class DenseIndex:
    """Embed knowledge entries once, then compare embedded queries by cosine similarity."""

    def __init__(self, docs: list[str], model: Any) -> None:
        self.model = model
        self.vectors = _vectors(model.encode(docs, normalize_embeddings=True)) if docs else []
        if len(self.vectors) != len(docs) or any(
            not vector or not all(math.isfinite(x) for x in vector) for vector in self.vectors
        ):
            raise ValueError("Embedding 模型返回了无效的知识库向量")
        if self.vectors and any(len(vector) != len(self.vectors[0]) for vector in self.vectors):
            raise ValueError("Embedding 模型返回的知识库向量维度不一致")

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self.vectors or top_k <= 0:
            return []
        encoded = _vectors(self.model.encode([query], normalize_embeddings=True))
        if len(encoded) != 1 or len(encoded[0]) != len(self.vectors[0]) or not all(
            math.isfinite(x) for x in encoded[0]
        ):
            raise ValueError("Embedding 模型返回了无效的查询向量")
        scored = [(index, cosine(encoded[0], vector)) for index, vector in enumerate(self.vectors)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
