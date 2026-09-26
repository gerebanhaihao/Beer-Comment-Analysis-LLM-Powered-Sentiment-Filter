"""Hybrid retrieval: BM25 and semantic embeddings, RRF, then CrossEncoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from beer_sentiment.config import AppConfig
from beer_sentiment.rag.dense import DenseIndex, load_embedding_model
from beer_sentiment.rag.knowledge import KnowledgeBase, KnowledgeEntry
from beer_sentiment.rag.sparse import BM25Index
from beer_sentiment.rag.tokenize import tokenize
from beer_sentiment.rules.normalize import extract_brands, normalize_ocr_noise


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse ordered document IDs using reciprocal rank fusion."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_index in enumerate(ranking):
            scores[doc_index] = scores.get(doc_index, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


@lru_cache(maxsize=4)
def load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("RAG 需要 sentence-transformers；请安装 beer-sentiment[rag]") from exc
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    """Score each query/entry pair with a dedicated sequence-pair model."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def rerank(self, query: str, entries: list[KnowledgeEntry]) -> list[float] | None:
        if not entries:
            return []
        try:
            raw = self.model.predict([(query, entry.text) for entry in entries])
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            scores = [float(score) for score in raw]
            if len(scores) != len(entries) or not all(math.isfinite(x) for x in scores):
                return None
            return scores
        except Exception:
            return None


@dataclass
class RetrievedEntry:
    entry: KnowledgeEntry
    score: float
    stage: str  # "rerank" | "rrf"


class HybridRetriever:
    """BM25 + embeddings -> RRF -> optional CrossEncoder reranking."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        rag_config: dict[str, Any] | None = None,
        *,
        embedding_model: Any | None = None,
        cross_encoder: Any | None = None,
        brand_config: AppConfig | None = None,
    ) -> None:
        cfg = rag_config or {}
        self.kb = knowledge_base
        self.brand_config = brand_config
        self.sparse = BM25Index(
            [tokenize(entry.text) for entry in knowledge_base.entries],
            k1=float(cfg.get("sparse", {}).get("k1", 1.5)),
            b=float(cfg.get("sparse", {}).get("b", 0.75)),
        )
        dense_cfg = cfg.get("dense", {})
        encoder = embedding_model or load_embedding_model(
            str(dense_cfg.get("model", "BAAI/bge-small-zh-v1.5"))
        )
        self.dense = DenseIndex([entry.text for entry in knowledge_base.entries], encoder)
        self.sparse_top_k = int(cfg.get("sparse", {}).get("top_k", 10))
        self.dense_top_k = int(dense_cfg.get("top_k", 10))
        self.rrf_k = int(cfg.get("rrf", {}).get("k", 60))
        rerank_cfg = cfg.get("rerank", {})
        self.rerank_top_n = int(rerank_cfg.get("top_n", 8))
        self.reranker = None
        if bool(rerank_cfg.get("enabled", True)):
            rerank_model = cross_encoder or load_cross_encoder(
                str(rerank_cfg.get("model", "BAAI/bge-reranker-base"))
            )
            self.reranker = CrossEncoderReranker(rerank_model)
        self.default_top_k = int(cfg.get("fewshot", {}).get("top_k", 5))

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedEntry]:
        final_k = self.default_top_k if top_k is None else top_k
        if final_k <= 0:
            return []
        # Keep the original query for embeddings. Add canonical brand names only
        # to BM25 so OCR errors and aliases can match knowledge-base terms.
        sparse_query = normalize_ocr_noise(query)
        if self.brand_config is not None:
            brands = extract_brands(sparse_query, self.brand_config)
            if brands:
                sparse_query += "\n" + " ".join(brands)
        sparse_ranking = [
            index for index, _ in self.sparse.search(tokenize(sparse_query), self.sparse_top_k)
        ]
        dense_ranking = [index for index, _ in self.dense.search(query, self.dense_top_k)]
        fused = rrf_fuse([sparse_ranking, dense_ranking], k=self.rrf_k)
        if not fused:
            return []

        candidate_indices = [index for index, _ in fused[: max(final_k, self.rerank_top_n)]]
        if self.reranker is not None and candidate_indices:
            candidates = [self.kb.entries[index] for index in candidate_indices]
            scores = self.reranker.rerank(query, candidates)
            if scores is not None:
                reranked = sorted(
                    zip(candidate_indices, scores), key=lambda pair: pair[1], reverse=True
                )
                return [
                    RetrievedEntry(self.kb.entries[index], score, "rerank")
                    for index, score in reranked[:final_k]
                ]

        return [
            RetrievedEntry(self.kb.entries[index], score, "rrf")
            for index, score in fused[:final_k]
        ]
