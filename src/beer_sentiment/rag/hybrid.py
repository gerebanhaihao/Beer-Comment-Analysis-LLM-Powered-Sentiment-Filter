"""混合检索：Dense + Sparse 双路召回、RRF 融合、Cross-Encoder（LLM 打分）重排序。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from beer_sentiment.rag.dense import DenseIndex
from beer_sentiment.rag.knowledge import KnowledgeBase, KnowledgeEntry
from beer_sentiment.rag.sparse import BM25Index
from beer_sentiment.rag.tokenize import tokenize

RERANK_SYSTEM_PROMPT = (
    "你是 Cross-Encoder 重排序器。给定一个待判定的帖子（查询）和若干知识库候选条目，"
    "请逐一评估每个候选与查询在语义上的相关程度，输出 0 到 1 的相关性分数"
    "（1 表示高度相关，0 表示无关）。只输出 JSON 对象。"
)


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：对多路召回的有序下标列表做融合。

    score(d) = sum(1 / (k + rank_i(d)))，k 为平滑常数。
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_index in enumerate(ranking):
            scores[doc_index] = scores.get(doc_index, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return fused


class LLMReranker:
    """用 LLM 给 (查询, 候选) 对打相关性分，扮演 Cross-Encoder 角色。"""

    def __init__(self, model_config: dict[str, Any], temperature: float = 0.0) -> None:
        self.model_config = model_config
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            from beer_sentiment.llm.openai_compat import create_client

            self._client = create_client(self.model_config)
        return self._client

    def rerank(self, query: str, entries: list[KnowledgeEntry]) -> list[float] | None:
        """返回与 entries 对齐的相关性分数列表；失败时返回 None（回退 RRF 顺序）。"""
        if not entries:
            return []
        lines = [f"{index}. {entry.text[:200]}" for index, entry in enumerate(entries, start=1)]
        user = (
            f"查询（待判定帖子）：\n{query[:1000]}\n\n"
            f"候选条目：\n" + "\n".join(lines) + "\n\n"
            '只输出 JSON：{"scores": [每个候选的 0-1 相关性分数，按顺序]}'
        )
        try:
            response = self._get_client().chat.completions.create(
                model=self.model_config.get("model", ""),
                messages=[
                    {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            data = json.loads(self._strip_code_fence(content))
            scores = data.get("scores")
            if not isinstance(scores, list) or len(scores) != len(entries):
                return None
            return [max(0.0, min(1.0, float(score))) for score in scores]
        except Exception:
            return None

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped


@dataclass
class RetrievedEntry:
    entry: KnowledgeEntry
    score: float
    stage: str  # "rerank" | "rrf"


class HybridRetriever:
    """Dense + Sparse 混合召回 -> RRF 融合 -> （可选）Cross-Encoder 重排序。"""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        rag_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        cfg = rag_config or {}
        self.kb = knowledge_base
        self.sparse = BM25Index(
            [tokenize(entry.text) for entry in knowledge_base.entries],
            k1=float(cfg.get("sparse", {}).get("k1", 1.5)),
            b=float(cfg.get("sparse", {}).get("b", 0.75)),
        )
        dense_cfg = cfg.get("dense", {})
        self.dense = DenseIndex(
            [tokenize(entry.text, ngram=3) for entry in knowledge_base.entries],
            dim=int(dense_cfg.get("dim", 256)),
        )
        self.sparse_top_k = int(cfg.get("sparse", {}).get("top_k", 10))
        self.dense_top_k = int(dense_cfg.get("dense", {}).get("top_k", 10))
        self.rrf_k = int(cfg.get("rrf", {}).get("k", 60))
        rerank_cfg = cfg.get("rerank", {})
        self.rerank_enabled = bool(rerank_cfg.get("enabled", False)) and model_config is not None
        self.rerank_top_n = int(rerank_cfg.get("top_n", 8))
        self.reranker = (
            LLMReranker(model_config, temperature=float(rerank_cfg.get("temperature", 0.0)))
            if self.rerank_enabled
            else None
        )
        self.default_top_k = int(cfg.get("fewshot", {}).get("top_k", 5))

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedEntry]:
        final_k = top_k or self.default_top_k
        sparse_ranking = [
            index for index, _ in self.sparse.search(tokenize(query), self.sparse_top_k)
        ]
        dense_ranking = [
            index for index, _ in self.dense.search(tokenize(query, ngram=3), self.dense_top_k)
        ]
        fused = rrf_fuse([sparse_ranking, dense_ranking], k=self.rrf_k)
        if not fused:
            return []

        candidate_indices = [index for index, _ in fused[: self.rerank_top_n]]
        if self.reranker is not None and candidate_indices:
            candidates = [self.kb.entries[index] for index in candidate_indices]
            scores = self.reranker.rerank(query, candidates)
            if scores is not None:
                reranked = sorted(
                    zip(candidate_indices, scores), key=lambda pair: pair[1], reverse=True
                )
                return [
                    RetrievedEntry(
                        entry=self.kb.entries[index], score=score, stage="rerank"
                    )
                    for index, score in reranked[:final_k]
                ]

        return [
            RetrievedEntry(
                entry=self.kb.entries[index],
                score=score,
                stage="rrf",
            )
            for index, score in fused[:final_k]
        ]
