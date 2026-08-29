"""RagJudge：在调用底层判定模型前，用 Hybrid RAG 检索知识库并注入参考上下文。"""

from __future__ import annotations

from beer_sentiment.llm.base import Judge
from beer_sentiment.models import JudgeResult
from beer_sentiment.rag.hybrid import HybridRetriever


class RagJudge(Judge):
    """装饰器式 Judge：检索 -> 拼 Few-shot 上下文 -> 委托内层模型判定。"""

    def __init__(self, inner: Judge, retriever: HybridRetriever, max_context_chars: int = 1500) -> None:
        self.inner = inner
        self.retriever = retriever
        self.max_context_chars = max_context_chars
        self.name = inner.name
        self.retrieval_stats = {"queries": 0, "rerank_hits": 0, "rrf_hits": 0}

    def judge(self, sample: str, context: str = "") -> JudgeResult:
        retrieved = self.retriever.retrieve(sample)
        self.retrieval_stats["queries"] += 1
        for item in retrieved:
            key = "rerank_hits" if item.stage == "rerank" else "rrf_hits"
            self.retrieval_stats[key] += 1

        blocks: list[str] = []
        if context:
            blocks.append(context)
        if retrieved:
            knowledge = self.retriever.kb.format_context([item.entry for item in retrieved])
            if len(knowledge) > self.max_context_chars:
                knowledge = knowledge[: self.max_context_chars] + "…"
            blocks.append(knowledge)
        merged_context = "\n\n".join(blocks)
        return self.inner.judge(sample, context=merged_context)
