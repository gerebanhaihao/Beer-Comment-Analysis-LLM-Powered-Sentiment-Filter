"""Hybrid RAG 模块测试：分词、BM25、稠密向量、RRF、检索器与 RagJudge。"""

from __future__ import annotations

from pathlib import Path

from beer_sentiment.llm.mock import MockJudge
from beer_sentiment.rag.dense import DenseIndex, cosine
from beer_sentiment.rag.hybrid import CrossEncoderReranker, HybridRetriever, rrf_fuse
from beer_sentiment.rag.judge import RagJudge
from beer_sentiment.rag.knowledge import KnowledgeBase
from beer_sentiment.rag.tokenize import tokenize

KB_PATH = Path(__file__).resolve().parents[1] / "config" / "knowledge_base.yaml"


class FakeEmbeddingModel:
    """Deterministic test double for the embedding model's batch API."""

    def encode(self, texts, normalize_embeddings=False):
        assert normalize_embeddings
        vectors = []
        for text in texts:
            vectors.append([
                float(any(word in text for word in ("乌苏", "上头", "后劲", "吐"))),
                float(any(word in text for word in ("百威", "杂质", "投诉"))),
                float(any(word in text for word in ("青岛", "营收", "下滑"))),
                float(any(word in text for word in ("足球", "比赛"))),
            ])
        return vectors


class FakeCrossEncoder:
    def predict(self, pairs):
        return [float("杂质" in query and "杂质" in text) for query, text in pairs]


def make_retriever(kb, config=None, cross_encoder=None):
    return HybridRetriever(
        kb, config or {}, embedding_model=FakeEmbeddingModel(),
        cross_encoder=cross_encoder or FakeCrossEncoder(),
    )


def test_tokenize_chinese_bigrams():
    tokens = tokenize("百威啤酒难喝")
    assert "百威" in tokens
    assert "啤酒" in tokens
    assert all(len(token) <= 2 for token in tokens if not token.isascii())


def test_bm25_ranks_relevant_doc_first():
    docs = [
        tokenize("教你三招识破勾兑啤酒，新国标科普"),
        tokenize("百威啤酒喝出杂质还拉肚子，投诉没人理"),
        tokenize("今天天气不错，适合出去露营"),
    ]
    from beer_sentiment.rag.sparse import BM25Index

    index = BM25Index(docs)
    hits = index.search(tokenize("百威 啤酒 杂质 投诉"), top_k=3)
    assert hits
    assert hits[0][0] == 1


def test_dense_index_relevant_doc_first():
    docs = ["怀旧：小时候绿瓶啤酒的味道", "乌苏啤酒后劲大上头，喝吐了", "足球比赛直播预告"]
    index = DenseIndex(docs, FakeEmbeddingModel())
    hits = index.search("乌苏 后劲 上头 吐", top_k=3)
    assert hits
    assert hits[0][0] == 1


def test_cosine_identical_vectors():
    vector = [0.6, 0.8]
    assert cosine(vector, vector) > 0.99


def test_rrf_fuse_prefers_docs_in_both_rankings():
    fused = rrf_fuse([[0, 1, 2], [2, 1, 0]], k=60)
    ranking = [index for index, _ in fused]
    # doc 0 与 doc 2 各有一次第一，得分相同；两路都排中间的 doc 1 得分最低
    assert ranking[-1] == 1
    assert ranking.index(0) < ranking.index(1)
    assert ranking.index(2) < ranking.index(1)


def test_knowledge_base_load_and_format():
    kb = KnowledgeBase.from_yaml(KB_PATH)
    assert len(kb) > 10
    entry = kb.get("ex-own-quality")
    assert entry is not None and entry.label == "blue"
    context = kb.format_context(kb.entries[:3])
    assert "示例" in context or "规则" in context


def test_retriever_finds_relevant_entries():
    kb = KnowledgeBase.from_yaml(KB_PATH)
    retriever = make_retriever(kb, {"fewshot": {"top_k": 5}})
    results = retriever.retrieve("百威啤酒喝出杂质拉肚子，投诉没人理")
    ids = {item.entry.id for item in results}
    assert "ex-own-quality" in ids or "rule-blue-vs-yellow" in ids
    assert len(results) <= 5


def test_reranker_failure_falls_back_to_rrf():
    class BrokenCrossEncoder:
        def predict(self, pairs):
            raise RuntimeError("inference failed")

    kb = KnowledgeBase.from_yaml(KB_PATH)
    config = {"fewshot": {"top_k": 5}}
    retriever = make_retriever(kb, config, BrokenCrossEncoder())
    results = retriever.retrieve("青岛啤酒营收下滑卖不动")
    assert results
    assert all(item.stage == "rrf" for item in results)


def test_cross_encoder_reorders_fused_candidates():
    kb = KnowledgeBase.from_yaml(KB_PATH)
    retriever = make_retriever(kb, {"fewshot": {"top_k": 3}})
    results = retriever.retrieve("百威啤酒喝出杂质，投诉无果")
    assert results[0].stage == "rerank"
    assert "杂质" in results[0].entry.text


def test_cross_encoder_rejects_invalid_scores():
    class InvalidCrossEncoder:
        def predict(self, pairs):
            return [float("nan")] * len(pairs)

    kb = KnowledgeBase.from_yaml(KB_PATH)
    reranker = CrossEncoderReranker(InvalidCrossEncoder())
    assert reranker.rerank("查询", kb.entries[:2]) is None


def test_rag_judge_injects_context(config):
    kb = KnowledgeBase.from_yaml(KB_PATH)
    retriever = make_retriever(kb, {"fewshot": {"top_k": 3}})
    captured = {}

    class CapturingJudge(MockJudge):
        def judge(self, sample, context=""):
            captured["context"] = context
            return super().judge(sample, context)

    rag_judge = RagJudge(CapturingJudge(config), retriever)
    result = rag_judge.judge("百威啤酒喝出杂质还拉肚子，投诉没人理")
    assert result.label is not None
    assert "知识库" in captured["context"]
    assert rag_judge.retrieval_stats["queries"] == 1
