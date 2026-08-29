"""Hybrid RAG 模块测试：分词、BM25、稠密向量、RRF、检索器与 RagJudge。"""

from __future__ import annotations

from pathlib import Path

from beer_sentiment.llm.mock import MockJudge
from beer_sentiment.rag.dense import DenseIndex, cosine, embed
from beer_sentiment.rag.hybrid import HybridRetriever, LLMReranker, rrf_fuse
from beer_sentiment.rag.judge import RagJudge
from beer_sentiment.rag.knowledge import KnowledgeBase
from beer_sentiment.rag.tokenize import tokenize

KB_PATH = Path(__file__).resolve().parents[1] / "config" / "knowledge_base.yaml"


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
    docs = [
        tokenize("怀旧：小时候绿瓶啤酒的味道"),
        tokenize("乌苏啤酒后劲大上头，喝吐了"),
        tokenize("足球比赛直播预告"),
    ]
    index = DenseIndex(docs, dim=128)
    hits = index.search(tokenize("乌苏 后劲 上头 吐"), top_k=3)
    assert hits
    assert hits[0][0] == 1


def test_embed_is_normalized():
    vector = embed(tokenize("青岛啤酒销量下滑"), dim=64)
    norm = sum(value * value for value in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_cosine_identical_vectors():
    vector = embed(tokenize("啤酒负面舆情"), dim=32)
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
    retriever = HybridRetriever(kb, {"fewshot": {"top_k": 5}}, model_config=None)
    results = retriever.retrieve("百威啤酒喝出杂质拉肚子，投诉没人理")
    ids = {item.entry.id for item in results}
    assert "ex-own-quality" in ids or "rule-blue-vs-yellow" in ids
    assert len(results) <= 5


def test_reranker_failure_falls_back_to_rrf():
    class BrokenReranker(LLMReranker):
        def _get_client(self):
            raise RuntimeError("no api")

    kb = KnowledgeBase.from_yaml(KB_PATH)
    config = {"fewshot": {"top_k": 5}}
    retriever = HybridRetriever(kb, config, model_config={"model": "x"})
    retriever.reranker = BrokenReranker({"model": "x"})
    retriever.rerank_enabled = True
    results = retriever.retrieve("青岛啤酒营收下滑卖不动")
    assert results
    assert all(item.stage == "rrf" for item in results)


def test_rag_judge_injects_context(config):
    kb = KnowledgeBase.from_yaml(KB_PATH)
    retriever = HybridRetriever(kb, {"fewshot": {"top_k": 3}}, model_config=None)
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
