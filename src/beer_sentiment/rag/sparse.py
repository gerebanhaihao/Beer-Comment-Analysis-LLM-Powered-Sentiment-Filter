"""稀疏召回：BM25（Okapi）索引。"""

from __future__ import annotations

import math
from collections import Counter


class BM25Index:
    """经典 BM25 稀疏检索，token 序列由外部给定。"""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tf: list[Counter[str]] = [Counter(doc) for doc in docs]
        self.doc_len = [len(doc) for doc in docs]
        self.avg_len = sum(self.doc_len) / len(docs) if docs else 0.0
        df: Counter[str] = Counter()
        for tf in self.doc_tf:
            df.update(tf.keys())
        self.idf = {
            term: math.log((len(docs) - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def score(self, doc_index: int, query_tokens: list[str]) -> float:
        tf = self.doc_tf[doc_index]
        total = 0.0
        for term in set(query_tokens):
            if term not in tf:
                continue
            freq = tf[term]
            denom = freq * (self.k1 + 1)
            norm = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[doc_index] / self.avg_len or 0))
            total += self.idf.get(term, 0.0) * denom / norm
        return total

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        """返回 (文档下标, 分数) 列表，按分数降序。"""
        scored = [
            (index, self.score(index, query_tokens))
            for index in range(len(self.doc_tf))
        ]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
