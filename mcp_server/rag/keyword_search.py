"""
Shared keyword search helper: pure-Python BM25, no embeddings, no vector
DB, and — deliberately — no external pip dependency.

BM25 is the standard keyword-ranking algorithm search engines have used
for decades (it's what Elasticsearch/Postgres full-text search build on
under the hood). It scores documents by term overlap with the query,
weighted by how rare/common each term is across the whole corpus.

This implementation is written from scratch in ~40 lines of pure Python
(no numpy, no rank_bm25) specifically so this add-on has ZERO new
dependencies to install — nothing new for the team to `pip install`,
nothing that can fail because a package index is unreachable.

This trades semantic understanding for simplicity: it won't know that
"vet" and "veterinarian" mean the same thing, but it needs zero external
calls and is easy to reason about and debug.
"""

import re
import math


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class KeywordStore:
    """
    upsert() to add records, query() to search — deliberately the same
    shape a vector store would have (upsert/query), so swapping in a
    real embeddings-based store later wouldn't require touching the
    tool code that calls this class.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.rows: list[dict] = []  # [{"payload": ..., "metadata": ...}, ...]
        self.k1 = k1
        self.b = b
        self._dirty = True
        self._doc_freqs: dict[str, int] = {}
        self._avg_doc_len = 0.0
        self._tokenized: list[list[str]] = []

    def upsert(self, payload, metadata: dict) -> None:
        self.rows.append({"payload": payload, "metadata": metadata})
        self._dirty = True

    @staticmethod
    def _as_text(payload) -> str:
        # payload can be a plain string or a dict with a text-ish field
        if isinstance(payload, str):
            return payload
        return payload.get("text") or payload.get("event_summary") or str(payload)

    def _rebuild_index(self) -> None:
        self._tokenized = [tokenize(self._as_text(r["payload"])) for r in self.rows]
        self._doc_freqs = {}
        for tokens in self._tokenized:
            for term in set(tokens):
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1
        total_len = sum(len(t) for t in self._tokenized)
        self._avg_doc_len = (total_len / len(self._tokenized)) if self._tokenized else 0.0
        self._dirty = False

    def _bm25_score(self, query_terms: list[str], doc_idx: int) -> float:
        tokens = self._tokenized[doc_idx]
        doc_len = len(tokens) or 1
        n_docs = len(self.rows)
        score = 0.0
        for term in query_terms:
            df = self._doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            tf = tokens.count(term)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self._avg_doc_len or 1))
            score += idf * (tf * (self.k1 + 1)) / (denom or 1)
        return score

    def query(self, query_text: str, top_k: int = 3, filter: dict | None = None) -> list[dict]:
        candidate_idxs = [
            i for i, r in enumerate(self.rows)
            if not filter or all(r["metadata"].get(k) == v for k, v in filter.items())
        ]
        if not candidate_idxs:
            return []

        if self._dirty:
            self._rebuild_index()

        query_terms = tokenize(query_text)
        query_set = set(query_terms)

        overlapping = [
            i for i in candidate_idxs
            if query_set & set(self._tokenized[i])
        ]
        ranked = sorted(overlapping, key=lambda i: self._bm25_score(query_terms, i), reverse=True)
        return [self.rows[i] for i in ranked[:top_k]]


if __name__ == "__main__":
    store = KeywordStore()
    store.upsert("Roof shows minor wear, recommend inspection before closing.", {"property_id": 1})
    store.upsert("Routine walkthrough, no notable issues found.", {"property_id": 5})
    results = store.query("roof condition", top_k=3, filter={"property_id": 1})
    print(results)