"""
M7.2 — Hybrid Retriever (Dense + BM25)

Combines two retrieval signals:
  - Dense:  VectorStore cosine similarity (semantic meaning)
  - Sparse: BM25 keyword frequency (exact term match)

Merging strategy: RRF (Reciprocal Rank Fusion)
  Each result gets a score = 1 / (k + rank) from each retriever.
  Scores are summed — a result appearing in both lists ranks higher.
  k=60 is the standard constant (dampens the effect of top ranks).

Why RRF instead of score normalisation?
  Dense scores (0~1 cosine) and BM25 scores (unbounded) live on
  different scales — normalising them is tricky and brittle.
  RRF only uses rank positions, so the scales never matter.
"""

import math
from rank_bm25 import BM25Okapi

from src.rag.vector_store import VectorStore


RRF_K = 60  # standard RRF constant


class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[dict]) -> None:
        """Build the BM25 index from the same chunks stored in ChromaDB."""
        self._chunks = chunks
        # BM25Okapi expects a list of token lists
        tokenised = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenised)
        print(f"  BM25 index built: {len(chunks)} documents")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k chunks using RRF over Dense + BM25 results."""
        if self._bm25 is None:
            raise RuntimeError("Call index() before search()")

        fetch_n = min(top_k * 4, len(self._chunks))  # fetch more candidates before merging

        # ── Dense results ────────────────────────────────────────────────────
        dense_results = self.vector_store.search(query, top_k=fetch_n)
        dense_rank = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_results)}

        # ── BM25 results ─────────────────────────────────────────────────────
        scores = self._bm25.get_scores(query.lower().split())
        # Get indices sorted by score descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        bm25_top = sorted_indices[:fetch_n]
        bm25_rank = {self._chunks[i]["chunk_id"]: rank + 1 for rank, i in enumerate(bm25_top)}

        # ── RRF merge ────────────────────────────────────────────────────────
        # Collect all candidate chunk_ids from both lists
        all_ids = set(dense_rank.keys()) | set(bm25_rank.keys())

        rrf_scores: dict[str, float] = {}
        for cid in all_ids:
            dense_score = 1 / (RRF_K + dense_rank[cid]) if cid in dense_rank else 0
            bm25_score  = 1 / (RRF_K + bm25_rank[cid])  if cid in bm25_rank  else 0
            rrf_scores[cid] = dense_score + bm25_score

        # Sort by RRF score descending, take top_k
        top_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        # Build result dicts — look up metadata from chunks list
        chunk_map = {c["chunk_id"]: c for c in self._chunks}
        results = []
        for cid in top_ids:
            c = chunk_map.get(cid, {})
            results.append({
                "chunk_id":   cid,
                "section_id": c.get("section_id", ""),
                "title":      c.get("title", ""),
                "text":       c.get("text", ""),
                "rrf_score":  round(rrf_scores[cid], 6),
                "in_dense":   cid in dense_rank,
                "in_bm25":    cid in bm25_rank,
            })
        return results
