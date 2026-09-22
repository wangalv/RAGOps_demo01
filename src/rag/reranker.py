"""
M7.3 — Reranker (Cross-Encoder)

Two-stage retrieval:
  Stage 1 (recall):    Dense retrieves top-N candidates (fast, coarse)
  Stage 2 (precision): Cross-Encoder rescores and reranks top-N (slow, precise)

Why Cross-Encoder is more accurate than Bi-Encoder:
  Bi-Encoder: query and document are encoded separately → one vector each
              → detail is lost in compression
  Cross-Encoder: query + document are concatenated and fed together
                 → model sees the full interaction between both texts
                 → outputs a single relevance score 0~1
"""

from sentence_transformers import CrossEncoder

RERANKER_MODEL        = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_LARGE  = "BAAI/bge-reranker-large"


class Reranker:
    def __init__(self):
        print(f"  Loading reranker: {RERANKER_MODEL}")
        self.model = CrossEncoder(RERANKER_MODEL, max_length=512)
        self._large_model: CrossEncoder | None = None  # lazy-load on first cascade use

    def _get_large_model(self) -> CrossEncoder:
        if self._large_model is None:
            print(f"  Loading cascade reranker: {RERANKER_MODEL_LARGE}")
            self._large_model = CrossEncoder(RERANKER_MODEL_LARGE, max_length=512)
        return self._large_model

    def cascade_rerank(self, query: str, candidates: list[dict],
                       mid_k: int = 10, top_k: int = 5) -> list[dict]:
        """Two-stage reranking: MiniLM → top mid_k → bge-reranker-large → top top_k."""
        if not candidates:
            return []
        # Stage 1: MiniLM prunes to mid_k
        stage1 = self.rerank(query, candidates, top_k=mid_k)
        # Stage 2: bge-reranker-large refines to top_k
        large = self._get_large_model()
        pairs  = [(query, c["text"]) for c in stage1]
        scores = large.predict(pairs)
        for chunk, score in zip(stage1, scores):
            chunk["rerank_score"] = round(float(score), 4)
        reranked = sorted(stage1, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k]

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        """Rerank candidates using Cross-Encoder relevance scores.

        Args:
            query:      the search query
            candidates: list of chunk dicts from Dense or Hybrid retrieval
            top_k:      how many to return after reranking

        Returns:
            top_k chunks sorted by Cross-Encoder score descending,
            each with an added 'rerank_score' field (0~1).
        """
        if not candidates:
            return []

        # Cross-Encoder expects list of (query, document) pairs
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        # Attach score to each candidate and sort
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = round(float(score), 4)

        reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k]
