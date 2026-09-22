"""
Ensemble Retriever — combines multiple retrievers via candidate pool merging.

Strategy:
  1. Each sub-retriever searches independently (top CANDIDATE_N each)
  2. All candidate pools are merged and deduplicated (union)
  3. A single Cross-encoder Reranker scores the merged pool against the
     original query and returns the top-k results

Why this works:
  Different retrievers find different chunks:
  - Dense finds semantically similar text
  - HyDE finds chunks matching formal legal vocabulary (e.g. "by-laws")
  - MultiQuery finds via rephrased variants (vocabulary breadth)
  - StepBack finds via abstracted topic (section-level coverage)
  The union of their candidate pools maximises recall before reranking.
"""

from src.rag.reranker import Reranker


class EnsembleRetriever:
    def __init__(self, retrievers: list[dict], reranker: Reranker, candidate_n: int = 20):
        """
        Args:
            retrievers: list of {"name": str, "fn": callable(query) -> list[dict]}
            reranker:   cross-encoder reranker
            candidate_n: how many candidates each sub-retriever fetches
        """
        self.retrievers  = retrievers
        self.reranker    = reranker
        self.candidate_n = candidate_n

    def search(self, query: str, final_k: int = 3) -> list[dict]:
        seen_ids: set[str] = set()
        candidates: list[dict] = []

        for r in self.retrievers:
            results = r["fn"](query, self.candidate_n)
            for chunk in results:
                if chunk["chunk_id"] not in seen_ids:
                    seen_ids.add(chunk["chunk_id"])
                    candidates.append(chunk)

        return self.reranker.rerank(query, candidates, top_k=final_k)
