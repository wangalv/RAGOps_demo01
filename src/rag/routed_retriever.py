"""
Routed Retriever: classifies the query then dispatches to the optimal retriever.

  Penalty      → LQE → Hybrid → Rerank
  Prohibition  → StepBack → Rerank
  CrossSection → MultiQuery → Rerank
  Other        → HyDE → Rerank
"""

from src.rag.query_router import classify_query
from src.rag.legal_query_expansion_retriever import LegalQueryExpansionRetriever
from src.rag.stepback_retriever import StepBackRetriever
from src.rag.multi_query_retriever import MultiQueryRetriever
from src.rag.hyde_retriever import HyDERetriever
from src.rag.reranker import Reranker


class RoutedRetriever:
    """Classifies each query and routes to the optimal retrieval strategy."""

    def __init__(
        self,
        lqe: LegalQueryExpansionRetriever,
        stepback: StepBackRetriever,
        multiquery: MultiQueryRetriever,
        hyde: HyDERetriever,
        reranker: Reranker,
        candidate_n: int = 20,
    ):
        self.lqe = lqe
        self.stepback = stepback
        self.multiquery = multiquery
        self.hyde = hyde
        self.reranker = reranker
        self.candidate_n = candidate_n

    def search(self, query: str, top_k: int = 5) -> tuple[list[dict], str]:
        """Classify query, route to optimal retriever, rerank, return top_k chunks.

        Returns:
            (chunks, route_label) — route_label for inspection/debugging.
        """
        qtype = classify_query(query)

        if qtype == "Penalty":
            candidates = self.lqe.search(query, top_k=self.candidate_n)[0]
            chunks = self.reranker.rerank(query, candidates, top_k=top_k)
        elif qtype == "Prohibition":
            chunks = self.stepback.search(query, per_query_k=self.candidate_n, final_k=top_k)[0]
        elif qtype == "CrossSection":
            chunks = self.multiquery.search(query, n_variants=3, per_query_k=self.candidate_n, final_k=top_k)[0]
        else:
            candidates = self.hyde.search(query, top_k=self.candidate_n)[0]
            chunks = self.reranker.rerank(query, candidates, top_k=top_k)

        return chunks, qtype
