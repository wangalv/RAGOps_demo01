"""
M7 補充 — Multi-Query Retrieval

Problem:
  A single query only expresses one phrasing of the information need.
  Chunks that use different vocabulary may be missed even though they
  are relevant.

Solution:
  Use an LLM to rewrite the query into N variants, retrieve independently
  for each, merge and deduplicate the candidate pool, then rerank using
  the original query.

Why rerank with the original query?
  The variants are only used to widen the candidate pool (Recall).
  Relevance scoring must be against the user's actual intent, not a
  paraphrase — so the Cross-Encoder always sees (original_query, chunk).

Drift mitigation (Method 1 — Prompt constraint):
  The prompt explicitly instructs the LLM to rephrase only and not
  introduce concepts absent from the original query.
  Experiment showed that a cosine similarity filter (Method 2) was too
  aggressive — it discarded useful variants like "compulsory notification
  duties" (sim=0.689) that retrieved correct chunks. Method 1 alone is
  the better tradeoff for this corpus.
"""

import os
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from src.rag.vector_store import VectorStore
from src.rag.reranker import Reranker

load_dotenv(dotenv_path=".env")

_client = OpenAI(
    base_url=os.environ["Z_AI_BASE_URL"],
    api_key=os.environ["Z_AI_API_KEY"],
)

# Method 1: "rephrase only" constraint prevents semantic drift.
# Without this, LLM turns "regulations" into "delegated legislation",
# flooding the candidate pool with delegation sections instead of s.140.
_VARIANT_PROMPT = """\
Generate {n} different search queries for finding information in Australian \
health services legislation.

Rules:
- Each query must target exactly the same information as the original.
- Rephrase only — do NOT introduce concepts or topics not present in the original query.
- Use different legal vocabulary or sentence structure, but keep the core meaning identical.

Return only the queries, one per line, no numbering or explanations.

Original query: {query}

Queries:"""


def generate_query_variants(query: str, n: int = 3) -> list[str]:
    """Call GLM to produce N rephrased versions of the query."""
    response = _client.chat.completions.create(
        model=os.environ["Z_AI_CHAT_MODEL"],
        messages=[{"role": "user", "content": _VARIANT_PROMPT.format(query=query, n=n)}],
        max_tokens=2000,
        temperature=0.7,
    )
    content = (response.choices[0].message.content or "").strip()
    return [line.strip() for line in content.splitlines() if line.strip()][:n]


class MultiQueryRetriever:
    """Retriever that searches with multiple query variants and merges results."""

    def __init__(self, vector_store: VectorStore, reranker: Reranker):
        self.store = vector_store
        self.reranker = reranker

    def search(self, query: str, n_variants: int = 3,
               per_query_k: int = 20, final_k: int = 3,
               ) -> tuple[list[dict], list[str]]:
        """Retrieve using the original query + N variants.

        per_query_k defaults to 20 (same as Dense@20→Reranked) so the
        original query's candidate pool is at least as wide as the baseline.

        Returns:
            (reranked_chunks, all_queries)
        """
        variants = generate_query_variants(query, n=n_variants)
        all_queries = [query] + variants

        seen_ids: set[str] = set()
        candidates: list[dict] = []

        for q in all_queries:
            results = self.store.search(q, top_k=per_query_k)
            for r in results:
                if r["chunk_id"] not in seen_ids:
                    seen_ids.add(r["chunk_id"])
                    candidates.append(r)

        # Rerank using the ORIGINAL query — variants only widened the pool
        reranked = self.reranker.rerank(query, candidates, top_k=final_k)
        return reranked, all_queries
