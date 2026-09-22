"""
M7 补充 — Step-back Prompting Retrieval

Problem:
  Some queries are too specific — the correct answer lives inside a broad
  section (e.g. "General Offences", "Ministerial Powers") that doesn't
  contain the exact terms in the user's query.  Direct retrieval misses it.

Step-back approach:
  1. Ask an LLM to rewrite the specific query as a more abstract, general
     question that covers the same topic area.
  2. Retrieve for BOTH the original query and the abstract query.
  3. Merge the two candidate pools, deduplicate, then rerank using the
     ORIGINAL query (so scoring reflects the user's actual intent).

How this differs from Multi-Query:
  - Multi-Query: rephrases sideways (same specificity, different words)
  - Step-back:   moves UP in abstraction (specific → general)
  Multi-Query expands vocabulary coverage; Step-back expands topic coverage.

When step-back helps:
  - "penalty for providing false information"
     → abstract: "what are the offence and penalty provisions in health legislation?"
     → catches General Offences section that doesn't mention "false information"

When it doesn't help:
  - Already-broad queries — the abstract version is barely different, no gain.
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

_STEPBACK_PROMPT = """\
You are an expert at legal research. Given a specific legal question, rewrite it \
as a more general, abstract question that covers the broader topic area.

The abstract question should:
- Be broader and more general than the original
- Cover the topic AREA, not just the specific detail
- Use general legal category terms (e.g. "penalty provisions", "ministerial powers", \
"reporting obligations")
- Be answerable by a general section or chapter of legislation, not just a specific clause

Return ONLY the abstract question — no explanation, no preamble.

Specific question: {query}

Abstract question:"""


def generate_stepback_query(query: str) -> str:
    """Call GLM to produce a more general, abstract version of the query."""
    response = _client.chat.completions.create(
        model=os.environ["Z_AI_CHAT_MODEL"],
        messages=[{"role": "user", "content": _STEPBACK_PROMPT.format(query=query)}],
        max_tokens=2000,
        temperature=0.3,
    )
    content = (response.choices[0].message.content or "").strip()
    for line in content.splitlines():
        line = line.strip()
        if line:
            return line
    return content


class StepBackRetriever:
    """Retriever that merges results from the original query and an abstracted version."""

    def __init__(self, vector_store: VectorStore, reranker: Reranker):
        self.store = vector_store
        self.reranker = reranker

    def search(self, query: str, per_query_k: int = 20,
               final_k: int = 3) -> tuple[list[dict], str]:
        """Retrieve using the original query and its step-back abstraction.

        Returns:
            (reranked_chunks, abstract_query) — abstract_query is returned for
            inspection so the caller can see what the LLM generated.
        """
        abstract_query = generate_stepback_query(query)

        # Retrieve independently for each query
        specific_results  = self.store.search(query,          top_k=per_query_k)
        abstract_results  = self.store.search(abstract_query, top_k=per_query_k)

        # Merge and deduplicate — specific first so its order is preserved when scores tie
        seen_ids: set[str] = set()
        candidates: list[dict] = []
        for r in specific_results + abstract_results:
            if r["chunk_id"] not in seen_ids:
                seen_ids.add(r["chunk_id"])
                candidates.append(r)

        # Rerank using the ORIGINAL query — the abstract query only widens the pool
        reranked = self.reranker.rerank(query, candidates, top_k=final_k)
        return reranked, abstract_query
