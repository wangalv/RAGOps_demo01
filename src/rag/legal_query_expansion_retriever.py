"""
Legal Query Expansion (LQE) retriever.

Instead of generating a hypothetical answer (HyDE), LQE rewrites the user's
query into the vocabulary the NSW Health Services Act 1997 actually uses.
The rewritten query is then sent to both BM25 (keyword match) and Dense
(semantic) retrieval, so both legs benefit from the vocabulary translation.

Key translations targeted:
  "fine" / "financial penalty"  →  "penalty units", "guilty of an offence"
  "prohibited" / "cannot"       →  "must not", "shall not"
  "term of office"              →  "holds office for", "period of appointment"
  "consequences"                →  "penalty", "offence", "liable"
"""

import os
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from src.rag.vector_store import VectorStore
from src.rag.hybrid_retriever import HybridRetriever

load_dotenv(dotenv_path=".env")

_client = OpenAI(
    base_url=os.environ["Z_AI_BASE_URL"],
    api_key=os.environ["Z_AI_API_KEY"],
)

_LQE_PROMPT = """\
Rewrite the user's question using the exact vocabulary of the NSW Health Services Act 1997.
Output ONLY the rewritten question — one sentence, no explanation, no options, no bullet points.

NSW Health Services Act vocabulary:
- "fine" / "financial penalty"  → "maximum penalty in penalty units" / "guilty of an offence"
- "prohibited" / "not allowed"  → "must not" / "shall not"
- "consequences" / "punished"   → "penalty" / "offence" / "liable"
- "term" / "how long serves"    → "holds office for" / "period of appointment"
- "set up" / "establish"        → "constitute" / "is to be established"
- "remove" / "fired"            → "removed from office" / "vacancy"

If no mapping applies, return the question unchanged.

Example:
IN:  What is the fine for not reporting an incident?
OUT: What is the maximum penalty in penalty units for a person guilty of an offence of failing to report an incident?

IN:  {query}
OUT:"""


def expand_query(query: str) -> str:
    """Rewrite a natural-language query using NSW Health Services Act vocabulary."""
    resp = _client.chat.completions.create(
        model=os.environ["Z_AI_CHAT_MODEL"],
        messages=[{"role": "user", "content": _LQE_PROMPT.format(query=query)}],
        temperature=0,
        max_tokens=2000,
    )
    return (resp.choices[0].message.content or "").strip()


class LegalQueryExpansionRetriever:
    """Retriever that translates query vocabulary before BM25+Dense retrieval.

    Flow: user query → LLM rewrite → HybridRetriever (BM25 + Dense + RRF)
    Both BM25 (keyword) and Dense (semantic) benefit from the translated query.
    """

    def __init__(self, hybrid_retriever: HybridRetriever):
        self.hybrid = hybrid_retriever

    def search(self, query: str, top_k: int = 5) -> tuple[list[dict], str]:
        """Retrieve using the vocabulary-expanded query.

        Returns:
            (chunks, expanded_query) — expanded_query returned for inspection.
        """
        expanded = expand_query(query)
        chunks = self.hybrid.search(expanded, top_k=top_k)
        return chunks, expanded
