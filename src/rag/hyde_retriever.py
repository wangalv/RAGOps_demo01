"""
M7.5 — HyDE (Hypothetical Document Embeddings)

Problem HyDE solves:
  User queries use everyday language ("licensing"), but legal documents use
  formal terminology ("registration", "authorisation"). Embedding the query
  directly produces a vector far from the matching chunks.

HyDE approach:
  1. Ask an LLM to generate a hypothetical passage that would answer the query,
     written in the style of the target document (formal legislation language)
  2. Embed the hypothetical passage instead of the raw query
  3. Use that embedding for retrieval

Why this works:
  The LLM knows that "licensing" in a healthcare legislation context means
  "registration" — it generates a passage using the correct legal vocabulary,
  so the resulting vector lands much closer to the actual chunks in ChromaDB.

Cost tradeoff:
  Every retrieval call = 1 LLM API call + 1 embedding.
  More expensive and slower than direct embedding; justified when vocabulary
  gap is large (lay queries against specialist documents).
"""

import os
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from src.rag.vector_store import VectorStore

load_dotenv(dotenv_path=".env")

_client = OpenAI(
    base_url=os.environ["Z_AI_BASE_URL"],
    api_key=os.environ["Z_AI_API_KEY"],
)

# Prompt instructs the LLM to write like a legislation section, not answer conversationally.
# The legal-style wording ("must not", "subject to", "in accordance with") pushes the
# generated text into the same vocabulary space as the Act.
_HYDE_PROMPT = """\
Write a short passage (2-4 sentences) from an Australian state health services \
legislation act that would answer the following question. \
Write as if it is an actual section from the legislation, using formal legal language \
("A person must...", "The Secretary may...", "subject to..."). \
Do not explain or summarise — write the passage itself only.

Question: {query}

Passage:"""

# Vocabulary-aware HyDE: lighter-touch prompt that naturally seeds NSW statutory vocabulary
# without NEVER-style constraints (which distract the LLM from the actual question).
# Provides a short vocabulary scaffold covering the most impactful mismatches:
#   "fine" -> "penalty units / guilty of an offence"
#   "prohibited" -> "must not / shall not"
#   "term" -> "holds office for / within X days"
_LEGAL_HYDE_PROMPT = """\
Write a short passage (2-4 sentences) from the NSW Health Services Act 1997 \
that directly answers the following question. \
Write as if it is the actual statutory text, using the Act's own vocabulary naturally: \
"may", "must", "must not", "is to", "subject to", "in accordance with", \
"penalty units", "guilty of an offence", "is appointed by the Minister", \
"holds office for", "means", "includes".

Do not explain or summarise — write the passage itself only.

Question: {query}

Passage:"""


def generate_hypothetical_doc(query: str, prompt_template: str = _HYDE_PROMPT) -> str:
    """Call GLM via Z.ai to produce a hypothetical legislation passage for the query."""
    response = _client.chat.completions.create(
        model=os.environ["Z_AI_CHAT_MODEL"],
        messages=[{"role": "user", "content": prompt_template.format(query=query)}],
        max_tokens=2000,
        temperature=0.3,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


class HyDERetriever:
    """Retriever that embeds a hypothetical document instead of the raw query."""

    def __init__(self, vector_store: VectorStore, prompt_template: str = _HYDE_PROMPT):
        self.store = vector_store
        self.prompt_template = prompt_template

    def search(self, query: str, top_k: int = 5) -> tuple[list[dict], str]:
        """Retrieve chunks using HyDE.

        Returns:
            (chunks, hypothetical_doc) — the hypothetical doc is returned so
            the caller can print it for inspection/debugging.
        """
        hypothetical_doc = generate_hypothetical_doc(query, self.prompt_template)

        # Embed the hypothetical doc, not the query
        embedding = self.store.model.encode([hypothetical_doc]).tolist()

        results = self.store.collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, self.store.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id":   results["ids"][0][i],
                "section_id": results["metadatas"][0][i]["section_id"],
                "title":      results["metadatas"][0][i]["title"],
                "text":       results["documents"][0][i],
                "score":      round(1 - results["distances"][0][i], 4),
            })

        return chunks, hypothetical_doc


class LegalHyDERetriever(HyDERetriever):
    """HyDE retriever with vocabulary-aware prompt tuned for NSW Health Services Act 1997.

    Uses a light-touch vocabulary scaffold (not NEVER-style constraints) to naturally
    seed statutory terminology into the hypothetical passage, closing embedding distance
    for Penalty ("penalty units"), Prohibition ("must not"), and Timeframe ("holds office for") queries.
    """

    def __init__(self, vector_store: VectorStore):
        super().__init__(vector_store, prompt_template=_LEGAL_HYDE_PROMPT)
