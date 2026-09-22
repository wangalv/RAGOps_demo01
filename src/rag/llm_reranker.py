"""
LLM Reranker — pointwise scoring with legal-aware prompt.

Takes top-k candidates (pre-filtered by cross-encoder) and asks the LLM
to score each on 0/1/2 scale, then returns top_k by score descending.

Designed to fix the failure modes identified in diagnostic:
  - Section title keyword trap (e.g. "Prohibition of PPP" != "what is prohibited")
  - Legal semantic equivalence ("fine" == "penalty units")
  - Answer-bearing vs merely topical passages
"""

import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from json_repair import repair_json

load_dotenv(dotenv_path=".env")

_client = OpenAI(
    base_url=os.environ["Z_AI_BASE_URL"],
    api_key=os.environ["Z_AI_API_KEY"],
)

_PROMPT = """\
You are a legal information retrieval reranker for a legislation RAG system.

Your task is to rerank the candidate legislative passages for the user's
question.

Your primary objective is to identify passages that contain the substantive
legal information needed to answer the question.

IMPORTANT RANKING RULES:

1. SUBSTANTIVE CONTENT OVER SECTION TITLE

Do NOT rank a passage highly merely because its section title contains words
similar to the user's question.

A section title is only contextual metadata. The substantive provision text
must support the relevance.

For example:
Query: "What activities are prohibited?"
A section titled "Prohibition of PPP" should NOT receive a high score merely
because it contains the word "Prohibition" if the substantive text does not
address the activity described in the question.

2. ANSWER-BEARING CONTENT OVER TOPICAL SIMILARITY

Prefer a passage that directly contains the information needed to answer the
question over a passage that merely discusses the same general topic.

Distinguish between:
- Direct answer-bearing provision
- Supporting/contextual provision
- Topically similar but non-answering provision

3. LEGAL SEMANTIC EQUIVALENCE

Do not require exact keyword matches.

Recognise legal concepts that may be expressed using different terminology.

Examples:

"fine" may correspond to:
- penalty
- maximum penalty
- penalty units
- liable to a penalty
- guilty of an offence

"prohibited" may correspond to:
- must not
- prohibited from
- may not
- restriction
- limitation
- not authorised
- outside the permitted authority

"how long" may correspond to:
- term of office
- period
- duration
- holds office for
- within X days
- for X years

4. PENALTY QUESTIONS

For questions asking about a fine or penalty, prefer provisions that actually
state the applicable penalty, penalty units, offence consequence, or maximum
penalty.

A provision that only states the underlying obligation should normally receive
a lower score.

5. PROHIBITION / RESTRICTION QUESTIONS

Do not require the word "prohibited" to appear in the passage.

Consider whether the substantive provision establishes:
- a prohibition
- a restriction
- a limitation
- a condition
- a "must not" requirement
- the boundary of an authority or permission

However, do not invent a prohibition that is not supported by the passage.

6. TIMEFRAME QUESTIONS

For questions asking "how long", "when", "how many days", "how many years",
or similar time-related questions, strongly prefer provisions that explicitly
state the relevant duration, deadline, period, date, or term.

7. SECTION / SUBSECTION REFERENCES

If the question explicitly refers to a section or subsection, treat matching
section identifiers as strong evidence.

Example:
Query: "How do sections 32 and 67 interact?"
A passage from section 32 or section 67 is structurally relevant.

8. CROSS-SECTION QUESTIONS

If the question requires multiple sections, recognise that a single passage
may provide only part of the answer.

Rank passages based on their contribution to answering the question.

Do not assume that the passage with the highest lexical similarity is the
complete answer.

9. DO NOT USE EXTERNAL KNOWLEDGE

Judge the candidates only from the information provided in the candidate
metadata and text.

Do not assume facts that are not supported by the candidate passages.

--------------------------------------------------
SCORING
--------------------------------------------------

Assign each candidate one of the following scores:

2 = ANSWER-BEARING
The substantive text directly contains the information required to answer the
question.

1 = SUPPORTING / CONTEXTUAL
The passage is relevant and may help answer the question, but does not itself
contain the key answer.

0 = IRRELEVANT
The passage does not materially help answer the question.

IMPORTANT:

A keyword match or section-title match alone is NOT sufficient for score 2.

When choosing between candidates with similar relevance, prefer the candidate
whose substantive text provides more direct and specific evidence for the
answer.

--------------------------------------------------
INPUT
--------------------------------------------------

USER QUESTION:
{query}

CANDIDATE PASSAGES:
{candidates}

Each candidate contains:
- Candidate ID
- Act / document
- Section number
- Subsection
- Section title / heading
- Substantive text

--------------------------------------------------
OUTPUT
--------------------------------------------------

Return ONLY valid JSON.

{{
  "results": [
    {{
      "id": "candidate_id",
      "score": 2,
      "reason": "The substantive provision directly states the information needed to answer the question."
    }}
  ]
}}"""


def _format_candidates(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks):
        cid = str(i + 1)
        section_id = c.get("section_id", "")
        title = c.get("title", "")
        # Extract subsection marker from title if present, e.g. "Constitution (1)" → "(1)"
        sub_match = re.search(r"\((\d+[A-Za-z]?)\)$", title)
        subsection = f"({sub_match.group(1)})" if sub_match else ""
        heading = re.sub(r"\s*\(\d+[A-Za-z]?\)$", "", title).strip()
        text = c.get("text", "")
        # Strip the heading line from text if it duplicates (first line often repeats title)
        text_lines = text.splitlines()
        if text_lines and heading and text_lines[0].strip().startswith(section_id):
            text_body = "\n".join(text_lines[1:]).strip()
        else:
            text_body = text.strip()

        lines.append(
            f"---\n"
            f"Candidate ID: {cid}\n"
            f"Act / document: NSW Health Services Act 1997\n"
            f"Section number: {section_id}\n"
            f"Subsection: {subsection}\n"
            f"Section title / heading: {heading}\n"
            f"Substantive text: {text_body[:600]}"
        )
    return "\n\n".join(lines)


def llm_rerank_legal(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank candidates using legal-aware LLM prompt. Returns top_k by score desc."""
    if not candidates:
        return []

    formatted = _format_candidates(candidates)
    prompt = _PROMPT.format(query=query, candidates=formatted)

    try:
        resp = _client.chat.completions.create(
            model=os.environ["Z_AI_CHAT_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4000,
        )
        raw = (resp.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        if not raw:
            print(f"    [LLMReranker] empty response from model, falling back")
            return candidates[:top_k]

        # Extract JSON object if model added preamble text
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            raw = json_match.group(0)

        # Try strict parse first, then json_repair for malformed LLM output
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            repaired = repair_json(raw)
            data = json.loads(repaired)
        results = data.get("results", [])

        # Build id → score map
        score_map: dict[str, int] = {}
        for r in results:
            cid = str(r.get("id", "")).strip()
            score_map[cid] = int(r.get("score", 0))

        # Attach scores to candidates
        for i, c in enumerate(candidates):
            c["llm_score"] = score_map.get(str(i + 1), 0)

        # Sort by LLM score desc, break ties by original cross-encoder score
        reranked = sorted(
            candidates,
            key=lambda c: (c["llm_score"], c.get("rerank_score", 0)),
            reverse=True,
        )
        return reranked[:top_k]

    except Exception as e:
        # Fallback: return original order
        print(f"    [LLMReranker] error: {e}, falling back to cross-encoder order")
        return candidates[:top_k]
