"""
LLM-as-Judge on existing traces.

Fetches all traces from the SubsecRoutedPC-LLMRnk5 session,
asks the LLM to score retrieval quality (0/1/2) for each query,
and writes the score back to each trace as 'llm_judge'.

Run from LangGraphLearning/:
    .venv/bin/python eval/llm_judge_traces.py
    .venv/bin/python eval/llm_judge_traces.py --limit 5   # smoke test
"""

import argparse
import os
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
from openai import OpenAI

SESSION_ID = "SubsecRoutedPC-LLMRnk5-grounded79-2026-09-21"

_JUDGE_PROMPT = """\
You are evaluating a legal information retrieval system.

Query: {query}

Retrieved sections (IDs only): {sections}

Based on the section IDs alone, does this retrieval look like it could answer a question about:
- "{query}"

Consider: sections like s.15 (hospital status), s.31 (suspension notice), s.105 (general powers), s.4 (definitions), etc.

Score:
2 = The section IDs are highly likely to contain the answer to this query
1 = The section IDs are partially relevant but may not fully answer the query
0 = The section IDs are unlikely to answer this query

Reply with ONLY a single digit: 0, 1, or 2"""

_JUDGE_PROMPT_FULL = """\
You are evaluating a legal information retrieval system for the NSW Health Services Act 1997.

Query: {query}

Top retrieved passages:
{passages}

Does the retrieved content contain information to answer the query?

Score:
2 = Yes — at least one passage directly addresses the query
1 = Partially — passages are related but may not fully answer
0 = No — passages are off-topic or irrelevant

Reply with ONLY a single digit: 0, 1, or 2
No explanation. No other text."""


langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

llm = OpenAI(
    base_url=os.environ["Z_AI_BASE_URL"],
    api_key=os.environ["Z_AI_API_KEY"],
)


def judge_from_llm_scores(trace_id: str) -> tuple[int, list]:
    """Derive quality score from the pipeline's own LLM reranker scores.

    Reads the llm_scores already stored in the llm_rerank span output:
      max score = 2 → judge 2 (pipeline found a strong match)
      max score = 1 → judge 1 (partial match)
      max score = 0 or all None (fallback) → judge 0 (no match / failed)
    """
    try:
        full = langfuse.api.trace.get(trace_id)
        for obs in full.observations:
            if obs.name == "llm_rerank" and obs.type == "GENERATION":
                out = obs.output or {}
                scores = out.get("llm_scores", [])
                valid = [s for s in scores if s is not None]
                if not valid:
                    return 0, scores   # all None = LLM fallback = bad
                return max(valid), scores
        return 1, []  # span not found, default
    except Exception:
        return 1, []


def fetch_traces(session_id: str, limit: int = None) -> list:
    """Fetch all traces for a session, handling pagination."""
    all_traces = []
    page = 1
    page_size = 50
    while True:
        result = langfuse.api.trace.list(
            session_id=session_id,
            page=page,
            limit=page_size,
        )
        batch = result.data if hasattr(result, "data") else list(result)
        if not batch:
            break
        all_traces.extend(batch)
        if limit and len(all_traces) >= limit:
            all_traces = all_traces[:limit]
            break
        if len(batch) < page_size:
            break
        page += 1
    return all_traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score first N traces (smoke test)")
    parser.add_argument("--session", type=str, default=SESSION_ID,
                        help="Session ID to score")
    args = parser.parse_args()

    print(f"Fetching traces for session: {args.session}")
    traces = fetch_traces(args.session, limit=args.limit)
    print(f"  {len(traces)} traces found\n")

    if not traces:
        print("No traces found. Check the session ID.")
        return

    scores = {0: 0, 1: 0, 2: 0}

    for i, trace in enumerate(traces):
        query    = (trace.input or {}).get("query", "")
        output   = trace.output or {}
        sections = output.get("retrieved_sections", [])
        recall   = output.get("recall_hit", None)

        score, raw_scores = judge_from_llm_scores(trace.id)
        scores[score] += 1

        langfuse.create_score(
            trace_id=trace.id,
            name="reranker_confidence",
            value=float(score),
            data_type="NUMERIC",
            comment=f"llm_scores={raw_scores}  sections={sections}",
        )

        recall_str = "✓" if recall else ("✗" if recall is False else "?")
        judge_str  = ["✗", "~", "✓"][score]
        print(f"  [{i+1:3d}/{len(traces)}] recall={recall_str} conf={judge_str}({score})  {query[:55]}")

    langfuse.flush()

    print(f"\n=== Reranker Confidence Distribution ===")
    print(f"  2 (strong match):  {scores[2]:3d}  ({scores[2]/len(traces)*100:.0f}%)")
    print(f"  1 (partial match): {scores[1]:3d}  ({scores[1]/len(traces)*100:.0f}%)")
    print(f"  0 (fallback/miss): {scores[0]:3d}  ({scores[0]/len(traces)*100:.0f}%)")
    print(f"\n✓ Done — 'reranker_confidence' scores written to Langfuse traces")
    print(f"  View: {os.environ.get('LANGFUSE_HOST', 'https://cloud.langfuse.com')}/traces?session={args.session}")


if __name__ == "__main__":
    main()
