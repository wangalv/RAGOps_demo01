"""
M5.3 Layer 3 — Simulated Query Generation + Evaluation

Generates 100 new queries across 8 query types using the Act's section list,
then runs HyDE→Reranked@5 on each and scores with LLM-as-Judge.

Output:
  - simulated_queries.xlsx  (queries + retrieved sections + judge scores)
  - Langfuse Experiment run: "SimulatedQueries-HyDE@5"

Run:
    .venv/bin/python eval/simulate_queries.py
    .venv/bin/python eval/simulate_queries.py --generate-only   # skip retrieval
    .venv/bin/python eval/simulate_queries.py --eval-only       # skip generation, use existing xlsx
"""

import argparse
import json
import os
import re
import sys
sys.path.insert(0, ".")

import openpyxl
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
from langfuse.openai import OpenAI

from src.rag.chunker import chunk_by_section
from src.rag.vector_store import VectorStore
from src.rag.reranker import Reranker
from src.rag.hyde_retriever import HyDERetriever

ACT_TEXT_FILE     = "tests/rag/health_services_act.txt"
PARENT_COLLECTION = "health-services-act-baseline"
OUTPUT_XLSX       = "simulated_queries.xlsx"
CANDIDATE_N       = 20
FINAL_K           = 5

_client = OpenAI(
    base_url=os.environ["Z_AI_BASE_URL"],
    api_key=os.environ["Z_AI_API_KEY"],
)

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

# ── Query type definitions ────────────────────────────────────────────────────

QUERY_TYPES = {
    "Procedural": {
        "description": "How to do something — process, steps, application, appeal procedures",
        "examples": [
            "What is the process for establishing a new local health district?",
            "How does a visiting practitioner appeal a suspension decision?",
        ],
        "count": 13,
    },
    "Definitional": {
        "description": "What something is defined as — meanings, scope, what qualifies",
        "examples": [
            "How is 'health service' defined under the Act?",
            "What constitutes a 'statutory health corporation'?",
        ],
        "count": 12,
    },
    "Penalty": {
        "description": "Consequences, offences, fines, penalties for non-compliance",
        "examples": [
            "What are the penalties for a board member who has a conflict of interest?",
            "What happens if someone provides false information to the Secretary?",
        ],
        "count": 12,
    },
    "Authority": {
        "description": "Who has power or responsibility — Minister, Secretary, Board, CEO, Governor",
        "examples": [
            "Who has the power to appoint the chief executive of a local health district?",
            "Which body can make by-laws governing a public hospital?",
        ],
        "count": 13,
    },
    "Timeframe": {
        "description": "Deadlines, time limits, notice periods, days/months requirements",
        "examples": [
            "Within how many days must an incident be reported to the Secretary?",
            "How long can a service contract run before it must be renewed?",
        ],
        "count": 12,
    },
    "Eligibility": {
        "description": "Conditions, prerequisites, qualifications to hold a position or exercise a right",
        "examples": [
            "What qualifications must a board member have to be appointed?",
            "Under what conditions can an affiliated health organisation be recognised?",
        ],
        "count": 12,
    },
    "CrossSection": {
        "description": "Questions requiring information from multiple sections or chapters",
        "examples": [
            "How do ministerial directions interact with a board's governance powers?",
            "What is the relationship between service contracts and clinical privilege decisions?",
        ],
        "count": 13,
    },
    "Prohibition": {
        "description": "What is not allowed, restricted, or prohibited — 'shall not', 'must not'",
        "examples": [
            "What are health board members prohibited from doing while in office?",
            "What activities cannot be delegated by the Secretary under this Act?",
        ],
        "count": 13,
    },
}

# ── Section index builder ─────────────────────────────────────────────────────

ACT_OVERVIEW = """\
The Health Services Act 1997 (NSW) covers:
- Chapter 1: Preliminary — definitions, objects of the Act, public health system overview
- Chapter 2: Structure of public health system — local health districts, statutory health corporations, affiliated health organisations, public hospitals
- Chapter 3: Local health districts — constitution, boards, board functions, board members, CEO appointment, delegation of functions, by-laws, ministerial directions
- Chapter 4: Statutory health corporations — constitution, boards, functions, governance, staff, assets
- Chapter 5: Affiliated health organisations — recognition, functions, accountability
- Chapter 6: Health services — visiting practitioners, service contracts (s.69-s.99), clinical privileges, appeal rights, reporting obligations for visiting practitioners charged with serious offences
- Chapter 7: Special provisions — complaints, investigation, suspension of services, ministerial intervention
- Chapter 8: Staff — employment, conditions, delegation
- Chapter 9: Finance — budget, accounts, audit, financial management
- Chapter 10: Miscellaneous — transfer of hospitals, dissolution, regulations, by-law making power (s.140)
- Schedules — listed health organisations, savings and transitional provisions

Key governance bodies: Minister for Health, Secretary of Health, local health district boards, statutory health corporation boards, Health Administration Corporation.
Key concepts: clinical privileges, visiting medical officers, service contracts, public health organisations, health support services, mandatory reporting.
"""


def build_section_index(chunks: list[dict]) -> str:
    return ACT_OVERVIEW

# ── Query generation ──────────────────────────────────────────────────────────

_GENERATE_PROMPT = """\
You are generating test queries for a legal information retrieval system \
based on the Health Services Act 1997 (NSW).

Below is a list of sections in the Act (section id: first line of content):
---
{section_index}
---

Generate exactly {total} queries spread across the following 8 query types. \
For each type, generate the specified number of queries.

Query types and counts:
{type_specs}

Requirements for all queries:
- Each query must be answerable by at least one section in the Act above
- Spread queries across different parts of the Act (don't cluster in one area)
- Vary phrasing — some formal legal language, some plain English
- All queries must be in ENGLISH
- Do NOT number the queries

Return a JSON object with this exact structure:
{{
  "Procedural": ["query1", "query2", ...],
  "Definitional": ["query1", "query2", ...],
  "Penalty": ["query1", "query2", ...],
  "Authority": ["query1", "query2", ...],
  "Timeframe": ["query1", "query2", ...],
  "Eligibility": ["query1", "query2", ...],
  "CrossSection": ["query1", "query2", ...],
  "Prohibition": ["query1", "query2", ...]
}}"""


def generate_all_queries(section_index: str) -> list[dict]:
    type_specs = "\n".join(
        f"  {qtype} ({cfg['count']} queries): {cfg['description']}"
        for qtype, cfg in QUERY_TYPES.items()
    )
    total = sum(cfg["count"] for cfg in QUERY_TYPES.values())

    prompt = _GENERATE_PROMPT.format(
        section_index=section_index,
        type_specs=type_specs,
        total=total,
    )

    print(f"  Calling LLM for all {total} queries (one request)...")
    response = _client.chat.completions.create(
        model=os.environ["Z_AI_CHAT_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=6000,
        temperature=0.8,
    )
    content = (response.choices[0].message.content or "").strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```[a-z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Raw response (first 500 chars): {content[:500]}")
        return []

    all_queries = []
    for qtype, cfg in QUERY_TYPES.items():
        queries = data.get(qtype, [])
        cap = cfg["count"]
        print(f"  {qtype:<15} → got {len(queries)} (want {cap})")
        for q in queries[:cap]:
            if isinstance(q, str) and q.strip():
                all_queries.append({"query": q.strip(), "query_type": qtype})
    return all_queries


def save_queries_xlsx(queries: list[dict], path: str):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Simulated Queries"
    ws.append(["ID", "Query Type", "Query", "Retrieved Sections", "Judge Scores", "Precision@5"])
    for i, row in enumerate(queries, 1):
        ws.append([i, row["query_type"], row["query"], "", "", ""])
    wb.save(path)
    print(f"  Saved {len(queries)} queries → {path}")


def load_queries_xlsx(path: str) -> list[dict]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    col = {name: i for i, name in enumerate(header)}
    queries = []
    for row in rows[1:]:
        if not row[col["ID"]]:
            continue
        queries.append({
            "id": row[col["ID"]],
            "query_type": row[col["Query Type"]],
            "query": row[col["Query"]],
        })
    return queries

# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are evaluating a legal document retrieval system.

User query: {query}

Retrieved section (from Health Services Act 1997):
---
{chunk_text}
---

Is this section relevant to answering the query? \
Consider it relevant if it contains information that directly helps answer the question, even partially.

You MUST respond in English with ONLY the single word YES or NO. No other text.
"""


def judge_chunk(query: str, chunk_text: str) -> int:
    response = _client.chat.completions.create(
        model=os.environ["Z_AI_CHAT_MODEL"],
        messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
            query=query,
            chunk_text=chunk_text[:600],
        )}],
        max_tokens=5,
        temperature=0,
    )
    answer = (response.choices[0].message.content or "").strip().lower()
    return 1 if answer.startswith("yes") else 0

# ── Evaluation ────────────────────────────────────────────────────────────────

def extract_section_prefix(chunk_id: str) -> str:
    m = re.match(r"^(s\.[^_]+)", chunk_id)
    return m.group(1) if m else chunk_id


SIM_DATASET_NAME = "NSW-HealthAct-SimulatedQueries-100"


def setup_sim_dataset(queries: list[dict]):
    try:
        langfuse.create_dataset(
            name=SIM_DATASET_NAME,
            description="100 LLM-generated queries across 8 types — no ground truth",
        )
        print(f"  Created dataset '{SIM_DATASET_NAME}'")
    except Exception:
        print(f"  Dataset '{SIM_DATASET_NAME}' already exists, reusing")

    for row in queries:
        langfuse.create_dataset_item(
            dataset_name=SIM_DATASET_NAME,
            input={"query": row["query"], "query_type": row["query_type"]},
            expected_output=None,
            id=f"simq-{row['id']}",
        )

    dataset = langfuse.get_dataset(SIM_DATASET_NAME)
    print(f"  {len(dataset.items)} items ready\n")
    return dataset


def run_evaluation(queries: list[dict], hyde: HyDERetriever, reranker: Reranker):
    dataset = setup_sim_dataset(queries)

    def task(item):
        query = item.input["query"]
        candidates, _ = hyde.search(query, top_k=CANDIDATE_N)
        chunks = reranker.rerank(query, candidates, top_k=FINAL_K)
        return {
            "retrieved_sections": [extract_section_prefix(c["chunk_id"]) for c in chunks],
            "retrieved_texts":    [c["text"][:600] for c in chunks],
        }

    def llm_judge_precision(*, input, output, expected_output, metadata=None, **kwargs):
        query  = (input or {}).get("query", "")
        texts  = (output or {}).get("retrieved_texts", [])
        if not texts:
            return {"name": "LLMJudge-Precision@5", "value": 0.0}
        scores = [judge_chunk(query, t) for t in texts]
        return {"name": "LLMJudge-Precision@5", "value": sum(scores) / len(scores)}

    print(f"  Running experiment 'SimulatedQueries-HyDE@5'...")
    result = langfuse.run_experiment(
        name=SIM_DATASET_NAME,
        run_name="SimulatedQueries-HyDE@5-v2",
        data=dataset.items,
        task=task,
        evaluators=[llm_judge_precision],
        max_concurrency=1,
    )

    # Collect results for local xlsx
    rows = []
    for ir in result.item_results:
        q_input = ir.item.input
        output  = ir.output or {}
        sections = output.get("retrieved_sections", [])
        score = next((e.value for e in ir.evaluations if e.name == "LLMJudge-Precision@5"), 0)
        rows.append({
            "id":                ir.item.id,
            "query_type":        q_input.get("query_type", ""),
            "query":             q_input.get("query", ""),
            "retrieved_sections": ", ".join(sections),
            "judge_scores":      "",
            "precision_at_5":    round(score, 2),
        })

    avg = sum(r["precision_at_5"] for r in rows) / len(rows) if rows else 0
    print(f"  avg LLMJudge-Precision@5 = {avg:.3f}  ({len(rows)} items)")
    if result.dataset_run_url:
        print(f"  → {result.dataset_run_url}")
    return rows


def save_results_xlsx(results: list[dict], path: str):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.append(["ID", "Query Type", "Query", "Retrieved Sections", "Judge Scores", "Precision@5"])
    for r in results:
        ws.append([
            r.get("id", ""),
            r["query_type"],
            r["query"],
            r.get("retrieved_sections", ""),
            r.get("judge_scores", ""),
            r.get("precision_at_5", ""),
        ])

    # Summary sheet
    ws2 = wb.create_sheet("Summary by Type")
    ws2.append(["Query Type", "Count", "Avg Precision@5"])
    from collections import defaultdict
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_type[r["query_type"]].append(r.get("precision_at_5", 0))
    for qtype, vals in sorted(by_type.items()):
        ws2.append([qtype, len(vals), round(sum(vals) / len(vals), 3)])

    wb.save(path)
    print(f"\n  Results saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate queries, skip retrieval+judging")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip generation, load existing simulated_queries.xlsx")
    args = parser.parse_args()

    print("Loading Act and building indexes...")
    with open(ACT_TEXT_FILE, encoding="utf-8") as f:
        act_text = f.read()
    chunks = chunk_by_section(act_text)

    parent_store = VectorStore(collection_name=PARENT_COLLECTION, persist_dir="./chroma_db")
    if parent_store.count() != len(chunks):
        parent_store.reset()
        parent_store.add_chunks(chunks)
    else:
        print(f"  Parent store: {parent_store.count()} chunks (reused)")

    reranker = Reranker()
    hyde     = HyDERetriever(vector_store=parent_store)

    # Step 1: Generate queries
    if not args.eval_only:
        print("\nGenerating queries...")
        section_index = build_section_index(chunks)
        queries = generate_all_queries(section_index)
        print(f"\n  Total generated: {len(queries)} queries")
        save_queries_xlsx(queries, OUTPUT_XLSX)

        if args.generate_only:
            print("\nDone — review simulated_queries.xlsx before running --eval-only")
            return
    else:
        print(f"\nLoading queries from {OUTPUT_XLSX}...")
        queries = load_queries_xlsx(OUTPUT_XLSX)
        print(f"  {len(queries)} queries loaded")

    # Step 2: Run HyDE@5 + LLM Judge
    print(f"\nRunning HyDE→Reranked@{FINAL_K} + LLM Judge on {len(queries)} queries...")
    results = run_evaluation(queries, hyde, reranker)

    # Step 3: Save + summarize
    output_path = OUTPUT_XLSX.replace(".xlsx", "_results.xlsx")
    save_results_xlsx(results, output_path)

    # Print summary
    from collections import defaultdict
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_type[r["query_type"]].append(r.get("precision_at_5", 0))

    print("\n── Summary by Query Type ──────────────────────")
    for qtype, vals in sorted(by_type.items()):
        avg = sum(vals) / len(vals)
        bar = "█" * int(avg * 20)
        print(f"  {qtype:<15} P@5={avg:.3f}  {bar}")

    all_vals = [r.get("precision_at_5", 0) for r in results]
    print(f"\n  Overall avg P@5 = {sum(all_vals)/len(all_vals):.3f}  ({len(all_vals)} queries)")
    print("\n✓ Done — check simulated_queries_results.xlsx for details")
    print("  Traces logged to Langfuse → Traces (filter by name: simulated-query-eval)")

    langfuse.flush()


if __name__ == "__main__":
    main()
