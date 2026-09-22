"""
M5.2 — RAG Retrieval Evaluation

Evaluates all M7 retrieval techniques against the Golden Set (GoldenSet_v2.xlsx).

Metrics computed per query:
  Recall@k    = |retrieved ∩ relevant| / |relevant|
  Precision@k = |retrieved ∩ relevant| / k

Averages are reported overall and broken down by Query Type.

Run from LangGraphLearning/:
    .venv/bin/python eval/run_eval.py

Results are printed to stdout. Techniques that call GLM (HyDE, MultiQuery,
Step-back) add ~10–15 seconds per query — the full run takes ~5–8 minutes.
Pass --fast to skip LLM-based techniques and only run local retrievers.
"""

import argparse
import os
import re
import sys
sys.path.insert(0, ".")

import openpyxl
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

from src.rag.chunker import chunk_by_section, chunk_parent_child
from src.rag.vector_store import VectorStore
from src.rag.reranker import Reranker
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.hyde_retriever import HyDERetriever
from src.rag.multi_query_retriever import MultiQueryRetriever
from src.rag.stepback_retriever import StepBackRetriever

ACT_TEXT_FILE      = "tests/rag/health_services_act.txt"
GOLDEN_SET_FILE    = "GoldenSet_v2.xlsx"
PARENT_COLLECTION  = "health-services-act-baseline"
CHILD_COLLECTION   = "health-services-act-children"
CANDIDATE_N        = 20
EVAL_K             = [3, 5]


# ── Golden Set loading ────────────────────────────────────────────────────────

def load_golden_set(path: str) -> list[dict]:
    """Load queries and relevant section annotations from the Excel file."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Golden Set v2"]

    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    col = {name: i for i, name in enumerate(header)}

    cases = []
    for row in rows[1:]:
        if not row[col["ID"]]:
            continue
        primary   = row[col["Primary Section"]] or ""
        secondary = row[col["Secondary Sections"]] or ""
        # Parse comma-separated section references like "s.105, s.106, s.107"
        all_relevant = []
        for ref in (primary + "," + secondary).split(","):
            ref = ref.strip()
            if ref and ref.lower() != "none":
                all_relevant.append(ref)

        cases.append({
            "id":         row[col["ID"]],
            "query":      row[col["Question"]],
            "relevant":   all_relevant,      # e.g. ["s.117", "s.99"]
            "primary":    primary.strip(),
            "query_type": row[col["Query Type"]],
            "difficulty": row[col["Difficulty"]],
        })
    return cases


# ── Section ID matching ───────────────────────────────────────────────────────

def extract_section_prefix(chunk_id: str) -> str:
    """Convert 's.117A_L3834' → 's.117A' for matching against Golden Set refs."""
    # chunk_id format: s.117A_L3834 or s.Chapter_3_L918
    m = re.match(r"^(s\.[^_]+)", chunk_id)
    return m.group(1) if m else chunk_id


def hits(retrieved: list[dict], relevant: list[str]) -> set[str]:
    """Return the subset of relevant sections found in the retrieved list."""
    retrieved_prefixes = {extract_section_prefix(r["chunk_id"]) for r in retrieved}
    return {sec for sec in relevant if sec in retrieved_prefixes}


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[dict], relevant: list[str], k: int) -> float:
    if not relevant:
        return 1.0
    found = hits(retrieved[:k], relevant)
    return len(found) / len(relevant)


def precision_at_k(retrieved: list[dict], relevant: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    found = hits(retrieved[:k], relevant)
    return len(found) / k


# ── Retrieval runners ─────────────────────────────────────────────────────────

def run_dense_baseline(store, query, k):
    return store.search(query, top_k=k)


def run_dense_reranked(store, reranker, query, candidate_n, final_k):
    candidates = store.search(query, top_k=candidate_n)
    return reranker.rerank(query, candidates, top_k=final_k)


def run_hybrid_reranked(hybrid, reranker, query, candidate_n, final_k):
    candidates = hybrid.search(query, top_k=candidate_n)
    return reranker.rerank(query, candidates, top_k=final_k)


def run_parent_child_reranked(child_store, parent_store, reranker, query, candidate_n, final_k):
    child_hits = child_store.search_children(query, top_k=candidate_n)
    parent_ids = [c["parent_id"] for c in child_hits]
    parents = child_store.get_parents_by_ids(parent_store, parent_ids)
    return reranker.rerank(query, parents, top_k=final_k)


def run_hyde_reranked(hyde, reranker, query, candidate_n, final_k):
    candidates, _ = hyde.search(query, top_k=candidate_n)
    for rank, r in enumerate(candidates, 1):
        r["prev_rank"] = rank
    return reranker.rerank(query, candidates, top_k=final_k)


def run_multiquery_reranked(mq, query, candidate_n, final_k):
    reranked, _ = mq.search(query, n_variants=3, per_query_k=candidate_n, final_k=final_k)
    return reranked


def run_stepback_reranked(sb, query, candidate_n, final_k):
    reranked, _ = sb.search(query, per_query_k=candidate_n, final_k=final_k)
    return reranked


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(cases, retrievers, ks):
    """Run all retrievers against all cases; return per-case results."""
    results = []
    total = len(cases)

    for i, case in enumerate(cases, 1):
        query    = case["query"]
        relevant = case["relevant"]
        row = {
            "id":         case["id"],
            "query":      query,
            "relevant":   relevant,
            "query_type": case["query_type"],
            "difficulty": case["difficulty"],
            "metrics":    {},
        }

        print(f"  [{i:02d}/{total}] {case['id']} — {query[:60]}...")

        trace_id = langfuse.create_trace_id()

        for name, fn in retrievers.items():
            try:
                retrieved = fn(query)
                metrics = {
                    f"R@{k}": recall_at_k(retrieved, relevant, k) for k in ks
                } | {
                    f"P@{k}": precision_at_k(retrieved, relevant, k) for k in ks
                }
                row["metrics"][name] = metrics

                for metric_name, value in metrics.items():
                    langfuse.create_score(
                        trace_id=trace_id,
                        name=f"{name}/{metric_name}",
                        value=value,
                        comment=f"query_id={case['id']} type={case['query_type']} diff={case['difficulty']}",
                    )

            except Exception as e:
                print(f"    ⚠️  {name} failed: {e}")
                row["metrics"][name] = {f"R@{k}": None for k in ks} | {f"P@{k}": None for k in ks}

        results.append(row)
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def avg(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def print_summary(results, technique_names, ks):
    metrics = [f"R@{k}" for k in ks] + [f"P@{k}" for k in ks]

    # Overall average
    print("\n" + "=" * 90)
    print("OVERALL AVERAGE (18 queries)")
    print("=" * 90)
    col_w = 10
    header = f"{'Technique':<28}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name in technique_names:
        row_vals = []
        for m in metrics:
            vals = [r["metrics"].get(name, {}).get(m) for r in results]
            row_vals.append(avg(vals))
        line = f"{name:<28}" + "".join(
            f"{v:>{col_w}.3f}" if v is not None else f"{'N/A':>{col_w}}" for v in row_vals
        )
        print(line)

    # Per query-type breakdown
    query_types = sorted({r["query_type"] for r in results})
    for qt in query_types:
        subset = [r for r in results if r["query_type"] == qt]
        print(f"\n── {qt} (n={len(subset)}) ──")
        print(f"{'Technique':<28}" + "".join(f"{m:>{col_w}}" for m in metrics))
        print("-" * len(header))
        for name in technique_names:
            row_vals = []
            for m in metrics:
                vals = [r["metrics"].get(name, {}).get(m) for r in subset]
                row_vals.append(avg(vals))
            line = f"{name:<28}" + "".join(
                f"{v:>{col_w}.3f}" if v is not None else f"{'N/A':>{col_w}}" for v in row_vals
            )
            print(line)

    # Per-case detail for R@3
    print("\n" + "=" * 90)
    print("PER-QUERY DETAIL — Recall@3")
    print("=" * 90)
    col_w2 = 8
    hdr = f"{'ID':<6}{'Type':<24}{'Diff':<8}" + "".join(f"{n[:col_w2]:>{col_w2}}" for n in technique_names)
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        vals = [r["metrics"].get(n, {}).get("R@3") for n in technique_names]
        line = (
            f"{r['id']:<6}"
            f"{r['query_type']:<24}"
            f"{r['difficulty']:<8}"
            + "".join(
                f"{v:>{col_w2}.2f}" if v is not None else f"{'N/A':>{col_w2}}" for v in vals
            )
        )
        print(line)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip LLM-based techniques (HyDE, MultiQuery, Step-back)")
    parser.add_argument("--only-dense", action="store_true",
                        help="Only run Dense@3 and Dense@5")
    args = parser.parse_args()

    print("Loading Golden Set...")
    cases = load_golden_set(GOLDEN_SET_FILE)
    print(f"  {len(cases)} queries loaded\n")

    print("Loading Act and building indexes...")
    with open(ACT_TEXT_FILE, encoding="utf-8") as f:
        act_text = f.read()

    parents, children = chunk_parent_child(act_text)
    chunks = chunk_by_section(act_text)

    parent_store = VectorStore(collection_name=PARENT_COLLECTION, persist_dir="./chroma_db")
    if parent_store.count() != len(chunks):
        parent_store.reset()
        parent_store.add_chunks(chunks)
    else:
        print(f"  Parent store: {parent_store.count()} chunks (reused)")

    child_store = VectorStore(collection_name=CHILD_COLLECTION, persist_dir="./chroma_db")
    if child_store.count() != len(children):
        child_store.reset()
        child_store.add_children(children)
    else:
        print(f"  Child store:  {child_store.count()} chunks (reused)")

    reranker = Reranker()
    hybrid   = HybridRetriever(vector_store=parent_store)
    hybrid.index(chunks)
    hyde     = HyDERetriever(vector_store=parent_store)
    mq       = MultiQueryRetriever(vector_store=parent_store, reranker=reranker)
    sb       = StepBackRetriever(vector_store=parent_store, reranker=reranker)

    max_k = max(EVAL_K)

    # Build retriever registry — each entry is a callable (query) -> list[dict]
    retrievers = {
        "Dense@3":           lambda q: run_dense_baseline(parent_store, q, 3),
        "Dense@5":           lambda q: run_dense_baseline(parent_store, q, 5),
    }

    if not args.only_dense:
        retrievers.update({
            "Dense→Reranked@3":  lambda q: run_dense_reranked(parent_store, reranker, q, CANDIDATE_N, 3),
            "Dense→Reranked@5":  lambda q: run_dense_reranked(parent_store, reranker, q, CANDIDATE_N, 5),
            "Hybrid→Reranked@3": lambda q: run_hybrid_reranked(hybrid, reranker, q, CANDIDATE_N, 3),
            "Hybrid→Reranked@5": lambda q: run_hybrid_reranked(hybrid, reranker, q, CANDIDATE_N, 5),
            "ParentChild→Rnk@3": lambda q: run_parent_child_reranked(child_store, parent_store, reranker, q, CANDIDATE_N, 3),
            "ParentChild→Rnk@5": lambda q: run_parent_child_reranked(child_store, parent_store, reranker, q, CANDIDATE_N, 5),
        })

    if not args.fast and not args.only_dense:
        retrievers.update({
            "HyDE→Reranked@3":   lambda q: run_hyde_reranked(hyde, reranker, q, CANDIDATE_N, 3),
            "HyDE→Reranked@5":   lambda q: run_hyde_reranked(hyde, reranker, q, CANDIDATE_N, 5),
            "MultiQuery→Rnk@3":  lambda q: run_multiquery_reranked(mq, q, CANDIDATE_N, 3),
            "MultiQuery→Rnk@5":  lambda q: run_multiquery_reranked(mq, q, CANDIDATE_N, 5),
            "StepBack→Rnk@3":    lambda q: run_stepback_reranked(sb, q, CANDIDATE_N, 3),
            "StepBack→Rnk@5":    lambda q: run_stepback_reranked(sb, q, CANDIDATE_N, 5),
        })
    else:
        print("  [--fast] Skipping HyDE, MultiQuery, Step-back\n")

    technique_names = list(retrievers.keys())

    print("\nRunning evaluation...")
    print(f"  {len(cases)} queries × {len(retrievers)} retrievers\n")

    results = evaluate(cases, retrievers, EVAL_K)
    print_summary(results, technique_names, EVAL_K)

    langfuse.flush()
    print("\n✓ Results uploaded to Langfuse")


if __name__ == "__main__":
    main()
