"""
M7.3 — Reranking Test

Run from LangGraphLearning/:
    .venv/bin/python tests/rag/test_m7_3_reranking.py

Shows three columns per query:
  Dense only (M7.1 baseline, top-3)
  Dense top-20 → Reranked top-3
  Hybrid top-20 → Reranked top-3

The "Before rank" column shows where the result sat BEFORE reranking —
if a result was rank 15 before and rank 1 after, the reranker rescued it.
"""

import sys
sys.path.insert(0, ".")

from src.rag.chunker import chunk_by_section
from src.rag.vector_store import VectorStore
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import Reranker

ACT_TEXT_FILE = "tests/rag/health_services_act.txt"
COLLECTION    = "health-services-act-baseline"
CANDIDATE_N   = 20   # how many candidates to fetch before reranking
FINAL_K       = 3    # how many to show after reranking

TEST_QUERIES = [
    "mandatory reporting obligations for health practitioners",
    "penalty for providing false information",
    "minister's power to make regulations",
    "patient rights and complaints procedure",
]


def print_table(label: str, results: list[dict], score_key: str) -> None:
    col_sec   = 18
    col_title = 40
    col_score = 8
    col_prev  = 6
    sep = f"+{'─'*4}+{'─'*col_sec}+{'─'*col_title}+{'─'*col_score}+{'─'*col_prev}+"
    print(f"  [{label}]")
    print(f"  {sep}")
    print(f"  │{'Rank':^4}│{'Section':^{col_sec}}│{'Title':^{col_title}}│{'Score':^{col_score}}│{'Prev':^{col_prev}}│")
    print(f"  {sep}")
    for rank, r in enumerate(results, 1):
        section = r["chunk_id"][:col_sec - 1]
        title   = r["title"][:col_title - 1]
        score   = f"{r.get(score_key, 0):.3f}"
        prev    = str(r.get("prev_rank", "-"))
        print(f"  │{rank:^4}│{section:<{col_sec}}│{title:<{col_title}}│{score:^{col_score}}│{prev:^{col_prev}}│")
    print(f"  {sep}")


def main():
    # ── Load + chunk ─────────────────────────────────────────────────────────
    print("Loading Act and building indexes...\n")
    with open(ACT_TEXT_FILE, encoding="utf-8") as f:
        act_text = f.read()
    chunks = chunk_by_section(act_text)

    # ── Dense store ──────────────────────────────────────────────────────────
    store = VectorStore(collection_name=COLLECTION, persist_dir="./chroma_db")
    if store.count() != len(chunks):
        store.reset()
        store.add_chunks(chunks)

    # ── Hybrid retriever ─────────────────────────────────────────────────────
    hybrid = HybridRetriever(vector_store=store)
    hybrid.index(chunks)

    # ── Reranker (downloads ~80MB on first run) ───────────────────────────────
    reranker = Reranker()

    # ── Run queries ───────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("Comparison: Dense@3  vs  Dense@20→Reranked  vs  Hybrid@20→Reranked")
    print("'Prev' = rank before reranking (shows how much reranker reshuffled)")
    print("=" * 75)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQuery {i}: \"{query}\"")

        # Dense top-3 (baseline)
        dense3 = store.search(query, top_k=FINAL_K)
        print_table("Dense top-3 (baseline)", dense3, "score")

        # Dense top-20 → rerank → top-3
        dense20 = store.search(query, top_k=CANDIDATE_N)
        for rank, r in enumerate(dense20, 1):
            r["prev_rank"] = rank
        dense_reranked = reranker.rerank(query, dense20, top_k=FINAL_K)
        print_table(f"Dense top-{CANDIDATE_N} → Reranked top-{FINAL_K}", dense_reranked, "rerank_score")

        # Hybrid top-20 → rerank → top-3
        hybrid20 = hybrid.search(query, top_k=CANDIDATE_N)
        for rank, r in enumerate(hybrid20, 1):
            r["prev_rank"] = rank
        hybrid_reranked = reranker.rerank(query, hybrid20, top_k=FINAL_K)
        print_table(f"Hybrid top-{CANDIDATE_N} → Reranked top-{FINAL_K}", hybrid_reranked, "rerank_score")


if __name__ == "__main__":
    main()
