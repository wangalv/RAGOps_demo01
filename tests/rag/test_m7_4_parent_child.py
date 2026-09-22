"""
M7.4 — Parent-Child Chunking Test

Run from LangGraphLearning/:
    .venv/bin/python tests/rag/test_m7_4_parent_child.py

Flow:
  1. Split Act into parent chunks (section-level, for generation)
     and child chunks (sentence-level, for retrieval)
  2. Store parents in collection A, children in collection B
  3. Query: search children → get parent_ids → fetch parents → rerank

Comparison per query:
  [Dense@3 baseline]            M7.1: section-level retrieval
  [Dense@20 → Reranked]         M7.3: section-level + reranking
  [Parent-Child → Reranked]     M7.4: sentence-level retrieval, section-level generation
"""

import sys
sys.path.insert(0, ".")

from src.rag.chunker import chunk_by_section, chunk_parent_child
from src.rag.vector_store import VectorStore
from src.rag.reranker import Reranker

ACT_TEXT_FILE    = "tests/rag/health_services_act.txt"
PARENT_COLLECTION = "health-services-act-baseline"
CHILD_COLLECTION  = "health-services-act-children"
CANDIDATE_N       = 20
FINAL_K           = 3

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
    print("Loading Act text...")
    with open(ACT_TEXT_FILE, encoding="utf-8") as f:
        act_text = f.read()

    # ── Build parent and child chunks ─────────────────────────────────────────
    parents, children = chunk_parent_child(act_text)
    print(f"  Parents: {len(parents)} section chunks")
    print(f"  Children: {len(children)} sentence chunks")

    # ── Parent store (section-level, reuse existing) ──────────────────────────
    parent_store = VectorStore(collection_name=PARENT_COLLECTION, persist_dir="./chroma_db")
    if parent_store.count() != len(parents):
        parent_store.reset()
        parent_store.add_chunks(parents)
    else:
        print(f"  Parent store: {parent_store.count()} chunks already indexed")

    # ── Child store (sentence-level, new collection) ──────────────────────────
    child_store = VectorStore(collection_name=CHILD_COLLECTION, persist_dir="./chroma_db")
    if child_store.count() != len(children):
        child_store.reset()
        child_store.add_children(children)
    else:
        print(f"  Child store: {child_store.count()} chunks already indexed")

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker = Reranker()

    # ── Run queries ───────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("Comparison: Dense@3  vs  Dense@20→Reranked  vs  Parent-Child→Reranked")
    print("=" * 75)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQuery {i}: \"{query}\"")

        # M7.1 baseline: section-level Dense top-3
        dense3 = parent_store.search(query, top_k=FINAL_K)
        print_table("Dense@3 baseline", dense3, "score")

        # M7.3: section-level Dense top-20 → rerank
        dense20 = parent_store.search(query, top_k=CANDIDATE_N)
        for rank, r in enumerate(dense20, 1):
            r["prev_rank"] = rank
        dense_reranked = reranker.rerank(query, dense20, top_k=FINAL_K)
        print_table(f"Dense@{CANDIDATE_N} → Reranked", dense_reranked, "rerank_score")

        # M7.4: child-level retrieval → fetch parents → rerank parents
        child_hits = child_store.search_children(query, top_k=CANDIDATE_N)
        parent_ids = [c["parent_id"] for c in child_hits]
        parent_candidates = child_store.get_parents_by_ids(parent_store, parent_ids)

        # tag prev_rank (order from child retrieval)
        for rank, r in enumerate(parent_candidates, 1):
            r["prev_rank"] = rank

        pc_reranked = reranker.rerank(query, parent_candidates, top_k=FINAL_K)
        print_table("Parent-Child → Reranked", pc_reranked, "rerank_score")


if __name__ == "__main__":
    main()
