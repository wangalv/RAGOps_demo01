"""
M7.2 — Hybrid Search Test

Run from LangGraphLearning/:
    .venv/bin/python tests/rag/test_m7_2_hybrid.py

Shows side-by-side comparison for each query:
  - Dense only (M7.1 baseline)
  - BM25 only
  - Hybrid (Dense + BM25 via RRF)

The "Source" column shows where each result came from:
  D   = Dense only
  B   = BM25 only
  D+B = appeared in both lists (RRF gives these the highest boost)
"""

import sys
sys.path.insert(0, ".")

from src.rag.chunker import chunk_by_section
from src.rag.vector_store import VectorStore
from src.rag.hybrid_retriever import HybridRetriever

ACT_TEXT_FILE = "tests/rag/health_services_act.txt"
COLLECTION    = "health-services-act-baseline"
TOP_K         = 3

TEST_QUERIES = [
    "mandatory reporting obligations for health practitioners",
    "penalty for providing false information",
    "minister's power to make regulations",
    "patient rights and complaints procedure",
]


def print_table(results: list[dict], mode: str) -> None:
    col_sec   = 18
    col_title = 42
    col_score = 8
    col_src   = 5
    sep = f"+{'─'*4}+{'─'*col_sec}+{'─'*col_title}+{'─'*col_score}+{'─'*col_src}+"
    print(f"  [{mode}]")
    print(f"  {sep}")
    print(f"  │{'Rank':^4}│{'Section':^{col_sec}}│{'Title':^{col_title}}│{'Score':^{col_score}}│{'Src':^{col_src}}│")
    print(f"  {sep}")
    for rank, r in enumerate(results, 1):
        section = r["chunk_id"][:col_sec - 1]
        title   = r["title"][:col_title - 1]
        if mode == "Hybrid":
            score = f"{r['rrf_score']:.5f}"
            src   = ("D+B" if r["in_dense"] and r["in_bm25"]
                     else "D" if r["in_dense"] else "B")
        else:
            score = f"{r.get('score', 0):.3f}"
            src   = mode[0]  # "D" or "B"
        print(f"  │{rank:^4}│{section:<{col_sec}}│{title:<{col_title}}│{score:^{col_score}}│{src:^{col_src}}│")
    print(f"  {sep}")


def main():
    # ── Load + chunk ─────────────────────────────────────────────────────────
    print("Loading Act and building indexes...\n")
    with open(ACT_TEXT_FILE, encoding="utf-8") as f:
        act_text = f.read()
    chunks = chunk_by_section(act_text)
    print(f"  {len(chunks)} chunks")

    # ── Dense (reuse existing ChromaDB collection) ───────────────────────────
    store = VectorStore(collection_name=COLLECTION, persist_dir="./chroma_db")
    if store.count() != len(chunks):
        store.reset()
        store.add_chunks(chunks)

    # ── BM25 ─────────────────────────────────────────────────────────────────
    hybrid = HybridRetriever(vector_store=store)
    hybrid.index(chunks)

    # ── BM25-only helper ─────────────────────────────────────────────────────
    from rank_bm25 import BM25Okapi
    tokenised = [c["text"].lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenised)

    def bm25_search(query: str, k: int) -> list[dict]:
        scores = bm25_index.get_scores(query.lower().split())
        top_i  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**chunks[i], "score": scores[i]} for i in top_i]

    # ── Run queries ───────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("Side-by-side comparison: Dense vs BM25 vs Hybrid")
    print("=" * 75)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQuery {i}: \"{query}\"")
        print_table(store.search(query, top_k=TOP_K),         mode="Dense")
        print_table(bm25_search(query, k=TOP_K),              mode="BM25")
        print_table(hybrid.search(query, top_k=TOP_K),        mode="Hybrid")


if __name__ == "__main__":
    main()
