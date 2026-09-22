"""
M7 补充 — Multi-Query Retrieval Test

Run from LangGraphLearning/:
    .venv/bin/python tests/rag/test_m7_6_multi_query.py

Comparison per query:
  [Dense@3 baseline]        direct query embedding, top-3
  [Dense@20 → Reranked]     M7.3: section-level + reranking
  [MultiQuery → Reranked]   original + 3 variants → merge → rerank

The generated query variants are printed under each query so you can
inspect whether GLM used different legal vocabulary.
"""

import sys
sys.path.insert(0, ".")

from src.rag.chunker import chunk_by_section
from src.rag.vector_store import VectorStore
from src.rag.reranker import Reranker
from src.rag.multi_query_retriever import MultiQueryRetriever

ACT_TEXT_FILE = "tests/rag/health_services_act.txt"
COLLECTION    = "health-services-act-baseline"
CANDIDATE_N   = 20
FINAL_K       = 3

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
    print("Loading Act and building index...\n")
    with open(ACT_TEXT_FILE, encoding="utf-8") as f:
        act_text = f.read()
    chunks = chunk_by_section(act_text)

    store = VectorStore(collection_name=COLLECTION, persist_dir="./chroma_db")
    if store.count() != len(chunks):
        store.reset()
        store.add_chunks(chunks)
    else:
        print(f"  Reusing {store.count()} indexed chunks\n")

    reranker  = Reranker()
    mq        = MultiQueryRetriever(vector_store=store, reranker=reranker)

    print("\n" + "=" * 75)
    print("Comparison: Dense@3  vs  Dense@20→Reranked  vs  MultiQuery→Reranked")
    print("Drift mitigation: prompt constraint only (per_query_k=20)")
    print("=" * 75)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQuery {i}: \"{query}\"")

        # Dense top-3 baseline
        dense3 = store.search(query, top_k=FINAL_K)
        print_table("Dense@3 baseline", dense3, "score")

        # Dense top-20 → rerank
        dense20 = store.search(query, top_k=CANDIDATE_N)
        for rank, r in enumerate(dense20, 1):
            r["prev_rank"] = rank
        dense_reranked = reranker.rerank(query, dense20, top_k=FINAL_K)
        print_table(f"Dense@{CANDIDATE_N} → Reranked", dense_reranked, "rerank_score")

        # Multi-Query → rerank (prompt constraint, per_query_k=20)
        print(f"  [Calling GLM to generate query variants...]")
        mq_reranked, all_queries = mq.search(
            query, n_variants=3, per_query_k=20, final_k=FINAL_K
        )
        print(f"  Variants generated:")
        for j, q in enumerate(all_queries):
            label = "original" if j == 0 else f"variant {j}"
            print(f"    [{label}] {q}")
        print_table("MultiQuery → Reranked", mq_reranked, "rerank_score")


if __name__ == "__main__":
    main()
