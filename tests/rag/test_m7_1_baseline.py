"""
M7.1 — Baseline RAG Pipeline Test

Run from LangGraphLearning/:
    .venv/bin/python tests/rag/test_m7_1_baseline.py

What this script does (step by step):
  1. Fetch one Act using the existing fetch_act_html tool (reuses M1/M2 code)
  2. Chunk the Act text by section using chunker.py
  3. Embed all chunks and store in ChromaDB using vector_store.py
  4. Run 5 test queries and show the top-3 retrieved chunks per query
  5. Print a manual recall checklist — you tick which results are relevant

This is the BASELINE. We measure Recall@3 and Recall@5 here,
then compare against M7.2 (Hybrid) and M7.3 (Reranking) later.
"""

import sys, json
sys.path.insert(0, ".")

from src.rag.chunker import chunk_by_section
from src.rag.vector_store import VectorStore

ACT_TEXT_FILE = "tests/rag/health_services_act.txt"
ACT_META_FILE = "tests/rag/health_services_act_meta.json"
COLLECTION = "health-services-act-baseline"

# 5 test queries — mix of exact terminology and plain-language paraphrasing
TEST_QUERIES = [
    "mandatory reporting obligations for health practitioners",
    "penalty for providing false information",
    "minister's power to make regulations",
    "patient rights and complaints procedure",
]


def main():
    # ── Step 1: Load Act from local file ────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading Act from local file...")
    print("=" * 60)
    with open(ACT_TEXT_FILE, encoding="utf-8") as f:
        act_text = f.read()
    with open(ACT_META_FILE, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"  Title : {meta['title']}")
    print(f"  Length: {len(act_text):,} characters")

    # ── Step 2: Chunk ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Chunking by section...")
    print("=" * 60)
    chunks = chunk_by_section(act_text)
    print(f"  Found {len(chunks)} sections")
    print("  First 5 sections:")
    for c in chunks[:5]:
        print(f"    {c['chunk_id']:12s} | {c['title'][:50]}")

    # ── Step 3: Embed + Store ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Embedding and storing in ChromaDB...")
    print("  (First run downloads ~80MB model — subsequent runs are instant)")
    print("=" * 60)
    store = VectorStore(collection_name=COLLECTION, persist_dir="./chroma_db")

    if store.count() == len(chunks):
        print(f"  Already indexed {store.count()} chunks — skipping re-embedding")
    else:
        store.reset()
        store.add_chunks(chunks)

    # ── Step 4: Query ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Running test queries (top-3 results each)")
    print("=" * 60)

    col_q     = 50   # query column width
    col_sec   = 12   # section id width
    col_title = 45   # title width
    col_score = 7    # score width
    sep = f"+{'─'*4}+{'─'*col_sec}+{'─'*col_title}+{'─'*col_score}+"

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQuery {i}: \"{query}\"")
        print(sep)
        print(f"│{'Rank':^4}│{'Section':^{col_sec}}│{'Title':^{col_title}}│{'Score':^{col_score}}│")
        print(sep)
        results = store.search(query, top_k=3)
        for rank, r in enumerate(results, 1):
            section = r['chunk_id'][:col_sec-1]
            title   = r['title'][:col_title-1]
            score   = f"{r['score']:.3f}"
            print(f"│{rank:^4}│{section:<{col_sec}}│{title:<{col_title}}│{score:^{col_score}}│")
        print(sep)

    # ── Step 5: Manual recall checklist ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Manual Recall Checklist")
    print("=" * 60)
    print("""
For each query above, review the retrieved sections and count:
  - How many of the top-3 results are ACTUALLY relevant?
  - Are there relevant sections NOT in the top-3? (= missed = low recall)

Record:
  relevant_retrieved / total_relevant = Recall@3

We will compare this number against M7.2 (Hybrid Search) later.
""")


if __name__ == "__main__":
    main()
