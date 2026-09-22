"""
Verify hard queries (方式2) actually fail Dense@3 before adding to Golden Set.

Run from LangGraphLearning/:
    .venv/bin/python eval/verify_hard_queries.py
"""

import sys
sys.path.insert(0, ".")

from src.rag.chunker import chunk_by_section
from src.rag.vector_store import VectorStore
import re

ACT_TEXT_FILE = "tests/rag/health_services_act.txt"
COLLECTION    = "health-services-act-baseline"

# 8 hard queries using informal English (方式2)
# Deliberately avoid vocabulary that matches section titles
HARD_QUERIES = [
    {
        "id": "H1",
        "query": "can anyone set up their own ambulance service and charge for it?",
        "relevant": ["s.67E"],
        "note": "informal 'set up/charge' vs legal 'unauthorised provision'",
    },
    {
        "id": "H2",
        "query": "can the government investigate a hospital if something goes wrong?",
        "relevant": ["s.123"],
        "note": "informal 'investigate' vs legal 'cause an inquiry to be held'",
    },
    {
        "id": "H3",
        "query": "can a hospital make its own internal rules?",
        "relevant": ["s.39"],
        "note": "informal 'rules' vs legal 'by-laws'; 'hospital' vs 'local health district'",
    },
    {
        "id": "H4",
        "query": "can a health worker be stood down while being investigated for misconduct?",
        "relevant": ["s.120A"],
        "note": "informal 'stood down' vs legal 'suspended'; 'investigated' vs 'pending decision'",
    },
    {
        "id": "H5",
        "query": "who decides how much public hospitals charge patients?",
        "relevant": ["s.69"],
        "note": "informal 'how much/charge' vs legal 'scale of fees'",
    },
    {
        "id": "H6",
        "query": "can the Minister remove someone from a hospital board?",
        "relevant": ["s.29"],
        "note": "informal 'remove' vs legal 'removal of members'; 'hospital board' vs 'local health district board'",
    },
    {
        "id": "H7",
        "query": "if someone is treated at hospital after an accident, can the hospital take money from their injury payout?",
        "relevant": ["s.72"],
        "note": "informal 'take money from payout' vs legal 'charge on damages'",
    },
    {
        "id": "H8",
        "query": "can a health worker be moved to another hospital if their role is made redundant?",
        "relevant": ["s.116C"],
        "note": "informal 'moved/role cut' vs legal 'transfer/on ground of redundancy'",
    },
]


def extract_section_prefix(chunk_id: str) -> str:
    m = re.match(r"^(s\.[^_]+)", chunk_id)
    return m.group(1) if m else chunk_id


def recall_at_k(retrieved, relevant, k):
    retrieved_prefixes = {extract_section_prefix(r["chunk_id"]) for r in retrieved[:k]}
    found = {sec for sec in relevant if sec in retrieved_prefixes}
    return len(found) / len(relevant)


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

    print(f"{'ID':<5} {'R@3':>5} {'R@5':>5}  Query")
    print("-" * 80)
    fails = []
    passes = []

    for case in HARD_QUERIES:
        results3 = store.search(case["query"], top_k=3)
        results5 = store.search(case["query"], top_k=5)
        r3 = recall_at_k(results3, case["relevant"], 3)
        r5 = recall_at_k(results5, case["relevant"], 5)

        status = "FAIL" if r3 == 0.0 else "PASS"
        print(f"{case['id']:<5} {r3:>5.2f} {r5:>5.2f}  {case['query'][:60]}")
        print(f"      Vocab gap: {case['note']}")
        print(f"      Top-3 retrieved: {[extract_section_prefix(r['chunk_id']) for r in results3]}")
        print()

        if r3 == 0.0:
            fails.append(case["id"])
        else:
            passes.append(case["id"])

    print("=" * 80)
    print(f"Dense@3 FAIL (good hard queries): {fails}")
    print(f"Dense@3 PASS (too easy, reconsider): {passes}")
    print(f"\n{len(fails)}/8 queries are genuinely hard for Dense@3")


if __name__ == "__main__":
    main()
