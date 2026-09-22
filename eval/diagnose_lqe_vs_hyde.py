"""
Compare LQE→Hybrid→Reranked@5 vs HyDE→Reranked@5 on the 16 previously-failing queries.
Shows which ones LQE recovered, which stayed broken, and which regressed.
"""
import os
import re
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.rag.chunker import chunk_by_section
from src.rag.vector_store import VectorStore
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import Reranker
from src.rag.hyde_retriever import HyDERetriever
from src.rag.legal_query_expansion_retriever import LegalQueryExpansionRetriever

import openpyxl
from collections import defaultdict

ACT_TEXT_FILE     = "tests/rag/health_services_act.txt"
SIM_LABELED_FILE  = "GoldenSet_from_100_Simulated_Queries.xlsx"
PARENT_COLLECTION = "health-services-act-baseline"
CANDIDATE_N       = 20
FINAL_K           = 5


def parse_expected_sections(raw) -> list[str]:
    if not raw:
        return []
    raw_str = str(raw).lower()
    if any(p in raw_str for p in ["no direct", "no general", "dictionary"]):
        return []
    sections = []
    for part in str(raw).replace("–", "-").replace("—", "-").split(";"):
        part = part.strip()
        m = re.match(r"s\s+(\d+[A-Za-z]*)", part, re.IGNORECASE)
        if m:
            sections.append(f"s.{m.group(1)}")
    return sections


def extract_section_prefix(chunk_id: str) -> str:
    m = re.match(r"^(s\.[^_]+)", chunk_id)
    return m.group(1) if m else chunk_id


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 1.0
    found = len({s for s in relevant if s in set(retrieved[:k])})
    return found / len(relevant)


def load_grounded(path: str) -> list[dict]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Golden Review"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    col = {name: i for i, name in enumerate(header)}
    cases = []
    for row in rows[1:]:
        if not row[col["ID"]]:
            continue
        relevant = parse_expected_sections(row[col["Expected Section(s)"]])
        if not relevant:
            continue
        cases.append({
            "id":         row[col["ID"]],
            "query":      row[col["Query"]],
            "relevant":   relevant,
            "query_type": row[col["Original Query Type"]],
            "grounded":   row[col["Grounded"]] or "",
        })
    return cases


print("Loading Act and indexes...")
with open(ACT_TEXT_FILE, encoding="utf-8") as f:
    act_text = f.read()

chunks = chunk_by_section(act_text)
parent_store = VectorStore(collection_name=PARENT_COLLECTION, persist_dir="./chroma_db")
if parent_store.count() != len(chunks):
    parent_store.reset()
    parent_store.add_chunks(chunks)
else:
    print(f"  {parent_store.count()} chunks (reused)")

reranker = Reranker()
hyde     = HyDERetriever(vector_store=parent_store)
hybrid   = HybridRetriever(vector_store=parent_store)
hybrid.index(chunks)
lqe      = LegalQueryExpansionRetriever(hybrid_retriever=hybrid)

print("\nLoading grounded queries...")
cases = load_grounded(SIM_LABELED_FILE)
print(f"  {len(cases)} grounded queries\n")

# ── Run both retrievers ────────────────────────────────────────────────────────
print("Running HyDE and LQE on all queries...")
results = []
for i, case in enumerate(cases):
    # HyDE
    hyde_cands, _ = hyde.search(case["query"], top_k=CANDIDATE_N)
    hyde_chunks   = reranker.rerank(case["query"], hyde_cands, top_k=FINAL_K)
    hyde_secs     = [extract_section_prefix(c["chunk_id"]) for c in hyde_chunks]
    hyde_r5       = recall_at_k(hyde_secs, case["relevant"], FINAL_K)

    # LQE
    lqe_cands, expanded = lqe.search(case["query"], top_k=CANDIDATE_N)
    lqe_chunks  = reranker.rerank(case["query"], lqe_cands, top_k=FINAL_K)
    lqe_secs    = [extract_section_prefix(c["chunk_id"]) for c in lqe_chunks]
    lqe_r5      = recall_at_k(lqe_secs, case["relevant"], FINAL_K)

    results.append({
        **case,
        "hyde_r5": hyde_r5, "hyde_secs": hyde_secs,
        "lqe_r5":  lqe_r5,  "lqe_secs":  lqe_secs,
        "expanded": expanded,
    })
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(cases)} done...")

# ── Summary ────────────────────────────────────────────────────────────────────
hyde_avg = sum(r["hyde_r5"] for r in results) / len(results)
lqe_avg  = sum(r["lqe_r5"]  for r in results) / len(results)
print(f"\n{'='*55}")
print(f"  HyDE avg R@5 : {hyde_avg:.3f}")
print(f"  LQE  avg R@5 : {lqe_avg:.3f}  (Δ {lqe_avg - hyde_avg:+.3f})")
print(f"{'='*55}")

# ── Focus on previously-failing queries (HyDE R@5 = 0) ───────────────────────
failed_by_hyde = [r for r in results if r["hyde_r5"] == 0.0]
recovered  = [r for r in failed_by_hyde if r["lqe_r5"] > 0.0]
still_fail = [r for r in failed_by_hyde if r["lqe_r5"] == 0.0]

print(f"\n── 之前失败的 {len(failed_by_hyde)} 条（HyDE R@5=0）─────────────────")
print(f"  LQE 救回 : {len(recovered)} 条")
print(f"  仍然失败 : {len(still_fail)} 条")

if recovered:
    print(f"\n  ✓ 救回的 {len(recovered)} 条：")
    for r in sorted(recovered, key=lambda x: x["query_type"]):
        print(f"\n    [{r['id']}] {r['query_type']}")
        print(f"    Q:        {r['query'][:80]}")
        print(f"    Expanded: {r['expanded'][:80]}")
        print(f"    Expected: {r['relevant']}")
        print(f"    HyDE got: {r['hyde_secs']}  R@5={r['hyde_r5']:.2f}")
        print(f"    LQE got:  {r['lqe_secs']}  R@5={r['lqe_r5']:.2f}")

print(f"\n  ✗ 仍然失败的 {len(still_fail)} 条：")
for r in sorted(still_fail, key=lambda x: x["query_type"]):
    print(f"\n    [{r['id']}] {r['query_type']}")
    print(f"    Q:        {r['query'][:80]}")
    print(f"    Expanded: {r['expanded'][:80]}")
    print(f"    Expected: {r['relevant']}")
    print(f"    LQE got:  {r['lqe_secs']}")

# ── Regressions (HyDE worked, LQE broke) ──────────────────────────────────────
regressions = [r for r in results if r["hyde_r5"] == 1.0 and r["lqe_r5"] < 1.0]
print(f"\n── LQE 导致退步的 {len(regressions)} 条（HyDE R@5=1 → LQE R@5<1）──────")
for r in sorted(regressions, key=lambda x: x["query_type"]):
    print(f"\n  [{r['id']}] {r['query_type']}")
    print(f"  Q:        {r['query'][:80]}")
    print(f"  Expanded: {r['expanded'][:80]}")
    print(f"  Expected: {r['relevant']}")
    print(f"  HyDE got: {r['hyde_secs']}  R@5={r['hyde_r5']:.2f}")
    print(f"  LQE got:  {r['lqe_secs']}   R@5={r['lqe_r5']:.2f}")
