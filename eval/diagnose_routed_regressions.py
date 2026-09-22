"""
Diagnose regressions in Routed→Reranked@5 vs HyDE→Reranked@5.

For each query where HyDE R@5=1 but Routed R@5<1, shows:
  - What the router classified it as (possible misclassification)
  - What the true query_type is (from Excel)
  - What the assigned retriever retrieved vs HyDE
  - Verdict: misclassification or retriever-switching degradation
"""
import os
import re
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.rag.chunker import chunk_by_section, chunk_parent_child
from src.rag.vector_store import VectorStore
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import Reranker
from src.rag.hyde_retriever import HyDERetriever
from src.rag.legal_query_expansion_retriever import LegalQueryExpansionRetriever
from src.rag.multi_query_retriever import MultiQueryRetriever
from src.rag.stepback_retriever import StepBackRetriever
from src.rag.routed_retriever import RoutedRetriever
from src.rag.query_router import classify_query

import openpyxl

ACT_TEXT_FILE     = "tests/rag/health_services_act.txt"
SIM_LABELED_FILE  = "GoldenSet_from_100_Simulated_Queries.xlsx"
PARENT_COLLECTION = "health-services-act-baseline"
CHILD_COLLECTION  = "health-services-act-children"
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
_, children = chunk_parent_child(act_text)

parent_store = VectorStore(collection_name=PARENT_COLLECTION, persist_dir="./chroma_db")
if parent_store.count() != len(chunks):
    parent_store.reset(); parent_store.add_chunks(chunks)
else:
    print(f"  {parent_store.count()} chunks (reused)")

child_store = VectorStore(collection_name=CHILD_COLLECTION, persist_dir="./chroma_db")
if child_store.count() != len(children):
    child_store.reset(); child_store.add_children(children)
else:
    print(f"  {child_store.count()} child chunks (reused)")

reranker  = Reranker()
hybrid    = HybridRetriever(vector_store=parent_store)
hybrid.index(chunks)
hyde      = HyDERetriever(vector_store=parent_store)
lqe       = LegalQueryExpansionRetriever(hybrid_retriever=hybrid)
mq        = MultiQueryRetriever(vector_store=parent_store, reranker=reranker)
sb        = StepBackRetriever(vector_store=parent_store, reranker=reranker)
routed    = RoutedRetriever(lqe=lqe, stepback=sb, multiquery=mq, hyde=hyde,
                            reranker=reranker, candidate_n=CANDIDATE_N)

print("\nLoading grounded queries...")
cases = load_grounded(SIM_LABELED_FILE)
print(f"  {len(cases)} grounded queries\n")

# ── Run HyDE and Routed on all queries ────────────────────────────────────────
print("Running HyDE and Routed on all queries...")
results = []
for i, case in enumerate(cases):
    # HyDE
    hyde_cands, _ = hyde.search(case["query"], top_k=CANDIDATE_N)
    hyde_chunks   = reranker.rerank(case["query"], hyde_cands, top_k=FINAL_K)
    hyde_secs     = [extract_section_prefix(c["chunk_id"]) for c in hyde_chunks]
    hyde_r5       = recall_at_k(hyde_secs, case["relevant"], FINAL_K)

    # Routed (also returns the route label)
    routed_chunks, route = routed.search(case["query"], top_k=FINAL_K)
    routed_secs   = [extract_section_prefix(c["chunk_id"]) for c in routed_chunks]
    routed_r5     = recall_at_k(routed_secs, case["relevant"], FINAL_K)

    results.append({
        **case,
        "hyde_r5":    hyde_r5,    "hyde_secs":    hyde_secs,
        "routed_r5":  routed_r5,  "routed_secs":  routed_secs,
        "route":      route,
    })
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(cases)} done...")

# ── Summary ────────────────────────────────────────────────────────────────────
hyde_avg   = sum(r["hyde_r5"]   for r in results) / len(results)
routed_avg = sum(r["routed_r5"] for r in results) / len(results)
print(f"\n{'='*55}")
print(f"  HyDE   avg R@5 : {hyde_avg:.3f}")
print(f"  Routed avg R@5 : {routed_avg:.3f}  (Δ {routed_avg - hyde_avg:+.3f})")
print(f"{'='*55}")

# ── Regressions ────────────────────────────────────────────────────────────────
regressions = [r for r in results if r["hyde_r5"] == 1.0 and r["routed_r5"] < 1.0]
print(f"\n── 退步的 {len(regressions)} 条（HyDE R@5=1 → Routed R@5<1）──────────")

for r in sorted(regressions, key=lambda x: x["query_type"]):
    true_type   = r["query_type"]
    routed_type = r["route"]
    verdict = "✓ 分类正确" if routed_type == true_type or (routed_type == "Other" and true_type not in ("Penalty","Prohibition","CrossSection")) \
              else f"✗ 误分类（真实={true_type}，路由={routed_type}）"

    print(f"\n  [{r['id']}] 真实类型={true_type}  路由判定={routed_type}  {verdict}")
    print(f"  Q:          {r['query'][:80]}")
    print(f"  Expected:   {r['relevant']}")
    print(f"  HyDE got:   {r['hyde_secs']}  R@5={r['hyde_r5']:.2f}")
    print(f"  Routed got: {r['routed_secs']}  R@5={r['routed_r5']:.2f}")

# ── Classification accuracy on all queries ─────────────────────────────────────
print(f"\n── 路由分类准确率（全部 {len(results)} 条）──────────────────────")
correct = 0
for r in results:
    true_t   = r["query_type"]
    routed_t = r["route"]
    is_correct = (routed_t == true_t) or \
                 (routed_t == "Other" and true_t not in ("Penalty","Prohibition","CrossSection"))
    if is_correct:
        correct += 1
print(f"  正确分类: {correct}/{len(results)} ({correct/len(results):.0%})")
