"""
Diagnose which queries fail in HyDE→Reranked@5 (R@5=0) on Grounded-77.
Groups failures by query type and grounded status.
"""
import os
import re
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.rag.chunker import chunk_by_section, chunk_parent_child
from src.rag.vector_store import VectorStore
from src.rag.reranker import Reranker
from src.rag.hyde_retriever import HyDERetriever

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
            continue  # skip unanswerable
        cases.append({
            "id":         row[col["ID"]],
            "query":      row[col["Query"]],
            "relevant":   relevant,
            "query_type": row[col["Original Query Type"]],
            "grounded":   row[col["Grounded"]] or "",
            "candidate":  row[col["Golden Set Candidate"]] or "",
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

print("\nLoading grounded queries...")
cases = load_grounded(SIM_LABELED_FILE)
print(f"  {len(cases)} grounded queries\n")

# ── Run retrieval and collect results ─────────────────────────────────────────

results = []
for i, case in enumerate(cases):
    candidates, _ = hyde.search(case["query"], top_k=CANDIDATE_N)
    chunks_out    = reranker.rerank(case["query"], candidates, top_k=FINAL_K)
    retrieved     = [extract_section_prefix(c["chunk_id"]) for c in chunks_out]
    r5            = recall_at_k(retrieved, case["relevant"], FINAL_K)
    results.append({**case, "retrieved": retrieved, "R@5": r5})
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(cases)} done...")

# ── Analysis ──────────────────────────────────────────────────────────────────

total   = len(results)
failed  = [r for r in results if r["R@5"] == 0.0]
partial = [r for r in results if 0 < r["R@5"] < 1.0]
perfect = [r for r in results if r["R@5"] == 1.0]

print(f"\n{'='*55}")
print(f"  Total queries : {total}")
print(f"  R@5 = 1.0    : {len(perfect)} ({len(perfect)/total:.0%})")
print(f"  0 < R@5 < 1  : {len(partial)} ({len(partial)/total:.0%})")
print(f"  R@5 = 0.0    : {len(failed)} ({len(failed)/total:.0%})  ← 完全失败")
print(f"{'='*55}")

# By query type
print("\n── 按 Query Type 分组 ──────────────────────────────")
by_type: dict[str, list] = defaultdict(list)
for r in results:
    by_type[r["query_type"]].append(r)

print(f"{'Type':<16} {'n':>3} {'R@5=0':>6} {'R@5=1':>6} {'avg R@5':>8}")
print("-" * 45)
for qtype in sorted(by_type):
    group = by_type[qtype]
    n      = len(group)
    fail   = sum(1 for r in group if r["R@5"] == 0.0)
    perf   = sum(1 for r in group if r["R@5"] == 1.0)
    avg    = sum(r["R@5"] for r in group) / n
    print(f"{qtype:<16} {n:>3} {fail:>6} {perf:>6} {avg:>8.3f}")

# By grounded status
print("\n── 按 Grounded 状态分组 ─────────────────────────────")
by_ground: dict[str, list] = defaultdict(list)
for r in results:
    by_ground[r["grounded"]].append(r)

print(f"{'Grounded':<20} {'n':>3} {'R@5=0':>6} {'avg R@5':>8}")
print("-" * 42)
for g in sorted(by_ground):
    group = by_ground[g]
    n     = len(group)
    fail  = sum(1 for r in group if r["R@5"] == 0.0)
    avg   = sum(r["R@5"] for r in group) / n
    print(f"{g:<20} {n:>3} {fail:>6} {avg:>8.3f}")

# Show all failures
print(f"\n── 完全失败的 {len(failed)} 条 query ──────────────────────────")
for r in sorted(failed, key=lambda x: x["query_type"]):
    print(f"\n  [{r['id']}] {r['query_type']} | {r['grounded']}")
    print(f"  Q: {r['query'][:90]}")
    print(f"  Expected : {r['relevant']}")
    print(f"  Retrieved: {r['retrieved']}")
