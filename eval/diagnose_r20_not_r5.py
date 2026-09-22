"""
诊断：R@20 找到但 R@5 找不到的 query

即：答案在 top-20 里，但 reranker 没把它排进前 5。
这说明 reranker 排序有问题，不是候选池缺失。

Run:
    .venv/bin/python eval/diagnose_r20_not_r5.py
"""

import os
import re
import sys
sys.path.insert(0, ".")

import openpyxl
from dotenv import load_dotenv
load_dotenv()

from src.rag.chunker import chunk_by_section, chunk_by_subsection, chunk_subsection_parent_child
from src.rag.vector_store import VectorStore
from src.rag.reranker import Reranker
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.hyde_retriever import generate_hypothetical_doc
from src.rag.legal_query_expansion_retriever import expand_query
from src.rag.stepback_retriever import generate_stepback_query
from src.rag.multi_query_retriever import generate_query_variants
from src.rag.query_router import classify_query

SIM_LABELED_FILE        = "GoldenSet_from_100_Simulated_Queries.xlsx"
ACT_TEXT_FILE           = "tests/rag/health_services_act.txt"
PARENT_COLLECTION       = "health-services-act-baseline"
SUBSEC_COLLECTION       = "health-services-act-subsections"
SUBSEC_CHILD_COLLECTION = "health-services-act-subsec-children"
CANDIDATE_N             = 20


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


def load_grounded_cases(path: str) -> list[dict]:
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
        })
    return cases


def extract_section_prefix(chunk_id: str) -> str:
    m = re.match(r"^(s\.[^_]+)", chunk_id)
    return m.group(1) if m else chunk_id


def main():
    print("Loading cases...")
    cases = load_grounded_cases(SIM_LABELED_FILE)
    print(f"  {len(cases)} grounded queries\n")

    print("Loading stores...")
    with open(ACT_TEXT_FILE, encoding="utf-8") as f:
        act_text = f.read()

    subsec_chunks, subsec_children = chunk_subsection_parent_child(act_text)

    subsec_store = VectorStore(collection_name=SUBSEC_COLLECTION, persist_dir="./chroma_db")
    print(f"  Subsec store: {subsec_store.count()} chunks")

    subsec_child_store = VectorStore(collection_name=SUBSEC_CHILD_COLLECTION, persist_dir="./chroma_db")
    print(f"  SubsecChild:  {subsec_child_store.count()} chunks")

    reranker = Reranker()
    print()

    def subsec_routed_pc(q: str, top_k: int = 20) -> list[dict]:
        qtype = classify_query(q)
        seen: set[str] = set()
        children: list[dict] = []

        def _collect(search_q: str) -> None:
            for c in subsec_child_store.search_children(search_q, top_k=CANDIDATE_N):
                if c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    children.append(c)

        if qtype == "Penalty":
            _collect(expand_query(q))
        elif qtype == "Prohibition":
            _collect(q)
            _collect(generate_stepback_query(q))
        elif qtype == "CrossSection":
            for sq in [q] + generate_query_variants(q, n=3):
                _collect(sq)
        else:
            _collect(generate_hypothetical_doc(q))

        parents = subsec_child_store.get_parents_by_ids(
            subsec_store, [c["parent_id"] for c in children]
        )
        return reranker.rerank(q, parents, top_k=top_k)

    print("=" * 70)
    print("  R@20 找到 但 R@5 找不到 的 query（reranker 排序失败）")
    print("=" * 70)

    found_in_r20_not_r5 = []
    total_r5_hit = 0
    total_r20_hit = 0

    for case in cases:
        q        = case["query"]
        relevant = set(case["relevant"])

        top20 = subsec_routed_pc(q, top_k=20)
        sections20 = [extract_section_prefix(c["chunk_id"]) for c in top20]
        sections5  = sections20[:5]

        hit5  = bool(relevant & set(sections5))
        hit20 = bool(relevant & set(sections20))

        if hit5:
            total_r5_hit += 1
        if hit20:
            total_r20_hit += 1

        if hit20 and not hit5:
            # 找到答案在哪个位置
            answer_ranks = []
            for rel in relevant:
                for rank, sec in enumerate(sections20, start=1):
                    if sec == rel:
                        answer_ranks.append((rel, rank, top20[rank-1].get("rerank_score", "?")))
                        break

            found_in_r20_not_r5.append({
                "case": case,
                "top20": top20,
                "sections20": sections20,
                "answer_ranks": answer_ranks,
            })

    print(f"\nR@5  = {total_r5_hit}/{len(cases)} = {total_r5_hit/len(cases):.3f}")
    print(f"R@20 = {total_r20_hit}/{len(cases)} = {total_r20_hit/len(cases):.3f}")
    print(f"\n'R@20找到但R@5没找到' 的 query 共 {len(found_in_r20_not_r5)} 条：\n")

    for item in found_in_r20_not_r5:
        case   = item["case"]
        top20  = item["top20"]
        sec20  = item["sections20"]
        ranks  = item["answer_ranks"]

        print(f"{'─'*70}")
        print(f"[{case['id']}] ({case['query_type']})")
        print(f"Query:    {case['query']}")
        print(f"Relevant: {case['relevant']}")
        print(f"答案位置: {ranks}")
        print()
        print("  Top-5 返回（错误排在前面）：")
        for i, c in enumerate(top20[:5], start=1):
            sec = extract_section_prefix(c["chunk_id"])
            marker = " ← ✓ 正确" if sec in case["relevant"] else ""
            print(f"    {i}. {sec}  score={c.get('rerank_score','?'):.4f}  {c['title'][:50]}{marker}")
        print()
        print("  答案 chunk 的内容（前200字）：")
        for rel, rank, score in ranks:
            chunk = top20[rank - 1]
            print(f"    Rank {rank} | {rel} | score={score:.4f}")
            print(f"    {chunk['text'][:200].strip()}")
            print()


if __name__ == "__main__":
    main()
