"""
Step 1: Check query routing classification accuracy on all 79 grounded queries.
Fast — only LLM classification calls, no retrieval.
"""
import re
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.rag.query_router import classify_query
import openpyxl

SIM_LABELED_FILE = "GoldenSet_from_100_Simulated_Queries.xlsx"
NON_OTHER = {"Penalty", "Prohibition", "CrossSection"}


def parse_expected_sections(raw) -> list[str]:
    if not raw:
        return []
    raw_str = str(raw).lower()
    if any(p in raw_str for p in ["no direct", "no general", "dictionary"]):
        return []
    sections = []
    for part in str(raw).replace("–", "-").replace("—", "-").split(";"):
        m = re.match(r"s\s+(\d+[A-Za-z]*)", part.strip(), re.IGNORECASE)
        if m:
            sections.append(f"s.{m.group(1)}")
    return sections


wb = openpyxl.load_workbook(SIM_LABELED_FILE, data_only=True)
ws = wb["Golden Review"]
rows = list(ws.iter_rows(values_only=True))
header = rows[0]
col = {name: i for i, name in enumerate(header)}

cases = []
for row in rows[1:]:
    if not row[col["ID"]]:
        continue
    if not parse_expected_sections(row[col["Expected Section(s)"]]):
        continue
    cases.append({
        "id":         row[col["ID"]],
        "query":      row[col["Query"]],
        "query_type": row[col["Original Query Type"]],
    })

print(f"Classifying {len(cases)} grounded queries...\n")
print(f"{'ID':<5} {'True Type':<16} {'Routed':<16} {'Result'}")
print("-" * 55)

misclassified = []
for i, case in enumerate(cases):
    routed = classify_query(case["query"])
    true_t = case["query_type"]

    correct = (routed == true_t) or \
              (true_t not in NON_OTHER and routed == "Other")

    status = "OK   " if correct else "WRONG"
    print(f"[{case['id']:<3}] {true_t:<16} {routed:<16} {status}  {case['query'][:45]}")

    if not correct:
        misclassified.append({**case, "routed": routed})

    if (i + 1) % 20 == 0:
        print(f"  --- {i+1}/{len(cases)} done ---")

print(f"\n{'='*55}")
print(f"  총 queries    : {len(cases)}")
print(f"  정확 분류     : {len(cases) - len(misclassified)} ({(len(cases)-len(misclassified))/len(cases):.0%})")
print(f"  误分类        : {len(misclassified)} ({len(misclassified)/len(cases):.0%})")
print(f"{'='*55}")

if misclassified:
    print(f"\n── 误分类详情 ──────────────────────────────────────")
    for r in misclassified:
        print(f"\n  [{r['id']}] 真实={r['query_type']}  路由判定={r['routed']}")
        print(f"  Q: {r['query'][:80]}")
