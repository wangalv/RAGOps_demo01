"""
One-time script: fetch Health Services Act and save to local file.

Run once:
    .venv/bin/python tests/rag/fetch_and_save_act.py

After this, all M7 tests read from tests/rag/health_services_act.txt
instead of hitting the network each time.
"""

import sys, json
sys.path.insert(0, ".")

from src.tools.fetch_legislation import fetch_act_html

ACT_URL = "https://legislation.nsw.gov.au/view/whole/html/inforce/current/act-1997-154"
OUT_TEXT = "tests/rag/health_services_act.txt"
OUT_META = "tests/rag/health_services_act_meta.json"

print(f"Fetching: {ACT_URL}")
result = fetch_act_html.invoke({"url": ACT_URL})

with open(OUT_TEXT, "w", encoding="utf-8") as f:
    f.write(result["text"])

with open(OUT_META, "w", encoding="utf-8") as f:
    json.dump({"title": result["title"], "url": result["url"]}, f, indent=2)

print(f"Saved text  → {OUT_TEXT}  ({len(result['text']):,} chars)")
print(f"Saved meta  → {OUT_META}")
print(f"Title: {result['title']}")
