"""
Run from the LangGraphLearning directory:
    .venv/bin/python tests/test_fetch_ministers.py

Tests the fetch_ministers_list tool function in isolation,
before wiring it into a LangGraph node.
"""

import sys
sys.path.insert(0, ".")

from src.tools.fetch_ministers import fetch_ministers_list

URL = "https://legislation.nsw.gov.au/view/whole/html/inforce/2025-05-16/sl-2023-0139"

def main():
    print(f"Fetching ministers from:\n  {URL}\n")
    ministers = fetch_ministers_list(URL)

    if not ministers:
        print("ERROR: no ministers found — selector may need adjusting")
        return

    print(f"Found {len(ministers)} ministers:\n")
    for i, name in enumerate(ministers, 1):
        print(f"  {i:2d}. {name}")

if __name__ == "__main__":
    main()
