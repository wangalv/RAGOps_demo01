"""
Run from the LangGraphLearning directory:
    .venv/bin/python tests/test_state.py
"""

import sys
sys.path.insert(0, ".")

from src.agents.state import PipelineState


def main():
    # Build the initial state that the pipeline starts with
    initial_state: PipelineState = {
        "source_url": "https://legislation.nsw.gov.au/view/whole/html/inforce/2025-05-16/sl-2023-0139",
        "source_pdf_path": "",
        "raw_html": "",
        "raw_text": "",
        "doc_title": "",
        "doc_metadata": {},
        "toc": [],
        "nodes": [],
        "extracted_nodes": [],
        "current_step": "init",
        "errors": [],
        "retry_count": 0,
        "is_valid": False,
    }

    print("=== PipelineState initialised ===\n")
    for key, value in initial_state.items():
        print(f"  {key:20s} = {repr(value)}")

    # Simulate what a node returns: only the fields it changed
    updated_state = {
        **initial_state,
        "current_step": "fetch_html",
        "doc_title": "Administrative Arrangements (Minns Ministry) Order 2023",
    }

    print("\n=== After simulated node update ===\n")
    print(f"  current_step = {updated_state['current_step']}")
    print(f"  doc_title    = {updated_state['doc_title']}")
    print("\n✓ State test passed")


if __name__ == "__main__":
    main()
