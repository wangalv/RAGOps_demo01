"""
Run from the LangGraphLearning directory:
    .venv/bin/python tests/test_pipeline.py
"""

import sys
sys.path.insert(0, ".")

from src.agents.pipeline import build_graph

URL = "https://legislation.nsw.gov.au/view/whole/html/inforce/2025-05-16/sl-2023-0139"
RUN_CONFIG = {"configurable": {"thread_id": "health-minister-run-2"}}

INITIAL_STATE = {
    "source_url": URL,
    "selected_minister": "Minister for Health",
    "ministers": [],
    "act_urls": [],
    "processed_acts": [],
    "raw_html": "",
    "raw_text": "",
    "nodes": [],
    "extracted_nodes": [],
    "current_step": "init",
    "errors": [],
    "retry_count": 0,
    "is_valid": False,
}


def main():
    graph = build_graph()

    print("Streaming pipeline — each line = one node completed:\n")

    # stream() yields one chunk per node as soon as it finishes
    for chunk in graph.stream(INITIAL_STATE, config=RUN_CONFIG):
        node_name = list(chunk.keys())[0]
        updated_fields = list(chunk[node_name].keys())
        print(f"  ✓ {node_name:25s} updated fields: {updated_fields}")

    print("\nDone.")


if __name__ == "__main__":
    main()
