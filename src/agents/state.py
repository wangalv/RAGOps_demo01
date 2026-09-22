"""
M1 Step 1 — PipelineState

State is the shared data container passed between every node in the Graph.
Each node reads from State, processes it, and returns only the fields it changed.
LangGraph merges those changes back into State before calling the next node.

TypedDict gives the dictionary a fixed schema so Python and LangGraph can
validate inputs and outputs at each node boundary.
"""

import operator
from typing import Annotated, TypedDict


class PipelineState(TypedDict):
    # ── Input ────────────────────────────────────────────────
    source_url: str           # URL of the Administrative Arrangements Order
    selected_minister: str    # Minister chosen by the user e.g. "Minister for Health"

    # ── Step 2a output: minister list ───────────────────────
    ministers: list[str]      # All ministers parsed from Schedule 1

    # ── Step 2b output: acts under selected minister ────────
    act_urls: list[dict]      # [{"title": "Health Services Act 1997", "url": "..."}, ...]

    # ── M2: parallel act processing results ─────────────────
    # Annotated + operator.add tells LangGraph to APPEND results from
    # each parallel branch instead of overwriting the field.
    processed_acts: Annotated[list[dict], operator.add]

    # ── Step 3 output: raw HTML per act ─────────────────────
    raw_html: str             # Raw HTML of a single fetched Act
    raw_text: str             # Plain text extracted from the HTML

    # ── Step 4 output: provision nodes ──────────────────────
    nodes: list[dict]         # Provision nodes split from the Act
    extracted_nodes: list[dict]  # Nodes enriched with obligations, penalties, cross_refs

    # ── Flow control ────────────────────────────────────────
    current_step: str         # Current pipeline step name, used for logging
    errors: list[str]         # Errors collected across steps
    retry_count: int          # Retry counter used by conditional routing in Step 4
    is_valid: bool            # Whether the latest extraction passed validation
