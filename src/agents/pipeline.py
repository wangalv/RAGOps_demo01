"""
M1 Step 3 — Conditional routing added to the pipeline

LangGraph feature: add_conditional_edges()
  Instead of a fixed edge, a router function reads State and returns
  the name of the next node to execute. This lets the graph branch
  based on data — pass/fail, retry logic, human review, etc.

Graph structure:
    START → fetch_ministers_node → fetch_acts_node → validate_node
                                          ↑                │
                                          │    retry_count < 2 & invalid
                                          └── retry_node ←─┘
                                                            │
                                                      valid → END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import PipelineState
from src.tools.fetch_legislation import fetch_ministers_list, fetch_acts_by_minister, fetch_act_html

MAX_RETRIES = 2


# ── Nodes ─────────────────────────────────────────────────────────────────────

def fetch_ministers_node(state: PipelineState) -> dict:
    print(f"[fetch_ministers_node] fetching {state['source_url']}")
    ministers = fetch_ministers_list.invoke({"url": state["source_url"]})
    print(f"[fetch_ministers_node] found {len(ministers)} ministers")
    return {"ministers": ministers, "current_step": "fetch_ministers"}


def fetch_acts_node(state: PipelineState) -> dict:
    minister = state["selected_minister"]
    print(f"[fetch_acts_node] fetching Acts for '{minister}' (attempt {state['retry_count'] + 1})")
    acts = fetch_acts_by_minister.invoke({
        "url": state["source_url"],
        "minister": minister,
    })
    print(f"[fetch_acts_node] found {len(acts)} Acts")
    return {"act_urls": acts, "current_step": "fetch_acts"}


def validate_node(state: PipelineState) -> dict:
    """Check that act_urls is non-empty and minister name was recognised."""
    acts = state.get("act_urls", [])
    valid = len(acts) > 0
    if valid:
        print(f"[validate_node] PASS — {len(acts)} Acts found")
    else:
        print(f"[validate_node] FAIL — no Acts found for '{state['selected_minister']}'")
    return {"is_valid": valid, "current_step": "validate"}


def retry_node(state: PipelineState) -> dict:
    count = state["retry_count"] + 1
    print(f"[retry_node] incrementing retry_count to {count}")
    return {"retry_count": count, "current_step": "retry"}


# ── Router function — this is the conditional edge ────────────────────────────

def route_after_validate(state: PipelineState):
    """
    Router for the conditional edge after validate_node.

    Key M2 concept: when returning a list of Send objects, LangGraph
    spawns each one as a parallel task — this is how Send API works.
    The router must live inside add_conditional_edges, not in a node.
    """
    if state["is_valid"]:
        print(f"[router] valid — spawning {len(state['act_urls'])} parallel tasks")
        return [
            Send("process_act", {"act_url": act["url"], "act_title": act["title"]})
            for act in state["act_urls"][:5]   # limit to 5 for learning purposes
        ]
    if state["retry_count"] < MAX_RETRIES:
        return "retry"
    print("[router] max retries reached — ending with empty result")
    return END


# ── M2: Send API node ─────────────────────────────────────────────────────────

def process_act_node(state: dict) -> dict:
    """
    Runs in parallel for each Act. Receives a small state with just
    act_url and act_title — not the full PipelineState.
    Returns a single-item list so operator.add appends it to processed_acts.
    """
    url   = state["act_url"]
    title = state["act_title"]
    print(f"[process_act_node] fetching: {title}")
    result = fetch_act_html.invoke({"url": url})
    return {
        "processed_acts": [{
            "title": title,
            "url":   url,
            "text":  result["text"][:500],   # store first 500 chars for now
        }]
    }


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("fetch_ministers", fetch_ministers_node)
    graph.add_node("fetch_acts",      fetch_acts_node)
    graph.add_node("validate",        validate_node)
    graph.add_node("retry",           retry_node)
    graph.add_node("process_act", process_act_node)  # M2: runs once per Act in parallel

    # Fixed edges
    graph.add_edge(START, "fetch_ministers")
    graph.add_edge("fetch_ministers", "fetch_acts")
    graph.add_edge("fetch_acts", "validate")
    graph.add_edge("retry", "fetch_acts")

    # Conditional edge from validate.
    # When route_after_validate returns a list[Send], LangGraph fans out in parallel.
    # When it returns the string "retry", LangGraph follows the "retry" mapping.
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"retry": "retry"},   # only need to map the string return values
    )

    # Each parallel process_act branch ends here
    graph.add_edge("process_act", END)

    # MemorySaver stores a State snapshot after every node completes.
    # In production replace with SqliteSaver or PostgresSaver for persistence.
    return graph.compile(checkpointer=MemorySaver())
