"""
M9 — RAGOps Tracing: SubsecRoutedPC→LLMRnk@5

Runs the best pipeline on Grounded-79, generating Langfuse Traces with
full span-level instrumentation, per-query recall scores, and dataset
run links so results appear in both Traces and Experiments.

Run from LangGraphLearning/:
    .venv/bin/python eval/trace_best_pipeline.py            # full 79 queries
    .venv/bin/python eval/trace_best_pipeline.py --limit 5  # smoke test
"""

import argparse
import datetime
import os
import re
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse, propagate_attributes

from src.rag.chunker import chunk_subsection_parent_child
from src.rag.vector_store import VectorStore
from src.rag.reranker import Reranker
from src.rag.query_router import classify_query
from src.rag.legal_query_expansion_retriever import expand_query
from src.rag.hyde_retriever import generate_hypothetical_doc
from src.rag.multi_query_retriever import generate_query_variants
from src.rag.stepback_retriever import generate_stepback_query
from src.rag.llm_reranker import llm_rerank_legal

ACT_TEXT_FILE         = "tests/rag/health_services_act.txt"
DATASET_NAME          = "NSW-HealthAct-SimulatedQueries-Grounded-77"
SUBSEC_COLLECTION     = "health-services-act-subsections"
SUBSEC_CHILD_COLLECTION = "health-services-act-subsec-children"
CANDIDATE_N = 20
RUN_NAME    = "SubsecRoutedPC→LLMRnk@5-traced"
SESSION_ID  = f"SubsecRoutedPC-LLMRnk5-grounded79-{datetime.date.today()}"

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)


def _section_prefix(chunk_id: str) -> str:
    m = re.match(r"^(s\.[^_]+)", chunk_id)
    return m.group(1) if m else chunk_id


def _recall_hit(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    if not relevant:
        return 1.0
    return 1.0 if any(s in set(retrieved[:k]) for s in relevant) else 0.0


def run_traced(subsec_store, subsec_child_store, reranker, items):
    hits = 0
    for i, item in enumerate(items):
        query    = item.input["query"]
        relevant = (item.expected_output or {}).get("relevant_sections", [])
        trace_id = langfuse.create_trace_id()

        with propagate_attributes(session_id=SESSION_ID), langfuse.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name=RUN_NAME,
            as_type="chain",
            input={"query": query},
            metadata={"dataset_item_id": item.id, "relevant_sections": relevant},
        ) as root:

            # ── 1. Query routing ──────────────────────────────────────────────
            with langfuse.start_as_current_observation(
                name="query_routing",
                as_type="span",
                input={"query": query},
            ) as span:
                qtype = classify_query(query)
                span.update(output={"query_type": qtype})

            # ── 2. Query augmentation ─────────────────────────────────────────
            with langfuse.start_as_current_observation(
                name="query_augmentation",
                as_type="span",
                input={"query": query, "query_type": qtype},
            ) as span:
                if qtype == "Penalty":
                    aug_queries = [expand_query(query)]
                elif qtype == "Prohibition":
                    aug_queries = [query, generate_stepback_query(query)]
                elif qtype == "CrossSection":
                    aug_queries = [query] + generate_query_variants(query, n=3)
                else:
                    aug_queries = [generate_hypothetical_doc(query)]
                span.update(output={"augmented_queries": aug_queries, "strategy": qtype})

            # ── 3. Subsection child retrieval ─────────────────────────────────
            seen: set[str] = set()
            children: list[dict] = []

            def _collect(search_q: str) -> None:
                for c in subsec_child_store.search_children(search_q, top_k=CANDIDATE_N):
                    if c["chunk_id"] not in seen:
                        seen.add(c["chunk_id"])
                        children.append(c)

            with langfuse.start_as_current_observation(
                name="child_retrieval",
                as_type="retriever",
                input={"queries": aug_queries, "top_k_per_query": CANDIDATE_N},
            ) as span:
                for sq in aug_queries:
                    _collect(sq)
                span.update(output={"n_children_retrieved": len(children)})

            # ── 4. Parent fetch (subsection-level) ────────────────────────────
            with langfuse.start_as_current_observation(
                name="parent_fetch",
                as_type="span",
                input={"n_children": len(children)},
            ) as span:
                parents = subsec_child_store.get_parents_by_ids(
                    subsec_store, [c["parent_id"] for c in children]
                )
                span.update(output={"n_parents": len(parents)})

            # ── 5. Cross-encoder rerank → top 10 ─────────────────────────────
            with langfuse.start_as_current_observation(
                name="cross_encoder_rerank",
                as_type="span",
                input={"n_candidates": len(parents), "target_k": 10},
            ) as span:
                top10 = reranker.rerank(query, parents, top_k=10)
                span.update(output={
                    "top_10_sections": [_section_prefix(c["chunk_id"]) for c in top10],
                })

            # ── 6. LLM legal rerank → top 5 ──────────────────────────────────
            with langfuse.start_as_current_observation(
                name="llm_rerank",
                as_type="generation",
                input={"query": query, "n_candidates": len(top10)},
                model=os.environ.get("Z_AI_CHAT_MODEL", "unknown"),
            ) as span:
                top5    = llm_rerank_legal(query, top10, top_k=5)
                sections = [_section_prefix(c["chunk_id"]) for c in top5]
                span.update(output={
                    "top_5_sections": sections,
                    "llm_scores": [c.get("llm_score") for c in top5],
                })

            # ── Root output ───────────────────────────────────────────────────
            hit = _recall_hit(sections, relevant, k=5)
            hits += int(hit)
            root.update(
                output={
                    "retrieved_sections": sections,
                    "relevant_sections":  relevant,
                    "recall_hit":         bool(hit),
                },
                metadata={"dataset_item_id": item.id, "query_type": qtype},
            )

        # Score the trace
        langfuse.create_score(
            trace_id=trace_id,
            name="recall@5",
            value=hit,
            data_type="NUMERIC",
            comment=f"relevant={relevant}  retrieved={sections}",
        )
        langfuse.create_score(
            trace_id=trace_id,
            name="recall_hit",
            value=hit,
            data_type="BOOLEAN",
        )

        # Link trace to Grounded-79 dataset run
        langfuse.api.dataset_run_items.create(
            run_name=RUN_NAME,
            dataset_item_id=item.id,
            trace_id=trace_id,
        )

        status = "✓" if hit else "✗"
        print(f"  [{i+1:3d}/{len(items)}] {status}  {query[:65]}")

    r5 = hits / len(items) if items else 0.0
    print(f"\n  Recall@5 = {r5:.3f}  ({hits}/{len(items)})")
    return r5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N queries (smoke test)")
    args = parser.parse_args()

    print(f"Loading dataset '{DATASET_NAME}' from Langfuse...")
    dataset = langfuse.get_dataset(DATASET_NAME)
    items   = dataset.items[: args.limit] if args.limit else dataset.items
    print(f"  {len(items)} queries\n")

    print("Loading Act and building subsection indexes...")
    with open(ACT_TEXT_FILE, encoding="utf-8") as f:
        act_text = f.read()

    subsec_chunks, subsec_children = chunk_subsection_parent_child(act_text)

    subsec_store = VectorStore(collection_name=SUBSEC_COLLECTION, persist_dir="./chroma_db")
    if subsec_store.count() != len(subsec_chunks):
        subsec_store.reset()
        subsec_store.add_chunks(subsec_chunks)
    else:
        print(f"  Subsec store:  {subsec_store.count()} chunks (reused)")

    subsec_child_store = VectorStore(collection_name=SUBSEC_CHILD_COLLECTION, persist_dir="./chroma_db")
    if subsec_child_store.count() != len(subsec_children):
        subsec_child_store.reset()
        subsec_child_store.add_children(subsec_children)
    else:
        print(f"  SubsecChild:   {subsec_child_store.count()} chunks (reused)")

    reranker = Reranker()
    print()

    print(f"Running {RUN_NAME} with full Langfuse tracing...\n")
    run_traced(subsec_store, subsec_child_store, reranker, items)

    langfuse.flush()
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    print(f"\n✓ Done")
    print(f"  Traces    → {host}/traces")
    print(f"  Dataset   → {host}/datasets/{DATASET_NAME}/runs/{RUN_NAME}")


if __name__ == "__main__":
    main()
