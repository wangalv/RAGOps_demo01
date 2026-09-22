---
name: project-aml-triage-demo
description: "AML triage demo validating arXiv 2604.19755's LLM-evidence-retrieval triage approach — architecture, dataset choice, ML-alert status"
metadata: 
  node_type: memory
  type: project
  originSessionId: c9f15588-c904-4aea-9aed-5a9d7f3eace4
  modified: 2026-09-20T00:13:31.326Z
---

Building a hands-on demo at `/Users/alvinwang/Documents/AI Development/aml-triage-demo/` to validate the paper [Explainable AML Triage with LLMs: Evidence Retrieval and Counterfactual Checks](https://arxiv.org/pdf/2604.19755) (Torres/Cheng/Hu, 2026) — a methodology/evaluation blueprint paper with no released code. Full plan recorded in `aml-triage-demo/PROJECT_PLAN.md` — read that file for current status before resuming this work.

**Why:** User wants to reproduce the paper's core claim — that RAG-grounded, evidence-retrieval LLM triage beats rule-only/ML-only alert triage — using real public data, not just plan it. Counterfactual-check layer is explicitly excluded from scope.

**Architecture:** Rule-based alert + ML-based alert (XGBoost) + Transaction Graph (Quantexa-style) + KYC + Policy + historical case → evidence retrieval → LLM triage. No counterfactual check.

**How to apply:** Before continuing this project in a future session, read `aml-triage-demo/PROJECT_PLAN.md` first — it has the authoritative up-to-date status (which of the 6 inputs are built, file locations, known bugs/fixes already applied). Key facts likely to still be true but worth re-verifying against the plan file: dataset is IBM AML World `HI-Small_Trans.csv` (not SynthAML10 — that dataset turned out to have no account identity/graph structure, wrong shape for this project); XGBoost only runs via the x86_64 `.venv-x86` (arm64 libomp is broken on this machine, Homebrew here is x86_64/Rosetta); ML-based alert (XGBoost v2, with per-day graph degree features) is done and judged "good enough" as an upstream candidate generator — deliberately not over-optimized further. See [[project_kg_explorer]] and [[project_tern_semantic_layer]] for unrelated prior projects — not linked in content, just other entries in this memory store.

**Planned next phase — Autonomous XGBoost Improvement (M10 in learning plan):** Apply the [karpathy/autoresearch](https://github.com/karpathy/autoresearch) pattern: AI agent autonomously edits `train_xgboost.py`, trains, evaluates `Recall@Precision≥0.30`, accepts/rejects, logs results, repeats. Agent explores feature engineering (time-window aggregations, cycle indicators, velocity features) and hyperparameters overnight. See [[project-langgraph-learning]] M10 for full design.
