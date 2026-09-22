---
type: source
title: LLM Wiki — Idea Doc
aliases: [LLM Wiki idea file]
tags: [meta, knowledge-management, llm]
created: 2026-09-22
updated: 2026-09-22
raw_path: raw/llm-wiki-idea-doc.md
ingested: 2026-09-22
---

The founding document for this vault. Describes [[llm-wiki-pattern]]: instead of re-deriving
answers from raw documents at query time (the [[retrieval-augmented-generation]] model), an LLM
incrementally builds and maintains a persistent, interlinked wiki that compounds as sources are
added.

## Key takeaways

- **Compounding vs. re-deriving.** The core distinction from plain RAG is that knowledge is
  synthesized once, on ingest, and then kept current — not rebuilt from fragments on every query.
- **Three layers.** `raw/` (immutable sources) → `wiki/` (LLM-owned, generated pages) → schema
  file (CLAUDE.md/AGENTS.md — the config that makes the LLM a disciplined maintainer).
- **Three operations.** Ingest (add + integrate a source), Query (answer + optionally file the
  answer back as a page), Lint (periodic health check for contradictions/orphans/gaps).
- **Two navigation files.** `index.md` (content-oriented catalog, read first when answering
  queries) and `log.md` (chronological, append-only, greppable by a consistent `## [date] type |`
  prefix).
- **Human/LLM division of labor.** Human curates sources, directs analysis, asks questions. LLM
  does all reading, summarizing, cross-referencing, and bookkeeping.
- **Intellectual lineage:** explicitly framed as realizing what [[vannevar-bush]]'s [[memex]]
  (1945) couldn't — the missing piece was maintenance, which the LLM now handles.
- **Tooling mentioned:** [[obsidian]] as the viewing/browsing layer, [[obsidian-web-clipper]] for
  sourcing, [[qmd]] as an optional local search engine at scale, [[marp]] and [[dataview]] as
  Obsidian plugins for output formats and dynamic frontmatter queries.
- Explicitly says the implementation (folder structure, schema, tooling) is meant to be
  instantiated collaboratively per-domain, not followed rigidly — this vault's CLAUDE.md is that
  instantiation.

## Open questions this raises

- What counts as an "entity" vs. a "concept" in a non-tech domain — not yet tested against real
  content beyond this meta-example.
- No search tooling ([[qmd]] or otherwise) is wired up yet; fine at this scale, worth revisiting
  once the wiki has more sources.
