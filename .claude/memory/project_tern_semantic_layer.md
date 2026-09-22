---
name: project-tern-semantic-layer
description: "TERN Semantic Layer pilot (TernSemanticLayer_v1) — architecture, real data sources, and the key empirical finding on where ontology actually adds LLM value"
metadata: 
  node_type: memory
  type: project
  originSessionId: aa565f0f-afa6-4e00-a7ef-5162f8983dfc
  modified: 2026-07-31T12:08:57.003Z
---

# TERN Semantic Layer Pilot

**Location**: `/Users/alvinwang/Documents/AI Development/TernSemanticLayer_v1`
**Goal**: demonstrate how a semantic layer built on a real ontology (TERN's `tern.ttl`, ecological field-survey data) can serve/improve LLM reasoning — not just RAG-style retrieval.

## What's built (as of 2026-07-31)

- **Ontology Service** (`src/ontology_service.py`): parses `tern.ttl` into Concepts/subClassOf/domain-range edges. 57 defined classes, 108 properties, 26 external refs.
- **Streamlit app** (`app.py`): ontology graph (streamlit-agraph) + instance overlay + "Map report to ontology" panel.
- **Report Mapper** (`src/report_mapper.py`): two-step LLM extraction (classify statement → class, then Python narrows candidate properties via `property_domain_edges`/`inferred_domain_edges`, LLM fills only from that narrowed list, Python rejects anything outside it). Uses **GLM via z.ai** (OpenAI-compatible endpoint, `Z_AI_API_KEY`/`Z_AI_BASE_URL`/`Z_AI_CHAT_MODEL=glm-5.2` in project-local `.env`), not Anthropic.
- **Real EcoPlots data pulled**: `data/ecoplots_vegetation.csv` (1232+ rows), loaded into `data/tern_ecoplots.db` (SQLite, table `vegetation_observations`). Pulled via the *actual* backend API (`POST https://ecoplots.tern.org.au/api/v1.0/ui/data`, body `{"query": {"feature_type": [...], "spatial": {...}}, "page_size": N}` — `page_size` up to 5000 works, must be a top-level sibling of `query`, not nested inside it). `observation_class`/`feature_class` fields in the raw data are literal `https://w3id.org/tern/ontologies/tern/...` URIs — same namespace as our parsed ontology, confirmed real (not just conceptual) alignment.

## Key empirical finding — where ontology actually helps an LLM (and where it doesn't)

### First test (small value space) — null result, but scope-limited

Ran a 3-way text-to-SQL comparison (`scripts/compare_sql_3way.py`) on the `stratum` column (only 5 distinct values: `upper 1/2/3`, `middle 1/2`), e.g. "上层植被" = "upper vegetation layer":

- **A. bare schema only** — often wrong (e.g. `WHERE stratum = 'middle'` → 0 rows, silently wrong).
- **B. schema + raw distinct sample values, no semantic explanation** — correct (`LIKE 'middle%'`).
- **C. schema + real tern:stratum skos:definition** (same example values as B, plus semantic explanation) — **identical SQL to B**, no additional benefit.

Initial (later corrected) conclusion was "ontology semantics add no value for value-guessing." **User correctly challenged this**: `stratum`'s value space is tiny (5 values) and self-describing (raw sample text already looks like the answer) — B had enough information to win trivially regardless of any ontology involvement. This result does not generalize; see below.

### Second test (larger, opaque value space) — ontology/semantic grounding wins clearly

Retested with `used_procedure` on a broader, unfiltered EcoPlots pull (`scripts/fetch_broad_dataset.py`, `scripts/compare_sql_vocab.py`, `data/tern_ecoplots_broad.db`): 7 distinct real values, but raw values are **opaque UUIDs** (e.g. `http://linked.data.gov.au/def/ausplots-cv/59ad2f10-...`) carrying zero guessable information — unlike `stratum`'s self-describing text. Resolved real labels via TERN's own controlled-vocabulary lookup service (`linked.data.gov.au`, live `skos:prefLabel`/`skos:definition`, not in `tern.ttl` itself — cached in `data/resolved_cv_labels.json`).

Two questions ("fuel load survey" data, "large tree survey" records):

- **A. bare schema** — 0 rows both times (obviously wrong, easy to catch).
- **B. schema + raw UUID samples, no labels** — **picked the wrong UUID both times** (guessed `8714be2a` = "ANUclimate derived temperature data" for both questions) but still returned plausible non-zero row counts (48, 530) — **silently wrong, worse than A because it looks like success**.
- **C. schema + resolved label lookup table** — picked the exactly correct UUID both times, matching the real-world meaning.

**Revised conclusion: ontology/semantic grounding's value for "guess the right data value" tasks is real, but conditional on the value space** — it only shows up when raw sample values are themselves uninformative (opaque codes/UUIDs, large cardinality). When raw values are small in number and self-describing (like `stratum`'s text labels), samples alone are sufficient and semantic definitions add nothing measurable. The earlier "ontology adds no value" framing was an artifact of testing only the easy case.

**Where the ontology's value IS real and demonstrated, independent of value-space size**: structural constraint. In `report_mapper.py`'s Level-1 narrowing, the LLM is only shown the candidate properties the ontology actually declares for a class (via `property_domain_edges`) — it structurally cannot select a property outside that list, and Python rejects/flags it if it tries. Not replicable by "just showing more examples," because it restricts the *space of valid structural claims*, not a specific data value.

**How to apply**: this pilot now has two independent, real demonstrations of ontology value: (1) structural constraint (always applies, mechanism-based), and (2) semantic value-grounding for opaque/high-cardinality value spaces (conditional — only matters when raw samples are uninformative; check value-space size/opacity before claiming this effect applies to a new column/dataset). Don't generalize either finding beyond its tested scope without checking the specific column's value-space characteristics first.
