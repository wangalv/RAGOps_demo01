---
name: project-kg-explorer
description: "NSW Legislation Knowledge Graph Explorer — key fixes, architecture decisions, and non-negotiable requirements"
metadata: 
  node_type: memory
  type: project
  originSessionId: ba514c41-891a-418d-a8f3-423e68287f3c
  modified: 2026-07-26T12:34:47.288Z
---

# NSW Legislation Knowledge Graph Explorer

**File**: `LegislationParser_v1/src/ux/kg_explorer.py`
**Stack**: Streamlit + streamlit-agraph (vis-react / vis.js) + Neo4j

## Non-negotiable requirements (do not change these)
1. Default graph = "expand all" state — all acts + provisions visible
2. Root Energy (left) and Root Environment (right) clearly separated
3. Before double-clicking, nodes draggable but auto spring-back
4. Double-click leaf/provision node extends next layer (obligations/penalties/entities); overall graph shape unchanged; no children → only shows details in right column, no graph change
5. During correspondence analysis, highlighted nodes light up WITHOUT changing overall graph shape

## Key architectural fixes (confirmed working as of 2026-07)

### Fix 1: Graph shuffle on every Streamlit rerun / double-click
**Root cause**: vis-react calls `network.setData(newDataSet)` on every rerun with a FRESH DataSet — vis.js assigns RANDOM positions to nodes without explicit x/y.
**Fix**: `_pin_kwargs()` returns `{"x": p["x"], "y": p["y"]}` for nodes already in `_saved_positions`. Regular-rerun branch uses `stabilization.enabled: False` — eliminates canvas blink entirely.

### Fix 2: Child nodes (obligations/penalties/entities) not expanding on double-click
**Root cause**: `load_provisions` had no ORDER BY, so Neo4j returned preamble sections ("Currency of version", "Notes—") as the first 3 provisions — these have ZERO children.
**Fix**: Added `ORDER BY child_count DESC, p.node_title` to `load_provisions` so the 3 provisions shown per act are those with the most children.

### Fix 3: StreamlitDuplicateElementId crash
**Root cause**: Two `st.button("Clear", use_container_width=True)` at lines 699 and 1024 generated the same auto-ID.
**Fix**: Added `key="clear_corr"` and `key="clear_edge"` respectively.

### Fix 4: Node highlighting disrupts graph shape during correspondence analysis
**Root cause**: `_in_reveal` branch used `stabilization.enabled: True, iterations: 30` → one blink per matched act. Even after switching to `stabilization.enabled: False` with high damping, live physics forces still ran on every `setData()` call, causing the graph to shuffle.
**Fix**: Set `physics.enabled: False` during `_in_reveal`. Physics engine completely off → vis.js places nodes exactly at saved x/y, zero forces → perfect graph shape preservation during colour-only changes.

## Physics config branches (lines 628–645)
- `_in_reveal`: `physics.enabled=False`, `stabilization.enabled=False` — physics completely off, colour-only changes, graph shape frozen
- `not _graph_fitted` (first render): `stabilization.enabled=True, iterations=1000, fit=True` — full layout + zoom to fit
- regular rerun: `damping=0.80`, `stabilization.enabled=False` — existing nodes get explicit x/y, spring-back via live physics

## JS patch (in streamlit-agraph frontend build)
`_kgPos` localStorage stores node positions. `stabilizationIterationsDone` event saves positions once (guarded by `window._kgPosSent`). `selectNode` / `doubleClick` send current positions with the event payload via `"nodeId|||{positions}"` or `"dbl::nodeId|||{positions}"`.

**Why:** [[feedback-vis-js-positions]]
