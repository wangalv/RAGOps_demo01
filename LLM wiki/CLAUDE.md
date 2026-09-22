# LLM Wiki — Schema

This vault is a personal, LLM-maintained knowledge base. You (Claude) own the `wiki/` layer
completely: you create pages, update them as new sources arrive, maintain cross-references, and
keep the whole thing internally consistent. The user owns sourcing, direction, and questions.
They read; you write.

Three layers:

1. **`raw/`** — immutable source material (articles, papers, transcripts, images). Never edit
   files in here. This is ground truth.
2. **`wiki/`** — everything you generate: summaries, entity pages, concept pages, synthesis,
   the index, the log.
3. **This file** — the schema. Update it yourself when the user corrects a convention or a new
   pattern emerges, so the next session inherits it.

## Folder conventions

```
raw/                  immutable sources. Never modified by Claude.
raw/assets/            downloaded images/attachments (Obsidian attachment folder, see below)
wiki/index.md          content-oriented catalog of every wiki page, by category
wiki/log.md            append-only chronological record of ingests/queries/lints
wiki/entities/         named, concrete things: people, orgs, products, tools, places
wiki/concepts/         abstract ideas, techniques, topics, patterns
wiki/sources/          one page per ingested source: summary + takeaways + link to raw file
wiki/synthesis/        evolving theses, comparisons, overviews, and query answers worth keeping
```

## File naming

- kebab-case, no spaces: `vannevar-bush.md`, `retrieval-augmented-generation.md`
- One concept/entity = one file. If a page would cover two distinct things, split it.
- Source pages are named after the source, not the topic: `wiki/sources/llm-wiki-idea-doc.md`.

## Frontmatter (every wiki page)

```yaml
---
type: entity | concept | source | synthesis
title: Human-readable title
aliases: [alternate names, acronyms]
tags: [freeform, lowercase, tags]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Source pages additionally include `raw_path: raw/<filename>` and `ingested: YYYY-MM-DD`.

## Linking

- Use Obsidian `[[wikilinks]]` for every reference to another wiki page. Link on first mention
  per page; don't over-link every subsequent occurrence.
- A new page should have at least one inbound link from an existing page (usually the source
  page that spawned it, or the index). Zero inbound links = orphan — flag it during lint.
- When a page's claim comes from a specific source, cite it with a link to that source's page
  in `wiki/sources/`, e.g. `(see [[llm-wiki-idea-doc]])`.

## Operations

### Ingest

Trigger: user drops a file in `raw/` (or pastes content to save there) and asks to ingest it, or
just references a new source in conversation.

1. Read the source in full (and any referenced images — read text first, then view images
   separately per the note below).
2. Summarize key takeaways back to the user in chat. Ask what to emphasize *before* writing
   pages, unless the user has clearly said "just go ahead" — default to staying involved rather
   than silently batch-processing, since that's the intended workflow here.
3. Create `wiki/sources/<slug>.md`: a summary page with takeaways and a link back to the raw
   file.
4. Check `wiki/index.md` for existing entity/concept pages this source touches. Update them —
   don't just append; integrate. If the new source contradicts an existing claim, don't silently
   overwrite it: add a note (e.g. a `> [!note] Contradiction` callout) pointing at both sources
   and dates, and flag it to the user in chat.
5. Create new entity/concept pages for anything notable that doesn't have one yet.
6. Update `wiki/index.md` (new/changed rows) and append an entry to `wiki/log.md`.
7. Report back: pages created, pages updated, any contradictions flagged.

A single source can reasonably touch 10-15 pages — that's expected, not a sign of scope creep.

### Query

Trigger: user asks a question against the wiki.

1. Read `wiki/index.md` first to find candidate pages — don't grep raw sources directly unless
   the wiki pages turn out to be insufficient (stale, missing detail), in which case fall back
   to the cited raw source.
2. Read the candidate pages, synthesize an answer, cite the wiki pages (and raw sources where
   relevant).
3. If the answer is a reusable synthesis — a comparison, an analysis, a connection the user
   will want again — offer to file it as a new `wiki/synthesis/` page rather than letting it
   disappear into chat scrollback. On a yes, create the page, link it from related pages and the
   index, and log it.

### Lint

Trigger: user asks for a health check ("lint the wiki").

Check for: orphan pages (no inbound links), contradictions between pages, claims a newer source
has superseded, concepts mentioned often but lacking their own page, missing cross-references,
gaps a web search could fill. Report findings and propose fixes; don't bulk-edit without
confirming, since lint can touch many files at once.

Log a lint pass in `wiki/log.md` even if it finds nothing to fix.

## Log format (`wiki/log.md`)

Append-only. Each entry starts with a consistent prefix so it's greppable:

```
## [YYYY-MM-DD] ingest | <Source Title>
- created: wiki/sources/foo.md, wiki/concepts/bar.md
- updated: wiki/concepts/baz.md (added §, noted contradiction with wiki/sources/older-source.md)
```

```
## [YYYY-MM-DD] query | <short question summary>
- answered from: wiki/concepts/bar.md, wiki/entities/baz.md
- filed as: wiki/synthesis/comparison-x-vs-y.md   (omit this line if not filed)
```

```
## [YYYY-MM-DD] lint | <one-line summary>
- issues found: ...
- fixed: ... / deferred: ...
```

`grep "^## \[" wiki/log.md | tail -5` gives the last 5 entries.

## Index format (`wiki/index.md`)

Organized by category with `###` subheadings matching the `wiki/` subfolders (Entities,
Concepts, Sources, Synthesis). Each row: link, one-line summary, optionally a date or tag.

```
- [[vannevar-bush]] — inventor of the Memex concept; forerunner to this whole pattern.
```

Update it on every ingest and every filed query/synthesis page.

## Images

Obsidian's attachment folder is set to `raw/assets/`. You can't read markdown with inline images
in one pass — read the page's text first, then view referenced images separately if you need
them for context.

## Conventions still to establish

This file should grow as the user's domain and preferences become clear — e.g. how deep to go
per source, what counts as an entity vs. a concept in their domain, output formats they want
(Marp decks, comparison tables) for filed queries. When the user corrects a convention, update
this file, don't just remember it for the session.

---

## Legislation Wiki

This vault is specialised for legislation documents. The standard wiki layers still apply.
When a source in `raw/legislation/` is ingested, follow the extended workflow below in addition
to the standard Ingest workflow.

### Additional folders

```
raw/legislation/                         source files (HTML/PDF/MD). Never edit.
wiki/legislation/<act-slug>/
  overview.md                            Act structure: parts, section count, purpose
  part-<n>-<slug>.md                     one page per Part
wiki/definitions/                        one page per defined term (drawn from definitions section)
wiki/obligations/                        obligation matrices (who must do what, under which section)
wiki/powers/                             powers matrices (who may do what, under which section)
```

**Part-page naming when Part numbers repeat across Chapters** (they do in most Acts — every
Chapter tends to restart at "Part 1"): name pages `ch<N>-part-<M>-<slug>.md`, not the bare
`part-<n>-<slug>.md` the schema above implies. A Chapter with no Parts gets a single
`ch<N>-<slug>.md` page instead of a Part page. First applied in the
[[health-services-act-1997-overview]] ingest.

### Frontmatter additions for legislation pages

Act overview (`wiki/legislation/<act-slug>/overview.md`):
```yaml
---
type: legislation-overview
title: "Health Services Act 1997 (NSW)"
act_id: act-1997-154
jurisdiction: NSW
url: https://legislation.nsw.gov.au/...
sections_total:
parts_total:
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Part page (`wiki/legislation/<act-slug>/part-<n>-<slug>.md`):
```yaml
---
type: legislation-part
act: "[[health-services-act-1997]]"
part_number:
part_title:
sections: [s.8, s.9, s.10]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Definition page (`wiki/definitions/<term>.md`):
```yaml
---
type: definition
title:
defined_in: "[[health-services-act-1997]]"
section: s.4
aliases: []
tags: []
created: YYYY-MM-DD
---
```

### Legislation Ingest workflow

Trigger: user drops a file in `raw/legislation/` and says "ingest [Act name]".

1. **Scan structure** — read the full document; identify title, jurisdiction, year, total
   parts and sections. Report the structure to the user before writing anything.

2. **Discuss scope** — ask which parts to go deep on vs. stub, unless the user says
   "full ingest". Default: extract all definitions + write overview + create Part stubs.

3. **`wiki/sources/<act-slug>.md`** — purpose of Act, key themes, parts count, raw file link.

4. **`wiki/legislation/<act-slug>/overview.md`** — Act structure table:

   | Part | Title | Sections | Theme |

   Include: purpose clause, key parties, penalties overview.

5. **`wiki/legislation/<act-slug>/part-<n>-<slug>.md` for each Part**:
   - Part purpose (2–3 sentences)
   - Section table: `| s.X | Heading | One-line summary |`
   - Key obligations and powers in this Part
   - Cross-references to other Parts/sections
   - `[[wikilinks]]` to relevant entity and definition pages

6. **`wiki/definitions/<term>.md` for every defined term** — exact statutory definition,
   plain-language explanation, wikilinks to sections that use the term.

7. **`wiki/entities/`** — create/update a page for every named organisation or role
   (Minister, Secretary, Board, etc.). Each page lists: powers, obligations, sections they appear in.

8. **`wiki/obligations/<act-slug>-obligations.md`** — extract every "must" / "shall" / "is
   required to" provision:

   | Actor | Obligation | Section | Condition |

9. **`wiki/powers/<act-slug>-powers.md`** — extract every "may" / "is entitled to" provision:

   | Actor | Power | Section | Condition |

10. **Update `wiki/index.md`** — add categories: Legislation, Definitions, Obligations, Powers.

11. **Append to `wiki/log.md`**: `## [DATE] ingest | <Act Title>`

### Deep-dive on a specific section

Trigger: "go deep on s.60" or "expand Part 6".

1. Read the section(s) in the raw file.
2. Expand the relevant Part page with full section-by-section breakdown.
3. Update obligations/powers matrices for any new provisions found.
4. Link from entity/definition pages that appear in this section.
5. Log as a `query` entry.

### Cross-reference conventions

- Section references always use format `s.60` (lowercase s, dot, number).
- Wikilinks to a specific section: `[[part-6-visiting-practitioners#s.60]]`
- When a section says "see s.X" or "subject to Part Y", add a `> See also: [[...]]` callout
  so cross-references are bidirectional.

### Handling multiple versions

If multiple versions of an Act exist in `raw/legislation/`:
- Name with version suffix: `health-services-act-1997-v2024.md`
- Add `version` and `in_force_from` to frontmatter.
- On new version ingest, flag changed sections with `> [!warning] Updated in [version]` callouts.
