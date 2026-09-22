# Log

Append-only. `grep "^## \[" wiki/log.md | tail -5` for the last 5 entries.

## [2026-09-22] ingest | LLM Wiki — Idea Doc
- created: wiki/sources/llm-wiki-idea-doc.md, wiki/concepts/llm-wiki-pattern.md,
  wiki/concepts/retrieval-augmented-generation.md, wiki/entities/vannevar-bush.md,
  wiki/entities/memex.md, wiki/entities/obsidian.md, wiki/entities/obsidian-web-clipper.md,
  wiki/entities/qmd.md, wiki/entities/marp.md, wiki/entities/dataview.md
- updated: wiki/index.md (all sections)
- no contradictions (first source in the vault)

## [2026-09-22] ingest | Health Services Act 1997 No 154 (NSW) — full deep ingest
- scope: user chose "full deep ingest, everything" after a structure scan (11 Chapters + 5A,
  ~453 provisions, 9 Schedules, ~68-term Dictionary); Schedules described in the overview but not
  individually paginated (reference data, not narrative provisions) — noted as a scope decision
  in wiki/sources/health-services-act-1997.md, not silently decided.
- created: raw/health_services_act.txt (saved source); wiki/sources/health-services-act-1997.md;
  wiki/legislation/health-services-act-1997/overview.md plus 30 Chapter/Part pages (ch1 through
  ch11, including ch5a-part-1 through ch5a-part-7); wiki/definitions/ (68 Dictionary term pages,
  generated via script from the extracted Dictionary text); wiki/entities/ (17 new pages: minister,
  health-secretary, local-health-district, statutory-health-corporation,
  affiliated-health-organisation, nsw-health-service, ambulance-service-of-nsw,
  health-administration-corporation, australian-medical-association-nsw,
  local-health-district-board, health-corporation-board, committee-of-review,
  medical-services-committee, ambulance-service-advisory-board, governor,
  industrial-relations-commission, chief-commissioner);
  wiki/obligations/health-services-act-1997-obligations.md;
  wiki/powers/health-services-act-1997-powers.md.
- updated: wiki/index.md (added Legislation, Definitions, Obligations, Powers categories);
  CLAUDE.md (added ch<N>-part-<M>-<slug>.md naming convention note, since Part numbers repeat
  across Chapters and the original schema only anticipated flat part-<n>-<slug>.md).
- no contradictions (first legislation source in the vault).
- open follow-ups: Schedule 3's individual affiliated health organisations not enumerated as
  entities; related Acts referenced throughout (Health Administration Act 1982, Health
  Practitioner Regulation National Law (NSW), State Debt Recovery Act 2018, etc.) don't have
  their own wiki pages yet — candidates for a future ingest if they become directly relevant.

## [2026-09-22] lint | index cleanup after external file deletions
- found during post-ingest verification: wiki/concepts/llm-wiki-pattern.md,
  wiki/concepts/retrieval-augmented-generation.md, wiki/entities/memex.md,
  wiki/entities/obsidian.md, wiki/entities/obsidian-web-clipper.md, and wiki/entities/qmd.md were
  no longer on disk (file timestamps put the change between the two ingest sessions, not caused
  by this session) — same window as the external CLAUDE.md edit that added the Legislation Wiki
  section.
- fixed: removed the now-dangling index.md rows for those six pages so the index only lists pages
  that exist.
- deferred: did not recreate the deleted pages or touch the still-dangling [[memex]] link inside
  wiki/entities/vannevar-bush.md — that's live content referencing a deleted page, which is a
  judgment call for the user, not something to silently patch.
