"""
M7.1 — Chunker

Splits raw Act text into section-level chunks.

Why section-level?
  Legal obligations always live inside a named section (e.g. "s.88 Mandatory reporting").
  Splitting at section boundaries preserves the legal unit of meaning —
  a sentence without its section heading loses context.

Output: list of Chunk dicts, each with:
  - chunk_id:   unique id e.g. "s.88"
  - section_id: same, used later for recall evaluation
  - title:      section heading text
  - text:       full text of the section (heading + body)
"""

import re


# NSW legislation plain text format:
#   Section number alone on one line, title on the next line.
#   Examples:
#     "1\nName of Act"
#     "Chapter 1\nPreliminary"
#     "88A\nMandatory reporting"
#
# This regex matches the NUMBER line; we then read the NEXT line as the title.
_HEADING_LINE = re.compile(
    r"^(Chapter\s+\d+|Part\s+\d+[A-Z]?|Division\s+\d+[A-Z]?|Subdivision\s+\d+[A-Z]?|\d+[A-Z]?)$"
)

# Matches a line that is ONLY a subsection marker: "(1)", "(1A)", etc.
# NSW legislation renders the marker on its own line, body on the next.
_SUBSECTION_MARKER = re.compile(r"^\((\d+[A-Za-z]?)\)$")


def chunk_by_section(text: str) -> list[dict]:
    """Split NSW Act plain text into section-level chunks.

    The NSW legislation website renders headings as:
      <section_number>\\n<section_title>\\n<body text...>

    We detect the number line, grab the next line as the title,
    and collect body text until the next heading.
    """
    lines = text.splitlines()
    n = len(lines)

    # Find all heading positions: (line_index, section_id, title)
    headings: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _HEADING_LINE.match(stripped) and i + 1 < n:
            next_line = lines[i + 1].strip()
            # Skip if next line is empty or looks like another number
            if next_line and not _HEADING_LINE.match(next_line):
                section_id = stripped.replace(" ", "_")
                headings.append((i, section_id, next_line))

    if not headings:
        return [{
            "chunk_id": "full",
            "section_id": "full",
            "title": "Full document",
            "text": text[:4000],
        }]

    chunks = []
    for idx, (line_i, section_id, title) in enumerate(headings):
        # Body starts after the title line (line_i + 2)
        body_start = line_i + 2
        body_end = headings[idx + 1][0] if idx + 1 < len(headings) else n
        body_lines = lines[body_start:body_end]
        body = f"{section_id} {title}\n" + "\n".join(body_lines).strip()

        # Cap at 1500 chars so embedding model isn't overloaded
        if len(body) > 1500:
            body = body[:1500]

        # Include line number to guarantee uniqueness —
        # the same section number (e.g. "Part 2") can appear in multiple Chapters.
        chunk_id = f"s.{section_id}_L{line_i}"
        chunks.append({
            "chunk_id": chunk_id,
            "section_id": section_id,
            "title": title,
            "text": body,
        })

    return chunks


def _split_into_subsections(section: dict) -> list[dict]:
    """Split a section chunk into numbered subsection chunks.

    NSW legislation puts the marker "(1)" alone on a line; body follows on
    the next line(s).  Returns a single-element list (the section itself) if
    fewer than 2 numbered subsections are found.
    """
    lines = section["text"].splitlines()

    sub_starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _SUBSECTION_MARKER.match(line.strip())
        if m:
            sub_starts.append((i, m.group(1)))

    if len(sub_starts) < 2:
        return [section]

    result = []
    for idx, (line_i, sub_num) in enumerate(sub_starts):
        end_i = sub_starts[idx + 1][0] if idx + 1 < len(sub_starts) else len(lines)
        # Skip the marker line itself ("(1)" alone); body starts on the next line
        sub_text = "\n".join(lines[line_i + 1:end_i]).strip()

        # Prepend section heading for context
        heading = f"{section['section_id']} {section['title']} ({sub_num})"
        full_text = f"{heading}\n{sub_text}"
        if len(full_text) > 1500:
            full_text = full_text[:1500]

        # Use positional index (s0, s1, …) to guarantee uniqueness —
        # some sections repeat (1) in nested lists.
        result.append({
            "chunk_id":  f"{section['chunk_id']}_s{idx}",
            "section_id": section["section_id"],
            "parent_section_chunk_id": section["chunk_id"],
            "title":     f"{section['title']} ({sub_num})",
            "text":      full_text,
        })

    return result


def chunk_by_subsection(text: str) -> list[dict]:
    """Split Act text into subsection-level chunks.

    Sections that contain numbered subsections (1), (2), … are split at that
    boundary.  Sections without them are kept as a single chunk (same as
    chunk_by_section).
    """
    sections = chunk_by_section(text)
    result: list[dict] = []
    for section in sections:
        result.extend(_split_into_subsections(section))
    return result


def chunk_subsection_parent_child(text: str) -> tuple[list[dict], list[dict]]:
    """Three-level hierarchy: section → subsection (parent) → sentence (child).

    Parents  = subsection chunks — returned to the LLM for generation.
    Children = individual sentences — used for precision retrieval.
    """
    parents = chunk_by_subsection(text)
    children: list[dict] = []

    for parent in parents:
        sentences = re.split(r"(?<=[.!?])\s+", parent["text"])
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        for i, sentence in enumerate(sentences):
            children.append({
                "chunk_id":   f"{parent['chunk_id']}_c{i}",
                "parent_id":  parent["chunk_id"],
                "section_id": parent["section_id"],
                "title":      parent["title"],
                "text":       sentence,
            })

    return parents, children


def chunk_parent_child(text: str) -> tuple[list[dict], list[dict]]:
    """Split Act text into two levels: parent (section) and child (sentence).

    Parent chunks  = same as chunk_by_section(), used for LLM generation
    Child chunks   = each sentence inside a parent, used for retrieval

    Each child stores its parent_id so we can fetch the full parent after retrieval.
    This lets us retrieve with precision (small child vectors) but generate with
    full context (large parent text).
    """
    parents = chunk_by_section(text)

    children: list[dict] = []
    for parent in parents:
        # Split parent body into sentences (split on ". " or ".\n")
        sentences = re.split(r"(?<=[.!?])\s+", parent["text"])
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

        for i, sentence in enumerate(sentences):
            child_id = f"{parent['chunk_id']}_c{i}"
            children.append({
                "chunk_id":  child_id,
                "parent_id": parent["chunk_id"],
                "section_id": parent["section_id"],
                "title":     parent["title"],
                "text":      sentence,
            })

    return parents, children
