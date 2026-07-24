# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""Split documents into citable chunks.

`receivables-agent` chunks on `##` headings and drops everything else, which
works because its one policy document was written for that. A client corpus is
not, so this module has two strategies and picks per document:

* **Heading-aware** when a document has Markdown headings — one chunk per
  section, heading kept in the text so the chunk stays self-describing and the
  heading can be cited. This is strictly better retrieval when it applies,
  because a section is a semantic unit and a fixed window is not.
* **Sliding window** otherwise, splitting on paragraph boundaries with overlap,
  because a PDF or a Word file usually has no reliable structure at all.

**Chunk IDs are deterministic** — `{source}::{ordinal}::{slug}`. Re-running an
ingest over unchanged documents produces identical IDs, so the indexer upserts
in place instead of appending duplicates, and the corpus fingerprint is stable
across runs. Both of those depend on this, so do not make IDs positional-only
or content-hashed: a one-word edit to a section must keep its ID and change its
*text*, which is what lets the indexer notice an update rather than seeing a
deletion plus an insertion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from stillroom.ingest.loaders import RawDocument

# A Markdown heading, level 1-3, on its own line.
_HEADING_RE = re.compile(r"^(#{1,3})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
# Blank-line paragraph break, tolerant of trailing whitespace.
_PARA_RE = re.compile(r"\n[ \t]*\n")

# Below this, a "heading-structured" document is really a flat one with a title
# on top, and the sliding window handles it better.
_MIN_SECTIONS_FOR_HEADING_MODE = 2


@dataclass(frozen=True)
class Chunk:
    """One retrievable, citable passage."""

    id: str
    source: str
    # The heading it came from, when there was one. Shown in citations; a client
    # recognises "Refund window" far faster than "chunk 14".
    heading: str | None
    text: str
    index: int


def _slug(text: str, limit: int = 40) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:limit] or "section"


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on Markdown headings into `(heading, body-including-heading)`."""
    matches = list(_HEADING_RE.finditer(text))
    if len(matches) < _MIN_SECTIONS_FOR_HEADING_MODE:
        return []

    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((match.group(2).strip(), body))
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs into ~`size`-char windows with `overlap` chars carried.

    Paragraph-first rather than character-first: cutting mid-sentence produces
    chunks that retrieve well and read terribly, and the client reads them —
    every answer shows its sources.
    """
    paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
    if not paragraphs:
        return []

    windows: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and len(candidate) > size:
            windows.append(current)
            # Carry the tail of the previous window so a fact spanning the seam
            # is retrievable from either side.
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        else:
            current = candidate

    if current:
        windows.append(current)

    # A single paragraph longer than the window (a wall-of-text PDF page) still
    # has to be broken, or it dominates every retrieval it appears in.
    out: list[str] = []
    for window in windows:
        if len(window) <= size * 2:
            out.append(window)
            continue
        step = size - overlap
        out.extend(
            window[i : i + size] for i in range(0, len(window), step) if window[i : i + size].strip()
        )
    return out


def chunk_document(document: RawDocument, *, chunk_chars: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk one document, heading-aware when it can be."""
    chunks: list[Chunk] = []
    sections = _split_sections(document.text)

    if sections:
        for i, (heading, body) in enumerate(sections):
            # A very long section still gets windowed, or heading mode would
            # hand the model a 20-page chapter as one "passage".
            pieces = _window(body, chunk_chars, chunk_overlap) if len(body) > chunk_chars * 2 else [body]
            for j, piece in enumerate(pieces):
                ordinal = f"{i}" if len(pieces) == 1 else f"{i}.{j}"
                chunks.append(
                    Chunk(
                        id=f"{document.source}::{ordinal}::{_slug(heading)}",
                        source=document.source,
                        heading=heading,
                        text=piece,
                        index=len(chunks),
                    )
                )
        return chunks

    for i, piece in enumerate(_window(document.text, chunk_chars, chunk_overlap)):
        chunks.append(
            Chunk(
                id=f"{document.source}::{i}::window",
                source=document.source,
                heading=None,
                text=piece,
                index=i,
            )
        )
    return chunks


def chunk_corpus(
    documents: list[RawDocument], *, chunk_chars: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk every document, preserving the (sorted) document order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(document, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
        )
    return chunks
