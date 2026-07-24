"""Chunking: deterministic IDs, heading awareness, and the windowing fallback."""

from __future__ import annotations

from stillroom.ingest.chunking import chunk_corpus, chunk_document
from stillroom.ingest.loaders import RawDocument

PARAMS = {"chunk_chars": 400, "chunk_overlap": 50}


def _doc(text: str, source: str = "doc.md") -> RawDocument:
    return RawDocument(source=source, text=text)


def test_headings_become_one_chunk_each_with_the_heading_kept():
    document = _doc("# Title\n\n## Alpha\n\nBody A.\n\n## Beta\n\nBody B.\n")

    chunks = chunk_document(document, **PARAMS)

    assert [c.heading for c in chunks] == ["Title", "Alpha", "Beta"]
    # The heading stays in the text, so a retrieved chunk is self-describing.
    assert "## Alpha" in chunks[1].text
    assert "Body A." in chunks[1].text


def test_chunk_ids_are_stable_across_runs():
    document = _doc("## Alpha\n\nBody A.\n\n## Beta\n\nBody B.\n")

    first = chunk_document(document, **PARAMS)
    second = chunk_document(document, **PARAMS)

    assert [c.id for c in first] == [c.id for c in second]
    # Deterministic IDs are what let the indexer upsert in place and prune.
    assert first[0].id == "doc.md::0::alpha"


def test_editing_a_section_keeps_its_id_and_changes_its_text():
    """An edit must read as an update, never as a delete plus an insert."""
    before = chunk_document(_doc("## Alpha\n\nOld body.\n\n## Beta\n\nB.\n"), **PARAMS)
    after = chunk_document(_doc("## Alpha\n\nNew body.\n\n## Beta\n\nB.\n"), **PARAMS)

    assert before[0].id == after[0].id
    assert before[0].text != after[0].text


def test_a_document_without_headings_falls_back_to_windows():
    text = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(40))

    chunks = chunk_document(_doc(text, source="plain.txt"), **PARAMS)

    assert len(chunks) > 1
    assert all(c.heading is None for c in chunks)
    assert all(c.id.endswith("::window") for c in chunks)


def test_a_single_heading_is_not_enough_for_heading_mode():
    """One heading means a title on a flat document, not a structured one."""
    text = "# Only title\n\n" + "\n\n".join(f"Para {i}." for i in range(30))

    chunks = chunk_document(_doc(text), **PARAMS)

    assert all(c.heading is None for c in chunks)


def test_an_oversized_paragraph_is_broken_up():
    """A wall-of-text PDF page must not dominate every retrieval it appears in."""
    chunks = chunk_document(_doc("word " * 3000, source="wall.txt"), **PARAMS)

    assert len(chunks) > 1
    assert all(len(c.text) <= PARAMS["chunk_chars"] * 2 for c in chunks)


def test_windows_overlap_so_a_seam_is_retrievable_from_either_side():
    text = "\n\n".join(f"Paragraph {i} " + "filler " * 20 for i in range(10))

    chunks = chunk_document(_doc(text, source="p.txt"), **PARAMS)

    tail = chunks[0].text[-PARAMS["chunk_overlap"] :].strip()
    assert tail and tail[:20] in chunks[1].text


def test_chunk_corpus_preserves_document_order():
    documents = [_doc("## A\n\nx\n", "a.md"), _doc("## B\n\ny\n", "b.md")]

    chunks = chunk_corpus(documents, **PARAMS)

    assert [c.source for c in chunks] == ["a.md", "b.md"]
