"""Ingest: loaders, idempotence, and pruning.

Pruning is the one with teeth. A client deletes a document because it is wrong
or because they are not allowed to keep it; if the chatbot goes on citing it,
that is a correctness bug with a compliance shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stillroom.index.store import get_collection, open_client, read_fingerprint
from stillroom.pipeline import ingest
from stillroom.ingest.loaders import UnreadableDocument, load_corpus, load_document


def test_load_corpus_reads_text_and_markdown(corpus: Path):
    documents, skipped = load_corpus(corpus, (".md", ".txt"))

    assert {d.source for d in documents} == {"handbook.md", "shipping.txt"}
    assert skipped == []


def test_sources_are_relative_so_they_do_not_leak_the_build_machine(corpus: Path):
    documents, _ = load_corpus(corpus, (".md",))

    assert documents[0].source == "handbook.md"
    assert not Path(documents[0].source).is_absolute()


def test_documents_come_back_sorted(tmp_path: Path):
    for name in ("c.md", "a.md", "b.md"):
        (tmp_path / name).write_text("## H\n\nbody\n", encoding="utf-8")

    documents, _ = load_corpus(tmp_path, (".md",))

    # Fingerprint stability must not depend on filesystem iteration order.
    assert [d.source for d in documents] == ["a.md", "b.md", "c.md"]


def test_an_unreadable_file_is_skipped_not_fatal(corpus: Path):
    (corpus / "scan.pdf").write_bytes(b"not really a pdf")

    documents, skipped = load_corpus(corpus, (".md", ".txt", ".pdf"))

    # One bad file in a hand-assembled folder must not deliver nothing.
    assert len(documents) == 2
    assert len(skipped) == 1
    assert "scan.pdf" in skipped[0]


def test_an_empty_file_is_refused(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")

    with pytest.raises(UnreadableDocument):
        load_document(path, tmp_path)


def test_ingest_indexes_and_fingerprints(config, embedding):
    result = ingest(config, embedding_function=embedding, embedding_name="test")

    assert result.documents == 2
    assert result.chunks > 0
    assert result.fingerprint

    client = open_client(config.index_path)
    assert read_fingerprint(client, config.collection, embedding) == result.fingerprint


def test_ingest_is_idempotent(config, embedding):
    first = ingest(config, embedding_function=embedding, embedding_name="test")
    second = ingest(config, embedding_function=embedding, embedding_name="test")

    assert first.fingerprint == second.fingerprint
    assert first.chunks == second.chunks
    assert second.pruned == 0

    collection = get_collection(open_client(config.index_path), config.collection, embedding)
    # Re-running must converge on the corpus, never grow.
    assert collection.count() == second.chunks


def test_deleting_a_document_prunes_its_passages(config, embedding):
    ingest(config, embedding_function=embedding, embedding_name="test")
    (Path(config.corpus.path) / "shipping.txt").unlink()

    result = ingest(config, embedding_function=embedding, embedding_name="test")

    assert result.pruned > 0
    collection = get_collection(open_client(config.index_path), config.collection, embedding)
    sources = {(m or {}).get("source") for m in collection.get()["metadatas"]}
    assert "shipping.txt" not in sources


def test_an_empty_corpus_is_refused(config, embedding, tmp_path):
    empty = tmp_path / "empty-corpus"
    empty.mkdir()
    config.corpus.path = str(empty)

    with pytest.raises(ValueError, match="No readable documents"):
        ingest(config, embedding_function=embedding, embedding_name="test")


def test_a_corpus_over_the_capacity_limit_warns_but_still_ingests(
    config, embedding, tmp_path, caplog
):
    """Warn by default: finding this mid-delivery starts a conversation about
    more room, it does not break the build."""
    config.capacity.max_documents = 1

    result = ingest(config, embedding_function=embedding, embedding_name="test")

    assert result.documents == 2
    assert "configured for 1 documents" in caplog.text


def test_capacity_can_be_enforced_hard(config, embedding):
    config.capacity.max_documents = 1
    config.capacity.warn_only = False

    with pytest.raises(ValueError, match="beyond the agreed capacity"):
        ingest(config, embedding_function=embedding, embedding_name="test")


def test_no_capacity_configured_means_no_limit(config, embedding):
    assert config.capacity.max_documents is None
    assert ingest(config, embedding_function=embedding, embedding_name="test").documents == 2
