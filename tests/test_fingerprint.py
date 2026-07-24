"""The corpus fingerprint — the thing the whole answer cache rests on.

The immutable-corpus rule allows caching a whole answer *only* because a corpus is immutable
between ingests. These tests pin the properties that make that argument true;
if one of them breaks, the cache starts serving confident lies.
"""

from __future__ import annotations

from stillroom.ingest.chunking import chunk_corpus
from stillroom.ingest.fingerprint import corpus_fingerprint
from stillroom.ingest.loaders import RawDocument

PARAMS = {"chunk_chars": 400, "chunk_overlap": 50}


def _fingerprint(documents: list[RawDocument], **overrides) -> str:
    params = {**PARAMS, **overrides}
    chunks = chunk_corpus(documents, **params)
    return corpus_fingerprint(chunks, **params, embedding_name="test")


def _docs(body: str = "Body A.") -> list[RawDocument]:
    return [
        RawDocument("a.md", f"## Alpha\n\n{body}\n"),
        RawDocument("b.md", "## Beta\n\nBody B.\n"),
    ]


def test_the_same_corpus_gives_the_same_fingerprint():
    assert _fingerprint(_docs()) == _fingerprint(_docs())


def test_changing_a_document_changes_the_fingerprint():
    assert _fingerprint(_docs()) != _fingerprint(_docs("Body A, revised."))


def test_document_order_does_not_change_the_fingerprint():
    """Filesystem walk order must not invalidate a client's whole cache."""
    forward = _docs()
    assert _fingerprint(forward) == _fingerprint(list(reversed(forward)))


def test_rechunking_changes_the_fingerprint():
    """Same bytes, different passages — the model would be shown other text."""
    assert _fingerprint(_docs()) != _fingerprint(_docs(), chunk_chars=800)


def test_changing_the_embedder_changes_the_fingerprint():
    """A new embedder changes what 'similar' means; old answers were computed
    under the old geometry."""
    chunks = chunk_corpus(_docs(), **PARAMS)
    a = corpus_fingerprint(chunks, **PARAMS, embedding_name="minilm")
    b = corpus_fingerprint(chunks, **PARAMS, embedding_name="other")

    assert a != b


def test_removing_a_document_changes_the_fingerprint():
    assert _fingerprint(_docs()) != _fingerprint(_docs()[:1])
