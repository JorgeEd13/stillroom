# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""Ingest orchestration: corpus directory -> chunked, fingerprinted index.

The one entry point a build calls, and the one the documented
re-ingest script wraps so the client can add documents themselves later.

This lives above `ingest/` and `index/` rather than inside either, because it is
the only module that needs both. Keeping it separate is what stops `index.store`
(which types against `Chunk`) and `ingest` (which feeds the store) from
importing each other.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from stillroom.answers.cache import answer_key, get_answer_cache
from stillroom.config import ClientConfig
from stillroom.index.embeddings import embedding_function_for
from stillroom.index.store import IngestResult, index_chunks, open_client, write_fingerprint
from stillroom.ingest.chunking import chunk_corpus
from stillroom.ingest.fingerprint import corpus_fingerprint
from stillroom.ingest.loaders import load_corpus
from stillroom.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)


class CorpusSnapshot(NamedTuple):
    """What the corpus on disk looks like right now, cheaply."""

    fingerprint: str
    documents: int


def ingest(
    config: ClientConfig,
    *,
    embedding_function=None,
    # ⚠️ **Defaults to the CONFIGURED embedder, not the shipped one.** It goes
    # into the corpus fingerprint, so a client who pins a different model and
    # gets `DEFAULT_EMBEDDING_NAME` written here has an index whose fingerprint
    # cannot tell the two apart — every cached answer from the old model would
    # survive the swap and be served as the new one's.
    embedding_name: str | None = None,
    model_name: str = "unknown",
) -> IngestResult:
    """Read, chunk, index and fingerprint the client's corpus.

    Safe to run repeatedly — the index converges on the corpus rather than
    growing. **Purges the answer cache of anything the new corpus invalidates**,
    which is the immutable-corpus boundary being enforced at the one moment it matters:
    a cached answer that survived a document change would be a confident lie.
    """
    embed = embedding_function or embedding_function_for(config)
    embedding_name = embedding_name or config.embedding.model

    documents, skipped = load_corpus(
        config.corpus.path, config.corpus.include, config.corpus.encoding
    )
    if not documents:
        raise ValueError(
            f"No readable documents found under {config.corpus.path!r} "
            f"(looked for {', '.join(config.corpus.include)})."
        )

    cap = config.capacity.max_documents
    if cap is not None and len(documents) > cap:
        message = (
            f"This build is configured for {cap} documents and the corpus has "
            f"{len(documents)}. The extra {len(documents) - cap} are beyond the "
            "agreed capacity."
        )
        if config.capacity.warn_only:
            # Loud, but it does not break a delivery mid-flight. It exists so the
            # agreed limit is observed rather than assumed.
            logger.warning("%s Ingesting anyway (warn_only).", message)
        else:
            raise ValueError(message)

    chunks = chunk_corpus(
        documents,
        chunk_chars=config.corpus.chunk_chars,
        chunk_overlap=config.corpus.chunk_overlap,
    )
    fingerprint = corpus_fingerprint(
        chunks,
        chunk_chars=config.corpus.chunk_chars,
        chunk_overlap=config.corpus.chunk_overlap,
        embedding_name=embedding_name,
    )

    client = open_client(config.index_path)
    _, pruned = index_chunks(client, config.collection, embed, chunks)
    write_fingerprint(client, config.collection, embed, fingerprint)

    if config.answer_cache.enabled:
        cache = get_answer_cache(
            client,
            config.answer_cache.collection,
            embed,
            threshold=config.answer_cache.threshold,
            fingerprint=fingerprint,
            key=answer_key(
                model_name=model_name,
                prompt_version=PROMPT_VERSION,
                k=config.retrieval.k,
            ),
        )
        dropped = cache.purge_stale()
        if dropped:
            logger.info("purged %d cached answers invalidated by this ingest", dropped)

    return IngestResult(
        documents=len(documents),
        chunks=len(chunks),
        pruned=pruned,
        fingerprint=fingerprint,
        skipped=skipped,
    )


def corpus_snapshot(
    config: ClientConfig, *, embedding_name: str | None = None
) -> CorpusSnapshot:
    """What the documents on disk WOULD produce, without embedding them.

    Reading and chunking a corpus is cheap; embedding it is not. Separating the
    two is what lets everything upstream ask *"has anything actually changed?"*
    before paying for an answer.

    ⚠️ **`refresh.py` states this rule and did not implement it.** Its own
    docstring: *"Never re-embed an unchanged corpus. The fingerprint already
    tells us whether anything changed… on a CPU-only box, re-embedding a
    1,500-document corpus every hour would eat the machine the assistant is
    supposed to be answering from."* The code ran a full `ingest()` first and
    compared the fingerprints afterwards — so a scheduled refresh re-embedded the
    entire corpus on every cycle, hourly, forever (leg B #22).
    """
    embedding_name = embedding_name or config.embedding.model
    documents, _ = load_corpus(
        config.corpus.path, config.corpus.include, config.corpus.encoding
    )
    chunks = chunk_corpus(
        documents,
        chunk_chars=config.corpus.chunk_chars,
        chunk_overlap=config.corpus.chunk_overlap,
    )
    return CorpusSnapshot(
        fingerprint=corpus_fingerprint(
            chunks,
            chunk_chars=config.corpus.chunk_chars,
            chunk_overlap=config.corpus.chunk_overlap,
            embedding_name=embedding_name,
        ),
        documents=len(documents),
    )


def indexed_fingerprint(config: ClientConfig) -> str | None:
    """What the index on disk currently holds, or None if never ingested."""
    from stillroom.index.store import read_fingerprint

    try:
        return read_fingerprint(
            open_client(config.index_path),
            config.collection,
            embedding_function_for(config),
        )
    except Exception:
        logger.warning("could not read the indexed fingerprint", exc_info=True)
        return None


def corpus_is_indexed(config: ClientConfig) -> bool:
    """Do the documents on disk match what was last indexed?"""
    indexed = indexed_fingerprint(config)
    return indexed is not None and indexed == corpus_snapshot(config).fingerprint
