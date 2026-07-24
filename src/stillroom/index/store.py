# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""The persistent document index, and the fingerprint that travels with it.

Idempotent by construction: chunk IDs are deterministic, so an ingest upserts
every chunk by ID and then **prunes** whatever IDs the current corpus no longer
produces. A deleted file cannot leave stale passages behind that the chatbot
would keep citing — which, for a client whose reason for buying is control over
their own documents, is a correctness bug with a compliance shape.

The corpus fingerprint is stored alongside the chunks, in a one-row metadata
collection. It has to be *persisted with the index* rather than recomputed at
startup, because recomputing it would mean re-reading and re-chunking the whole
corpus on every boot — and on a client machine the corpus directory may not even
be mounted at serving time. The index is the artifact; the fingerprint is part
of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction

from stillroom.ingest.chunking import Chunk

# MiniLM (and the hashing fallback) are both best compared with cosine distance.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}
# Chroma requires a non-empty embedding for every record, so the fingerprint row
# gets a throwaway document; it is never queried by similarity, only by ID.
_FINGERPRINT_ID = "corpus_fingerprint"


@dataclass(frozen=True)
class IngestResult:
    """What an ingest actually did — reported to the client in the runbook."""

    documents: int
    chunks: int
    pruned: int
    fingerprint: str
    skipped: list[str]


def open_client(index_path: str) -> ClientAPI:
    return chromadb.PersistentClient(path=index_path)


def get_collection(
    client: ClientAPI, name: str, embedding_function: EmbeddingFunction
) -> Collection:
    return client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function,
        metadata=_COLLECTION_METADATA,
    )


def _meta_collection(client: ClientAPI, name: str, embedding_function: EmbeddingFunction) -> Collection:
    return client.get_or_create_collection(
        name=f"{name}__meta",
        embedding_function=embedding_function,
        metadata=_COLLECTION_METADATA,
    )


def write_fingerprint(
    client: ClientAPI, name: str, embedding_function: EmbeddingFunction, fingerprint: str
) -> None:
    """Persist the corpus fingerprint next to the index it describes."""
    _meta_collection(client, name, embedding_function).upsert(
        ids=[_FINGERPRINT_ID],
        documents=["corpus fingerprint"],
        metadatas=[{"fingerprint": fingerprint}],
    )


def read_fingerprint(
    client: ClientAPI, name: str, embedding_function: EmbeddingFunction
) -> str | None:
    """The fingerprint of the indexed corpus, or None if never ingested."""
    result = _meta_collection(client, name, embedding_function).get(ids=[_FINGERPRINT_ID])
    metadatas = result.get("metadatas") or []
    if not metadatas:
        return None
    return (metadatas[0] or {}).get("fingerprint")


def index_chunks(
    client: ClientAPI,
    name: str,
    embedding_function: EmbeddingFunction,
    chunks: list[Chunk],
) -> tuple[Collection, int]:
    """Upsert `chunks` and prune anything the corpus no longer produces.

    Returns the collection and the number of pruned chunks.
    """
    if not chunks:
        raise ValueError("Refusing to index an empty corpus — nothing would be answerable.")

    collection = get_collection(client, name, embedding_function)
    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"source": c.source, "heading": c.heading or "", "index": c.index} for c in chunks
        ],
    )

    current = {c.id for c in chunks}
    existing = set(collection.get(include=[])["ids"])
    stale = existing - current
    if stale:
        collection.delete(ids=list(stale))

    return collection, len(stale)
