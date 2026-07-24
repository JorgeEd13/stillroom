# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""Curated instant answers, and the rule that keeps them honest.

A local model on ordinary hardware is slow — that is the honest cost of keeping
the documents in the building — so the most-asked questions get answered in
about a second with **no model
call at all**.

`receivables-agent` caches the *plan* and never the answer, because its ledger
is mutable and a frozen number lies. This engine caches the whole answer, and
that is the same rule reaching a different conclusion rather than a relaxation
of it: a document corpus is immutable between ingests, so a cached answer is
exact until the documents change. **Everything therefore rests on invalidation.**

Two guards, and both are here because either one alone is insufficient:

1. **The corpus fingerprint.** A cached answer stores the fingerprint of the
   corpus it was computed over. A re-ingest produces a new fingerprint and every
   entry keyed to the old one is dead — not "probably stale", dead, checked on
   every read. This is the immutable-corpus boundary in code.

2. **The answer key.** The fingerprint covers the *documents*; it does not cover
   the model, the prompt, or `k`. Change the model and every cached answer was
   written by a model that is no longer installed. So entries also carry an
   answer key over those, and a mismatch is a miss.

A stale entry is **deleted on read**, not merely skipped. Skipping leaves a
growing pile of entries that look like cache content and are not, and the next
person to read this code will reasonably assume a `count()` means something.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction

_CACHE_METADATA = {"hnsw:space": "cosine"}


# Words that flip a question's meaning while barely moving its embedding.
# English and Portuguese, because those are the languages this ships in; a
# corpus in another one needs this list extended, and the honest consequence of
# not extending it is a cache that can answer the opposite question.
_NEGATIONS = frozenset(
    {
        "not", "no", "never", "without", "except", "excluding", "unless",
        "isn't", "aren't", "doesn't", "don't", "won't", "cannot", "can't",
        "não", "nao", "nunca", "sem", "exceto", "salvo", "nenhum", "nenhuma",
    }
)


def _polarity_differs(question: str, cached: str) -> bool:
    """Do these two questions disagree about negation?

    A deliberately crude test — presence, not parsing. It is a **guard on a
    shortcut**, not a semantic analyser: the cost of being wrong in one
    direction is a cache miss and a few extra seconds, and in the other it is an
    instant, confident, inverted answer. Crude and asymmetric beats clever here.
    """
    words = set(re.findall(r"[\w']+", question.lower()))
    cached_words = set(re.findall(r"[\w']+", cached.lower()))
    return bool(words & _NEGATIONS) != bool(cached_words & _NEGATIONS)


def answer_key(
    *,
    model_name: str,
    prompt_version: str,
    k: int,
    standing_context: str | None = None,
) -> str:
    """Everything outside the corpus that would change an answer.

    `standing_context` is hashed in rather than stored: it is the client's own
    background notes, it changes answers without changing a single
    document, and the corpus fingerprint cannot see it.
    """
    raw = json.dumps(
        {
            "model": model_name,
            "prompt": prompt_version,
            "k": k,
            "standing": hashlib.sha256(
                (standing_context or "").encode("utf-8")
            ).hexdigest()[:16],
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CachedAnswer:
    """An answer served without a model call."""

    question: str
    answer: str
    citations: list[dict]
    # True for a question the client listed and we baked at build time; False
    # for one the engine learned by answering it once. The distinction is shown
    # in the UI, because "one of your curated questions" and "something I
    # happened to answer before" earn different amounts of trust.
    curated: bool


class AnswerCache:
    """Semantic question -> answer cache over a ChromaDB collection.

    Lookup is cosine similarity on the question embedding, with a conservative
    threshold: a miss costs one model call, a false hit answers a question the
    client did not ask and looks confident doing it.
    """

    def __init__(
        self,
        collection: Collection,
        *,
        threshold: float,
        fingerprint: str,
        key: str,
    ) -> None:
        self._collection = collection
        self._threshold = threshold
        self._fingerprint = fingerprint
        self._key = key

    def lookup(self, question: str) -> CachedAnswer | None:
        """Return a valid cached answer, or None.

        Deletes the entry when it is a semantic hit but its corpus or answer key
        has moved on — the documents changed, so that answer is no longer true.
        """
        if self._collection.count() == 0:
            return None

        result = self._collection.query(query_texts=[question], n_results=1)
        ids = result["ids"][0]
        if not ids:
            return None

        distance = (result.get("distances") or [[None]])[0][0]
        if distance is None or distance > (1.0 - self._threshold):
            return None

        cached_question = result["documents"][0][0]
        if _polarity_differs(question, cached_question):
            # ⚠️ **Embeddings do not encode negation**. "What is the
            # refund window?" and "What is NOT the refund window?" sit on top of
            # each other in the vector space — measured hitting this cache at a
            # 0.90 threshold and answering the second with the first's answer:
            # *"The refund window is 30 days"*, served instantly, with no model
            # call and no way for the client to see it happened.
            #
            # A confidently wrong cached answer is the worst output this product
            # can produce, because the cache is the one path with no model in it
            # to hedge. A missed cache hit costs seconds; this costs the claim.
            return None

        metadata = result["metadatas"][0][0] or {}
        if (
            metadata.get("fingerprint") != self._fingerprint
            or metadata.get("key") != self._key
        ):
            # Stale: the corpus or the model moved. Remove it rather than leave
            # a dead entry that still counts towards the collection size.
            self._collection.delete(ids=[ids[0]])
            return None

        answer = metadata.get("answer")
        if not answer:
            return None

        return CachedAnswer(
            question=result["documents"][0][0],
            answer=answer,
            citations=json.loads(metadata.get("citations") or "[]"),
            curated=bool(metadata.get("curated")),
        )

    def warm(
        self, question: str, answer: str, citations: list[dict], *, curated: bool = False
    ) -> None:
        """Store an answer against the current corpus and answer key.

        Idempotent by question text: the ID is the question, so re-baking the
        same curated question updates in place rather than duplicating.
        """
        self._collection.upsert(
            ids=[question],
            documents=[question],
            metadatas=[
                {
                    "answer": answer,
                    "citations": json.dumps(citations),
                    "fingerprint": self._fingerprint,
                    "key": self._key,
                    "curated": curated,
                }
            ],
        )

    def purge_stale(self) -> int:
        """Drop every entry that no longer matches the corpus and answer key.

        Called after a re-ingest. `lookup` already refuses stale entries, so this
        is housekeeping rather than correctness — but it means the count the
        client sees in the runbook is the number of questions that will actually
        come back instantly, which is the number that was agreed.
        """
        existing = self._collection.get()
        ids = existing.get("ids") or []
        metadatas = existing.get("metadatas") or []

        stale = [
            entry_id
            for entry_id, metadata in zip(ids, metadatas)
            if (metadata or {}).get("fingerprint") != self._fingerprint
            or (metadata or {}).get("key") != self._key
        ]
        if stale:
            self._collection.delete(ids=stale)
        return len(stale)

    def count(self) -> int:
        return self._collection.count()


def get_answer_cache(
    client: chromadb.api.ClientAPI,
    collection_name: str,
    embedding_function: EmbeddingFunction,
    *,
    threshold: float,
    fingerprint: str,
    key: str,
) -> AnswerCache:
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata=_CACHE_METADATA,
    )
    return AnswerCache(
        collection, threshold=threshold, fingerprint=fingerprint, key=key
    )
