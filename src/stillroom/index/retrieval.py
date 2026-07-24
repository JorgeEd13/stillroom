# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""Retrieval, and the citations that come back with it.

Every answer this engine produces cites the source it came from — that is in the
product's central promise, not a feature. The citation is assembled here
rather than parsed out of the model's reply on purpose: a model asked to cite
its sources will sometimes cite a plausible-sounding document that was never
retrieved, and the client cannot tell the difference. Citations built from the
retrieval result cannot be hallucinated, because they are not generated.

The relevance floor is the other half. Chroma always returns `k` nearest
neighbours, even when the nearest one is unrelated — ask a handbook corpus about
the weather and you get five passages back with terrible scores. Handing those
to a model and hoping it declines is how "it made something up" reviews happen.
`Retrieval.grounded` reports whether anything cleared the floor, and the engine
refuses to call the model at all when nothing did.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass

from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+", re.UNICODE)


# ⚠️ **THE COMPENSATING STAGE THAT WAS MEASURED AND REJECTED — read this before
# adding one**. The embedder swap predicted that a *relative* margin against the
# best hit would catch what no absolute floor can, and the intuition is good:
# a question the documents really answer should stand out from the field.
#
# It is backwards. Measured over 72 question/corpus pairs on a realistic corpus:
#
#     signal                  lowest on-topic   highest far   highest nonsense
#     top (absolute)                    0.569         0.428              0.698
#     top - runner-up                   0.005         0.033              0.094
#     top - mean(retrieved k)           0.055         0.039              0.157
#     top - mean(whole corpus)          0.165         0.106              0.256
#     z-score of top                    2.475         2.609              3.252
#
# Every relative signal scores lexical **nonsense higher than the weakest real
# question**, and two of them score the *far* band higher too. The reason is
# structural rather than incidental: a genuine question about refunds matches
# several refund passages, so its top hit stands out **less**; nonsense about
# refunds spikes on exactly one. A margin gate would therefore refuse real
# questions and admit the band it was built for.
#
# **So the first gate is an absolute floor and nothing else, and the nonsense
# band is the model's grounding refusal alone.** That is the name-the-language finding
# unchanged and now measured under the new embedder, and it has to be said out
# loud rather than implied: *"How many moons does a refund have?"* reaches the
# model, and if the model answers it, nothing upstream stopped it.


class CorpusSpeller:
    """One spelling retry against the client's own indexed vocabulary.

    Never rewrites a question — `engine` searches with the original, and only
    tries this when the floor has already refused. See `SpellingRetryConfig`
    for the measurement and for why a rewrite was the wrong shape.
    """

    def __init__(self, documents: list[str], *, min_token_length: int = 5, cutoff: float = 0.80):
        self._min = min_token_length
        self._cutoff = cutoff
        self._vocabulary = {
            word
            for document in documents
            for word in _WORD.findall(document.lower())
            if len(word) >= 4
        }

    def correction(self, question: str) -> str | None:
        """A respelled question, or None when nothing was worth changing."""
        if not self._vocabulary:
            return None
        tokens, changed = [], False
        for token in _WORD.findall(question.lower()):
            if len(token) < self._min or token in self._vocabulary:
                tokens.append(token)
                continue
            near = difflib.get_close_matches(
                token, self._vocabulary, n=1, cutoff=self._cutoff
            )
            if near and near[0] != token:
                tokens.append(near[0])
                changed = True
            else:
                tokens.append(token)
        return " ".join(tokens) if changed else None


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, with what a client needs to check it."""

    source: str
    heading: str | None
    text: str
    similarity: float

    def label(self) -> str:
        """How this passage is named in a citation."""
        return f"{self.source} — {self.heading}" if self.heading else self.source


@dataclass(frozen=True)
class Retrieval:
    """The result of one search.

    ⚠️ **`passages` is everything Chroma returned; `relevant` is what the floor
    admitted, and only the second is ever used**. The distinction was
    missing and it shipped: the floor decided *whether to call the model* and
    then every retrieved chunk was handed to it anyway, and listed to the client
    as a source.

    Observed in a delivered container asking about international shipping: one
    passage at **0.82** answered it, and the reply came back with four more
    "sources" underneath at 0.15, -0.004, -0.014 and -0.016. Negative cosine
    similarity is not a weak match, it is an unrelated one — and the product
    being built is the one that does not improvise. Showing four unrelated
    documents beneath a correct answer undermines exactly the thing the client
    is paying for, and the module docstring above already made the argument
    about the model's context without it being applied.

    The raw list is kept rather than discarded because a *near miss* is the
    useful thing to see when tuning a client's floor — the number that explains
    a refusal is the highest score that failed.
    """

    passages: tuple[Passage, ...]
    # True when at least one passage cleared the relevance floor. False means
    # the corpus has nothing to say about this question.
    grounded: bool
    # The floor these passages were scored against, kept so `relevant` can be
    # derived here instead of at each call site — a filter re-implemented in
    # three places is a filter that will be forgotten in one of them.
    min_similarity: float = 0.0

    @property
    def relevant(self) -> tuple[Passage, ...]:
        """The passages that actually cleared the floor.

        This is what the model is shown and what the client is shown. Nothing
        below the floor is either grounds for an answer or evidence for one.
        """
        return tuple(p for p in self.passages if p.similarity >= self.min_similarity)

    def citations(self) -> list[dict[str, str | float]]:
        """Citations as plain data, for the API and the UI."""
        return [
            {
                "source": p.source,
                "heading": p.heading or "",
                "snippet": p.text[:280],
                "similarity": round(p.similarity, 4),
            }
            for p in self.relevant
        ]


def search(
    collection: Collection, question: str, *, k: int, min_similarity: float
) -> Retrieval:
    """Retrieve the `k` nearest passages and mark whether any are relevant."""
    if collection.count() == 0:
        return Retrieval(passages=(), grounded=False)

    result = collection.query(query_texts=[question], n_results=k)
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    # Chroma returns cosine *distance*; similarity is 1 - distance.
    distances = (result.get("distances") or [[None] * len(documents)])[0]

    # ⚠️ **A passage with no text is not a passage, and dropping it here is the
    # difference between a stale answer and a 500** (leg B #17). Chroma can
    # return a hit whose document is `None` when another process deleted the
    # chunk between the query and the read — which is exactly what the Standard
    # documented re-ingest does, in a second container, against a running
    # assistant. Nothing rejected it, so `None` travelled all the way to
    # `prompts._strip_markers` and every question 500'd until a restart.
    #
    # Dropped rather than raised: a corpus being rewritten underneath us is a
    # transient the client can survive, and the honest outcome is an answer from
    # fewer passages — or the ordinary refusal, if none are left.
    kept = [
        (document, meta, distance)
        for document, meta, distance in zip(documents, metadatas, distances)
        if isinstance(document, str) and document.strip()
    ]
    if len(kept) != len(documents):
        logger.warning(
            "%d of %d retrieved passages had no text and were dropped; the index "
            "is probably being re-ingested by another process",
            len(documents) - len(kept),
            len(documents),
        )

    passages = tuple(
        Passage(
            source=(meta or {}).get("source", "unknown"),
            heading=(meta or {}).get("heading") or None,
            text=document,
            similarity=1.0 - distance if distance is not None else 0.0,
        )
        for document, meta, distance in kept
    )

    grounded = any(p.similarity >= min_similarity for p in passages)
    return Retrieval(
        passages=passages, grounded=grounded, min_similarity=min_similarity
    )
