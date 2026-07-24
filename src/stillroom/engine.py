# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""The engine: retrieve, then answer. Deliberately not a ReAct loop.

`receivables-agent` runs a LangGraph ReAct agent, and it should: it has *two*
tools — guarded SQL over a ledger and RAG over a policy — and choosing between
them (or using both) is a real decision that only a model can make.

**This engine has one source of truth**, so there is no routing decision left to
make, and carrying the loop over would inherit its costs without its
justification. Those costs are not theoretical on a
small local model:

* **Latency, multiplied.** A ReAct turn is at minimum two model calls — decide
  to search, then answer. On client hardware that is the difference between
  eight seconds and sixteen, on the slowest path the client has.
* **Small models over-call and loop.** The one failure `receivables-agent`
  needed a step cap, a wall-clock budget and a dedup tracker to contain. None of
  that machinery is needed if the retrieval is not the model's decision.
* **Cacheability.** A fixed pipeline has one deterministic shape, so the whole
  answer can be cached against a corpus fingerprint. A loop's shape
  varies per run.

What is kept from the earlier project is the part users see: streaming, so the system
narrates rather than appearing frozen, and grounded citations on every answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from stillroom.answers.cache import AnswerCache, answer_key, get_answer_cache
from stillroom.config import ClientConfig
from stillroom.conversation import Turn, retrieval_query, trim
from stillroom.index.embeddings import embedding_function_for
from stillroom.index.retrieval import CorpusSpeller, Retrieval, search
from stillroom.index.store import get_collection, open_client, read_fingerprint
from stillroom.ingest.loaders import read_plain_text
from stillroom.prompts import (
    PROMPT_VERSION,
    build_user_prompt,
    no_answer,
    system_prompt,
)
from stillroom.provider import build_chat_model, model_label

logger = logging.getLogger(__name__)

# Any letter or digit, in any script. See `Engine.retrieve` for the measured
# regression this guards.
_HAS_WORD_CHARACTER = re.compile(r"\w", re.UNICODE)


def _read_standing_context(config: ClientConfig) -> str | None:
    """Load the client's background notes, or None.

    A missing file is a hard error rather than a shrug: the path is only in the
    config because somebody put it there, and a build that silently drops the
    client's source of truth answers plausibly and slightly wrongly forever.
    """
    if not config.context.path:
        return None

    path = Path(config.context.path)
    if not path.is_file():
        raise FileNotFoundError(
            f"context.path is set to {config.context.path!r} but there is no file "
            "there. Remove the setting, or ship the notes with the build."
        )

    text = read_plain_text(path)[: config.context.max_chars]
    return text.strip() or None


class NotIngested(RuntimeError):
    """Raised when the index has no corpus fingerprint — nothing was ingested."""


@dataclass
class Answer:
    """One answer, and everything needed to check it."""

    text: str
    citations: list[dict] = field(default_factory=list)
    # "cache" = served instantly from a curated or learned answer; "model" = a
    # live model call; "no-match" = nothing relevant retrieved, no model call.
    served_by: str = "model"
    curated: bool = False


class Engine:
    """A configured engine over one client's index.

    Built once and reused: opening the index and constructing the model are
    startup costs, not per-question costs.
    """

    def __init__(
        self,
        config: ClientConfig,
        *,
        chat_model: BaseChatModel | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        self.config = config
        embed = embedding_function or embedding_function_for(config)

        client = open_client(config.index_path)
        fingerprint = read_fingerprint(client, config.collection, embed)
        if fingerprint is None:
            raise NotIngested(
                f"No corpus indexed at {config.index_path!r}. Run an ingest first."
            )

        self._fingerprint = fingerprint
        # Kept so the live fingerprint can be re-read later. A re-ingest by
        # ANOTHER process moves the index under this one, and nothing here
        # would otherwise notice (leg B #17).
        self._client = client
        self._embed = embed
        self._collection = get_collection(client, config.collection, embed)
        self._speller_cache: CorpusSpeller | None = None
        self._standing = _read_standing_context(config)
        # Constructing the model resolves `auto` against the hardware, so it
        # happens once here rather than on every question.
        self._model = chat_model if chat_model is not None else build_chat_model(config)
        self._model_name = (
            model_label(config) if chat_model is None else "injected"
        )

        self._cache: AnswerCache | None = None
        if config.answer_cache.enabled:
            self._cache = get_answer_cache(
                client,
                config.answer_cache.collection,
                embed,
                threshold=config.answer_cache.threshold,
                fingerprint=fingerprint,
                key=answer_key(
                    model_name=self._model_name,
                    prompt_version=PROMPT_VERSION,
                    k=config.retrieval.k,
                    # The standing context changes answers without changing a
                    # single document, so it belongs in the key exactly as the
                    # model and the prompt do. Editing it and leaving
                    # the cache is the immutable-corpus failure with a different cause.
                    standing_context=self._standing,
                ),
            )

    @property
    def fingerprint(self) -> str:
        """The corpus this process is SERVING — fixed for its lifetime.

        It has to be fixed: the answer cache is keyed on it, so a value that
        moved under a running engine would serve answers written against a
        corpus that no longer exists.
        """
        return self._fingerprint

    @property
    def indexed_fingerprint(self) -> str | None:
        """The corpus the index on disk currently HOLDS. Re-read every call.

        ⚠️ **These two can differ, and when they do the client is being served
        documents they think they replaced** (leg B #17). Two ways it happens:

        * the documented re-ingest, which the runbook has the
          client run in a **second container** against a running assistant;
        * the scheduled refresh, which re-ingests in *this*
          process — correctly, and still leaves `fingerprint` behind, because
          it was captured in `__init__`.

        The second is the quiet one: no crash, no restart, and `/api/health`
        reporting a corpus that is several refreshes old. Reading it fresh is a
        single metadata `get`, which is why this is a property and not a cached
        value.
        """
        try:
            return read_fingerprint(self._client, self.config.collection, self._embed)
        except Exception:  # an index that vanished mid-flight is not fatal here
            logger.warning("could not re-read the corpus fingerprint", exc_info=True)
            return None

    @property
    def serving_current_corpus(self) -> bool | None:
        """Is what we serve what the index holds? None when it cannot be read."""
        indexed = self.indexed_fingerprint
        if indexed is None:
            return None
        return indexed == self._fingerprint

    @property
    def cache(self) -> AnswerCache | None:
        return self._cache

    def retrieve(self, question: str, history: tuple[Turn, ...] = ()) -> Retrieval:
        """Search, and on a refusal try the question's spelling once.

        ⚠️ **The order matters and it is the whole safety argument.** The
        original question is always searched first, and the retry only ever
        runs when that already failed the floor. So the retry cannot lower the
        gate for anything — it can only rescue a question the gate had already
        refused, and the better of the two results wins. A correction that
        makes things worse is discarded by construction, which is what makes it
        safe to correct against a vocabulary that contains no question words.
        """
        # ⚠️ **A REGRESSION THE SWAP CAUSED, found by re-running the troll
        # battery**. That battery recorded *"degenerate input
        # short-circuited by the floor with no model call"*. Under `bge-m3` it
        # is not: measured against the English corpus, `"???"` scores **0.518**
        # and a question of pure whitespace scores **0.588** — both clear of a
        # 0.50 floor, so both reach the model.
        #
        # That is the compressed range in its purest form: with nothing to be
        # about, the vector lands near the corpus centroid, and under this
        # embedder the corpus centroid is *high*. No floor fixes it, because the
        # score is not low.
        #
        # Refused here rather than in `search` so it costs neither the embedding
        # call nor the model call, which is the property that battery recorded.
        # A word character is the test, so it holds in every script — Han, Hangul
        # and Cyrillic are all `\\w`.
        if not _HAS_WORD_CHARACTER.search(question):
            return Retrieval(passages=(), grounded=False)

        query = retrieval_query(question, history)
        retrieval = search(
            self._collection,
            query,
            k=self.config.retrieval.k,
            min_similarity=self.config.retrieval.min_similarity,
        )
        if retrieval.grounded or not self.config.retrieval.spelling_retry.enabled:
            return retrieval

        respelled = self._speller.correction(query)
        if respelled is None:
            return retrieval

        retry = search(
            self._collection,
            respelled,
            k=self.config.retrieval.k,
            min_similarity=self.config.retrieval.min_similarity,
        )
        if not retry.grounded:
            return retrieval

        logger.info(
            "nothing cleared the floor for %r; the corpus vocabulary respelled "
            "it as %r and that did",
            query,
            respelled,
        )
        return retry

    @property
    def _speller(self) -> CorpusSpeller:
        """Built once per engine, from the indexed text it will correct against.

        Lazy because the happy path never needs it: a build whose questions all
        clear the floor never reads the corpus back out of Chroma.
        """
        if self._speller_cache is None:
            settings = self.config.retrieval.spelling_retry
            documents = self._collection.get(include=["documents"])["documents"] or []
            self._speller_cache = CorpusSpeller(
                [d for d in documents if isinstance(d, str)],
                min_token_length=settings.min_token_length,
                cutoff=settings.cutoff,
            )
        return self._speller_cache

    def _history(self, history: tuple[Turn, ...]) -> tuple[Turn, ...]:
        """Whatever the caller sent, reduced to what this build will carry."""
        conversation = self.config.conversation
        if not conversation.enabled:
            return ()
        return trim(
            history,
            max_turns=conversation.max_turns,
            max_chars=conversation.max_chars_per_turn,
        )

    def _messages(
        self, question: str, retrieval: Retrieval, history: tuple[Turn, ...] = ()
    ) -> list:
        # `relevant`, never `passages`: a chunk below the floor is unrelated
        # text, and putting it in the prompt is how a model is talked into
        # improvising.
        passages = [(p.label(), p.text) for p in retrieval.relevant]
        return [
            SystemMessage(content=system_prompt(self.config.language)),
            HumanMessage(
                content=build_user_prompt(
                    question, passages, history, self._standing
                )
            ),
        ]

    def ask(
        self,
        question: str,
        *,
        use_cache: bool = True,
        history: tuple[Turn, ...] = (),
    ) -> Answer:
        """Answer a question. The synchronous path, used by the CLI and tests."""
        history = self._history(history)

        # ⚠️ **A conversation turn is never cached, in either direction**
        #. The cache is keyed on the question text alone, and a
        # follow-up's text does not identify it: "how long does it take?" means
        # one thing after a question about refunds and another after one about
        # shipping. Reading the cache would serve the wrong conversation's
        # answer; writing to it would store that ambiguity for everybody else,
        # in a cache that is shared and outlives the session.
        #
        # The instant answers are unaffected: a curated question
        # is asked fresh, which is exactly the case this still covers.
        if history:
            use_cache = False

        if use_cache and self._cache is not None:
            hit = self._cache.lookup(question)
            if hit is not None:
                return Answer(
                    text=hit.answer,
                    citations=hit.citations,
                    served_by="cache",
                    curated=hit.curated,
                )

        retrieval = self.retrieve(question, history)
        if not retrieval.grounded:
            # No model call at all: nothing retrieved cleared the relevance
            # floor, so there is nothing for a model to ground an answer in.
            return Answer(
                text=no_answer(self.config.language),
                citations=[],
                served_by="no-match",
            )

        reply = self._model.invoke(self._messages(question, retrieval, history))
        citations = retrieval.citations()
        text = enforce_citations(_text_of(reply), citations)

        if not text:
            # The model produced nothing but citation markers. Refusing is
            # honest; rendering "[1]" as an answer is not. Deliberately NOT
            # cached — a failed generation must not be served again.
            logger.warning("the model returned no answer text; refusing instead")
            return Answer(
                text=no_answer(self.config.language), citations=[], served_by="no-match"
            )

        if use_cache and self._cache is not None:
            self._cache.warm(question, text, citations)

        return Answer(text=text, citations=citations, served_by="model")

    async def astream(
        self, question: str, history: tuple[Turn, ...] = ()
    ) -> AsyncIterator[dict]:
        """Stream one answer as events, so the UI can show it working.

        A local model is slow, and a silent wait reads as a hang. The event
        stream is what turns that wait into visible progress — the same contract
        the earlier project's UI consumes: `cached` / `sources` / `token` / `answer` /
        `error`, then the caller closes the stream.
        """
        history = self._history(history)
        try:
            # Same rule as `ask`, and for the same reason.
            if not history and self._cache is not None:
                hit = self._cache.lookup(question)
                if hit is not None:
                    yield {
                        "type": "cached",
                        "curated": hit.curated,
                        "reply": hit.answer,
                        "citations": hit.citations,
                    }
                    return

            retrieval = self.retrieve(question, history)
            if not retrieval.grounded:
                yield {
                    "type": "answer",
                    "reply": no_answer(self.config.language),
                    "citations": [],
                }
                return

            citations = retrieval.citations()
            yield {"type": "sources", "citations": citations}

            parts: list[str] = []
            async for piece in self._model.astream(
                self._messages(question, retrieval, history)
            ):
                token = _text_of(piece)
                if token:
                    parts.append(token)
                    yield {"type": "token", "text": token}

            # The tokens went out as they arrived; the final `answer` event is
            # what the UI renders, and it is the one that must be clean. Same
            # post-condition as `ask`, applied at the one point both paths have.
            text = enforce_citations("".join(parts), citations)
            if not text:
                logger.warning("the model streamed no answer text; refusing instead")
                yield {
                    "type": "answer",
                    "reply": no_answer(self.config.language),
                    "citations": [],
                }
                return

            if not history and self._cache is not None:
                self._cache.warm(question, text, citations)

            yield {"type": "answer", "reply": text, "citations": citations}
        except Exception as exc:  # never leave the stream half-open
            logger.exception("streaming answer failed")
            yield {"type": "error", "message": str(exc)}

    def bake_curated(self, questions: tuple[str, ...] | None = None) -> list[tuple[str, bool]]:
        """Pre-compute the client's curated questions into the cache.

        Run at build time, after ingest, so the questions the client actually
        asks come back in about a second from the very first request — not after
        a warm-up window they would experience as "it was slow at first".

        Returns `(question, grounded)` per question. **A question that retrieves
        nothing is reported, not silently baked**: it means their corpus does not
        answer something they told us they ask, and that is a finding to raise
        with them before handover rather than a `NO_ANSWER` they discover later.
        """
        if self._cache is None:
            raise RuntimeError("Answer cache is disabled; nothing to bake.")

        results: list[tuple[str, bool]] = []
        for question in questions or self.config.answer_cache.curated:
            retrieval = self.retrieve(question)
            if not retrieval.grounded:
                results.append((question, False))
                continue

            reply = self._model.invoke(self._messages(question, retrieval))
            self._cache.warm(
                question, _text_of(reply), retrieval.citations(), curated=True
            )
            results.append((question, True))

        return results


# ⚠️ **Golden rule 4 governed the citation LIST and nothing governed the answer
# BODY, and both halves of that gap shipped** (leg B #15, #19).
#
# The list is built from retrieval and is trustworthy. The `[n]` markers inside
# the sentence are written by the model, and nothing checked them, so a client
# could be shown:
#
#   "…within 14 days … [2] and [1a6234b4]."
#
# `1a6234b4` is the per-request fence nonce from `build_user_prompt`:
# the injection defence, printed twice per passage next to the number the model
# is told to cite, and duly cited. It resolves to nothing in the sources shown
# underneath — on a product whose whole point is *every answer shows its sources*.
#
# The same gap let an answer be the single string `"[1]"`: five citations
# attached, no sentence, rendered as if it were an answer.
#
# Temperature is 0, so neither is a fluke — for the question that triggers it, it
# triggers every time.
_CITATION_MARKER = re.compile(r"\[([^\]\s]{1,24})\]")

# Conjunctions left stranded when the marker after them is dropped. Covers the
# languages the chrome ships in; an unlisted language loses the tidy-up, not the
# correctness — the marker is still removed.
_TRAILING_CONNECTIVE = (
    r"(?:\s*,)?\s+(?:and|or|e|ou|y|o|et|und|oder)\s*([.,;:]|$)"
)


def enforce_citations(text: str, citations: list[dict]) -> str:
    """Drop citation markers the client could not resolve, and refuse an empty answer.

    A post-condition on output, which is where golden rule 4 already lives.
    Markers are *removed* rather than the answer rejected: the sentence is
    usually right and one stray token should not cost the client their answer.
    An answer that is *nothing but* markers is a different thing — it is a failed
    generation, and saying so is better than rendering `[1]` as prose.
    """
    valid = {str(i) for i in range(1, len(citations) + 1)}

    def keep(match: re.Match) -> str:
        return match.group(0) if match.group(1) in valid else ""

    cleaned = _CITATION_MARKER.sub(keep, text)
    # Tidy what a removal leaves behind. "…[2] and [1a6234b4]." would otherwise
    # end "…[2] and." — the stray token is gone and the sentence is still visibly
    # broken, which reads as a bug rather than as an answer.
    cleaned = re.sub(_TRAILING_CONNECTIVE, r"\1", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    if not _CITATION_MARKER.sub("", cleaned).strip():
        return ""
    return cleaned


def _text_of(message: Any) -> str:
    """Message content as text, tolerating list-of-blocks content."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)
