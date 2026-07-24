# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""Embedding functions for the document index.

The default is **local**, and that is the quiet half of the whole product. An
embedder that reached out to a cloud service would send every chunk of the
corpus off the machine at ingest time, falsifying the product's central
claim before the chatbot answered a single question. The provider is the loud
part of "nothing leaves the building"; the embedder is the part nobody thinks to
check.

## The model is `bge-m3`, served by the client's own Ollama

It used to be ChromaDB's bundled ONNX `all-MiniLM-L6-v2`, and the swap is
A measured trade rather than an upgrade:

- **What it buys.** MiniLM is English-centric. A Portuguese question against a
  Portuguese corpus scored 0.646–0.704 only because Portuguese shares script and
  lexis with English; Chinese scored **0.040–0.205 against a 0.25 floor** — every
  question refused, and the refusal is indistinguishable from *"your documents do
  not cover that"*. `bge-m3` puts every one of those in the 0.59–0.81
  band, in both directions. The documents'-language rule stops being a policy the
  product cannot honour.
- **What it costs.** It compresses the whole similarity range upward, so the old
  0.25 floor stopped separating anything at all — **all 27 benchmark cells
  passed** under it. See `retrieval`, where the re-derived floor and the
  compensating margin live, and `benchmarks/retrieval_floor.py`, which measured
  them.
- **⚠️ What it changes architecturally.** `ingest` now needs Ollama, which it
  did not before — only `bake` did. That is a new failure mode on the client's
  first run, and it is why `doctor` and both launcher preflights probe the
  embedding model by name rather than probing "Ollama".

**Why Ollama and not `sentence-transformers`:** the client already runs Ollama —
it is a stated prerequisite — so this costs no torch, no image growth,
and **no download from Chroma's S3**, which was one of the four failure modes the
base tarball exists to remove. It also un-does part of that shipping decision: the weights are
no longer inside the image we hand over, so we are not their redistributor.

`DeterministicEmbeddingFunction` is a dependency-free hashing vectorizer. It is
not semantic, but it is offline and reproducible, so the test suite indexes and
retrieves without downloading a model or reaching Ollama.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.error
import urllib.request

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

logger = logging.getLogger(__name__)

# Scripts written without spaces between words: Han, Hiragana, Katakana,
# Hangul. A word-run regex sees one token per *sentence* in these, and a
# sentence token shares nothing with the same question phrased differently — so
# they are split per character, with bigrams added because a single Han
# character is closer to a letter than to a word.
_SCRIPTLESS = r"぀-ヿ㐀-䶿一-鿿豈-﫿가-힯"

# ⚠️ **This used to be `[a-z0-9]+`, and that made the fixture lie**.
# Chinese, Japanese, Korean, Arabic and Russian matched *nothing*, so every one
# of them embedded to the **zero vector** — which Chroma happily indexes and
# which retrieves at similarity 0.0 against everything. A multilingual
# retrieval test written against it would have passed by returning garbage, in
# the phase whose entire purpose is to make multilingual retrieval real.
# Fixed first, before the embedder was touched, for exactly that reason.
_TOKEN_RE = re.compile(rf"[{_SCRIPTLESS}]+|(?:(?![{_SCRIPTLESS}])\w)+", re.UNICODE)
_SCRIPTLESS_CHAR = re.compile(rf"[{_SCRIPTLESS}]")


def _tokenise(text: str) -> list[str]:
    """Tokens whose overlap approximates shared vocabulary, in any script."""
    tokens: list[str] = []
    for run in _TOKEN_RE.findall(text.lower()):
        if _SCRIPTLESS_CHAR.match(run):
            tokens.extend(run)
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run)
    return tokens

# ⚠️ **Named by VERSION, not by family**. `bge-m3` alone is not a
# version, and the bill of materials is a contractual document. `567m` and
# `latest` resolve to the same digest today (verified 2026-07-22,
# sha256:79076464…) — pinning the one that cannot move is the point.
DEFAULT_EMBEDDING_NAME = "bge-m3:567m"

# Named in the corpus fingerprint: changing the embedder changes what "similar"
# means, which invalidates every cached answer produced under the old one. That
# is why this swap happened before any client existed.
DEFAULT_EMBEDDING_DIMENSIONS = 1024

# Chunks per `/api/embed` call. A corpus is embedded in one `ingest`, and the
# A large corpus is thousands of documents — one request carrying every chunk of that
# is a multi-megabyte body and a request that either times out or dies in a
# proxy, with the whole ingest lost rather than one batch of it.
_BATCH = 32
# Generous on purpose: this runs on the hardware the product *targets*,
# which is frequently CPU-only, and a first call also pays for loading a 1.2 GB
# model into memory.
_TIMEOUT_SECONDS = 600


def _split_in_half(text: str) -> tuple[str, str] | None:
    """Split at the word boundary nearest the middle, or None if too short."""
    if len(text.strip()) < 40:
        return None
    middle = len(text) // 2
    cut = text.rfind(" ", 0, middle) or -1
    if cut <= 0:
        cut = text.find(" ", middle)
    if cut <= 0 or cut >= len(text) - 1:
        return None
    return text[:cut].strip(), text[cut + 1 :].strip()


def _mean_unit_vector(vectors: list[list[float]]) -> list[float]:
    total = [sum(components) for components in zip(*vectors)]
    norm = sum(v * v for v in total) ** 0.5
    return [v / norm for v in total] if norm else total


def _passage_ref(text: str) -> str:
    """A content-free way to name the passage that tripped an upstream defect.

    ⚠️ **The obvious thing here — printing the first N characters — puts client
    document text into a log line and an exception message** (audit, 2026-07-23).
    That is content leaving the passage and landing somewhere it was never meant
    to: the container log, and any support bundle a client later sends us. On a
    product whose whole promise is that the documents stay in the building, a
    debug convenience must not be the one path that copies them out.

    Length plus a short SHA-256 prefix is reproducible — the same passage yields
    the same ref every run — so it identifies *which* chunk hit ollama#16625 for
    anyone reproducing it against the same corpus, and reveals nothing about what
    the chunk says.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{len(text)} chars, sha256:{digest}"


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be reached or does not answer.

    A distinct type because the two callers want opposite things from it.
    `ingest` must stop — an index built from nothing is worse than no index.
    `doctor` must catch it and print a sentence a non-technical client can act
    on, which is why every message raised here names the address, the model and
    the command that fixes it.

    ⚠️ **`passage_rejected` separates two worlds that must never be confused,
    and conflating them was a real defect caught by its own test.** The
    split-and-retry below exists for *one* failure: the model read the request
    and produced `NaN` for this particular text. Every other failure — Ollama
    unreachable, the model not pulled, a truncated reply — is about the
    **deployment**, not the passage. The first version split on all of them, so
    an unreachable Ollama told the client *"this is an upstream defect, not a
    problem with your document"* while retrying their corpus one halved passage
    at a time. That is leg B #13 exactly: one condition, two worlds, and the
    message asserting the wrong one.
    """

    def __init__(self, message: str, *, passage_rejected: bool = False) -> None:
        super().__init__(message)
        self.passage_rejected = passage_rejected


def default_embedding_function(
    base_url: str | None = None, model: str | None = None
) -> EmbeddingFunction:
    """The client's own Ollama, embedding locally. No key, no cloud, no S3.

    `base_url` follows the same config → `OLLAMA_HOST` → loopback resolution as
    the chat model, and for the same delivered-defect reason: in the
    container `localhost` is the container. Callers that hold a `ClientConfig`
    should pass `config.embedding.effective_base_url()`; the bare default exists
    for the one-off scripts and is the laptop answer, not the deliverable's.
    """
    from stillroom.config import DEFAULT_OLLAMA_URL, resolve_ollama_url

    return OllamaEmbeddingFunction(
        base_url=base_url or resolve_ollama_url(None) or DEFAULT_OLLAMA_URL,
        model=model or DEFAULT_EMBEDDING_NAME,
    )


def embedding_function_for(config) -> EmbeddingFunction:
    """The embedder *this engagement* is configured to use.

    ⚠️ **Prefer this to `default_embedding_function()` anywhere a `ClientConfig`
    is in scope.** The bare default resolves to the laptop answer, and a caller
    that reaches for it in the deliverable rebuilds the wrong-computer defect one layer
    down: the index would be written by one address and read by another, the
    embedder's name would still match, and nothing would report a problem.
    """
    return default_embedding_function(
        base_url=config.embedding.effective_base_url(), model=config.embedding.model
    )


@register_embedding_function
class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embeds through the client's Ollama, over `/api/embed`.

    ⚠️ **Written here rather than taken from `chromadb.utils`, and the reason is
    the error message.** Chroma's version needs the `ollama` Python package and
    raises whatever that package raises — which, measured against an unreachable
    Ollama, is a `ConnectionError` traceback. This is the first thing a client's
    very first `ingest` can fail on, in front of somebody who did not choose to
    be looking at a terminal, and leg B #13 already cost this project a release
    for telling a client to pull a model they had while the real fault was the
    network. So: `urllib`, no new dependency, and one honest sentence per world.

    Batched, because a whole corpus in one request is a request that dies in the
    middle and takes the ingest with it.
    """

    def __init__(
        self,
        base_url: str,
        model: str = DEFAULT_EMBEDDING_NAME,
        timeout: int = _TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def __call__(self, input: Documents) -> Embeddings:
        vectors: list[list[float]] = []
        for start in range(0, len(input), _BATCH):
            batch = list(input[start : start + _BATCH])
            try:
                vectors.extend(self._embed_batch(batch))
            except EmbeddingError as exc:
                # An unreachable Ollama is not going to become reachable one
                # passage at a time, and retrying the whole corpus against it
                # turns a five-second failure into a very long one.
                if not exc.passage_rejected:
                    raise
                # One unembeddable passage fails the whole batch, so a batch
                # failure says nothing about which text caused it. Fall back to
                # one at a time; slow, and only on the unhappy path.
                for text in batch:
                    vectors.append(self._embed_one_resiliently(text))
        return vectors

    def _embed_one_resiliently(self, text: str) -> list[float]:
        """Embed one text, splitting it if the model refuses to read it.

        ⚠️ **This exists for an upstream defect, and it is a workaround — say so
        wherever it is described**. Ollama's `bge-m3` GGUF returns
        `NaN` components for some ordinary passages (numerical overflow in
        attention on particular token sequences, F16). Ollama cannot serialise
        `NaN`, so the request dies with `HTTP 500 … json: unsupported value:
        NaN`. Measured on 0.24.0: **5 of 32 English passages, 0 of 32
        Portuguese**, deterministic per text, no content-preserving edit avoids
        it — a trailing space, collapsing newlines and stripping the Markdown
        heading all still fail. Splitting at a word boundary recovered 5 of 5.
        Upstream: ollama/ollama#16625, open.

        **Splitting is safe in a way that dropping is not.** The passage still
        reaches the index, in halves, so the client's question can still find
        it. A dropped passage would make the assistant answer *"that is not in
        your documents"* about a document it was given — the silent-wrong-answer
        class this engine refuses everywhere else.

        The vector returned for the whole passage is the **mean of its halves**,
        renormalised. That is an approximation, and it is the honest one
        available: the alternative is indexing the halves as separate chunks,
        which would make the stored chunk IDs disagree with the chunker that
        produced them and break the pruning contract in `store.index_chunks`.
        """
        try:
            return self._embed_batch([text])[0]
        except EmbeddingError as exc:
            if not exc.passage_rejected:
                raise

        halves = _split_in_half(text)
        if halves is None:
            raise EmbeddingError(
                f"The embedding model {self._model!r} will not read this passage "
                "and it is too short to split further. This is an upstream "
                "defect in Ollama's bge-m3 build (ollama/ollama#16625), not a "
                "problem with your document. Upgrading Ollama, or configuring "
                "another embedding model, is the way out. The passage "
                f"({_passage_ref(text)}) cannot be split further.",
                passage_rejected=True,
            )

        logger.warning(
            "the embedding model refused a passage and it was split to get "
            "around an upstream defect (ollama/ollama#16625); passage %s",
            _passage_ref(text),
        )
        vectors = [self._embed_one_resiliently(half) for half in halves]
        return _mean_unit_vector(vectors)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            f"{self._base_url}/api/embed",
            data=json.dumps({"model": self._model, "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400]
            # 404 is Ollama's answer for "that model is not pulled", and it is
            # the failure a client actually hits: they pulled the chat model
            # because the runbook said so, and this is a SECOND model.
            if exc.code == 404:
                raise EmbeddingError(
                    f"The embedding model {self._model!r} is not available on the "
                    f"Ollama at {self._base_url}. Your documents cannot be read "
                    f"without it. Pull it there with:  ollama pull {self._model}"
                ) from exc
            raise EmbeddingError(
                f"The Ollama at {self._base_url} refused to embed with "
                f"{self._model!r} (HTTP {exc.code}): {body}",
                # A 500 is how the NaN defect surfaces: the model answered, and
                # the answer would not serialise. That is about THIS text, so
                # splitting it is worth trying. Nothing else here is.
                passage_rejected=exc.code >= 500,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EmbeddingError(
                f"Nothing is answering at {self._base_url} from this process, so "
                "your documents cannot be read. If Ollama is running, it is not "
                "reachable from here — a container cannot see a service bound to "
                "the host's loopback address. Set OLLAMA_HOST=0.0.0.0 on the "
                "Ollama host and allow the Docker bridge range through its "
                "firewall, or point embedding.base_url at an address this "
                f"process can reach. ({exc})"
            ) from exc

        embeddings = payload.get("embeddings")
        # ⚠️ A short or empty list here would become zero vectors downstream,
        # which index and retrieve silently — the exact failure the offline
        # fixture had (see `_TOKEN_RE`). Refuse instead.
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingError(
                f"The Ollama at {self._base_url} returned "
                f"{len(embeddings) if isinstance(embeddings, list) else 'no'} "
                f"embeddings for {len(texts)} passages. Refusing to index a "
                "corpus that would be partly blank."
            )
        return [[float(x) for x in vector] for vector in embeddings]

    @staticmethod
    def name() -> str:
        return "stillroom_ollama"

    def get_config(self) -> dict:
        return {"base_url": self._base_url, "model": self._model, "timeout": self._timeout}

    @classmethod
    def build_from_config(cls, config: dict) -> "OllamaEmbeddingFunction":
        return cls(
            base_url=config.get("base_url", ""),
            model=config.get("model", DEFAULT_EMBEDDING_NAME),
            timeout=config.get("timeout", _TIMEOUT_SECONDS),
        )


@register_embedding_function
class DeterministicEmbeddingFunction(EmbeddingFunction[Documents]):
    """Offline hashing bag-of-words embedding, for tests.

    Tokenises to words, hashes each into `dim` buckets and L2-normalises, so
    cosine similarity reflects shared vocabulary. Enough for a test to assert
    that the right passage is retrieved; not enough for production, which is why
    it is never the configured default.

    ⚠️ **It shares no property with the shipped embedder except "returns a unit
    vector".** It is lexical, so it cannot cross languages and its absolute
    scores mean nothing — the shipped `bge-m3` compresses everything upward. **No calibrated constant may be derived from this class**, and
    none is: `min_similarity` and the relative margin are measured by
    `benchmarks/retrieval_floor.py` against the real model, and the offline
    tests exercise the *mechanism* against explicit scores instead.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _tokenise(text):
            bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[bucket] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def name() -> str:
        return "deterministic_hashing"

    def get_config(self) -> dict:
        return {"dim": self._dim}

    @classmethod
    def build_from_config(cls, config: dict) -> "DeterministicEmbeddingFunction":
        return cls(dim=config.get("dim", 256))
