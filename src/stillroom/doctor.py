# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""`stillroom doctor` — prove the client's side is set up, and change nothing.

**Every check here is a defect that actually happened.** That is the whole
design constraint (the wrong-computer defects superseded the host-side scanner idea): a checklist
written from imagination catches the problems somebody thought of, and this
product's failures have consistently been the ones nobody thought of. So the
rows below are seeded from the break-it campaign — leg A's fault injection and
leg B's difficult-client scenarios — and nothing is here because it seemed
prudent.

**It runs INSIDE the container, and that is not incidental.** The recurring
defect of this entire project is code reasoning about the wrong computer:

* the engine measured the container's RAM to pick a model for the host's GPU;
* the engine called `localhost` from inside the container, where that address is
  the container;
* `start.sh` checks Ollama from the **host**, where it is reachable, and hands
  off to a container where it is not (leg B #12).

A host-side binary can answer none of the questions that matter, because the
question is always *"can **this** process, from **here**, reach the thing it is
configured to reach?"* — and the only way to answer it is to ask from here.

**Read-only, always** (verify never install). Doctor pulls nothing,
writes nothing, and fixes nothing. It reports, and every failing line names the
command or the action that resolves it.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from stillroom.config import ClientConfig, OllamaModel
from stillroom.hardware import list_downloaded, ollama_answers, running_in_container
from stillroom.provider import ProviderError, resolve_model_name

Status = Literal["ok", "warn", "fail"]

_MARK = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}


@dataclass(frozen=True)
class Check:
    """One question, its answer, and — when it is bad news — what to do."""

    name: str
    status: Status
    detail: str
    # Shown indented under a WARN or FAIL. Never a stack trace, never an exit
    # code: a client reads this over a shoulder, and so does a support email.
    action: str | None = None

    # ⚠️ **Set this on any check whose detail quotes the client's documents.**
    #
    # The engagement is built on never receiving the client's documents, and the
    # report is the one artifact that travels back to us. Three checks earn
    # their keep by quoting the corpus — skipped filenames, headings that appear
    # in two documents, and curated questions that find nothing — and each of
    # those is document content leaving the building inside a support message.
    #
    # `--share` swaps in this version, which must keep the DIAGNOSIS (how many,
    # how bad, what to do) and drop the CONTENT. A redacted report nobody can
    # act on would just get sent unredacted instead.
    share_detail: str | None = None


def _ok(name: str, detail: str) -> Check:
    return Check(name, "ok", detail)


# --------------------------------------------------------------- the model ---


def check_model(config: ClientConfig) -> list[Check]:
    """Can this build reach the model it is configured to use, from here?

    Leg A defects 1, 2 and 4; leg B #8, #11 and #13. The four failures this
    separates were, for a while, a single indistinguishable "it does not work".
    """
    model = config.model
    if not isinstance(model, OllamaModel):
        return [
            Check(
                "model",
                "warn",
                f"Bring-your-own-key build ({model.provider}); the connection is "
                "not tested here.",
                f"A live check would spend the client's own credit. Confirm "
                f"{model.api_key_env} is set in .env if answers fail.",
            )
        ]

    checks: list[Check] = []
    base_url = model.effective_base_url()
    in_container = running_in_container()

    # ⚠️ Leg B #8 — the check that would have caught the poisoned template. The
    # engine is right to let config beat environment (the GPU-box engagement
    # depends on it); what is wrong is a *loopback* address written into a
    # config that will run in a container, where it can only ever mean "me".
    if in_container and model.base_url and model.ollama_is_on_this_machine():
        checks.append(
            Check(
                "config.base_url",
                "fail",
                f"client.toml pins model.base_url to {model.base_url!r}, and this "
                "process is in a container — that address is the container "
                "itself, not the machine running Ollama.",
                "Delete the base_url line from client.toml. Unset, the container "
                "supplies the right address in OLLAMA_HOST. Set it only to name "
                "a DIFFERENT machine, e.g. http://gpu-box.internal:11434.",
            )
        )

    tags = list_downloaded(base_url)
    if not tags:
        # ⚠️ Leg B #13 — "unreachable" and "nothing pulled" were reported as the
        # same thing, and the sentence asserted the wrong one. One extra probe
        # separates them, and they need opposite actions from the client.
        reachable = ollama_answers(base_url)
        if not reachable:
            checks.append(
                Check(
                    "ollama",
                    "fail",
                    f"Nothing is answering at {base_url} from inside this container.",
                    _unreachable_action(base_url, in_container),
                )
            )
            return checks
        checks.append(
            Check(
                "ollama",
                "fail",
                f"Ollama is answering at {base_url} but has no models installed.",
                "Run: ollama pull qwen2.5:7b",
            )
        )
        return checks

    checks.append(_ok("ollama", f"answering at {base_url}, {len(tags)} model(s) installed"))

    try:
        resolved = resolve_model_name(model)
    except ProviderError as exc:
        checks.append(
            Check(
                "model",
                "fail",
                str(exc),
                "Pull one of the models this build supports, or pin model.name "
                "in client.toml to a tag that Ollama already has.",
            )
        )
        return checks

    if resolved not in tags:
        checks.append(
            Check(
                "model",
                "fail",
                f"This build resolves to {resolved}, which is not installed.",
                f"Run: ollama pull {resolved}",
            )
        )
    else:
        checks.append(_ok("model", f"{resolved} is installed and will be used"))

    return checks


def _unreachable_action(base_url: str, in_container: bool) -> str:
    """The sentence that would have saved leg B #11.

    On Linux, `host.docker.internal` is the docker0 bridge, and Ollama binds
    `127.0.0.1` by default — so a perfectly healthy Ollama is invisible from a
    container, and the launcher's host-side check passes while it is. Docker
    Desktop proxies host loopback and does not have this problem, which is
    exactly why it was never seen during development.
    """
    if in_container and "host.docker.internal" in base_url:
        return (
            "If Ollama is running, it is not reachable from a container. Two "
            "causes, both seen on Linux and both invisible from the host: (1) "
            "Ollama listens only on loopback by default — set OLLAMA_HOST=0.0.0.0 "
            "and restart it, via a systemd override on Linux; (2) the host "
            "firewall drops traffic from the Docker bridge — allow it on the "
            "docker0 interface. Docker Desktop normally needs neither. If Ollama "
            "is not running at all, start it and run this again."
        )
    return "Start Ollama, wait a few seconds, then run this again."


# ----------------------------------------------------------- the embedder ---


def check_embedding(config: ClientConfig) -> list[Check]:
    """Can this build READ the client's documents?

    ⚠️ **A check that did not need to exist before this phase, and that is
    exactly why it is easy to miss.** Until `bge-m3`, embedding was an ONNX
    model inside the image: it could not fail, so nothing checked it. Now it is
    a **second model** on the client's Ollama, and the client has no reason to
    know that — the runbook told them to pull the *chat* model. So the most
    likely first-run failure on any delivery after this one is an `ingest` that
    dies with "model not found" for a model nobody mentioned.

    Probed **live**, not by listing tags. The tag list says the weights are on
    disk; it does not say the server can load them, and this whole project's
    method is that reading about the artifact is not running it.

    It is checked **separately from the chat model** because the two addresses
    can differ: a BYOK engagement has no chat-model Ollama at all and still
    needs this one (`EmbeddingConfig`).
    """
    from stillroom.index.embeddings import EmbeddingError, embedding_function_for

    base_url = config.embedding.effective_base_url()
    model = config.embedding.model

    try:
        vector = embedding_function_for(config)(["a short probe of the embedding model"])
    except EmbeddingError as exc:
        message = str(exc)
        # The two worlds again (leg B #13): "not pulled" and "not reachable"
        # need opposite actions from the client, and the message already
        # distinguishes them — this only has to not flatten them back together.
        if "not available" in message:
            return [
                Check(
                    "embedding",
                    "fail",
                    f"{model} is not installed on the Ollama at {base_url}. Your "
                    "documents cannot be read without it.",
                    f"Run: ollama pull {model}   (this is a SECOND model, "
                    "separate from the one that writes the answers)",
                )
            ]
        return [
            Check(
                "embedding",
                "fail",
                f"The embedding model at {base_url} did not answer, so the "
                "documents cannot be read.",
                _unreachable_action(base_url, running_in_container()),
            )
        ]

    if not vector or not any(vector[0]):
        # A zero vector indexes and retrieves in perfect silence, scoring 0.0
        # against everything — the failure mode the offline fixture had.
        return [
            Check(
                "embedding",
                "fail",
                f"{model} returned an empty vector for a plain English probe.",
                "This model cannot be used for retrieval. Configure a different "
                "embedding.model, or report it — it is not something your "
                "documents caused.",
            )
        ]

    return [_ok("embedding", f"{model} answered at {base_url}, {len(vector[0])} dimensions")]


# --------------------------------------------------------------- the index ---


def check_index(config: ClientConfig) -> list[Check]:
    """Is there an index, and can this process still write to it?"""
    from stillroom.index.embeddings import embedding_function_for
    from stillroom.index.store import open_client, read_fingerprint

    path = Path(config.index_path)
    checks: list[Check] = []

    if not path.exists():
        return [
            Check(
                "index",
                "fail",
                f"No index directory at {path}.",
                "Run an ingest: stillroom ingest --config client.toml",
            )
        ]

    if not os.access(path, os.W_OK):
        checks.append(
            Check(
                "index.writable",
                "fail",
                f"{path} is not writable by this process (uid {os.getuid()}).",
                "A re-ingest will fail. Check the volume's ownership in "
                "docker-compose.yml.",
            )
        )
    else:
        checks.append(_ok("index.writable", str(path)))

    try:
        fingerprint = read_fingerprint(
            open_client(config.index_path), config.collection, embedding_function_for(config)
        )
    except Exception as exc:  # an index that exists but cannot be opened
        return checks + [
            Check("index.corpus", "fail", f"The index could not be read: {exc}",
                  "Delete the index volume and re-ingest.")
        ]

    if fingerprint is None:
        checks.append(
            Check(
                "index.corpus",
                "fail",
                "The index exists but holds no corpus.",
                "Run an ingest: stillroom ingest --config client.toml",
            )
        )
    else:
        checks.append(_ok("index.corpus", f"fingerprint {fingerprint[:16]}"))

    return checks


# -------------------------------------------------------------- the corpus ---


def check_corpus(config: ClientConfig) -> list[Check]:
    """Are the documents there, readable, and not quietly contradicting each other?"""
    from stillroom.ingest.loaders import load_corpus

    path = Path(config.corpus.path)
    if not path.exists():
        return [
            Check(
                "corpus",
                "fail",
                f"No documents folder at {path}.",
                "Check the documents volume in docker-compose.yml.",
            )
        ]

    documents, skipped = load_corpus(
        config.corpus.path, config.corpus.include, config.corpus.encoding
    )
    checks: list[Check] = []

    if not documents:
        return [
            Check(
                "corpus",
                "fail",
                f"{path} contains no readable documents "
                f"(looking for {', '.join(config.corpus.include)}).",
                "Put the documents in that folder, then re-ingest.",
            )
        ]

    checks.append(_ok("corpus", f"{len(documents)} readable document(s)"))

    if skipped:
        checks.append(
            Check(
                "corpus.skipped",
                "warn",
                f"{len(skipped)} file(s) are not in the assistant's knowledge: "
                + "; ".join(skipped[:5])
                + ("…" if len(skipped) > 5 else ""),
                "Each line says why. A scanned PDF is a picture of a page and "
                "has no text to read; convert it or supply a text original.",
                share_detail=(
                    f"{len(skipped)} file(s) are not in the assistant's "
                    "knowledge (names withheld; run without --share to see them)."
                ),
            )
        )

    checks.extend(_check_contradictions(documents))
    checks.append(_check_language(documents, config.language))
    checks.append(_check_index_matches_corpus(config, documents))
    return checks


# ⚠️ **Leg B #21 — doctor printed both halves of this and did not notice.**
#
#     [OK  ] index.corpus   fingerprint 70fb6387a6c5b0ee      <- 2 documents
#     [OK  ] corpus         3 readable document(s)
#
# Two green lines, one of them describing the documents on disk and the other
# describing an index built before the third arrived. Measured after adding a
# file and re-running the launcher, which is what a client will actually do —
# and `docker compose up -d` does not rebuild an image that already exists, so
# the new document was simply not in the assistant's knowledge, silently.
#
# This is the same drift as the serves-versus-holds distinction, one step earlier in the chain: there,
# what-is-served vs what-is-indexed; here, what-is-indexed vs what-is-on-disk.
# Both are invisible without comparing, and both are the failure a client reports
# as "it does not know about the document I added".
def _check_index_matches_corpus(config: ClientConfig, documents) -> Check:
    from stillroom.index.embeddings import embedding_function_for
    from stillroom.index.store import open_client, read_fingerprint
    from stillroom.pipeline import corpus_snapshot

    try:
        indexed = read_fingerprint(
            open_client(config.index_path), config.collection, embedding_function_for(config)
        )
    except Exception:
        return Check("corpus.indexed", "warn", "The index could not be read to compare.",
                     "Run an ingest: stillroom ingest --config client.toml")

    if indexed is None:
        return Check("corpus.indexed", "fail", "Nothing has been indexed yet.",
                     "Run an ingest: stillroom ingest --config client.toml")

    # The same computation `ingest` performs, so the two are comparable by
    # construction rather than by a heuristic like counting files.
    on_disk = corpus_snapshot(config).fingerprint

    if on_disk == indexed:
        return _ok("corpus.indexed", "the index matches the documents on disk")

    return Check(
        "corpus.indexed",
        "fail",
        "The documents on disk have changed since they were last indexed — the "
        "assistant does not know about the current versions.",
        "Re-read them: stillroom ingest --config client.toml, then restart the "
        "assistant (stop, then start).",
    )


# ⚠️ Leg B #14 — the cheapest honest signal that a corpus disagrees with itself.
#
# Two documents carrying the same heading with different bodies is what a stale
# handbook beside its replacement looks like, and the measured behaviour is that
# the engine surfaces the conflict or buries it **depending on how the question
# is phrased**: asked directly it named both figures and said they conflicted;
# asked as part of a compound question it dropped one silently and, in one run,
# invented a qualifier to reconcile them that appears in neither document.
#
# There is no answer-time fix for that — it is a small model doing its best with
# contradictory evidence. There IS a delivery-time fix, and this is it: say so
# before handover, while the client can still tell us which document wins.
def _check_contradictions(documents: Iterable) -> list[Check]:
    by_heading: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        for heading in _headings(document.text):
            by_heading[heading.lower()].add(document.source)

    shared = {h: s for h, s in by_heading.items() if len(s) > 1}
    if not shared:
        return [_ok("corpus.conflicts", "no heading appears in more than one document")]

    listed = "; ".join(
        f"{heading!r} in {', '.join(sorted(sources))}"
        for heading, sources in sorted(shared.items())[:5]
    )
    return [
        Check(
            "corpus.conflicts",
            "warn",
            f"{len(shared)} heading(s) appear in more than one document: {listed}"
            + ("…" if len(shared) > 5 else ""),
            "If two documents state the same policy differently, the assistant "
            "may answer from either, and will not always say they disagree. "
            "Remove the superseded document, or record which one wins in the "
            "standing context file.",
            share_detail=(
                f"{len(shared)} heading(s) appear in more than one document "
                "(headings and filenames withheld)."
            ),
        )
    ]


_HEADING_RE = re.compile(r"^#{1,3}[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def _headings(text: str) -> list[str]:
    return [m.group(1).strip() for m in _HEADING_RE.finditer(text)]


# ⚠️ Leg B #18 — measured, and it is not a near miss.
#
# On a Portuguese corpus, Portuguese questions retrieve at 0.646–0.704 and
# English questions at 0.076–0.220 against a 0.25 floor: 0 of 5 relevant, every
# one refused with no model call. Cross-language retrieval is absent, not weak.
# `language` in the config sets the REPLY language and does nothing to
# retrieval — so a build can be configured "pt-BR", answer Portuguese
# perfectly, and refuse every English question about the same documents.
#
# A warning rather than a failure: a corpus whose language differs from the
# chrome language is a legitimate deployment. It is just never what somebody
# meant to build, and it is invisible until a user asks in the wrong language.
_STOPWORDS = {
    "en": {"the", "and", "of", "to", "is", "for", "within", "any", "which", "with"},
    "pt": {"de", "que", "para", "não", "com", "uma", "dos", "das", "são", "até"},
    "es": {"de", "que", "para", "con", "una", "los", "las", "son", "hasta", "el"},
    "fr": {"de", "que", "pour", "avec", "une", "les", "des", "sont", "est", "dans"},
}


def _check_language(documents: Iterable, configured: str) -> Check:
    words = Counter()
    for document in documents:
        words.update(re.findall(r"[a-zà-ÿ]+", document.text.lower()))

    scores = {
        code: sum(words[w] for w in stopwords) for code, stopwords in _STOPWORDS.items()
    }
    detected, best = max(scores.items(), key=lambda kv: kv[1])
    if best == 0:
        return _ok("corpus.language", "not enough text to judge; not checked")

    family = configured.split("-")[0].lower()
    if family == detected:
        return _ok("corpus.language", f"documents and client language agree ({configured})")

    return Check(
        "corpus.language",
        "warn",
        f"The documents look like '{detected}' and this build is configured "
        f"language = {configured!r}.",
        "The assistant only finds documents when the question is asked in the "
        "language the documents are written in — it cannot search across "
        "languages. Questions in the configured language will be refused as "
        "'not in your documents'. Set language to match the documents, and "
        "write the curated questions and the eval suite in that language too.",
    )


# ------------------------------------------------------------- the curated ---


def check_curated(config: ClientConfig) -> list[Check]:
    """Do the client's own headline questions still find anything?

    `bake` already fails a build whose curated questions cannot be answered — but it costs a model call each, and by then the image is being
    built. This is the same question asked for free: does the question retrieve
    anything above the floor at all?
    """
    if not config.answer_cache.enabled or not config.answer_cache.curated:
        return []

    from stillroom.index.embeddings import embedding_function_for
    from stillroom.index.retrieval import search
    from stillroom.index.store import get_collection, open_client

    try:
        collection = get_collection(
            open_client(config.index_path), config.collection, embedding_function_for(config)
        )
    except Exception:
        return []  # check_index already reported this

    unfindable = [
        question
        for question in config.answer_cache.curated
        if not search(
            collection,
            question,
            k=config.retrieval.k,
            min_similarity=config.retrieval.min_similarity,
        ).grounded
    ]

    if not unfindable:
        return [
            _ok("curated", f"all {len(config.answer_cache.curated)} question(s) find documents")
        ]

    return [
        Check(
            "curated",
            "fail",
            f"{len(unfindable)} curated question(s) find nothing in the documents: "
            + "; ".join(repr(q) for q in unfindable),
            "Either the documents that answer them are missing, or the question "
            "asks for something they do not cover. If the documents are not in "
            "English, the questions must not be either.",
            share_detail=(
                f"{len(unfindable)} of {len(config.answer_cache.curated)} curated "
                "question(s) find nothing in the documents (questions withheld). "
                "Numbered from the list you supplied: "
                + ", ".join(
                    str(i + 1)
                    for i, q in enumerate(config.answer_cache.curated)
                    if q in unfindable
                )
                + "."
            ),
        )
    ]


# ---------------------------------------------------------- the capacity ---


def check_capacity(config: ClientConfig) -> list[Check]:
    """Does the agreed document limit actually hold?

    ⚠️ **This became load-bearing when the client gained a re-ingest they run
    themselves.** `warn_only` defaults to True, and that default was reasoned for
    *delivery* — discovering an over-limit corpus mid-build should start a
    conversation, not break a build I am standing in front of. Post-handover the
    same default means the opposite thing: the client adds documents, the ingest
    logs a warning nobody reads, and the agreed limit quietly stops existing.

    Two moments, opposite correct answers, one flag — so the honest move is to
    surface it at delivery rather than pick silently.
    """
    capacity = config.capacity
    if capacity.max_documents is None:
        return [
            Check(
                "capacity",
                "warn",
                "No document limit is configured, so the agreed limit is not "
                "enforced.",
                "Set capacity.max_documents to the limit agreed for this "
                "engagement.",
            )
        ]

    if capacity.warn_only:
        return [
            Check(
                "capacity",
                "warn",
                f"The limit of {capacity.max_documents} documents is advisory "
                "(warn_only), so a client re-ingest above it will succeed.",
                "For a delivered build set capacity.warn_only = false: an "
                "over-limit re-ingest then fails cleanly and the assistant keeps "
                "serving the documents it already had.",
            )
        ]

    return [_ok("capacity", f"limit of {capacity.max_documents} documents is enforced")]


# ------------------------------------------------------- the delivered shape ---


def check_delivery(config: ClientConfig, directory: str = ".") -> list[Check]:
    """Does this deployment still match what was handed over?

    ⚠️ **Evidence, never enforcement.** This reports and returns; it does not
    block, modify or delete. The engine ships every capability to every
    deployment, so the agreed limits live in a config file rather than behind a
    lock — and a lock is not available to us, because the container runs on
    hardware the client owns with the source public for them to audit. What *is*
    available is saying plainly, in a report anybody can read, that the running
    deployment is no longer the one that was agreed.

    A missing manifest is `unverifiable`, never `tampered`. Most drift is
    innocent — a client raising their own document limit after outgrowing it is
    a conversation, not an accusation — and a check that cried breach would stop
    being read.
    """
    from stillroom import integrity

    manifest = integrity.load(directory)
    if manifest is None:
        return [
            Check(
                "delivery",
                "warn",
                "No delivery manifest, so this build cannot be compared with "
                "what was handed over.",
                "Expected for a build made before manifests, or one rebuilt "
                "without it. Ask for a fresh manifest to re-establish a baseline.",
            )
        ]

    drifts = integrity.compare(manifest, config)
    if not drifts:
        return [_ok("delivery", "configuration matches the delivered baseline")]

    return [
        Check(
            "delivery",
            "warn",
            "This deployment no longer matches what was delivered — "
            + "; ".join(str(d) for d in drifts)
            + ".",
            "If these changes were agreed, ask for an updated manifest so the "
            "baseline matches. If they were not, the delivered configuration is "
            "the one covered by the engagement.",
        )
    ]


# ----------------------------------------------------------------- the run ---


def run(config: ClientConfig) -> list[Check]:
    """Every check, in the order a failure cascades.

    Model first: a build whose model is unreachable cannot answer anything, so a
    warning about heading collisions underneath that is noise.

    The embedder comes second, and ahead of the index, because since the move to bge-m3 it
    is what *builds* the index. A failure there explains every check below it.
    """
    checks: list[Check] = []
    checks.extend(check_model(config))
    checks.extend(check_embedding(config))
    checks.extend(check_index(config))
    checks.extend(check_corpus(config))
    checks.extend(check_curated(config))
    checks.extend(check_capacity(config))
    checks.extend(check_delivery(config))
    return checks


def format_report(checks: list[Check], share: bool = False) -> str:
    """Plain text, because a client reads this and a JSON blob is not readable.

    `share=True` produces the version a client can send back to us. The whole
    engagement rests on their documents never leaving the building, and this
    report is the only artifact that travels the other way — so a filename or a
    heading quoted here would be the one place the promise leaks, in a message
    the client sent *because we asked them to*.
    """
    width = max((len(c.name) for c in checks), default=0)
    lines = []
    if share:
        lines += [
            "Shareable report — content withheld.",
            "Filenames, document headings and question text are replaced with "
            "counts.",
            "Run `stillroom doctor` without --share to see them on this machine.",
            "",
        ]
    for check in checks:
        detail = check.share_detail if (share and check.share_detail) else check.detail
        lines.append(f"[{_MARK[check.status]}] {check.name.ljust(width)}  {detail}")
        if check.action:
            lines.append(f"{' ' * (width + 9)}-> {check.action}")

    failed = [c for c in checks if c.status == "fail"]
    warned = [c for c in checks if c.status == "warn"]

    lines.append("")
    if failed:
        lines.append(
            f"{len(failed)} problem(s) will stop this assistant working. "
            "Each line above says what to do."
        )
    elif warned:
        lines.append(
            f"Everything needed to run is in place, with {len(warned)} thing(s) "
            "worth knowing about."
        )
    else:
        lines.append("Everything checks out.")
    return "\n".join(lines)


def worst(checks: list[Check]) -> Status:
    for status in ("fail", "warn"):
        if any(c.status == status for c in checks):
            return status  # type: ignore[return-value]
    return "ok"
