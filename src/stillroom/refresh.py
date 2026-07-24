# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""Scheduled refresh: keeping the index current, unattended.

"Scheduled refresh" means: the client drops documents into their folder and the
assistant notices, without anyone running a command. It is `ingest` + `bake` on
a timer, and the interesting part is entirely in the failure modes.

**It is a loop in the app, not a cron job.** A cron entry would need a second
deployment artifact, per-OS (Task Scheduler on Windows, launchd on macOS, cron
on Linux), installed on a machine where "verify, never install" is the rule. A thread inside the process the client is already running ships with
the container and works identically everywhere.

**Three things it must never do**, each learned from what the alternative costs:

1. **Never leave the assistant unanswerable.** A refresh that throws must be
   caught and logged, and the previous index must keep serving. A client whose
   chatbot went dark overnight because a colleague saved a corrupt PDF has a
   worse problem than a slightly stale answer.
2. **Never re-embed an unchanged corpus.** The fingerprint already tells us
   whether anything changed. Skipping is not an optimisation here — on a CPU-only
   box, re-embedding a 1,500-document corpus every hour would eat the machine the
   assistant is supposed to be answering from.
3. **Never rebake silently on a broken corpus.** If a curated question stops
   being answerable after a refresh, that is the loud-bake finding arriving
   post-handover — log it loudly so it reaches someone.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from stillroom.config import ClientConfig
from stillroom.pipeline import corpus_snapshot, ingest
from stillroom.provider import model_label

logger = logging.getLogger(__name__)


@dataclass
class RefreshResult:
    """What one refresh cycle did. Returned for tests and for the log line."""

    ran: bool
    changed: bool
    fingerprint: str | None = None
    documents: int = 0
    rebaked: int = 0
    unanswerable: tuple[str, ...] = ()
    error: str | None = None


def refresh_once(config: ClientConfig, *, engine_factory=None) -> RefreshResult:
    """Re-ingest, and rebake the curated answers only if the corpus changed.

    Never raises: a scheduled task that dies takes the schedule with it, and the
    client finds out weeks later.
    """
    try:
        before = _current_fingerprint(config)

        # ⚠️ **Rule 2 above was stated and not implemented** (leg B #22). This
        # used to run a full `ingest()` and compare fingerprints afterwards —
        # so an unchanged corpus was re-embedded in its entirety on every cycle,
        # hourly, on exactly the deployments with the largest corpora and the
        # least appetite for the machine being busy. Reading and chunking is
        # cheap; embedding is the expensive half, and now only the cheap half
        # runs when nothing has changed.
        snapshot = corpus_snapshot(config)
        if before is not None and snapshot.fingerprint == before:
            logger.info(
                "refresh: corpus unchanged (%s docs), nothing to do", snapshot.documents
            )
            return RefreshResult(
                ran=True, changed=False, fingerprint=before, documents=snapshot.documents
            )

        result = ingest(config, model_name=model_label(config))
        changed = result.fingerprint != before

        if not changed:
            logger.info("refresh: corpus unchanged (%s docs), nothing to do", result.documents)
            return RefreshResult(
                ran=True, changed=False, fingerprint=result.fingerprint,
                documents=result.documents,
            )

        logger.info(
            "refresh: corpus changed -> %d documents, %d passages, %d pruned",
            result.documents, result.chunks, result.pruned,
        )

        # The ingest already purged the answers this change invalidated, so the
        # curated set has to be rebuilt or instant answers quietly degrade to
        # "the first person to ask waits".
        rebaked, unanswerable = _rebake(config, engine_factory)
        if unanswerable:
            logger.error(
                "refresh: %d curated question(s) can no longer be answered from the "
                "documents: %s", len(unanswerable), "; ".join(unanswerable),
            )

        return RefreshResult(
            ran=True, changed=True, fingerprint=result.fingerprint,
            documents=result.documents, rebaked=rebaked, unanswerable=unanswerable,
        )
    except Exception as exc:
        # The previous index is untouched and still serving.
        logger.exception("refresh failed; the assistant keeps serving the previous index")
        return RefreshResult(ran=True, changed=False, error=str(exc))


def _current_fingerprint(config: ClientConfig) -> str | None:
    from stillroom.index.embeddings import embedding_function_for
    from stillroom.index.store import open_client, read_fingerprint

    try:
        embed = embedding_function_for(config)
        return read_fingerprint(open_client(config.index_path), config.collection, embed)
    except Exception:  # never ingested yet, or an unreadable index
        return None


def _rebake(config: ClientConfig, engine_factory) -> tuple[int, tuple[str, ...]]:
    if not config.answer_cache.enabled or not config.answer_cache.curated:
        return 0, ()

    from stillroom.engine import Engine

    engine = (engine_factory or Engine)(config)
    results = engine.bake_curated()
    baked = [q for q, ok in results if ok]
    missed = tuple(q for q, ok in results if not ok)
    return len(baked), missed


class RefreshScheduler:
    """Runs `refresh_once` on an interval, in a daemon thread.

    Daemon so it can never hold the process open on shutdown, and `stop()` uses
    an Event rather than a sleep loop so a container gets a prompt SIGTERM
    instead of waiting out the interval.
    """

    def __init__(self, config: ClientConfig, *, interval_s: float, engine_factory=None) -> None:
        self.config = config
        self.interval_s = interval_s
        self._engine_factory = engine_factory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last: RefreshResult | None = None

    def _loop(self) -> None:
        # Wait first: startup already has a fresh index, and refreshing at boot
        # would make every restart pay for a full re-embed.
        while not self._stop.wait(self.interval_s):
            self.last = refresh_once(self.config, engine_factory=self._engine_factory)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="stillroom-refresh", daemon=True
        )
        self._thread.start()
        logger.info("scheduled refresh every %.0f s", self.interval_s)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
