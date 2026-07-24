"""Scheduled refresh: keeping the index current, unattended.

The failure-mode tests matter more than the happy path. A scheduled task that
dies takes the schedule with it, and the client finds out weeks later that their
assistant has been answering from a corpus nobody has updated since handover.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from stillroom.engine import Engine
from stillroom.pipeline import ingest
from stillroom.refresh import RefreshScheduler, refresh_once
from tests.conftest import FakeChatModel


@pytest.fixture
def factory(config, embedding, fake_model):
    """Build engines with the offline embedder and a stub model."""

    def make(cfg):
        return Engine(cfg, chat_model=fake_model, embedding_function=embedding)

    return make


@pytest.fixture
def offline(monkeypatch, embedding):
    """Force the deterministic embedder through every path refresh touches.

    Patched at the *source* module too: `refresh._current_fingerprint` imports
    it inside the function, so patching only the importing modules misses it.
    """
    import stillroom.engine
    import stillroom.index.embeddings
    import stillroom.pipeline
    import stillroom.refresh

    for module in (
        stillroom.index.embeddings,
        stillroom.pipeline,
        stillroom.engine,
    ):
        monkeypatch.setattr(module, "embedding_function_for", lambda config: embedding)
    monkeypatch.setattr(stillroom.refresh, "model_label", lambda config: "stub")


def _seed(config, embedding):
    """Ingest exactly the way `refresh_once` does.

    `embedding_name` participates in the fingerprint, so seeding with a
    different one guarantees the first refresh reports a spurious change.
    """
    return ingest(config, embedding_function=embedding)


def _edit(config, old: str, new: str) -> None:
    handbook = Path(config.corpus.path) / "handbook.md"
    handbook.write_text(
        handbook.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )


def test_an_unchanged_corpus_is_detected_and_skipped(config, embedding, offline, factory):
    """Re-embedding hourly on a CPU box would eat the machine it serves from."""
    _seed(config, embedding)

    result = refresh_once(config, engine_factory=factory)

    assert result.ran and not result.changed
    assert result.documents == 2


def test_a_changed_corpus_is_reingested_and_rebaked(config, embedding, offline, factory):
    _seed(config, embedding)
    _edit(config, "30 days", "14 days")

    result = refresh_once(config, engine_factory=factory)

    assert result.changed
    # The ingest purged the invalidated answers, so the curated set must be
    # rebuilt or instant answers silently degrade.
    assert result.rebaked == 1
    assert result.unanswerable == ()


def test_a_curated_question_that_stops_being_answerable_is_reported(
    config, embedding, offline, factory
):
    """The the loud-bake rule finding, arriving after handover — it must reach someone."""
    _seed(config, embedding)
    # Gut the section the curated question depends on.
    handbook = Path(config.corpus.path) / "handbook.md"
    handbook.write_text("## Unrelated\n\nNothing about money here.\n", encoding="utf-8")

    result = refresh_once(config, engine_factory=factory)

    assert result.changed
    assert result.unanswerable == ("What is the refund window?",)


def test_a_broken_refresh_never_raises_and_keeps_the_old_index(
    config, embedding, offline, factory
):
    """A refresh that throws must not take the schedule — or the service — down."""
    _seed(config, embedding)
    engine = Engine(config, chat_model=FakeChatModel(), embedding_function=embedding)
    assert engine.ask("What is the refund window?").text

    # The corpus directory disappears (an unmounted share is the realistic case).
    for path in Path(config.corpus.path).iterdir():
        path.unlink()

    result = refresh_once(config, engine_factory=factory)

    assert result.error is not None
    assert not result.changed
    # The previously built index is untouched and still answering.
    assert engine.ask("What is the refund window?").text


def test_the_scheduler_runs_and_stops_cleanly(config, embedding, offline, factory):
    _seed(config, embedding)
    ran = threading.Event()

    scheduler = RefreshScheduler(config, interval_s=0.05, engine_factory=factory)
    original = scheduler._loop

    def loop():
        original()

    scheduler._loop = loop
    scheduler.start()
    for _ in range(100):
        if scheduler.last is not None:
            ran.set()
            break
        threading.Event().wait(0.02)
    scheduler.stop(timeout=2)

    assert ran.is_set(), "the scheduler never completed a cycle"
    assert scheduler._thread is None


def test_the_scheduler_does_not_refresh_immediately_on_start(config, embedding, offline, factory):
    """Startup already has a fresh index; refreshing at boot would make every
    restart pay for a full re-embed."""
    _seed(config, embedding)

    scheduler = RefreshScheduler(config, interval_s=30, engine_factory=factory)
    scheduler.start()
    threading.Event().wait(0.1)
    scheduler.stop(timeout=2)

    assert scheduler.last is None


def test_the_scheduler_thread_is_a_daemon(config, offline, factory):
    """So it can never hold the process open on shutdown."""
    scheduler = RefreshScheduler(config, interval_s=30, engine_factory=factory)
    scheduler.start()
    try:
        assert scheduler._thread.daemon
    finally:
        scheduler.stop(timeout=2)


def test_an_unchanged_corpus_is_never_re_embedded(config, embedding, offline, factory, monkeypatch):
    """Leg B #22 — the module's own rule 2, which it did not implement.

    The old order was ingest-then-compare, so every cycle paid the full
    embedding cost to discover nothing had changed. On a scheduled refresh that is
    hourly, on the clients with the largest corpora.
    """
    _seed(config, embedding)

    called = []
    monkeypatch.setattr(
        "stillroom.refresh.ingest",
        lambda *a, **k: called.append(1),
    )

    result = refresh_once(config, engine_factory=factory)

    assert result.ran and not result.changed
    assert result.documents == 2
    assert called == [], "an unchanged corpus must not reach ingest at all"
