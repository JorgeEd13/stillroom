"""Shared fixtures. Every test in this suite runs offline.

No model is downloaded, no Ollama is contacted, no network call is made:
embeddings come from the deterministic hashing function and the chat model is a
stub. That is a hard requirement rather than a preference — this engine's whole
claim is that it works without reaching anything, and a test suite that quietly
needed the network would be unable to notice if that stopped being true.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, Iterator

import pytest
from langchain_core.messages import AIMessage

from stillroom.config import ClientConfig
from stillroom.index.embeddings import DeterministicEmbeddingFunction


class NetworkAccessInTest(AssertionError):
    """A test opened a socket. See `_offline`."""


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make "the suite runs offline" enforced instead of merely intended.

    ⚠️ **This exists because the docstring above was briefly false and nothing
    noticed.** Adding a reachability probe to `/api/health` made the
    API tests contact Ollama for real; they passed, in full green, because the
    development machine happens to have Ollama running. On any other machine
    they would have failed for a reason having nothing to do with the code, and
    on this one they were quietly asserting against a live service.

    That is the precise failure this engine cannot afford to be relaxed about:
    the product's central claim is that it works without reaching anything, and
    a suite that can silently use the network is unable to notice when that
    stops being true.

    Opting out is deliberate and per-test — see `allow_network`.
    """

    def refuse(*args: Any, **kwargs: Any):
        raise NetworkAccessInTest(
            "This test tried to open a socket. The suite runs offline "
            "(see conftest). Stub the call, or use the `allow_network` fixture "
            "if the test is specifically about network behaviour."
        )

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def allow_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt back into real sockets, for a test that is *about* network failure.

    There is exactly one legitimate use: proving that an unreachable Ollama
    degrades correctly, which is only convincing against a genuinely closed
    port. Request this fixture and the refusal above is lifted for that test.
    """
    monkeypatch.undo()


@pytest.fixture(autouse=True)
def _canned_model_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/api/health` now asks Ollama whether the model is really there.

    That is a network call on a route almost every API and UI test hits, so it
    is canned here rather than in each of them. Tests that are *about* the probe
    call `probe_model` directly with their own stubs — the interesting cases are
    its answers, not its plumbing.
    """
    from stillroom import api

    monkeypatch.setattr(
        api, "probe_model", lambda _config: (True, "The model is loaded (stubbed).")
    )


class FakeChatModel:
    """A chat model that records its calls and returns a scripted reply."""

    def __init__(self, reply: str = "The refund window is 30 days [1].") -> None:
        self.reply = reply
        self.calls: list[Any] = []

    def invoke(self, messages: list) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.reply)

    async def astream(self, messages: list):
        self.calls.append(messages)
        for word in self.reply.split(" "):
            yield AIMessage(content=word + " ")


@pytest.fixture
def embedding() -> DeterministicEmbeddingFunction:
    return DeterministicEmbeddingFunction()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A small, structured corpus with headings the retriever can find."""
    root = tmp_path / "documents"
    root.mkdir()

    (root / "handbook.md").write_text(
        "# Staff handbook\n\n"
        "## Refund window\n\n"
        "Customers may request a refund within 30 days of purchase. "
        "Refunds after 30 days require a manager to approve them.\n\n"
        "## Notice period\n\n"
        "Employees must give four weeks of written notice before leaving.\n\n"
        "## Expense approval\n\n"
        "Any expense above 500 must be approved by the finance director.\n",
        encoding="utf-8",
    )
    (root / "shipping.txt").write_text(
        "Orders are dispatched within two working days.\n\n"
        "International shipping takes ten to fifteen working days to arrive.\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def config(tmp_path: Path, corpus: Path) -> ClientConfig:
    return ClientConfig.model_validate(
        {
            "client": "Test Ltd",
            "index_path": str(tmp_path / "index"),
            "corpus": {"path": str(corpus)},
            # The hashing embedder is lexical, so its similarities sit on a
            # different scale from MiniLM's — measured against this fixture,
            # relevant questions score 0.24-0.31 and irrelevant ones 0.09-0.13.
            # 0.20 separates them. The production default (0.25) is tuned for
            # MiniLM and is not what is being exercised here.
            "retrieval": {"k": 3, "min_similarity": 0.20},
            "answer_cache": {"curated": ("What is the refund window?",)},
        }
    )


@pytest.fixture
def fake_model() -> FakeChatModel:
    return FakeChatModel()
