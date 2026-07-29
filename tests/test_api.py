"""The HTTP surface, exercised offline against a stub engine.

Auth is the one to care about. This service answers questions about the exact
documents the client paid to keep private, and it is deployed with nothing in
front of it — no gateway, no cloud identity, just whatever this file does.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from stillroom.api import create_app
from stillroom.config import ClientConfig
from stillroom.engine import Answer

API_KEY = "test-key-123"


class StubEngine:
    """Records the history it was handed, so the route's wiring is testable."""

    fingerprint = "abc123"

    def __init__(self) -> None:
        self.seen_history: tuple = ()

    def ask(self, question: str, *, history: tuple = ()) -> Answer:
        self.seen_history = history
        return Answer(
            text="30 days [1].",
            citations=[{"source": "handbook.md", "heading": "Refund window"}],
            served_by="model",
        )

    async def astream(self, question: str, history: tuple = (), *, use_cache: bool = True):
        self.seen_history = history
        self.seen_use_cache = use_cache
        yield {"type": "sources", "citations": []}
        yield {"type": "token", "text": "30 days"}
        yield {"type": "answer", "reply": "30 days [1].", "citations": []}


def _client(engine: Any, **ui: Any) -> TestClient:
    payload: dict[str, Any] = {
        "client": "Acme",
        "api_key": API_KEY,
        "corpus": {"path": "/tmp/docs"},
    }
    if ui:
        payload["ui"] = ui
    config = ClientConfig.model_validate(payload)
    # `base_url` matters: the app refuses unknown Host headers (DNS rebinding —
    # see `create_app`), and TestClient's default host is `testserver`. Pointing
    # it at a real allowed host is what keeps this suite honest about the guard
    # rather than quietly disabling it.
    return TestClient(
        create_app(config, engine_builder=lambda _: engine), base_url="http://localhost"
    )


@pytest.fixture
def client() -> Any:
    # The default fixture is `open` mode — the single-machine default. Entered as
    # a context manager so the lifespan actually runs — otherwise the engine is
    # never built and every route sees a 503.
    with _client(StubEngine()) as test_client:
        yield test_client


@pytest.fixture
def keyed_client() -> Any:
    # `key` mode: the shared-office-network deployment, where the key is the
    # boundary. The rejection tests belong here — asserting rejection in `open`
    # mode encoded the bug where open mode demanded a key it never sends
    # (ADR-069).
    with _client(StubEngine(), access="key") as test_client:
        yield test_client


def test_health_is_open_so_a_container_can_probe_it(client: TestClient):
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["fingerprint"] == "abc123"
    # Derived from the config, so the runbook's privacy claim is checkable.
    assert body["documents_stay_on_premises"] is True
    assert "local" in body["privacy_posture"]


def test_asking_without_a_key_is_rejected_in_key_mode(keyed_client: TestClient):
    response = keyed_client.post("/api/ask", json={"question": "refunds?"})

    assert response.status_code == 401


def test_asking_with_a_wrong_key_is_rejected_in_key_mode(keyed_client: TestClient):
    response = keyed_client.post(
        "/api/ask", json={"question": "refunds?"}, headers={"X-API-Key": "wrong"}
    )

    assert response.status_code == 401


def test_streaming_also_requires_a_key_in_key_mode(keyed_client: TestClient):
    """The streaming route is the one clients actually use — it must not be
    the one that forgot to check."""
    response = keyed_client.post("/api/ask/stream", json={"question": "refunds?"})

    assert response.status_code == 401


def test_open_mode_answers_the_ask_route_without_a_key(client: TestClient):
    """The default single-machine mode: the page sends no key by design
    (ADR-029), the boundary is the loopback binding, and requiring a key here
    bounced a client to a gate on their first question (ADR-069). Both ask
    routes must answer keyless in open mode."""
    assert client.post("/api/ask", json={"question": "refunds?"}).status_code == 200
    assert (
        client.post("/api/ask/stream", json={"question": "refunds?"}).status_code == 200
    )


def test_a_valid_key_gets_an_answer_with_citations(client: TestClient):
    response = client.post(
        "/api/ask", json={"question": "refunds?"}, headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "30 days [1]."
    assert body["citations"][0]["source"] == "handbook.md"
    assert body["served_by"] == "model"


def test_streaming_returns_sse_events_then_done(client: TestClient):
    response = client.post(
        "/api/ask/stream", json={"question": "refunds?"}, headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "sources"' in response.text
    assert '"type": "answer"' in response.text
    assert response.text.rstrip().endswith("[DONE]")


def test_an_empty_question_is_rejected_by_validation(client: TestClient):
    response = client.post(
        "/api/ask", json={"question": ""}, headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 422


def test_a_broken_stream_ends_with_an_error_event_not_a_hang():
    class BrokenEngine(StubEngine):
        async def astream(self, question: str, history: tuple = (), *, use_cache: bool = True):
            yield {"type": "sources", "citations": []}
            raise RuntimeError("ollama is not running")

    with _client(BrokenEngine()) as client:
        response = client.post(
            "/api/ask/stream",
            json={"question": "refunds?"},
            headers={"X-API-Key": API_KEY},
        )

    assert '"type": "error"' in response.text
    assert response.text.rstrip().endswith("[DONE]")


def test_the_regenerate_flag_reaches_the_engine_as_a_cache_bypass():
    """The UI's "answer again" control, end to end through the route.

    Without this the button renders, posts, and silently returns the same
    cached answer it was pressed to escape — which is the failure it exists to
    fix, wearing a working-looking UI.
    """
    engine = StubEngine()
    # As a context manager: the lifespan has to run or the engine is never
    # built and every route answers 503 instead of exercising the wiring.
    with _client(engine) as client:
        client.post("/api/ask/stream", json={"question": "What is the refund window?"})
        assert engine.seen_use_cache is True

        client.post(
            "/api/ask/stream",
            json={"question": "What is the refund window?", "fresh": True},
        )
        assert engine.seen_use_cache is False
