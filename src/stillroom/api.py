# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""The local HTTP service the client's browser talks to.

Deliberately small. It binds to the client's own machine or their own network,
and every design choice here follows from that being the *whole* deployment —
there is no gateway in front of it, no cloud auth, and nobody else to blame for
a default.

* **The engine is built once**, in `lifespan`. Constructing it opens the index
  and resolves the model against the hardware; doing that per request would add
  seconds to every question on exactly the hardware that can least afford it.
* **API key on every data route.** "It's only on the local network" is not a
  security boundary in an office, and this service answers questions about the
  documents the client most wanted kept private.
* **Streaming is the primary route.** A local model is slow; the stream is what
  makes that legible as progress rather than as a hang.
* **The engine is injectable**, so the whole HTTP stack is testable offline
  against a stub — no model, no network.
"""

from __future__ import annotations

import json
import secrets
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from stillroom import ui as ui_assets
from stillroom.config import ClientConfig, OllamaModel
from stillroom.conversation import Turn
from stillroom.engine import Engine
from stillroom.hardware import list_downloaded
from stillroom.provider import ProviderError, resolve_model_name

# The only files this service will ever hand out, by exact name. A directory
# mount would serve whatever lands in that folder later; this list cannot.
_STATIC = {
    "app.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "favicon.svg": "image/svg+xml",
}

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

EngineBuilder = Callable[[ClientConfig], Any]


class HistoryTurn(BaseModel):
    """One earlier exchange, as the page remembers it."""

    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # ⚠️ **The conversation lives in the browser, not on the server** (ADR-046).
    #
    # It keeps this service stateless — no sessions, no per-user storage, no
    # transcript of the client's most confidential questions sitting in a volume
    # after they close the tab — and it makes *New chat* real in one line: the page
    # drops what it holds and there is nothing else to clear.
    #
    # The cost is that history is user-composed, so it is untrusted input: it is
    # length-capped here, turn-capped in `conversation.trim`, and
    # marker-stripped in `prompts`. In a single-tenant local deployment a user
    # who forges their own history has misled only themselves (ADR-045).
    history: tuple[HistoryTurn, ...] = ()
    # "Answer this again, properly." Set by the regenerate control in the UI.
    #
    # ⚠️ It exists because a wrong answer is CACHED. Every live answer is warmed,
    # so a confidently wrong one is served instantly and identically to everyone
    # afterwards, and until this there was no way for anybody to dislodge it
    # short of a re-ingest. On a small local model that is how one bad answer
    # becomes permanent (found 2026-07-29, `dgp-05`).
    #
    # Safe to expose: a single-tenant service on the client's own hardware,
    # where the worst a user can do with it is spend their own CPU.
    fresh: bool = False

    def turns(self) -> tuple[Turn, ...]:
        return tuple(Turn(question=t.question, answer=t.answer) for t in self.history)


class AskResponse(BaseModel):
    # ⚠️ `reply` IS MARKDOWN. Any UI consuming it MUST render it.
    #
    # This is landmine 3 from `sistema/achados/receivables-agent.md`: the
    # showcase's chat UI renders answers as plain text, so `**bold**` and `##`
    # appear literally on screen. It reads as unpolished, and it is not
    # cosmetic — it is what **blocked using the live demo screenshot as the
    # product screenshot**, because the one image meant to show quality
    # showed raw syntax.
    #
    # The model emits Markdown and instructing it not to would cost real
    # formatting in long answers, so the fix belongs in the renderer. Stated
    # here because this schema is what the UI is written against.
    reply: str
    citations: list[dict]
    served_by: str
    curated: bool


class HealthResponse(BaseModel):
    ready: bool
    # Surfaced so an operator can confirm the running service is serving the
    # corpus they think it is — the first thing to check when a client says
    # "it's answering from the old handbook".
    fingerprint: str | None = None
    # The central claim, derived from the config rather than promised.
    # True for a local model AND for a client-hosted endpoint on their own
    # network — both keep the documents in the building (ADR-023).
    documents_stay_on_premises: bool
    # The same fact in a sentence the client can read.
    privacy_posture: str

    # ⚠️ **`ready` above is not what the word suggests, and this pair exists
    # because that gap shipped** (ADR-039). `ready` means "the engine object was
    # constructed" — it never touches the model, so a build whose Ollama is
    # unreachable reports `ready: true`, serves the page, answers **baked**
    # questions correctly, and fails on the first real one. The runbook sends
    # the client to this exact URL to check the thing is working.
    #
    # `ready` keeps its meaning rather than being widened: the container's
    # healthcheck and the launcher's wait loop both read this route, and a model
    # that is merely restarting must not read as a dead service. So the model is
    # a **separate** fact, and this route stays 200 either way.
    #
    # None = not probed. A bring-your-own-key build is never probed, because
    # "check the model is reachable" would mean a billable call to the client's
    # own account every 30 seconds.
    model_reachable: bool | None = None
    # The same fact in a sentence, and when it is False it names the fix.
    model_status: str

    # ⚠️ **`fingerprint` above is what this process is SERVING, and it can be
    # older than the index on disk** (leg B #17). It is captured when the engine
    # is built and cannot move afterwards, because the answer cache is keyed on
    # it. So the field whose docstring promises an operator can "confirm the
    # running service is serving the corpus they think it is" was, on its own,
    # unable to notice the one case it exists for.
    #
    # Both paths reach it and neither announces itself: the documented
    # runbook has the client re-ingest from a second container, and the Advanced
    # refresh re-ingests on a timer with no restart at all — after which this route
    # reported a corpus several refreshes old, indefinitely, with nothing wrong
    # on the surface.
    #
    # None = the index could not be read.
    corpus_current: bool | None = None
    # The same fact in a sentence, and when it is False it names the fix.
    corpus_status: str


def _default_builder(config: ClientConfig) -> Any:
    return Engine(config)


def _corpus_state(engine: Any) -> tuple[bool | None, str]:
    """Is the running service serving the documents that are on disk?

    Split out from the route because the sentence matters more than the flag:
    "your documents were re-read but the assistant is still using the old ones"
    is a thing the client can act on in ten seconds, and it is invisible
    otherwise — no crash, no error, just answers about a handbook they replaced.
    """
    if engine is None:
        return None, "The assistant is still starting up."

    current = getattr(engine, "serving_current_corpus", None)
    if current is None:
        return None, "The document index could not be read."
    if current:
        return True, "The assistant is answering from your current documents."
    return False, (
        "Your documents have been re-read since the assistant started, so it is "
        "still answering from the previous versions. Restart it (stop, then "
        "start) to pick up the changes."
    )


def probe_model(config: ClientConfig) -> tuple[bool | None, str]:
    """Can this build actually reach the model it is configured to use?

    Read-only and cheap: it lists the tags the target Ollama has, which answers
    both halves of the question — the service is up, *and* the model this build
    resolves to is present on it. Those are the two ways the delivered container
    fails while looking healthy.

    Every message here is read by the client, so each one says what to do next
    rather than what went wrong.
    """
    model = config.model
    if not isinstance(model, OllamaModel):
        return None, (
            "This build uses your own API key, so the assistant does not test "
            "the connection on its own."
        )

    base_url = model.effective_base_url()
    tags = list_downloaded(base_url)
    if not tags:
        return False, (
            f"The local model service is not answering at {base_url}. "
            "Open Ollama, wait a few seconds, then reload this page."
        )

    try:
        resolved = resolve_model_name(model)
    except ProviderError as exc:
        return False, str(exc)

    if resolved not in tags:
        return False, (
            f"The local model service is running, but the model this assistant "
            f"uses ({resolved}) is not installed on it. Run: ollama pull {resolved}"
        )

    return True, f"The model {resolved} is loaded and running on this machine."


def create_app(
    config: ClientConfig, engine_builder: EngineBuilder = _default_builder
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine_builder(config)

        # Scheduled refresh lives inside the process the operator
        # already runs, so there is no second per-OS artifact to install on a
        # machine where the rule is verify-never-install (ADR-003).
        scheduler = None
        if config.refresh.enabled:
            from stillroom.refresh import RefreshScheduler

            scheduler = RefreshScheduler(
                config, interval_s=config.refresh.interval_minutes * 60
            )
            scheduler.start()
        app.state.refresh = scheduler

        yield

        if scheduler is not None:
            scheduler.stop()
        app.state.engine = None

    app = FastAPI(
        title=f"{config.client} — document assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ⚠️ The Host header is checked because "it only listens on localhost" is not
    # the boundary it sounds like. A page on the public internet can point its
    # own hostname at 127.0.0.1 (DNS rebinding) and then talk to this service
    # from inside an employee's browser, with the browser's same-origin rules
    # satisfied — reading answers about the documents this whole product exists
    # to keep in the building. Refusing unknown Host values costs nothing and
    # closes it. A client behind a reverse proxy adds their name to
    # `ui.allowed_hosts`.
    #
    # There is deliberately **no CORS middleware anywhere in this file.** Without
    # `Access-Control-Allow-Origin`, a browser refuses to let another site read
    # a response from this one. Adding it "to make testing easier" would undo
    # the protection above; a test asserts it stays absent.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(config.ui.allowed_hosts))

    # Client-supplied brand assets are read once, at startup, rather than per
    # request: a missing file then fails the build loudly instead of showing a
    # broken image at handover, and no request ever reaches the filesystem with
    # a path derived from anything but this config.
    logo_bytes: bytes | None = None
    logo_type = "image/svg+xml"
    if config.ui.theme.logo:
        logo_path = config.resolve_asset(config.ui.theme.logo)
        if not logo_path.is_file():
            raise FileNotFoundError(
                f"ui.theme.logo points at {logo_path}, which does not exist."
            )
        logo_bytes = logo_path.read_bytes()
        logo_type = "image/png" if logo_path.suffix.lower() == ".png" else logo_type

    custom_css = ""
    if config.ui.theme.custom_css:
        css_path = config.resolve_asset(config.ui.theme.custom_css)
        if not css_path.is_file():
            raise FileNotFoundError(
                f"ui.theme.custom_css points at {css_path}, which does not exist."
            )
        custom_css = css_path.read_text(encoding="utf-8")

    def require_api_key(key: str | None = Security(_API_KEY_HEADER)) -> None:
        # In `open` access mode the boundary is the network binding, not a key
        # (ADR-029): the page is published on loopback, "can reach it" means
        # "is at this machine", and the UI deliberately sends no `X-API-Key`.
        # Requiring one here anyway is what bounced an open-mode client to a gate
        # on their first question — the server demanded a key the page is
        # designed never to send. The key gate is a `key`-mode feature; here it
        # must stand down, or open mode cannot answer at all.
        if config.ui.access == "open":
            return
        # `key` mode: constant-time compare so a wrong key cannot be guessed
        # byte by byte.
        if not key or not secrets.compare_digest(key, config.api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key.",
            )

    def get_engine(request: Request) -> Any:
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The assistant is not ready.",
            )
        return engine

    # Deliberately sync: `probe_model` makes a blocking call with a short
    # timeout, and on an `async def` route that would stall the event loop for
    # every other request. FastAPI runs a plain `def` in a threadpool.
    @app.get("/api/health", response_model=HealthResponse, tags=["meta"])
    def health(request: Request) -> HealthResponse:
        engine = getattr(request.app.state, "engine", None)
        reachable, status = probe_model(config)
        current, corpus_status = _corpus_state(engine)
        return HealthResponse(
            ready=engine is not None,
            fingerprint=getattr(engine, "fingerprint", None),
            documents_stay_on_premises=config.documents_stay_on_premises(),
            privacy_posture=config.privacy_posture(),
            model_reachable=reachable,
            model_status=status,
            corpus_current=current,
            corpus_status=corpus_status,
        )

    @app.post(
        "/api/ask",
        response_model=AskResponse,
        tags=["ask"],
        dependencies=[Depends(require_api_key)],
    )
    async def ask(req: AskRequest, engine: Any = Depends(get_engine)) -> AskResponse:
        answer = engine.ask(req.question, history=req.turns())
        return AskResponse(
            reply=answer.text,
            citations=answer.citations,
            served_by=answer.served_by,
            curated=answer.curated,
        )

    @app.post(
        "/api/ask/stream",
        tags=["ask"],
        dependencies=[Depends(require_api_key)],
    )
    async def ask_stream(
        req: AskRequest, engine: Any = Depends(get_engine)
    ) -> StreamingResponse:
        """Server-Sent Events: `cached` / `sources` / `token` / `answer` / `error`."""

        async def event_source() -> AsyncIterator[str]:
            def sse(obj: dict[str, Any]) -> str:
                return f"data: {json.dumps(obj)}\n\n"

            try:
                async for event in engine.astream(
                    req.question, req.turns(), use_cache=not req.fresh
                ):
                    yield sse(event)
            except Exception as exc:  # never leave the stream half-open
                yield sse({"type": "error", "message": str(exc)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/suggestions", tags=["ask"], dependencies=[Depends(require_api_key)])
    async def suggestions() -> dict[str, list[str]]:
        """The curated questions, for the one-click chips.

        ⚠️ **Behind the key, unlike the rest of the page's chrome.** This list is
        the client's own most-asked questions, and read together it is a summary
        of what the business worries about — which supplier, which clause, which
        formula. The interface strings are not sensitive; this is.

        It also serves as the gate's verification call: the browser checks a key
        here before storing it, so a wrong key is rejected at the door rather
        than on the team's first real question.
        """
        if not config.ui.suggestions:
            return {"questions": []}
        return {"questions": list(config.answer_cache.curated)}

    if not config.ui.enabled:
        # A headless build — the client integrates the API themselves. Nothing
        # below is registered, so the page genuinely does not exist rather than
        # being served and hidden.
        return app

    @app.get("/", response_class=HTMLResponse, tags=["ui"], include_in_schema=False)
    async def index() -> Response:
        return Response(
            content=ui_assets.read_asset("index.html"),
            media_type="text/html; charset=utf-8",
            # The page is a shell; the client's own text arrives from /api/ui.
            # Never cached, so a re-delivered build is never a stale page.
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/ui/theme.css", tags=["ui"], include_in_schema=False)
    async def theme_css() -> Response:
        """The client's look, rendered from their config on every request.

        Generated rather than stored so that changing an accent colour is an
        edit to a TOML file and a browser refresh — no rebuild, no bundler, no
        container image. That is the whole reason this UI has no build step.
        """
        return Response(
            content=ui_assets.render_theme_css(config),
            media_type="text/css; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/ui/custom.css", tags=["ui"], include_in_schema=False)
    async def custom_stylesheet() -> Response:
        """The per-client escape hatch, loaded last so it can override anything.

        Empty unless the engagement supplied one. It is CSS and never JS: a
        client's brand file can restyle every element on the page without being
        able to touch what the page *does* with their documents.
        """
        return Response(content=custom_css, media_type="text/css; charset=utf-8")

    @app.get("/ui/logo", tags=["ui"], include_in_schema=False)
    async def logo() -> Response:
        if logo_bytes is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return Response(content=logo_bytes, media_type=logo_type)

    @app.get("/ui/{name}", tags=["ui"], include_in_schema=False)
    async def static_asset(name: str) -> Response:
        media_type = _STATIC.get(name)
        if media_type is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return Response(content=ui_assets.read_asset(name), media_type=media_type)

    @app.get("/api/ui", tags=["ui"])
    async def ui_config() -> dict[str, Any]:
        """Everything the page needs before anybody has authenticated.

        Open by necessity — in `access = "key"` mode the browser has to render
        the key prompt, in the client's language, before it has a key. It
        carries chrome and the privacy sentence, and deliberately **not** the
        curated questions (see `/api/suggestions`).
        """
        return ui_assets.ui_payload(config)

    return app
