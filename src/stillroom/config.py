# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""Client configuration — a client is a config file, not a fork.

The whole economic premise of this lane is *build once, configure per client*. So everything that varies between engagements lives here, in a TOML
file, and nothing that varies lives in the engine's code.

Two structural choices in this module are load-bearing, not stylistic:

**1. There is no fallback provider field. At all.** `receivables-agent` ships
`PRIMARY_PROVIDER` / `FALLBACK_PROVIDER` with an active fallback, which is the
right default for a public demo. On a client build it is a data leak with a
config flag in front of it: when the local model fails, the client's retrieved
passages go to a cloud provider — precisely the thing they paid to prevent,
precisely when nobody is watching, and the system looks healthy while doing it.
The no-cloud-fallback rule requires that path to be **absent rather than disabled**, so `model` is
a single value, not a chain. A local-model failure fails loudly and locally
because there is nowhere for it to fall to.

**2. Bring-your-own-key is a separate `kind`, and it carries an env var name,
never a key.** A client who supplies their own key is making an informed choice — the same mechanism as the rejected fallback, the opposite
ethics, and the difference is entirely that somebody chose it in advance. It
is selectable only by writing it into the config on purpose.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

# The embedder owns its own name; this module only pins it as a default. The
# import is safe in this direction — `index.embeddings` reaches back into this
# module only from inside a function, precisely to keep it that way.
from stillroom.index.embeddings import DEFAULT_EMBEDDING_NAME

# The sentinel that turns on hardware-aware model selection: resolve the model
# from the machine's actual memory at startup rather than pinning a tag that is
# wrong on half the machines it ships to.
AUTO_MODEL = "auto"

# Where Ollama is when nothing says otherwise. Correct on a laptop; the
# deliverable overrides it via `OLLAMA_HOST` because a container's `localhost`
# is the container.
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def resolve_ollama_url(base_url: str | None) -> str:
    """Config → `OLLAMA_HOST` → loopback, in that order.

    Shared by the chat model and the embedding model because they resolve the
    same way and must not drift. Read at call time, not at load time, so one
    config file works on a laptop and in a container — which is the point.
    """
    if base_url:
        return base_url
    return os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL

# Theme colours are interpolated into a stylesheet, so they are matched rather
# than trusted. See `ThemeConfig`.
_HEX_COLOUR = re.compile(r"#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})")


class StrictModel(BaseModel):
    """Base for every config model: reject unknown keys rather than ignoring them.

    Two reasons, and the second is the real one. A typo (`chunk_char`) that is
    silently ignored costs an afternoon of wondering why a setting does nothing.
    And a config that *accepts* `fallback_provider` by ignoring it would let a
    future operator believe they had configured something the no-cloud-fallback rule requires to
    not exist. Unknown key, loud error.
    """

    model_config = ConfigDict(extra="forbid")


class OllamaModel(StrictModel):
    """A model served by a local Ollama. The default, and the whole pitch.

    ⚠️ **`base_url` is unset by default, and that is the fix for a delivered
    defect.** It used to default to `http://localhost:11434`, which is correct
    on a laptop and *wrong in the deliverable*: the engine runs in a container,
    where `localhost` is the container itself and Ollama is on the host. The
    compose file knew the right answer and exported `OLLAMA_HOST` — and nothing
    in this package ever read it. So every delivered build called an address
    with nothing behind it, and only the **baked** answers worked, because those
    never touch the model. Found 2026-07-21 by running the image, not reading it.

    **Resolution order, and why it is this order** (`effective_base_url`):

    1. an explicit `base_url` in the client's config,
    2. `OLLAMA_HOST` from the environment,
    3. `http://localhost:11434`.

    Config beats environment on purpose. The inverse is compose landmine 4 — an
    `environment:` block silently overriding a file somebody deliberately
    edited — and here it has a real client behind it: the engagement that points
    at **a GPU box elsewhere on their network** writes that address in the
    config, and a deployment default must not quietly redirect it. Unset means
    "wherever this deployment says", which is what a container should say.
    """

    kind: Literal["ollama"] = "ollama"
    base_url: str | None = None
    # "auto" picks the largest catalog model that fits the detected memory.
    name: str = AUTO_MODEL
    # Sizes the context window. None = whatever the Ollama server defaults to.
    num_ctx: int | None = None
    # Pins the model in memory between turns so the prompt cache on the long
    # system prefix survives — a near-free latency win on the local path, which
    # matters more here than in the demo because local *is* the product.
    keep_alive: str | None = "10m"

    def effective_base_url(self) -> str:
        """Where this deployment should actually call Ollama.

        Config → `OLLAMA_HOST` → loopback. See the class docstring for why the
        config wins. Read at call time rather than at load time so a container
        and a laptop can share one config file, which is the point.
        """
        return resolve_ollama_url(self.base_url)

    def ollama_is_on_this_machine(self) -> bool:
        """True when Ollama runs on the same machine as this process.

        The only consumer is `auto` model selection, and it is load-bearing
        there: reading RAM and VRAM tells you about **this** machine, and if
        the model runs somewhere else that reading describes the wrong
        computer. In the deliverable it always does — the engine is in a
        container and Ollama is on the host — which is exactly how `auto` came
        to recommend a 14B model for a 6 GB card.
        """
        host = urlparse(self.effective_base_url()).hostname or ""
        return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class ByokModel(StrictModel):
    """A cloud model the client pays for directly, with their own key.

    Only ever configured after a failed hardware check, and only in writing.
    It is never a default and never a suggestion.

    The key itself is **not** stored here — `api_key_env` names the environment
    variable to read it from, so a client config file can be committed to their
    own repo without becoming a secret.

    **Four providers, and the fourth is the one that matters.** A client who has
    already lost the local-hardware argument arrives with a vendor they have
    *already* chosen — an existing contract, a procurement decision, a data
    residency requirement. Being unable to serve that is a lost sale for a
    reason that is pure implementation. `openai_compatible` plus `base_url`
    covers most of the rest of the market (Groq, Together, OpenRouter, Mistral,
    DeepSeek, Azure OpenAI, or a vLLM the client hosts themselves) without this
    file growing a branch per vendor.
    """

    kind: Literal["byok"] = "byok"
    provider: Literal["google", "openai", "anthropic", "openai_compatible"]
    name: str
    api_key_env: str
    # Required for `openai_compatible`, ignored otherwise.
    base_url: str | None = None

    # ⚠️ **The privacy claim is not binary, and this flag is where that lives.**
    #
    # "Local" and "cloud" are the two cases the pitch talks about, but there is
    # a third and it is a real client: a **GPU box on their own network**. An
    # OpenAI-compatible server (vLLM, Ollama, LM Studio, TGI) running on their
    # hardware is a cloud *SDK* talking to *their* machine — so the documents
    # never leave the building and the central claim survives intact, even
    # though `kind = "byok"`.
    #
    # It is **explicit and defaults to False** rather than inferred from the
    # address. Guessing from a private IP range would be wrong in both
    # directions — a VPN'd cloud host looks private, an on-prem box behind an
    # internal DNS name looks public — and the failure mode of guessing wrong is
    # telling a client their confidential documents stay in the building when
    # they do not. **That claim is never inferred. Somebody asserts it.**
    on_premises: bool = False

    @model_validator(mode="after")
    def _refuse_inline_secrets(self) -> "ByokModel":
        # A key pasted into `api_key_env` would be silently accepted and then
        # silently committed. Names are short and have no secret shape.
        if len(self.api_key_env) > 64 or " " in self.api_key_env:
            raise ValueError(
                "api_key_env must be the NAME of an environment variable, not a key."
            )
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError(
                "provider='openai_compatible' requires base_url (the endpoint to call)."
            )
        if self.on_premises and not self.base_url:
            # on_premises asserts "this endpoint is the client's own hardware".
            # Without an address that assertion is about nothing — and the
            # managed providers cannot be on-premises by definition.
            raise ValueError(
                "on_premises=true requires base_url — it asserts that a specific "
                "endpoint is the client's own hardware. A managed provider "
                "(google/openai/anthropic) can never be on-premises."
            )
        return self


ModelConfig = Annotated[OllamaModel | ByokModel, Field(discriminator="kind")]


class CorpusConfig(StrictModel):
    """Where the client's documents are and how they get chunked."""

    # Directory scanned recursively at ingest. Never inside this repo.
    path: str
    # Extensions to ingest. Scanned images are out of scope: these
    # loaders read digital text and do no OCR.
    include: tuple[str, ...] = (".md", ".txt", ".csv", ".pdf", ".docx", ".xlsx")
    # ⚠️ Force a text encoding instead of detecting one. Left unset,
    # `.txt` and `.md` are decoded by byte-order mark, then strict UTF-8, then a
    # detector restricted to Western codepages. That detector was measured
    # getting a short Brazilian document wrong (cp1250 for cp1252, so `ção`
    # became `ăo`), and a client corpus exported wholesale from one legacy
    # system is exactly the case where the answer is already known. Naming it
    # here beats guessing per file. Any codec Python knows: "cp1252",
    # "iso-8859-1", "cp1251", "shift_jis".
    encoding: str | None = None
    # Target chunk size in characters, for documents with no heading structure.
    chunk_chars: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)

    @model_validator(mode="after")
    def _overlap_fits(self) -> "CorpusConfig":
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("chunk_overlap must be smaller than chunk_chars.")
        return self


class ConversationConfig(StrictModel):
    """How much of the conversation the engine carries.

    Defaults are deliberately small. The context window is a fixed budget shared
    with the retrieved passages, and the passages are the part that must not be
    squeezed — a follow-up that fits at the cost of the paragraph it should have
    quoted is a worse answer, not a longer one.
    """

    enabled: bool = True
    # Exchanges kept. Six covers "and after that?" chains without letting a long
    # session push the sources out of the window.
    max_turns: int = Field(default=6, ge=0, le=20)
    # Per question and per answer. A follow-up needs an earlier answer's
    # *topic*, which is at the front of it, not the whole text.
    max_chars_per_turn: int = Field(default=600, ge=100, le=4000)


class StandingContextConfig(StrictModel):
    """A small always-present document: the client's own source of truth.

    It plays the same role a project's own `MEMORY.md` plays for a
    working session. Facts that are true of the *organisation* rather than of any
    one document — what the company does, what its abbreviations mean, which of
    two conflicting handbooks supersedes the other, who "the board" is. Retrieval
    cannot supply these reliably because nobody writes a document titled *"things
    everyone here already knows"*, and their absence is what makes an assistant
    read as an outsider.

    It is configuration, but producing it is an interview — extracting what a client believes is too obvious to write down
    is the slowest hour of an engagement and the one that most changes the
    result.

    ⚠️ **It is background, not evidence.** It is never cited, because it was
    never retrieved, and an answer that rests on it alone is an answer with no
    source behind it. `prompts` states that boundary to the model in as many
    words, and `max_chars` keeps it from crowding out the passages that *are*
    evidence.
    """

    path: str | None = None
    max_chars: int = Field(default=2000, ge=100, le=8000)


class EmbeddingConfig(StrictModel):
    """The model that turns documents and questions into vectors.

    ⚠️ **A separate axis from `model`, and it has to be.** The obvious shortcut
    is to borrow the chat model's `base_url`, and it breaks the one engagement
    shape that most needs the config to be explicit: a **BYOK** client has no `OllamaModel` at all, and under the old embedder they
    needed no Ollama either. Now they do — embedding is local by construction,
    because an embedder calling a cloud service would ship every chunk of the
    corpus off the machine at ingest time, which is the central claim.

    **So this is a prerequisite change, not a config field.** Ollama is now
    required for *every* build including BYOK ones, and that belongs in the
    qualifier and the runbook, not only here.
    """

    # ⚠️ Pinned by version. Changing it is a corpus-invalidating
    # change — it is in the fingerprint — so it is a per-engagement decision
    # made once, at build, not a knob to turn on a running deployment.
    model: str = DEFAULT_EMBEDDING_NAME
    # Unset means "wherever this deployment says", exactly as for the chat
    # model: config → OLLAMA_HOST → loopback. See `OllamaModel` for the
    # delivered defect that ordering exists to fix.
    base_url: str | None = None

    def effective_base_url(self) -> str:
        return resolve_ollama_url(self.base_url)


class RetrievalConfig(StrictModel):
    """How much context an answer is allowed to stand on.

    ⚠️ **`min_similarity` was re-derived for `bge-m3` and it moved a long way**. It is measured, not chosen — reproduce it with
    `python benchmarks/retrieval_floor.py --sweep`, and do not adjust it for a
    client without re-running that.
    """

    k: int = Field(default=5, ge=1, le=20)
    # Cosine similarity a chunk must reach to count as relevant at all. Below
    # this for *every* chunk, the engine says it does not know instead of
    # asking the model to improvise over unrelated text (see `engine`).
    #
    # ⚠️ **0.25 was MiniLM's, and it gated nothing under either model.** Under
    # bge-m3 all 27 cells of the language benchmark passed it. And measured on a
    # *realistic* corpus rather than four paragraphs, MiniLM's own far band
    # reached **0.521** — so the shipped floor was not a working gate before the
    # swap either. That was never visible because it had only ever been measured
    # on four paragraphs.
    #
    # 0.50 is the middle of the band that admits all 36 on-topic questions and
    # refuses all 12 far-band ones (workable range 0.450–0.550). It also refuses
    # 11 of 14 plausible-but-absent questions, which the old floor never did.
    min_similarity: float = Field(default=0.50, ge=0.0, le=1.0)
    # ⚠️ **A relative margin was measured as the compensating stage and
    # REJECTED**. It is not a knob that was left off — it is a design
    # that the data says is worse than nothing, and `index.retrieval` records
    # why so nobody adds it back on intuition.
    #
    # What *is* here instead: below the floor, the question is retried once
    # against the client's own corpus vocabulary. See `SpellingRetryConfig`.
    spelling_retry: "SpellingRetryConfig" = Field(default_factory=lambda: SpellingRetryConfig())


class SpellingRetryConfig(StrictModel):
    """One retry against the corpus vocabulary before refusing.

    **The problem it solves.** Raising the floor to 0.50 buys a working refusal
    gate and costs the typo tolerance F3.5 measured: a heavily mistyped question
    scores 0.314, below the floor, and is refused as though the documents did
    not cover it. And no floor fixes that — the *far* band reaches 0.428, so
    mangled-but-real scores **below** unrelated-but-clean. That conflict exists
    for all three embedders tested; it is a property of multilingual embedders,
    not of bge-m3.

    **Why it is a retry and never a rewrite.** The first version corrected the
    question and searched with the correction, and it made clean questions
    *worse*: `what` → `that`, because question words are not in a corpus of
    policies. So the corrected form is only ever tried as a **second** search,
    and the better of the two results wins — a bad correction can then only be
    discarded, never applied.

    **Why the corpus is the dictionary.** No word list ships, in any language.
    The vocabulary is the client's own indexed text, so `refnud` → `refund`
    works for the same reason `apolice` → `apólice` would: the right word is in
    their documents. It cannot invent a word the corpus does not contain, which
    is what keeps it from rescuing nonsense.

    **Measured** (bge-m3, realistic corpus, floor 0.50): the worst typo case
    goes 0.314 → 0.555, and **no far-band or adjacent question is lifted over
    the floor at any parameter setting tested**. Costs nothing on the happy
    path — it only runs when the floor has already refused.
    """

    enabled: bool = True
    # Shorter tokens are skipped: at three or four characters, edit distance
    # stops discriminating and every short word has a neighbour in the corpus.
    min_token_length: int = Field(default=5, ge=3, le=12)
    # `difflib` similarity a candidate must reach to be considered a correction
    # of a token. Measured insensitive between 0.75 and 0.80; the higher value
    # is kept because the failure it guards is a false correction.
    cutoff: float = Field(default=0.80, ge=0.5, le=1.0)


class AnswerCacheConfig(StrictModel):
    """The curated instant answers.

    Whole answers are cached here, which `receivables-agent` deliberately does
    not do. That is not a relaxation of its rule, it is the same rule reaching a
    different conclusion on different data: it caches the *plan* and re-runs it
    live because a ledger is mutable and a frozen number lies. A document corpus
    is immutable *between ingests*, so the whole answer is exact until the
    documents change — and busting the cache on re-ingest is what keeps that
    true. See `answers.cache`, where the fingerprint check lives.
    """

    enabled: bool = True
    collection: str = "answer_cache"
    # Cosine similarity a question must reach to reuse a cached answer. High on
    # purpose: a miss just costs a model call, a false hit answers the wrong
    # question confidently.
    threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    # The client's own most-asked questions, baked at build time.
    # MUST be written in the language their documents are in —
    # these are cache *keys*, matched by embedding against the corpus, so a
    # translated question silently misses and drops every asker onto the slow
    # model, which is the exact failure this feature exists to prevent.
    curated: tuple[str, ...] = ()


class RefreshConfig(StrictModel):
    """Scheduled refresh: keeping the index current, unattended.

    Off by default: a deployment that did not ask for it must not get a background thread
    re-embedding their corpus. Enabled per engagement, like everything else.
    """

    enabled: bool = False
    # How often to re-check the corpus. Hourly is the sensible floor — an ingest
    # that finds nothing changed is cheap, but one that finds a 1,500-document
    # corpus rewritten is not, and on a CPU-only box that competes with the
    # assistant the client is trying to use.
    interval_minutes: int = Field(default=60, ge=5, le=10_080)


class CapacityConfig(StrictModel):
    """The agreed document limit, enforced at ingest.

    A corpus size is agreed before the work starts. Without a check here that
    is a number in a document that nobody ever compares against reality — and the direction it fails is always the same:
    the corpus grows past what was agreed, the ingest quietly succeeds, and
    the limit stops existing.

    `warn_only` is the default because discovering this mid-delivery should
    start a conversation, not break the build.
    """

    max_documents: int | None = None
    warn_only: bool = True


class ThemeConfig(StrictModel):
    """How the client's assistant *looks* — every value becomes a CSS variable.

    **This table is the front-end's whole customisation surface, and it is a
    surface on purpose.** The visible product is what a non-technical buyer
    forms an opinion from, so "make it match our brand" has to be a config edit
    an hour before handover, not a code change and a rebuild. Every token below
    lands in the stylesheet the server renders at `/ui/theme.css`; nothing in
    the UI's own CSS hard-codes a colour, a radius or a duration.

    **Enums, not free strings, everywhere except the colours.** Two reasons and
    both are load-bearing:

    1. **CSS injection.** A custom property is interpolated into a stylesheet,
       so a value like `red; } body { display: none` would escape the rule it
       was meant to fill. Colours are therefore matched against a strict hex
       pattern and everything else is a closed set.
    2. **Nothing leaves the building — including the UI.** A free-text font
       family invites a webfont, and a webfont is the client's browser telling
       a third party, on every single page view, which internal tool they are
       using. The font choices here resolve to stacks already on the machine.

    `custom_css` is the escape hatch for anything the tokens do not cover. It
    is CSS, never JS, and it is read from a path **outside this repo** (a client's brand assets are client material and never become fixtures here).
    """

    # Which appearance the page opens in. "system" follows the operating
    # system; the in-page toggle overrides it and persists per browser.
    mode: Literal["system", "light", "dark"] = "system"

    # The base palette. Each preset ships a light *and* a dark pair, because a
    # client build cannot assume which one an employee's laptop is set to.
    preset: Literal["slate", "warm", "forest", "mono"] = "slate"

    # Optional per-client accent overrides, one per appearance. Left unset, the
    # preset's own accents are used. Hex only — see the class docstring.
    accent: str | None = None
    accent_dark: str | None = None

    # Resolved to a font stack of faces that already exist on the machine.
    font: Literal["system", "serif", "mono", "rounded"] = "system"
    # The radius family: soft (8px), sharp (0), pill (999px).
    shape: Literal["soft", "sharp", "pill"] = "soft"
    # Comfortable is right for a browser tab kept open all day; compact fits
    # more history on a laptop screen.
    density: Literal["comfortable", "compact"] = "comfortable"
    # The page backdrop. All four are drawn in CSS — no image is fetched.
    background: Literal["plain", "gradient", "grid", "dots"] = "plain"
    # Motion budget. "none" is not only an accessibility setting: on the
    # CPU-only hardware this product targets, animation competes with the model.
    # `prefers-reduced-motion` is honoured on top of whatever is set here.
    effects: Literal["full", "subtle", "none"] = "full"

    # Optional client logo shown in the header. A path to a .svg or .png next
    # to the config file. Rendered in an <img>, which does not execute script
    # even when the file is an SVG.
    logo: str | None = None
    # Optional stylesheet appended after the generated tokens, so it can
    # override any of them. Path to a .css file next to the config.
    custom_css: str | None = None

    @model_validator(mode="after")
    def _validate_colours_and_assets(self) -> "ThemeConfig":
        for field_name in ("accent", "accent_dark"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not _HEX_COLOUR.fullmatch(value):
                raise ValueError(
                    f"{field_name} must be a hex colour such as '#3b5bdb'. Got "
                    f"{value!r}. Values are interpolated into a stylesheet, so "
                    "arbitrary CSS is not accepted."
                )
        if self.logo is not None and not self.logo.lower().endswith((".svg", ".png")):
            raise ValueError("logo must be a .svg or .png file.")
        if self.custom_css is not None and not self.custom_css.lower().endswith(".css"):
            raise ValueError("custom_css must be a .css file.")
        return self


class UiConfig(StrictModel):
    """The web page the client's team opens — the thing they actually bought.

    Every deployment gets the page, so this is not optional scope;
    `languages` is what a bilingual deployment means in practice.

    ⚠️ **`access` is the honest name for a real trade-off.** The default,
    `open`, asks the visitor for nothing: they open the address and it works,
    which is the only acceptable experience for a non-technical team. The
    boundary in that mode is the **network binding** — the compose file
    publishes on loopback, so "can reach the page" means "is sitting at this
    machine". Ask the browser for nothing *and* publish on the office network
    and there is no boundary at all, which is what `key` exists for: the team
    pastes, once per browser, the key that already lives in the config file.

    The key is never embedded in the page in either mode. A page that carries
    the credential that guards it is not authenticated, it is decorated.
    """

    enabled: bool = True

    # Header text. Defaults are derived from `client` when these are unset, so
    # a minimal config still produces a page with the client's name on it.
    title: str | None = None
    intro: str | None = None

    # UI chrome languages, in menu order; the first is the default. One locale
    # means no language switcher is rendered at all.
    #
    # ⚠️ **Chrome only.** Answers come back in the language the client's
    # *documents* are written in, and are never machine-translated. Translating
    # the interface must not imply translated answers — the UI says so out
    # loud, in the same place the switcher lives.
    languages: tuple[str, ...] = ("en",)

    access: Literal["open", "key"] = "open"

    # Host headers this service will answer to. A browser enforces same-origin
    # against a *name*, so a page on the public internet that resolves its own
    # hostname to 127.0.0.1 can otherwise reach a loopback-only service in the
    # employee's browser and read the answers (DNS rebinding). Any real hostname
    # or reverse-proxy name the client uses must be added here.
    allowed_hosts: tuple[str, ...] = (
        "localhost",
        "127.0.0.1",
        "[::1]",
        "host.docker.internal",
    )

    # Show the curated questions as one-click chips. The instant-answer set is
    # a feature nobody can see was not delivered.
    suggestions: bool = True

    theme: ThemeConfig = Field(default_factory=ThemeConfig)

    @model_validator(mode="after")
    def _validate_languages(self) -> "UiConfig":
        from stillroom.ui import AVAILABLE_LOCALES

        if not self.languages:
            raise ValueError("ui.languages must list at least one locale.")
        unknown = [lang for lang in self.languages if lang not in AVAILABLE_LOCALES]
        if unknown:
            raise ValueError(
                f"Unknown UI locale(s): {', '.join(unknown)}. Available: "
                f"{', '.join(sorted(AVAILABLE_LOCALES))}. An extra interface "
                "language is a translated string file, added per engagement."
            )
        if len(set(self.languages)) != len(self.languages):
            raise ValueError("ui.languages contains a duplicate.")
        return self


class ClientConfig(StrictModel):
    """One engagement, end to end."""

    client: str
    # UI/answer language hint, passed to the model. Answers come out in the
    # documents' language regardless; this only steers phrasing.
    language: str = "en"

    model: ModelConfig = Field(default_factory=OllamaModel)
    # Separate from `model` on purpose — see `EmbeddingConfig`. A BYOK build has
    # no chat-model Ollama to borrow an address from, and still needs this one.
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    corpus: CorpusConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    answer_cache: AnswerCacheConfig = Field(default_factory=AnswerCacheConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    context: StandingContextConfig = Field(default_factory=StandingContextConfig)
    refresh: RefreshConfig = Field(default_factory=RefreshConfig)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)
    ui: UiConfig = Field(default_factory=UiConfig)

    # Persistent index location. Client data — outside this repo, always.
    index_path: str = "index"
    collection: str = "documents"
    # Auth for the local HTTP service. Local-network-only, but "local network"
    # is not a security boundary in an office.
    api_key: str = "change-me"

    # Where this config was read from. Client-supplied assets (a logo, a brand
    # stylesheet) are named relative to their own config file, because that is
    # how a person thinks about a folder they were handed — not relative to
    # whatever directory the service happened to be started from.
    _source_dir: Path | None = PrivateAttr(default=None)

    @classmethod
    def load(cls, path: str | Path) -> "ClientConfig":
        """Read a client config from TOML."""
        source = Path(path)
        raw = source.read_bytes()
        config = cls.model_validate(tomllib.loads(raw.decode("utf-8")))
        config._source_dir = source.parent.resolve()
        return config

    def resolve_asset(self, name: str) -> Path:
        """A client-supplied file, resolved next to their config file."""
        base = self._source_dir or Path.cwd()
        return (base / name).resolve()

    def uses_local_model(self) -> bool:
        """True when the model runs on this very machine.

        Narrower than `documents_stay_on_premises` — a client-hosted GPU box is
        not "local" but the documents still never leave their building. Use this
        only when the distinction is genuinely *this machine*; use the other for
        anything that touches the privacy claim.
        """
        return self.model.kind == "ollama"

    def documents_stay_on_premises(self) -> bool:
        """True when nothing this build does sends text to a third party.

        **This is the product's central claim, made checkable.** The handover
        runbook states it per client and the rehearsal asserts it, so
        it must be derived from the config rather than remembered.

        Three cases, not two: a local Ollama, a client-hosted
        OpenAI-compatible endpoint on their own network, or a managed cloud
        provider. The first two keep the claim; only the third breaks it.
        """
        if self.model.kind == "ollama":
            return True
        return bool(self.model.on_premises)

    def privacy_posture_kind(self) -> str:
        """Which of the three postures this build is, as a token to translate.

        The sentence below is written in English for the runbook and the health
        route. The *page* cannot use it: a bilingual build would
        then show the one sentence the whole claim rests on in the wrong
        language, in the header, above every answer. So the UI is handed this
        token plus `privacy_posture_detail()` and composes the sentence from its
        own string file.
        """
        if self.model.kind == "ollama":
            return "local"
        return "on_premises" if self.model.on_premises else "third_party"

    def privacy_posture_detail(self) -> str | None:
        """The endpoint or provider named in the posture sentence, if any."""
        if self.model.kind == "ollama":
            return None
        return self.model.base_url if self.model.on_premises else self.model.provider

    def privacy_posture(self) -> str:
        """One line for the runbook and the health endpoint.

        Written for the client to read, because at handover they do.
        """
        if self.model.kind == "ollama":
            return "local — the model runs on this machine; documents never leave it"
        if self.model.on_premises:
            return (
                f"on-premises — the model runs on your own network ({self.model.base_url}); "
                "documents never leave your building"
            )
        return (
            f"third-party — documents are sent to {self.model.provider} using your own "
            "API key, as you requested"
        )
