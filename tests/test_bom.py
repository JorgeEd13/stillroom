"""The §6.2 bill of materials.

These tests guard a **contractual** artifact, so they are less about code being
correct than about the document never asserting something nobody checked. The
failure mode that matters is a bill of materials that looks complete and is not.
"""

from __future__ import annotations

import pytest

from stillroom import bom, hardware
from stillroom.config import ClientConfig


@pytest.fixture
def ollama_config(config: ClientConfig) -> ClientConfig:
    """The default engagement: a local model, pinned so the test never depends
    on what this machine happens to be able to run."""
    return config.model_copy(
        update={"model": config.model.model_copy(update={"name": "llama3.1:8b"})}
    )


def test_every_row_fills_in_all_four_required_fields(ollama_config):
    """§6.2 asks for name+version, licence, modified, and how incorporated. A
    blank in any of them is a row that does not discharge the obligation."""
    for item in bom.build(ollama_config):
        assert item.name
        assert item.version
        assert item.licence
        assert item.modified
        assert item.incorporation


def test_the_engine_row_states_the_licence_the_client_actually_gets(ollama_config):
    engine = bom.build(ollama_config)[0]

    assert engine.name == "stillroom"
    assert "PolyForm Shield 1.0.0" in engine.licence
    # Never "open source" — it is not OSI-approved, and the client's own lawyer
    # reading it as open source is the misunderstanding worth pre-empting.
    assert "NOT open source" in engine.licence


def test_the_model_row_carries_use_terms_and_not_distribution_terms(ollama_config):
    """We ship no weights — the client's Ollama pulls them. Telling a client
    they must display "Built with Llama" on their website would be inventing an
    obligation, which is the exact thing §6.2 forbids."""
    (model,) = [i for i in bom.build(ollama_config) if i.name == "llama3.1:8b"]

    assert model.obligations == hardware.LLAMA_31_COMMUNITY.use_obligations
    assert not any("Built with Llama" in ob for ob in model.obligations)
    assert "not redistributed by Freelancer" in model.incorporation


def test_a_model_outside_the_vetted_catalog_blocks_delivery(config):
    """A config may pin anything. Anything we never read the licence for must
    stop the delivery rather than appear in the document as a fact."""
    pinned = config.model_copy(
        update={"model": config.model.model_copy(update={"name": "mystery-model:70b"})}
    )
    (offender,) = [i for i in bom.blocked(bom.build(pinned)) if "mystery" in i.name]

    assert "nobody has read its licence" in offender.blocked_because


def test_the_embedding_model_is_in_the_document_at_all(ollama_config):
    """It ships in EVERY build, unlike the chat model — and it was missing from
    every licence conversation until §6.2's "other third-party materials" was
    read literally."""
    (row,) = [
        i for i in bom.build(ollama_config) if bom.DEFAULT_EMBEDDING_NAME in i.name
    ]

    # Named by its UPSTREAM model, because that is whose licence governs it.
    assert bom._EMBEDDING_UPSTREAM in row.name


def test_the_embedding_weights_are_not_credited_to_a_library_s_licence(ollama_config):
    """A model's licence is never its host library's, or its runner's.

    When the embedder was Chroma's MiniLM the trap was "ChromaDB is Apache-2.0";
    now that it is served by Ollama the same trap is "Ollama is MIT". Both
    describe *code*. The weights are a separate work under the terms of the
    repository that published them, and the row must name that repository."""
    (row,) = [
        i for i in bom.build(ollama_config) if bom.DEFAULT_EMBEDDING_NAME in i.name
    ]

    assert bom._EMBEDDING_UPSTREAM in row.name
    assert "chromadb" not in row.licence.lower()
    assert "ollama" not in row.licence.lower()


def test_the_embedding_row_is_named_by_VERSION_not_by_family(ollama_config):
    """⚠️ The licence-verification gate: "bge-m3" is a family, not a version.

    Llama 3 / 3.1 / 3.2 / 3.3 are four different agreements that disagree with
    each other, and the discipline learned there applies to every model in the
    document. The tag is what the config pins and what the client actually
    pulls, so the tag is what the bill of materials has to say."""
    (row,) = [
        i for i in bom.build(ollama_config) if bom.DEFAULT_EMBEDDING_NAME in i.name
    ]

    assert ":" in bom.DEFAULT_EMBEDDING_NAME, "the pinned name carries no version"
    assert bom.DEFAULT_EMBEDDING_NAME in row.version


def test_the_embedding_weights_are_NO_LONGER_redistributed_by_us(ollama_config):
    """⚠️ **This assertion is the exact inverse of the one it replaces**, and
    that is the point.

    Shipping the base image as a tarball made us the redistributor of the
    embedding weights, and this test used to assert that the row said so.
    `bge-m3` takes the weights back out of the image — the client's own Ollama
    pulls them, exactly as it pulls the chat model — so the load-bearing
    sentence *"we never distribute weights"* is true again for every model.

    **Stating an obligation that does not exist is the same defect as omitting
    one that does.** A row still claiming we hand these over, or still pointing
    at THIRD-PARTY-NOTICES.md for a model that is no longer in the image, would
    be a contractual document describing a delivery that did not happen."""
    (row,) = [
        i for i in bom.build(ollama_config) if bom.DEFAULT_EMBEDDING_NAME in i.name
    ]

    text = " ".join(row.obligations)
    assert "Redistributed to you" not in text
    assert "THIRD-PARTY-NOTICES.md" not in text
    assert "None fall on you from us" in text
    # The client obtains them the same way as the chat model, and the row says so.
    assert "own distribution channel" in text
    assert "NOT inside the image" in row.incorporation


def test_the_chat_model_is_still_NOT_redistributed(ollama_config):
    """The other half, and the reason this is one item rather than a policy
    change. The client's own Ollama still pulls the chat model, so Llama's and
    Gemma's Distribution-triggered conditions still attach to nobody here."""
    (row,) = [i for i in bom.build(ollama_config) if "llama3.1" in i.name.lower()]

    assert "not redistributed by Freelancer" in row.incorporation


def test_nothing_in_a_default_build_is_left_uncertified(ollama_config):
    """The state to hold: a default engagement generates a bill of materials
    that can actually be handed over.

    It was RED until 2026-07-21 on the embedding model, and the way it went
    green matters — **not** by someone deciding ChromaDB's Apache-2.0 covered
    the weights (it does not; they are not in the wheel), but by archiving a
    document stating the licence of *that model*.

    ⚠️ The suite runs from a source checkout, where the dependency closure
    genuinely cannot be read — so that one blocker is expected here and is the
    guard working. What must be empty is everything else."""
    blocked = bom.blocked(bom.build(ollama_config))
    environmental = [i for i in blocked if "not installed in this environment" in i.blocked_because]

    assert blocked == environmental, "something is blocked for a LICENCE reason"


def test_the_embedding_licence_cites_the_document_it_was_read_from(ollama_config):
    """Same rule the model catalogue lives under: a legal fact needs a quote.

    ⚠️ **This pair is the trap the embedder swap named, and it is why the test is
    explicit about both halves.** `_EMBEDDING_LICENCE` was `Apache-2.0`, which
    is MiniLM's; `bge-m3` is MIT. Swapping the embedder and leaving either
    constant behind would put a **false licence statement into a contractual
    document** — and nothing else in the suite would have noticed, because the
    document renders perfectly well with the wrong licence in it.

    Two sources, kept apart on purpose: the model card is
    a *declaration* of the licence, the FlagEmbedding file is the licence
    *text*, and they live in different repositories under different copyright
    holders. Collapsing them would hide a seam that was deliberately named."""
    assert bom._EMBEDDING_LICENCE == "MIT"
    assert bom._EMBEDDING_LICENCE_SOURCE.startswith("legal/")
    assert bom._EMBEDDING_LICENCE_TEXT_SOURCE.startswith("legal/")
    assert "bge-m3" in bom._EMBEDDING_LICENCE_SOURCE


def test_the_embedding_licence_and_the_embedder_cannot_drift_apart(ollama_config):
    """The guard the last swap did not have.

    A licence constant is only true of one model. Tying the two together in an
    assertion means the next embedder change fails here rather than shipping a
    document about the model it replaced."""
    assert bom.DEFAULT_EMBEDDING_NAME.split(":")[0] in bom._EMBEDDING_LICENCE_SOURCE
    assert bom.DEFAULT_EMBEDDING_NAME.split(":")[0] in bom._EMBEDDING_UPSTREAM.lower()


def test_a_byok_engagement_does_not_claim_to_license_the_provider(config):
    """The client holds that account and that agreement. Passing a licence
    through would be asserting a right we do not have."""
    # Validated rather than `model_copy`d: an update dict is not coerced, so the
    # config would keep a plain dict where a ByokModel belongs.
    byok = ClientConfig.model_validate(
        {
            "client": config.client,
            "index_path": config.index_path,
            "corpus": {"path": config.corpus.path},
            "model": {
                "kind": "byok",
                "provider": "anthropic",
                "name": "claude-opus-4-8",
                "api_key_env": "ANTHROPIC_API_KEY",
                "on_premises": False,
            },
        }
    )
    (row,) = [i for i in bom.build(byok) if "anthropic" in i.name]

    assert "client's own agreement" in row.licence
    assert "No licence is granted or passed through" in row.licence


def test_the_rendered_document_lists_every_item_and_its_obligations(ollama_config):
    items = bom.build(ollama_config)
    document = bom.render(items, "Acme Ltd")

    assert "# Bill of materials — Acme Ltd" in document
    assert "section 6.2" in document
    for item in items:
        assert item.name in document
    # The obligations the client inherits get their own section, in words.
    assert "Terms that apply to you" in document
    assert "Acceptable Use Policy" in document


def test_a_pipe_in_a_field_cannot_break_the_table(ollama_config):
    """A licence string containing `|` would silently mangle the Markdown table
    and drop columns from a document a lawyer reads."""
    item = bom.BomItem(
        name="weird",
        version="1|2",
        licence="MIT | Apache-2.0",
        modified="No.",
        incorporation="test",
    )
    row = [
        line for line in bom.render([item], "X").splitlines() if line.startswith("| weird")
    ][0]

    # Count only the pipes that actually delimit a cell — an escaped `\|` is
    # content. 5 cells => 6 delimiters.
    delimiters = len([
        i for i, ch in enumerate(row) if ch == "|" and (i == 0 or row[i - 1] != "\\")
    ])

    assert delimiters == 6
    assert r"\|" in row


def test_project_scoped_copyleft_would_stop_a_delivery(monkeypatch):
    """§6.2 forbids delivering anything that obliges the CLIENT to license or
    disclose their own IP. Nothing in the closure hits this today — the guard is
    for the dependency added in a year, when this reasoning is gone.

    ⚠️ Weak (file-scoped) copyleft is a different thing and must NOT block:
    `certifi`, `orjson` and `tqdm` are MPL-2.0, whose §3.3 expressly permits
    combining into a Larger Work under other terms. Blocking those would stop
    every delivery over a non-problem."""
    assert bom._STRONG_COPYLEFT.search("GPL-3.0-only")
    assert bom._STRONG_COPYLEFT.search("AGPL-3.0")
    assert bom._STRONG_COPYLEFT.search("SSPL-1.0")

    for permitted in ("MPL-2.0 AND MIT", "Apache-2.0", "BSD-3-Clause", "MIT"):
        assert not bom._STRONG_COPYLEFT.search(permitted), permitted


def test_the_real_closure_carries_no_blocking_licence():
    """Run against whatever is actually installed. If a future dependency bump
    pulls in strong copyleft or something undeclared, this is what says so."""
    offenders = [i for i in bom._dependency_items() if i.blocked_because]

    assert offenders == [] or all(
        "not installed in this environment" in i.blocked_because for i in offenders
    ), [(i.name, i.licence) for i in offenders]


def test_an_installed_extra_is_kept_but_an_uninstalled_one_is_dropped(monkeypatch):
    """The delivered image installs `.[ollama,docs]`, so pypdf, python-docx and
    openpyxl are redistributed in the tarball and MUST be declared. The walker
    used to skip EVERY `extra ==` requirement, so the loaders shipped undeclared
    for months. The filter is resolution, not the word "extra": an
    extra that is actually installed is kept; one that is not is dropped."""
    from importlib import metadata as md

    real = md.distribution

    class FakeRoot:
        # `certifi` really is installed here; declaring it through an extra must
        # still include it. The made-up package resolves to nothing and is gone.
        requires = ["certifi ; extra == 'docs'", "no-such-pkg-zzz ; extra == 'dev'"]

    def fake(name: str):
        return FakeRoot() if name == "stillroom" else real(name)

    monkeypatch.setattr("stillroom.bom.metadata.distribution", fake)

    closure = bom.dependency_distributions()

    assert "certifi" in closure  # installed extra -> kept
    assert "no-such-pkg-zzz" not in closure  # unresolved extra -> dropped
