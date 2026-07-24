"""`stillroom doctor` — one test per defect the check exists because of.

The rule this file is written under is the same one that shaped `doctor.py`: a
check earns its place by having failed in reality, so every test below names the
campaign finding it comes from. A test here that cannot name one is a test for a
check that should not exist.

Offline, like the rest of the suite (golden rule 7): no Ollama, no network, no
model. The container-side facts these checks depend on — "am I in a container",
"what has that Ollama pulled" — are exactly the things a delivered build gets
wrong, so they are injected rather than measured.
"""

from __future__ import annotations

import pytest

from stillroom import doctor
from stillroom.config import ClientConfig
from stillroom.doctor import Check


def _config(tmp_path, **overrides) -> ClientConfig:
    base = {
        "client": "Acme",
        "index_path": str(tmp_path / "index"),
        "corpus": {"path": str(tmp_path / "documents")},
        "model": {"kind": "ollama"},
    }
    base.update(overrides)
    return ClientConfig.model_validate(base)


def _status(checks: list[Check], name: str) -> str | None:
    for check in checks:
        if check.name == name:
            return check.status
    return None


def _detail(checks: list[Check], name: str) -> str:
    return next(c.detail + " " + (c.action or "") for c in checks if c.name == name)


# ----------------------------------------------------------- the embedder ---
#
# ⚠️ These exist because the check itself is new. Before `bge-m3` the
# embedder was an ONNX model inside the image and could not fail, so nothing
# checked it — and a check nobody thought to write is the same as a check that
# does not exist. The likely first-run failure on every delivery after this one
# is an ingest dying on a model the client was never told to pull.


def _embedding_raising(monkeypatch, exc):
    """Patched on the module, because `check_embedding` imports at call time."""
    from stillroom.index import embeddings

    def boom(config):
        def call(texts):
            raise exc

        return call

    monkeypatch.setattr(embeddings, "embedding_function_for", boom)


def test_an_unpulled_embedding_model_names_the_pull_command(tmp_path, monkeypatch):
    """The client pulled the CHAT model because the runbook said so. This is a
    second one, and nothing they have read mentions it."""
    from stillroom.index.embeddings import EmbeddingError

    _embedding_raising(
        monkeypatch,
        EmbeddingError("The embedding model 'bge-m3:567m' is not available on the Ollama"),
    )
    checks = doctor.check_embedding(_config(tmp_path))

    assert _status(checks, "embedding") == "fail"
    assert "ollama pull bge-m3:567m" in _detail(checks, "embedding")
    assert "SECOND model" in _detail(checks, "embedding")


def test_an_unreachable_embedding_ollama_is_a_DIFFERENT_diagnosis(tmp_path, monkeypatch):
    """Leg B #13's lesson, applied to the new dependency: "not pulled" and "not
    reachable" need opposite actions, and flattening them is what cost a
    release."""
    from stillroom.index.embeddings import EmbeddingError

    _embedding_raising(monkeypatch, EmbeddingError("Nothing is answering at http://x"))
    checks = doctor.check_embedding(_config(tmp_path))

    assert _status(checks, "embedding") == "fail"
    assert "ollama pull" not in _detail(checks, "embedding")


def test_a_zero_vector_from_the_embedder_is_a_failure(tmp_path, monkeypatch):
    """⚠️ The failure the offline fixture had, and it is silent by nature: a
    corpus of zero vectors indexes cleanly and retrieves at 0.0 against
    everything, so every question is refused and nothing reports a fault."""
    from stillroom.index import embeddings

    monkeypatch.setattr(
        embeddings, "embedding_function_for", lambda config: (lambda texts: [[0.0] * 8])
    )
    checks = doctor.check_embedding(_config(tmp_path))

    assert _status(checks, "embedding") == "fail"
    assert "empty vector" in _detail(checks, "embedding")


def test_a_working_embedder_reports_its_dimensions(tmp_path, monkeypatch):
    """Probed live rather than by listing tags: the tag list says the weights
    are on disk, not that the server can load them."""
    from stillroom.index import embeddings

    monkeypatch.setattr(
        embeddings, "embedding_function_for", lambda config: (lambda texts: [[0.1] * 1024])
    )
    checks = doctor.check_embedding(_config(tmp_path))

    assert _status(checks, "embedding") == "ok"
    assert "1024" in _detail(checks, "embedding")


# --------------------------------------------------------------- the model ---


def test_a_loopback_base_url_in_a_container_is_a_failure(tmp_path, monkeypatch):
    """Leg B #8 — the poisoned template, which no test over `src/` could see.

    The engine is right that config beats environment; what is wrong is a
    loopback address written into a config destined for a container, where it
    can only ever name the container.
    """
    monkeypatch.setattr(doctor, "running_in_container", lambda: True)
    monkeypatch.setattr(doctor, "list_downloaded", lambda url, timeout=2.0: set())
    monkeypatch.setattr(doctor, "ollama_answers", lambda url, timeout=3.0: False)

    checks = doctor.check_model(_config(tmp_path, model={"kind": "ollama", "base_url": "http://localhost:11434"}))

    assert _status(checks, "config.base_url") == "fail"
    assert "Delete the base_url line" in _detail(checks, "config.base_url")


def test_the_same_base_url_outside_a_container_is_fine(tmp_path, monkeypatch):
    """A laptop install is a legitimate deployment and must not be nagged."""
    monkeypatch.setattr(doctor, "running_in_container", lambda: False)
    monkeypatch.setattr(doctor, "list_downloaded", lambda url, timeout=2.0: {"qwen2.5:7b"})
    monkeypatch.setattr(doctor, "resolve_model_name", lambda model: "qwen2.5:7b")

    checks = doctor.check_model(_config(tmp_path, model={"kind": "ollama", "base_url": "http://localhost:11434"}))

    assert _status(checks, "config.base_url") is None


def test_a_gpu_box_elsewhere_is_never_flagged(tmp_path, monkeypatch):
    """The engagement the three privacy states exist for. A non-loopback address is the point."""
    monkeypatch.setattr(doctor, "running_in_container", lambda: True)
    monkeypatch.setattr(doctor, "list_downloaded", lambda url, timeout=2.0: {"qwen2.5:7b"})
    monkeypatch.setattr(doctor, "resolve_model_name", lambda model: "qwen2.5:7b")

    checks = doctor.check_model(
        _config(tmp_path, model={"kind": "ollama", "base_url": "http://gpu-box.internal:11434"})
    )

    assert _status(checks, "config.base_url") is None
    assert _status(checks, "ollama") == "ok"


def test_unreachable_and_empty_are_different_diagnoses(tmp_path, monkeypatch):
    """Leg B #13 — these were one message, and it asserted the wrong one.

    Measured against an Ollama holding three models, the client was told to
    `ollama pull` a model they already had.
    """
    monkeypatch.setattr(doctor, "running_in_container", lambda: True)
    monkeypatch.setattr(doctor, "list_downloaded", lambda url, timeout=2.0: set())

    monkeypatch.setattr(doctor, "ollama_answers", lambda url, timeout=3.0: False)
    unreachable = doctor.check_model(_config(tmp_path))

    monkeypatch.setattr(doctor, "ollama_answers", lambda url, timeout=3.0: True)
    empty = doctor.check_model(_config(tmp_path))

    assert "Nothing is answering" in _detail(unreachable, "ollama")
    assert "no models installed" in _detail(empty, "ollama")
    assert "ollama pull" in _detail(empty, "ollama")


def test_the_linux_loopback_trap_is_named_in_the_action(tmp_path, monkeypatch):
    """Leg B #11 — the defect that made the whole compose path dead on Linux.

    `host.docker.internal` is the docker0 bridge; Ollama binds 127.0.0.1. The
    remedy is one sentence and the client cannot guess it.
    """
    monkeypatch.setenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    monkeypatch.setattr(doctor, "running_in_container", lambda: True)
    monkeypatch.setattr(doctor, "list_downloaded", lambda url, timeout=2.0: set())
    monkeypatch.setattr(doctor, "ollama_answers", lambda url, timeout=3.0: False)

    checks = doctor.check_model(_config(tmp_path))

    assert "OLLAMA_HOST=0.0.0.0" in _detail(checks, "ollama")


def test_a_byok_build_is_not_probed(tmp_path):
    """Leg A's reasoning: a liveness check would spend the client's own credit."""
    checks = doctor.check_model(
        _config(
            tmp_path,
            model={
                "kind": "byok",
                "provider": "google",
                "name": "gemini-2.0-flash",
                "api_key_env": "CLIENT_API_KEY",
            },
        )
    )

    assert _status(checks, "model") == "warn"
    assert "CLIENT_API_KEY" in _detail(checks, "model")


# -------------------------------------------------------------- the corpus ---


def test_a_heading_in_two_documents_is_reported(tmp_path):
    """Leg B #14 — a stale handbook beside its replacement.

    Measured: the engine names the conflict when asked directly and buries it
    when asked as part of a compound question, once inventing a qualifier to
    reconcile the two. There is no answer-time fix; this is the delivery-time
    one.
    """
    from stillroom.ingest.loaders import RawDocument

    documents = [
        RawDocument(source="old.md", text="# Policy\n\n## Refund window\n\n14 days."),
        RawDocument(source="new.md", text="# Handbook\n\n## Refund window\n\n30 days."),
    ]

    checks = doctor._check_contradictions(documents)

    assert checks[0].status == "warn"
    assert "refund window" in checks[0].detail
    assert "old.md" in checks[0].detail and "new.md" in checks[0].detail


def test_distinct_headings_are_not_reported(tmp_path):
    from stillroom.ingest.loaders import RawDocument

    documents = [
        RawDocument(source="a.md", text="# A\n\n## Refunds\n\nx"),
        RawDocument(source="b.md", text="# B\n\n## Holidays\n\ny"),
    ]

    assert doctor._check_contradictions(documents)[0].status == "ok"


def test_a_portuguese_corpus_on_an_english_build_is_flagged(tmp_path):
    """Leg B #18 — measured at 0.076–0.220 against a 0.25 floor.

    Cross-language retrieval is absent, not weak, and nothing in the product
    said so. `language` sets the reply, never the search.
    """
    from stillroom.ingest.loaders import RawDocument

    documents = [
        RawDocument(
            source="politica.md",
            text=(
                "# Política de Reembolso\n\n"
                "O cliente pode solicitar reembolso em até 30 dias corridos "
                "após a entrega. Uma taxa de reposição de 10% é cobrada sobre "
                "os itens que não são devolvidos com a embalagem original."
            ),
        )
    ]

    check = doctor._check_language(documents, "en")

    assert check.status == "warn"
    assert "'pt'" in check.detail
    assert "cannot search across" in (check.action or "")


def test_a_matching_language_passes(tmp_path):
    from stillroom.ingest.loaders import RawDocument

    documents = [
        RawDocument(
            source="policy.md",
            text=(
                "# Refund Policy\n\nThe customer may request a refund within 30 "
                "days of delivery, and the restocking fee is charged for any "
                "item which is not returned with the original packaging."
            ),
        )
    ]

    assert doctor._check_language(documents, "en").status == "ok"


def test_a_regional_tag_matches_its_family(tmp_path):
    """`pt-BR` documents on a `pt-BR` build must not warn."""
    from stillroom.ingest.loaders import RawDocument

    documents = [
        RawDocument(
            source="politica.md",
            text=(
                "# Política\n\nO prazo de reembolso é de 30 dias para que o "
                "cliente possa solicitar, com uma taxa que não é cobrada."
            ),
        )
    ]

    assert doctor._check_language(documents, "pt-BR").status == "ok"


# --------------------------------------------------------------- the index ---


def test_a_missing_index_says_to_ingest(tmp_path):
    checks = doctor.check_index(_config(tmp_path))

    assert _status(checks, "index") == "fail"
    assert "stillroom ingest" in _detail(checks, "index")


def test_a_missing_documents_folder_is_a_failure(tmp_path):
    checks = doctor.check_corpus(_config(tmp_path))

    assert _status(checks, "corpus") == "fail"


# -------------------------------------------------------------- the report ---


def test_the_exit_status_distinguishes_a_warning_from_a_failure():
    """A launcher must not refuse to start over a heading collision."""
    warn_only = [Check("a", "ok", ""), Check("b", "warn", "")]
    with_failure = [Check("a", "ok", ""), Check("b", "fail", "")]

    assert doctor.worst(warn_only) == "warn"
    assert doctor.worst(with_failure) == "fail"
    assert doctor.worst([Check("a", "ok", "")]) == "ok"


def test_every_bad_line_carries_an_action():
    """The client reads this. A diagnosis they cannot act on is a support email."""
    report = doctor.format_report(
        [Check("ollama", "fail", "Nothing is answering.", "Start Ollama, then try again.")]
    )

    assert "-> Start Ollama" in report
    assert "1 problem(s) will stop this assistant working" in report


def test_a_clean_run_says_so_plainly():
    assert "Everything checks out." in doctor.format_report([Check("a", "ok", "fine")])


# ----------------------------------------------------- the shareable report ---
#
# The engagement is built on never receiving the client's documents, and this
# report is the ONE artifact that travels back to us. So the test is not "does
# --share look tidy" but "can a filename, a heading or a question survive it".


def _content_bearing_checks(config):
    """The three checks that quote the corpus, each with something to hide."""
    return [
        doctor.Check(
            "corpus.skipped",
            "warn",
            "2 file(s) are not in the assistant's knowledge: "
            "Q3 Redundancies.pdf; Merger Due Diligence.docx",
            "Convert them or supply a text original.",
            share_detail="2 file(s) are not in the assistant's knowledge "
            "(names withheld; run without --share to see them).",
        ),
        doctor.Check(
            "corpus.conflicts",
            "warn",
            "1 heading(s) appear in more than one document: "
            "'Severance terms' in Handbook.docx, Q3 Redundancies.pdf",
            "Remove the superseded document.",
            share_detail="1 heading(s) appear in more than one document "
            "(headings and filenames withheld).",
        ),
        doctor.Check(
            "curated",
            "fail",
            "1 curated question(s) find nothing in the documents: "
            "'How much severance is Roberto owed?'",
            "The documents that answer them may be missing.",
            share_detail="1 of 3 curated question(s) find nothing in the "
            "documents (questions withheld). Numbered from the list you "
            "supplied: 2.",
        ),
    ]


SECRETS = (
    "Q3 Redundancies.pdf",
    "Merger Due Diligence.docx",
    "Handbook.docx",
    "Severance terms",
    "Roberto",
)


def test_the_shareable_report_leaks_no_document_content():
    report = doctor.format_report(_content_bearing_checks(None), share=True)

    for secret in SECRETS:
        assert secret not in report, f"--share leaked {secret!r}"


def test_the_shareable_report_still_says_what_is_wrong():
    """A redacted report nobody can act on just gets sent unredacted instead."""
    report = doctor.format_report(_content_bearing_checks(None), share=True)

    assert "2 file(s) are not in the assistant's knowledge" in report
    assert "1 heading(s) appear in more than one document" in report
    # The index survives, so we can map it back to the list the client sent us.
    assert "Numbered from the list you supplied: 2." in report
    assert "content withheld" in report.lower()


def test_the_local_report_still_names_everything():
    """Redaction is for the copy that LEAVES. On their own machine, full detail."""
    report = doctor.format_report(_content_bearing_checks(None), share=False)

    for secret in SECRETS:
        assert secret in report


def test_a_check_without_a_share_version_is_shown_unchanged():
    """Checks that carry no document content need no redacted twin."""
    checks = [doctor.Check("ollama", "ok", "answering at http://host:11434")]

    assert "answering at http://host:11434" in doctor.format_report(checks, share=True)
