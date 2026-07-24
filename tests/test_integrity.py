# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""The delivery manifest reports drift, and never does anything worse than that.

The tests that matter here are the NEGATIVE ones. A manifest is a document meant
to be used in a disagreement, so the ways it must not fail are:

* it must not cry breach when it cannot read itself,
* it must not cry breach over formatting or wording nobody agreed to,
* it must never block, modify or delete anything.

A false accusation is worse than a missed one: the client is right there, and
being told they tampered when they did not is the end of the engagement.
"""

from __future__ import annotations

import json

import pytest

from stillroom import integrity
from stillroom.config import ClientConfig


@pytest.fixture()
def config(tmp_path) -> ClientConfig:
    return ClientConfig.model_validate(
        {
            "client": "Acme Ltd",
            "index_path": str(tmp_path / "index"),
            "api_key": "k" * 20,
            "corpus": {"path": str(tmp_path / "documents")},
            "capacity": {"max_documents": 50, "warn_only": False},
            "answer_cache": {"curated": ["What is the refund window?"]},
            "ui": {"languages": ["en"]},
            "refresh": {"enabled": False},
        }
    )


def test_an_untouched_deployment_reports_no_drift(config):
    assert integrity.compare(integrity.build(config), config) == []


def test_raising_the_document_limit_is_reported(config):
    manifest = integrity.build(config)
    config.capacity.max_documents = 5000

    drifts = integrity.compare(manifest, config)

    assert [d.field for d in drifts] == ["max_documents"]
    assert drifts[0].delivered == 50
    assert drifts[0].current == 5000


def test_switching_on_scheduled_refresh_is_reported(config):
    manifest = integrity.build(config)
    config.refresh.enabled = True

    assert [d.field for d in integrity.compare(manifest, config)] == ["scheduled_refresh"]


def test_growing_the_prepared_set_is_reported(config):
    """Adding prepared questions is the likeliest quiet upgrade."""
    manifest = integrity.build(config)
    config.answer_cache.curated = tuple(f"Question {n}?" for n in range(50))

    drifts = integrity.compare(manifest, config)

    assert [d.field for d in drifts] == ["curated_answers"]
    assert (drifts[0].delivered, drifts[0].current) == (1, 50)


def test_adding_a_second_interface_language_is_reported(config):
    manifest = integrity.build(config)
    config.ui.languages = ("en", "pt-BR")

    assert [d.field for d in integrity.compare(manifest, config)] == ["ui_languages"]


def test_turning_the_capacity_limit_back_to_advisory_is_reported(config):
    """Softening enforcement is itself the change worth seeing."""
    manifest = integrity.build(config)
    config.capacity.warn_only = True

    assert [d.field for d in integrity.compare(manifest, config)] == ["capacity_enforced"]


def test_rewording_a_prepared_question_is_NOT_drift(config):
    """The manifest records the agreement, not the prose.

    A client improving the wording of their own question changed nothing that
    was agreed, and a report that flagged it would train them to ignore reports.
    """
    manifest = integrity.build(config)
    config.answer_cache.curated = ("How long do I have to ask for a refund?",)

    assert integrity.compare(manifest, config) == []


def test_a_manifest_from_an_unknown_schema_reports_no_drift(config):
    """Never invent a breach out of a document we failed to understand."""
    manifest = integrity.build(config)
    manifest["schema"] = 999
    config.capacity.max_documents = 5000

    assert integrity.compare(manifest, config) == []


def test_a_corrupt_manifest_is_absent_rather_than_damning(tmp_path):
    (tmp_path / integrity.MANIFEST_NAME).write_text("{not json", encoding="utf-8")

    assert integrity.load(tmp_path) is None


def test_no_manifest_at_all_loads_as_none(tmp_path):
    assert integrity.load(tmp_path) is None


def test_the_manifest_carries_no_document_content(config):
    """It travels to us in support bundles, so it must hold nothing confidential.

    The prepared questions are the client's own business questions -- read
    together they summarise what a business worries about -- so the manifest
    records how MANY there are and never what they say.
    """
    rendered = integrity.render(integrity.build(config))

    assert "refund" not in rendered.lower()
    assert json.loads(rendered)["agreed"]["curated_answers"] == 1


def test_doctor_reports_drift_without_failing_the_run(config, tmp_path):
    """Evidence, not enforcement: drift is a warning, never a `fail`."""
    from stillroom import doctor

    (tmp_path / integrity.MANIFEST_NAME).write_text(
        integrity.render(integrity.build(config)), encoding="utf-8"
    )
    config.capacity.max_documents = 5000

    checks = doctor.check_delivery(config, str(tmp_path))

    assert [c.status for c in checks] == ["warn"]
    assert "no longer matches" in checks[0].detail
    assert "5000" in checks[0].detail


def test_doctor_calls_a_missing_manifest_unverifiable_not_tampered(config, tmp_path):
    from stillroom import doctor

    checks = doctor.check_delivery(config, str(tmp_path))

    assert [c.status for c in checks] == ["warn"]
    assert "cannot be compared" in checks[0].detail
    for word in ("tamper", "breach", "violation"):
        assert word not in checks[0].detail.lower()
