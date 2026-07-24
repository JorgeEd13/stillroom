"""Config: the structural guarantees, not the field list.

The important assertions here are the ones about what the config *cannot*
express. The no-cloud-fallback rule requires the silent cloud fallback to be absent rather than
disabled, and "absent" is only meaningful if it is checkable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stillroom.config import ByokModel, ClientConfig, OllamaModel

MINIMAL = {"client": "Acme", "corpus": {"path": "/tmp/docs"}}


def test_the_default_model_is_local():
    config = ClientConfig.model_validate(MINIMAL)

    assert isinstance(config.model, OllamaModel)
    assert config.uses_local_model()


def test_there_is_no_fallback_provider_field_at_all():
    """The no-cloud-fallback guarantee, asserted structurally.

    A config that cannot name a second provider cannot silently fall through to
    one. If someone adds a `fallback` field, this test is the tripwire.
    """
    fields = set(ClientConfig.model_fields)

    assert not {f for f in fields if "fallback" in f}
    assert not {f for f in OllamaModel.model_fields if "fallback" in f}


def test_extra_provider_config_is_rejected_rather_than_ignored():
    with pytest.raises(ValidationError):
        ClientConfig.model_validate(
            {**MINIMAL, "model": {"kind": "ollama", "fallback_provider": "gemini"}}
        )


def test_byok_must_be_selected_explicitly():
    config = ClientConfig.model_validate(
        {
            **MINIMAL,
            "model": {
                "kind": "byok",
                "provider": "google",
                "name": "gemini-2.0-flash",
                "api_key_env": "CLIENT_KEY",
            },
        }
    )

    assert isinstance(config.model, ByokModel)
    # The handover runbook states this per client; it must be derivable.
    assert not config.uses_local_model()


def test_an_inline_secret_in_the_api_key_env_field_is_refused():
    """A pasted key would be silently accepted and then silently committed."""
    with pytest.raises(ValidationError):
        ByokModel.model_validate(
            {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "api_key_env": "sk-" + "x" * 80,
            }
        )


def test_overlap_larger_than_the_chunk_is_refused():
    with pytest.raises(ValidationError):
        ClientConfig.model_validate(
            {**MINIMAL, "corpus": {"path": "/tmp/docs", "chunk_chars": 300, "chunk_overlap": 300}}
        )


def test_config_loads_from_toml(tmp_path):
    path = tmp_path / "client.toml"
    path.write_text(
        'client = "Acme"\n'
        '[corpus]\npath = "/tmp/docs"\n'
        "[answer_cache]\ncurated = [\"How do refunds work?\"]\n",
        encoding="utf-8",
    )

    config = ClientConfig.load(path)

    assert config.client == "Acme"
    assert config.answer_cache.curated == ("How do refunds work?",)
