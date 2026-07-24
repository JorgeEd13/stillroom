"""Provider construction, including every bring-your-own-key branch.

⚠️ **Honest scope.** Only the `ollama` and `google` SDKs are installed in this
environment, so the OpenAI and Anthropic branches are verified by **dispatch**
— the right client, constructed with the right arguments — via injected stub
modules. That proves the routing and the argument wiring, which is where the
bugs in this file would be. It does **not** prove those SDKs behave as expected
against a live endpoint, and the first client build on one of them must be
rehearsed for real before handover.
"""

from __future__ import annotations

import sys
import types

import pytest

from stillroom.config import ByokModel, ClientConfig
from stillroom.provider import ProviderError, build_chat_model, model_label

BASE = {"client": "Acme", "corpus": {"path": "/tmp/docs"}}


def _config(**model) -> ClientConfig:
    return ClientConfig.model_validate({**BASE, "model": {"kind": "byok", **model}})


@pytest.fixture
def fake_sdks(monkeypatch):
    """Install stub provider SDKs and record how they were constructed."""
    calls: dict[str, dict] = {}

    def make(module_name: str, class_name: str, key: str):
        module = types.ModuleType(module_name)

        class Client:
            def __init__(self, **kwargs):
                calls[key] = kwargs

        setattr(module, class_name, Client)
        monkeypatch.setitem(sys.modules, module_name, module)

    make("langchain_openai", "ChatOpenAI", "openai")
    make("langchain_anthropic", "ChatAnthropic", "anthropic")
    make("langchain_google_genai", "ChatGoogleGenerativeAI", "google")
    return calls


def test_a_missing_key_fails_with_an_actionable_message(monkeypatch):
    monkeypatch.delenv("CLIENT_KEY", raising=False)
    config = _config(provider="google", name="gemini-2.0-flash", api_key_env="CLIENT_KEY")

    with pytest.raises(ProviderError, match="CLIENT_KEY"):
        build_chat_model(config)


def test_the_key_is_read_from_the_environment_not_the_config(monkeypatch, fake_sdks):
    monkeypatch.setenv("CLIENT_KEY", "secret-value")
    config = _config(provider="google", name="gemini-2.0-flash", api_key_env="CLIENT_KEY")

    build_chat_model(config)

    assert fake_sdks["google"]["google_api_key"] == "secret-value"
    # The secret must never be a field on the config object itself.
    assert "secret-value" not in config.model_dump_json()


def test_anthropic_is_supported(monkeypatch, fake_sdks):
    """A client arriving with an existing Anthropic contract is a real case."""
    monkeypatch.setenv("CLIENT_KEY", "sk-ant-xxx")
    config = _config(provider="anthropic", name="claude-sonnet-5", api_key_env="CLIENT_KEY")

    build_chat_model(config)

    assert fake_sdks["anthropic"]["model"] == "claude-sonnet-5"
    assert fake_sdks["anthropic"]["temperature"] == 0


def test_openai_is_supported(monkeypatch, fake_sdks):
    monkeypatch.setenv("CLIENT_KEY", "sk-xxx")
    config = _config(provider="openai", name="gpt-4o-mini", api_key_env="CLIENT_KEY")

    build_chat_model(config)

    assert fake_sdks["openai"]["model"] == "gpt-4o-mini"
    # None means "the SDK's own default endpoint".
    assert fake_sdks["openai"]["base_url"] is None


def test_an_openai_compatible_endpoint_is_pointed_at_the_given_base_url(monkeypatch, fake_sdks):
    """Covers Groq, Together, OpenRouter, Azure, or a vLLM on the client's LAN."""
    monkeypatch.setenv("CLIENT_KEY", "key")
    config = _config(
        provider="openai_compatible",
        name="llama-3.3-70b",
        api_key_env="CLIENT_KEY",
        base_url="https://api.groq.com/openai/v1",
    )

    build_chat_model(config)

    assert fake_sdks["openai"]["base_url"] == "https://api.groq.com/openai/v1"


def test_openai_compatible_without_a_base_url_is_refused():
    """Silently defaulting to OpenAI would send the client's documents to a
    vendor they did not choose."""
    with pytest.raises(Exception, match="base_url"):
        _config(provider="openai_compatible", name="x", api_key_env="K")


def test_the_model_label_separates_two_endpoints_serving_the_same_model_name():
    """Otherwise cached answers written by one would be served as the other's."""
    groq = _config(
        provider="openai_compatible",
        name="llama-3.3-70b",
        api_key_env="K",
        base_url="https://api.groq.com/openai/v1",
    )
    together = _config(
        provider="openai_compatible",
        name="llama-3.3-70b",
        api_key_env="K",
        base_url="https://api.together.xyz/v1",
    )

    assert model_label(groq) != model_label(together)


def test_a_managed_cloud_build_does_not_claim_the_documents_stay_put():
    """The handover runbook states this per client, so it must be derivable."""
    config = _config(provider="anthropic", name="claude-sonnet-5", api_key_env="K")

    assert not config.uses_local_model()
    assert not config.documents_stay_on_premises()
    assert "third-party" in config.privacy_posture()


def test_a_client_hosted_endpoint_KEEPS_the_privacy_claim():
    """The GPU-box case: their hardware, their network, documents never leave.

    Not "local" — the model is not on this machine — but the central claim of
    the central claim survives, and reporting otherwise understates what the client
    actually bought.
    """
    config = _config(
        provider="openai_compatible",
        name="llama-3.3-70b",
        api_key_env="K",
        base_url="http://gpu-box.internal:8000/v1",
        on_premises=True,
    )

    assert not config.uses_local_model()
    assert config.documents_stay_on_premises()
    assert "on-premises" in config.privacy_posture()


def test_on_premises_cannot_be_asserted_without_an_endpoint():
    """It is a claim ABOUT an address; without one it is a claim about nothing.
    A managed provider can never be on-premises."""
    with pytest.raises(Exception, match="on_premises"):
        _config(provider="anthropic", name="claude-sonnet-5", api_key_env="K", on_premises=True)


def test_on_premises_is_never_inferred_from_the_address():
    """Guessing wrong means telling a client their documents stay in the
    building when they do not. Somebody asserts it, explicitly."""
    looks_internal = _config(
        provider="openai_compatible",
        name="m",
        api_key_env="K",
        base_url="http://192.168.1.50:8000/v1",
    )

    assert not looks_internal.documents_stay_on_premises()


@pytest.mark.parametrize("provider", ["google", "openai", "anthropic"])
def test_every_provider_branch_is_reachable(monkeypatch, fake_sdks, provider):
    monkeypatch.setenv("CLIENT_KEY", "key")

    build_chat_model(_config(provider=provider, name="m", api_key_env="CLIENT_KEY"))

    assert provider in fake_sdks
