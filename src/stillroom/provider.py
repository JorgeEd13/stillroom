# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""Build the chat model. One model. No fallback chain exists.

`receivables-agent` wires a primary and a fallback with LangChain's
`with_fallbacks`, and that is good engineering for a public demo — a free Space
that dies when Ollama is unreachable is a broken portfolio piece.

**Here it would be a data leak.** If the local model fails and a cloud fallback
picks up, the client's retrieved passages — the confidential documents they paid
to keep in the building — go to a third party, at the exact moment nobody is
watching, and the system reports success. The no-cloud-fallback rule requires that path to be
absent rather than disabled, so this module never constructs more than one
model and there is no code here that could chain them.

A local-model failure therefore raises. That is the correct behaviour: "the
model is down" is a problem the client can fix in a minute, and it is a strictly
better outcome than an answer that quietly cost them their compliance position.
"""

from __future__ import annotations

import logging
import os

from langchain_core.language_models.chat_models import BaseChatModel

from stillroom.config import AUTO_MODEL, ByokModel, ClientConfig, OllamaModel
from stillroom.hardware import (
    best_downloaded,
    ollama_answers,
    recommend_model,
    running_in_container,
)

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when the configured model cannot be constructed."""


def resolve_model_name(model: OllamaModel) -> str:
    """Resolve the `auto` sentinel against the machine that will run the model.

    A pinned tag is wrong on half the machines it ships to — too big and it will
    not load, too small and the client paid for answers worse than their
    hardware can produce. `auto` reads the actual memory and picks.

    ⚠️ **"The machine that will run the model" is not always this one, and
    getting that wrong shipped a real defect**. In the deliverable the
    engine is in a container and Ollama is on the host, so `detect_profile()`
    measures the container: no `nvidia-smi`, therefore no GPU, therefore the
    budget falls back to 80% of RAM — which is *larger* than the VRAM it should
    have used. Measured on the development machine: the host resolves
    `qwen2.5:7b` against a real 6 GB card, and the container resolves
    `qwen2.5:14b` against an imagined 11.7 GB. That is a ~9 GB download starting
    silently under the client's first question, for a model their card cannot
    hold.

    So the rule is **never guess about a machine you cannot see**:

    - Bare metal, Ollama here → hardware detection is valid, use it (unchanged).
    - Anything else → the only trustworthy fact is what that Ollama has
      **already pulled**, which is host truth and needs no measurement. Pick the
      best of those. If it has none of ours, raise: a loud error naming the
      command to run beats a silent multi-gigabyte download in front of the
      client, and it is the same reasoning the no-cloud-fallback rule applies to the missing
      fallback.

    ⚠️ **"In a container" is checked separately from "Ollama is remote", and it
    has to be.** At build time the image is built with `--network host`, so
    Ollama's address *is* `localhost` while the GPU is still invisible — the one
    combination that looks like a laptop and is not. Testing only the address
    would leave the build resolving one model and the runtime another, and that
    disagreement deletes the baked answers (see `running_in_container`).
    """
    if model.name != AUTO_MODEL:
        return model.name

    base_url = model.effective_base_url()

    if model.ollama_is_on_this_machine() and not running_in_container():
        chosen = recommend_model(base_url)
        logger.info("hardware-aware model selection resolved to %s", chosen)
        return chosen

    chosen = best_downloaded(base_url)
    if chosen is None:
        # ⚠️ **Two different worlds, and this message used to assert the wrong
        # one** (leg B #13). `best_downloaded` returns None both when that
        # Ollama has nothing useful and when it cannot be reached at all —
        # measured, the client was told to pull a model they already had, while
        # the actual fault was that a container cannot reach a service bound to
        # the host's loopback. One extra probe, two honest sentences.
        if not ollama_answers(base_url):
            raise ProviderError(
                f"Nothing is answering at {base_url} from this process. If "
                "Ollama is running, it is not reachable from here — a container "
                "cannot see a service bound to the host's loopback address. Set "
                "OLLAMA_HOST=0.0.0.0 on the Ollama host and allow the Docker "
                "bridge through its firewall, or point model.base_url at an "
                "address this process can actually reach."
            )
        raise ProviderError(
            f"model.name is 'auto', but the memory of the machine running Ollama "
            f"at {base_url} cannot be measured from this process — and that "
            "Ollama has none of the models this engine ships with pulled. Pull "
            "one there (for example `ollama pull qwen2.5:7b`), or pin model.name "
            "in the config to the tag it should use."
        )
    logger.info(
        "hardware is not measurable from here (%s); selected %s from what that "
        "Ollama has already pulled",
        base_url,
        chosen,
    )
    return chosen


def build_chat_model(config: ClientConfig) -> BaseChatModel:
    """Construct the single configured chat model."""
    model = config.model

    if isinstance(model, OllamaModel):
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - install-time failure
            raise ProviderError(
                "Local model support needs the 'ollama' extra (langchain-ollama)."
            ) from exc

        kwargs: dict = {}
        if model.num_ctx:
            kwargs["num_ctx"] = model.num_ctx
        if model.keep_alive:
            kwargs["keep_alive"] = model.keep_alive

        return ChatOllama(
            model=resolve_model_name(model),
            base_url=model.effective_base_url(),
            temperature=0,
            **kwargs,
        )

    if isinstance(model, ByokModel):
        return _build_byok_model(model)

    raise ProviderError(f"Unknown model configuration: {model!r}")


def _build_byok_model(model: ByokModel) -> BaseChatModel:
    """Construct a client-supplied cloud model.

    Reached only when a client explicitly asked for this in writing after a
    failed hardware check. The key is read from the
    environment; it is never in the config file.

    Each branch imports lazily and reports the missing extra by name. A client
    build installs **one** provider SDK, not four — and the person hitting the
    error is usually me, mid-build, wanting to know which `pip install` fixes it.
    """
    api_key = os.environ.get(model.api_key_env)
    if not api_key:
        raise ProviderError(
            f"Environment variable {model.api_key_env!r} is not set. "
            "This build is configured to use your own API key, and that key is "
            "read from the environment rather than stored in the config."
        )

    try:
        if model.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model.name, google_api_key=api_key, temperature=0
            )

        if model.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model.name, api_key=api_key, temperature=0)

        from langchain_openai import ChatOpenAI

        # `openai_compatible` is the same client pointed elsewhere — Groq,
        # Together, OpenRouter, Azure, or a vLLM on the client's own network.
        # `base_url=None` for plain "openai" means the SDK's own default.
        return ChatOpenAI(
            model=model.name,
            api_key=api_key,
            base_url=model.base_url,
            temperature=0,
        )
    except ImportError as exc:  # pragma: no cover - install-time failure
        extra = {
            "google": "langchain-google-genai",
            "anthropic": "langchain-anthropic",
            "openai": "langchain-openai",
            "openai_compatible": "langchain-openai",
        }[model.provider]
        raise ProviderError(
            f"Provider {model.provider!r} needs {extra}. "
            f"Install it with: pip install '.[byok]'  (or pip install {extra})"
        ) from exc


def model_label(config: ClientConfig) -> str:
    """A stable name for the configured model, for the cache's answer key.

    Includes `base_url` for an OpenAI-compatible endpoint: the same model name
    served by two different providers is two different models, and cached
    answers written by one must not be served as the other's.
    """
    model = config.model
    if isinstance(model, OllamaModel):
        return f"ollama:{resolve_model_name(model)}"
    if model.base_url:
        return f"{model.provider}:{model.base_url}:{model.name}"
    return f"{model.provider}:{model.name}"
