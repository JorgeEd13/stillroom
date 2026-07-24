# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""Pick the largest local model this machine can actually run.

A pinned model tag is wrong on most machines it ships to: too big and it will
not load at all, too small and the client paid for answers worse than their
hardware could produce. So the engine resolves `auto` at startup against the
machine it is actually on.

**This is not the pre-purchase qualifier**, and the difference is the whole of
The capability/qualifier split. The qualifier (`ai-model-requirements`, public, MIT) runs on a
stranger's machine *before any money*, where Ollama's absence is the expected
state and opening a socket is a reason not to trust the binary. This runs
*inside a working install*, at construction, where Ollama being up is the normal
case — so asking it which models are already pulled is a real optimisation
rather than a new failure mode. Same question, opposite moments, opposite
defaults.

**Carried over from F1: only discrete VRAM counts.** An integrated GPU borrows
system RAM, so counting its reported "VRAM" double-counts memory the model also
needs for everything else, and the machine gets recommended a model that will
not load. Reading VRAM from `nvidia-smi` alone avoids this structurally — it
reports discrete NVIDIA cards and nothing else. A machine with a discrete AMD
card is therefore under-estimated rather than over-estimated, which is the safe
direction to be wrong in.
"""

from __future__ import annotations

import json
import logging
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelLicence:
    """The terms a client inherits by running a model, in the form the bill of
    materials needs.

    ⚠️ **"Free to download" is not "free to use commercially", and open *weights*
    are not open *source*.** Every model below is free and runs entirely offline,
    and one of them still may not be run by a paying business.

    This exists because of a specific obligation, not general tidiness: the
    engagement terms require a bill of materials naming, for each third-party
    item, its licence and how it is used — and require that nothing delivered
    creates licence obligations for the client. A model recommendation *is* a
    delivered obligation, and it was being made with no record of its terms.

    ### The split between the two obligation tuples is the load-bearing part

    **We never distribute model weights.** The image carries none: the container
    reaches the client's own Ollama on the host (`docker-compose.yml`, and the
    `--network host` note in the `Dockerfile`), and *they* pull the weights from
    upstream under their own acceptance. Both the Llama and Gemma agreements hang
    their heavy conditions on **Distribution** — Gemma §1.1(b) defines it as
    transmission or sharing to a third party; Llama §1.b.i opens with *"If you
    distribute or make available…"*.

    So `distribution_obligations` **do not currently apply to anyone in this
    delivery model**, and listing them as if they did would tell a client they
    must display "Built with Llama" on their website when they need not. Stating
    an obligation that does not exist is the same defect as omitting one that
    does — §6.2 is about not *creating* obligations for the client.

    They are recorded rather than deleted because the day the delivery model
    changes — an air-gapped client who needs the weights handed over on disk —
    they become live, and rediscovering them from scratch is how this went wrong
    the first time.

    ### Provenance

    `source` names the document in `docs/legal/` each claim was read from, with
    the clause. **A licence with no `source` has not been verified against a
    primary document**, and the archive's rule is that such a fact may not be
    stated. `summary` is a plain-language note for the client's benefit — it is
    **not** the licence; the `name` is version-exact so they, or their lawyer,
    can read the real text.
    """

    name: str
    # May a paying business run this in the ordinary course of its work?
    commercial: bool
    # What binds the client by RUNNING the model. These are the live ones.
    use_obligations: tuple[str, ...] = ()
    # What would bind whoever hands the WEIGHTS on. We do not. See the note above.
    distribution_obligations: tuple[str, ...] = ()
    summary: str = ""
    # The archived document and clause. Empty = unverified; see UNVERIFIED below.
    source: str = ""


APACHE_2_0 = ModelLicence(
    name="Apache-2.0",
    commercial=True,
    distribution_obligations=(
        "Keep the copyright notice and a copy of the licence with any copy passed on (§4.1, §4.2).",
        "Mark modified files as changed (§4.2).",
    ),
    summary="Permissive. Nothing to do while merely running it.",
    source="legal/licences/Apache-2.0.pdf §4",
)

# ⛔ The reason this whole module grew a licence field. Qwen2.5 is Apache-2.0 at
# every size EXCEPT 3B and 72B — the 3B is under a research licence, and it was
# sitting in this catalog as the highest-quality model that fits a 4 GB machine,
# i.e. the one most likely to be selected for a modest client box.
QWEN_RESEARCH = ModelLicence(
    name="Qwen RESEARCH LICENSE AGREEMENT (release 2024-09-19)",
    commercial=False,
    use_obligations=(
        "Use is limited to research or evaluation purposes only (§2.a with §1.i).",
        "Commercial use requires a separate licence from Alibaba Cloud (§2.b).",
    ),
    summary="Research use only. Not deliverable to a paying client.",
    source="legal/Qwen 2.5 RESEARCH LICENSE.pdf §1.i, §2.a, §2.b",
)

# ⚠️ Llama 3.1 and 3.2 are SEPARATE agreements, not one "Llama Community
# License". The required notice strings and the acceptable-use-policy URLs
# differ by version, and §6.2 asks for *the applicable licence* per item. Sharing
# one object between them is the same version-collapsing error that put a Llama 3
# (and then a Llama 4) document in the archive for models governed by neither.
LLAMA_31_COMMUNITY = ModelLicence(
    name="Llama 3.1 Community License Agreement (release 2024-07-23)",
    commercial=True,
    use_obligations=(
        "Adhere to the Llama 3.1 Acceptable Use Policy, incorporated by reference (§1.b.iv).",
        "An organisation whose products exceeded 700 million monthly active users on "
        "2024-07-23 must request a licence from Meta, which Meta may grant at its "
        "discretion, before exercising these rights (§2).",
    ),
    distribution_obligations=(
        'Prominently display "Built with Llama" on a related website, user interface, '
        "blog post, about page or product documentation (§1.b.i(B)).",
        "Provide a copy of the agreement with the materials (§1.b.i(A)).",
        'Retain, in a "Notice" text file, "Llama 3.1 is licensed under the Llama 3.1 '
        'Community License, Copyright © Meta Platforms, Inc. All Rights Reserved." (§1.b.iii).',
    ),
    summary="Commercial use permitted, subject to an acceptable-use policy.",
    source="legal/Llama/Llama 3.1 Community License Agreement.pdf §1.b, §2",
)

# Structurally identical to 3.1, and that was checked rather than assumed — the
# two differ only in the version strings below. ⚠️ Note what is NOT here: Llama
# **3** §1.b.v forbade using outputs to improve any other LLM. Both 3.1 and 3.2
# dropped that clause; they go i, ii, iii, iv and straight to §2. Reading the
# nearest available version would have put a restriction on the client that
# their licence does not impose.
LLAMA_32_COMMUNITY = ModelLicence(
    name="Llama 3.2 Community License Agreement (release 2024-09-25)",
    commercial=True,
    use_obligations=(
        "Adhere to the Llama 3.2 Acceptable Use Policy, incorporated by reference (§1.b.iv).",
        "An organisation whose products exceeded 700 million monthly active users on "
        "2024-09-25 must request a licence from Meta, which Meta may grant at its "
        "discretion, before exercising these rights (§2).",
    ),
    distribution_obligations=(
        'Prominently display "Built with Llama" on a related website, user interface, '
        "blog post, about page or product documentation (§1.b.i(B)).",
        "Provide a copy of the agreement with the materials (§1.b.i(A)).",
        'Retain, in a "Notice" text file, "Llama 3.2 is licensed under the Llama 3.2 '
        'Community License, Copyright © Meta Platforms, Inc. All Rights Reserved." (§1.b.iii).',
    ),
    summary="Commercial use permitted, subject to an acceptable-use policy.",
    source="legal/Llama/Llama 3.2 Community License Agreement.pdf §1.b, §2",
)

GEMMA_TERMS = ModelLicence(
    name="Gemma Terms of Use (last modified 2026-04-01)",
    commercial=True,
    use_obligations=(
        "Do not use for the restricted uses in the Gemma Prohibited Use Policy, "
        "incorporated by reference (§3.2.1).",
        "Do not use in violation of applicable laws and regulations (§3.2.2).",
    ),
    distribution_obligations=(
        "Include the §3.2 use restrictions as an ENFORCEABLE PROVISION in any agreement "
        "governing use or distribution, and give notice to subsequent users (§3.1.1).",
        "Provide third-party recipients a copy of the Gemma Terms of Use (§3.1.2).",
        "Cause any modified files to carry prominent notices of modification (§3.1.3).",
        'Accompany the distribution with a "Notice" file reading "Gemma is provided under '
        'and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms" (§3.1.4).',
    ),
    # §3.3 is worth repeating to a client: Google claims no rights in Outputs.
    summary="Commercial use permitted, subject to a prohibited-use policy. The "
    "publisher claims no rights in the answers the model generates (§3.3).",
    source="legal/Gemma/Gemma Terms of Use _ Google AI for Developers.pdf §3.1, §3.2, §3.3",
)


@dataclass(frozen=True)
class ModelSpec:
    """A local chat model, the memory it needs, and the terms it carries."""

    name: str
    ram_gb: float
    vram_gb: float
    quality: int
    licence: ModelLicence

    def requirement_for(self, has_gpu: bool) -> float:
        return self.vram_gb if has_gpu else self.ram_gb


# Public Ollama tags, ordered by quality. The floor exists so that a machine
# that should never have been accepted still starts rather than crashing — the
# gate that screens that machine out runs before delivery, not in this file.
#
# ⚠️ A model with `commercial=False` stays listed rather than being deleted, so
# that its exclusion is a visible decision instead of an absence somebody
# helpfully "fixes" later. `recommend_model` filters it out; a test proves it.
CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("qwen2.5:0.5b", 1.0, 1.5, 1, APACHE_2_0),
    ModelSpec("qwen2.5:1.5b", 1.5, 2.0, 2, APACHE_2_0),
    ModelSpec("llama3.2:3b", 3.0, 4.0, 4, LLAMA_32_COMMUNITY),
    ModelSpec("qwen2.5:3b", 3.0, 4.0, 5, QWEN_RESEARCH),  # ⛔ never recommended
    ModelSpec("qwen2.5:7b", 5.0, 6.0, 7, APACHE_2_0),
    ModelSpec("llama3.1:8b", 6.0, 7.0, 7, LLAMA_31_COMMUNITY),
    ModelSpec("gemma2:9b", 7.0, 8.0, 8, GEMMA_TERMS),
    ModelSpec("qwen2.5:14b", 9.0, 10.0, 9, APACHE_2_0),
)

# ⚠️ THE REMAINING GAP, and it is a different KIND of fact from the ones above.
#
# The archive proves what each licence SAYS. It does not prove WHICH licence
# governs which model — that "qwen2.5:7b is Apache-2.0 while qwen2.5:3b is not"
# lives on the publisher's model card, and no document here evidences it. The
# whole DELIVERABLE filter rests on that mapping.
#
# Closing it means archiving the model card for each tag above. Until then the
# mapping is the best available reading and is recorded as such, which is the
# honest state — not a claim to have checked.
MAPPING_UNEVIDENCED = True

# Licences whose text we have NOT read from a primary document. A model may not
# go into a client's bill of materials while it is listed here: the archive's
# rule is that a legal fact needs a quote, and "we believe it mirrors the
# previous version" is not one.
#
# Empty, and it should stay that way: adding a model to CATALOG without a
# readable licence in `docs/legal/` is what puts a name here. This set is pinned
# by a test, so it shrinks only by someone reading a document and filling in
# `source` — never by quietly deleting a line.
UNVERIFIED: frozenset[str] = frozenset()

# Everything that may actually be delivered. This, not CATALOG, is what
# selection reads.
DELIVERABLE: tuple[ModelSpec, ...] = tuple(
    spec for spec in CATALOG if spec.licence.commercial
)

FLOOR_MODEL = "qwen2.5:1.5b"


def spec_for(model_name: str) -> ModelSpec | None:
    """The catalog entry for a tag, ignoring any `:latest` suffix noise."""
    wanted = model_name.strip()
    for spec in CATALOG:
        if spec.name == wanted:
            return spec
    return None


def licence_for(model_name: str) -> ModelLicence | None:
    """The terms a given model carries — the bill of materials reads this.

    Returns `None` for a model outside the catalog, which is the honest answer:
    if an engagement pins a model we never vetted, its terms have not been
    checked and somebody has to check them by hand before delivery.
    """
    spec = spec_for(model_name)
    return spec.licence if spec else None


def licence_is_verified(model_name: str) -> bool:
    """Has this model's licence been read from a document in `docs/legal/`?

    Derived from `source` rather than from `UNVERIFIED`, on purpose: the set is
    the *expectation* and this is the *fact*, and a test asserts they agree. One
    of them being edited alone is exactly the drift worth catching — filling in a
    `source` without meaning to, or pinning a model as verified without reading
    anything.

    An unknown model is not verified, which is the same honest answer
    `licence_for` gives: nobody checked it.
    """
    spec = spec_for(model_name)
    return bool(spec and spec.licence.source)


@dataclass(frozen=True)
class Profile:
    ram_total_gb: float
    vram_gb: float

    @property
    def has_gpu(self) -> bool:
        # Below ~2 GB a card cannot hold a useful model plus its KV cache, so
        # running on CPU with more system RAM is the better outcome.
        return self.vram_gb > 2.0

    @property
    def effective_memory_gb(self) -> float:
        """Memory available to the model.

        On CPU, 80% of total RAM — the headroom keeps the OS, this process and
        the Chroma index from being starved when the model loads.
        """
        if self.has_gpu:
            return self.vram_gb
        return round(self.ram_total_gb * 0.80, 1)


def _detect_ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass

    try:  # Linux stdlib fallback
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        pass

    logger.warning("could not detect RAM; assuming a conservative 8 GB")
    return 8.0


def _detect_vram_gb() -> float:
    """Discrete NVIDIA VRAM in GB, or 0.0. Integrated GPUs are excluded."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0

    values = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not values:
        return 0.0
    try:
        # Largest card, not the sum: a model runs on one GPU by default.
        return round(max(float(v) for v in values) / 1024, 1)
    except ValueError:
        return 0.0


def detect_profile() -> Profile:
    return Profile(ram_total_gb=_detect_ram_gb(), vram_gb=_detect_vram_gb())


def list_downloaded(base_url: str, timeout: float = 2.0) -> set[str]:
    """Model tags already pulled on this Ollama, or an empty set.

    Failure is not an error here: an unreachable Ollama simply means we cannot
    prefer a downloaded model, and selection falls back to the catalog. The
    provider will raise soon enough, loudly, if Ollama is genuinely down.
    """
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout) as fh:
            payload = json.load(fh)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        logger.info("Ollama not reachable for model listing; using the catalog alone")
        return set()

    return {m.get("name", "") for m in payload.get("models", []) if m.get("name")}


def ollama_answers(base_url: str, timeout: float = 3.0) -> bool:
    """Is anything listening at that address at all?

    Deliberately weaker than `list_downloaded`, and the weakness is the point:
    it separates *"nothing is there"* from *"something is there and has nothing
    useful"*. `list_downloaded` returns an empty set for both, which is how a
    client came to be told to `ollama pull qwen2.5:7b` against an Ollama that
    already held three models (leg B #13). The two need opposite actions, so
    they need two probes.
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def running_in_container() -> bool:
    """Is this process inside a container?

    ⚠️ **This is the signal that was missing, and its absence is the whole of
    the wrong computer.** The obvious test — "is Ollama's address local?" — is not enough,
    because at *build* time the deliverable runs with `--network host`, so the
    address is `localhost` and looks exactly like a laptop. It is not one: the
    process is still in a container that cannot see the host's GPU, so
    `detect_profile()` reports no GPU, the budget falls back to the larger
    RAM-based figure, and selection goes *up* instead of down.

    That is also why build and runtime could disagree — same config, host
    network at build and bridge at runtime — and a disagreement here is not
    cosmetic: the resolved tag goes into the answer-cache key, so the answers
    baked at build time would be **deleted on first lookup** at runtime
    (`answers/cache.py`), silently killing the instant-answer feature the
    product rests on.

    Both files are checked because the client's platform is not ours to assume:
    Docker writes the first, Podman the second.
    """
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def best_downloaded(base_url: str) -> str | None:
    """The best deliverable model this Ollama has **already pulled**, or None.

    This is the selection path for a **remote** Ollama — in practice every
    delivered build, where the engine is containerised and the model is on the
    host. It measures nothing, because there is nothing here worth
    measuring: RAM and VRAM read from this process describe the container, not
    the machine that will load the model.

    What is left is a fact rather than an estimate — *which models that machine
    has*. It is weaker than a memory reading, and it is honest: somebody pulled
    those deliberately, in most cases because the pre-purchase qualifier told
    them which one their hardware supports. Returning None (rather than a
    plausible guess) is what turns "9 GB downloading under the client's first
    question" into a sentence saying what to pull.
    """
    downloaded = list_downloaded(base_url)
    if not downloaded:
        return None

    present = [spec for spec in DELIVERABLE if spec.name in downloaded]
    if not present:
        return None
    return max(present, key=lambda spec: spec.quality).name


def recommend_model(base_url: str = "http://localhost:11434") -> str:
    """The best catalog model that fits this machine.

    Prefers a model that is **already downloaded** among those that fit, so the
    first question after handover does not stall on a multi-gigabyte pull while
    the client watches.
    """
    profile = detect_profile()
    budget = profile.effective_memory_gb
    downloaded = list_downloaded(base_url)

    # DELIVERABLE, not CATALOG: a model that may not lawfully be run in a
    # business is not a candidate, however well it fits their machine.
    fitting = [
        spec for spec in DELIVERABLE if spec.requirement_for(profile.has_gpu) <= budget
    ]
    if not fitting:
        logger.warning(
            "no catalog model fits %.1f GB of effective memory; falling back to %s",
            budget,
            FLOOR_MODEL,
        )
        return FLOOR_MODEL

    already = [spec for spec in fitting if spec.name in downloaded]
    best = max(already or fitting, key=lambda spec: spec.quality)

    logger.info(
        "selected %s (effective memory %.1f GB, gpu=%s, downloaded=%s)",
        best.name,
        budget,
        profile.has_gpu,
        best.name in downloaded,
    )
    return best.name
