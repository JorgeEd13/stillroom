# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""The bill of materials that ships with every delivery.

**This is a contractual artifact, not documentation.** The engagement terms
require that, with *each* delivery of Work Product, the freelancer separately
provides a bill of materials identifying all Background Technology and other
third-party materials incorporated into the Work Product, giving for each:

    (a) the name and any associated version number,
    (b) the applicable licence or licensing terms,
    (c) whether the item has been modified by Freelancer,
    (d) how the item has been incorporated into, is used by, or is relied
        upon by the Work Product.

`BomItem` has exactly those four fields, in that order, because a hand-written
list drifts from what was actually installed and a client's lawyer reads the
list rather than the lockfile. Versions come from the *installed distributions*,
so the document describes the build that was handed over.

### Why this refuses rather than warns

`stillroom bom` exits non-zero when any item cannot be certified — same posture
as `stillroom eval`, and for the same reason: it is a delivery gate. An item
whose licence nobody has read from a primary document may not be asserted in a
document that goes to the client. The archive's rule is that a legal fact needs
a quote; a bill of materials is nothing *but* legal facts.

⚠️ **Generate it from the built container, not a source checkout.** Dependency
versions come from installed metadata, so a checkout cannot produce a real one —
and it says so and blocks, rather than printing a confident document with no
libraries in it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import metadata

from stillroom import hardware
from stillroom.config import ByokModel, ClientConfig, OllamaModel
from stillroom.index.embeddings import DEFAULT_EMBEDDING_NAME

# The licence granted to the client for the engine itself. Kept as a constant
# because it appears in the BoM, in `pyproject.toml` and in `LICENSE.md`, and
# those three disagreeing is the kind of thing nobody notices until it matters.
ENGINE_LICENCE = "PolyForm Shield 1.0.0 (source-available; NOT open source)"

# ⛔ Third-party items whose licence we believe we know but have NOT read from a
# document in `docs/legal/`. Anything here blocks delivery.
#
# The embedding model was the instructive one. Every licence conversation had
# been about the *chat* model, which varies per machine — while that one ships in
# **every single build**, is a third-party ML model with its own terms, and had
# never been looked at. It was found by asking what §6.2's "and other
# third-party materials" actually covers, not by anybody remembering it.
#
# Empty, and the way it emptied is the point: `onnx-minilm-l6-v2` sat here until
# the upstream model card was archived (2026-07-21). "ChromaDB is Apache-2.0"
# did **not** close it — that covers ChromaDB's code, and these weights are not
# in the wheel. What closed it was a document stating the licence of *this
# model*, which is a different kind of fact from the licence text itself.
_UNARCHIVED: dict[str, str] = {}

# Where the weights actually come from. **This pair moved with the embedder and
# it had to move in the same commit**: the values below used to
# describe ChromaDB's MiniLM under Apache-2.0, and leaving either behind while
# swapping the model would put a false licence statement into a document the
# client is handed at delivery — the exact failure the licence gate exists to prevent.
_EMBEDDING_UPSTREAM = "BAAI/bge-m3"
# ⚠️ **No longer a URL we fetch from.** The old value was Chroma's S3 bucket,
# because the weights arrived in our build. Now the client's own Ollama pulls
# them, exactly as it pulls the chat model — see `_embedding_item`.
_EMBEDDING_SOURCE_URL = "the Ollama library, pulled by your own Ollama"

# The chain, walked to its origin and archived, every link quoted in the embedder decision's
# addenda: the model card declares `license: mit`; its code (`FlagEmbedding`) is
# MIT © 2022 staoxiao; its base model `xlm-roberta-large` declares MIT; and that
# card names its own origin release, `fairseq`, whose LICENSE is MIT ©
# Facebook, Inc. The last link is the authoritative one — the xlm-roberta card
# says in as many words that Hugging Face wrote it, not the releasing team.
_EMBEDDING_LICENCE = "MIT"
_EMBEDDING_LICENCE_SOURCE = "legal/licences/bge-m3 model card.md"
# Kept separately because the model card is a *declaration* and this is the
# licence *text*. The embedder decision named that seam rather than smoothing it.
_EMBEDDING_LICENCE_TEXT_SOURCE = "legal/licences/bge-m3 FlagEmbedding LICENSE"


@dataclass(frozen=True)
class BomItem:
    """One row. The first four fields are §6.2(a)–(d), in order."""

    name: str
    version: str
    licence: str
    modified: str
    incorporation: str
    # What the CLIENT must do because this item is present. Usually empty.
    obligations: tuple[str, ...] = field(default_factory=tuple)
    # Non-empty means this row may not be certified. See `_UNARCHIVED`.
    blocked_because: str = ""


def _engine_item() -> BomItem:
    try:
        version = metadata.version("stillroom")
    except metadata.PackageNotFoundError:  # running from a source tree
        version = "0.0.0+source"

    return BomItem(
        name="stillroom",
        version=version,
        licence=ENGINE_LICENCE,
        modified="Authored by Freelancer; this is the Background Technology itself.",
        incorporation=(
            "The entire application: ingestion, retrieval, answer generation, "
            "caching and the local web service."
        ),
    )


def _chat_model_item(config: ClientConfig) -> BomItem:
    """The model that writes the answers.

    ⚠️ It is **not redistributed by us**. The client's own Ollama pulls the
    weights from upstream under their own acceptance of the publisher's terms,
    so what belongs here are the terms that bind them by *running* it — never
    the distribution duties, which attach to nobody in this delivery model.
    """
    model = config.model

    if isinstance(model, ByokModel):
        # Nothing of the provider's is incorporated — the client holds the
        # account and the agreement. Saying "no licence obligation" would be
        # wrong; the honest entry names whose terms actually govern.
        return BomItem(
            name=f"{model.provider}: {model.name}",
            version="as served by the provider",
            licence=(
                "The client's own agreement with the provider. No licence is "
                "granted or passed through by Freelancer."
            ),
            modified="No.",
            incorporation=(
                "Called over the network, with the client's own API key, to "
                "generate answers from passages retrieved locally."
            ),
        )

    assert isinstance(model, OllamaModel)
    from stillroom.provider import resolve_model_name

    name = resolve_model_name(model)
    licence = hardware.licence_for(name)

    if licence is None:
        return BomItem(
            name=name,
            version="as pulled by Ollama",
            licence="UNKNOWN — outside the vetted catalog",
            modified="No.",
            incorporation="Generates answers from passages retrieved locally.",
            blocked_because=(
                "this model is not in hardware.CATALOG, so nobody has read its "
                "licence; it must be checked by hand before delivery"
            ),
        )

    return BomItem(
        name=name,
        version="as pulled by Ollama",
        licence=licence.name,
        modified="No.",
        incorporation=(
            "Run locally by the client's own Ollama to generate answers from "
            "passages retrieved locally. The weights are downloaded by the "
            "client from the publisher and are not redistributed by Freelancer."
        ),
        obligations=licence.use_obligations,
        blocked_because=(
            ""
            if hardware.licence_is_verified(name)
            else "its licence has not been read from a document in docs/legal/"
        ),
    )


def _embedding_item() -> BomItem:
    """The model that turns text into vectors.

    ⚠️ **It is a separate third-party item from the libraries, not part of one.**
    That was learned when it was Chroma's bundled MiniLM: ChromaDB is Apache-2.0
    and that covers ChromaDB's *code*, while the weights were a different work
    under different terms, fetched from a different place. Treating a library's
    licence as its model's licence is the same mistake as treating "free to
    download" as "free to use commercially" — one layer further down, and easier
    to miss because the library really is permissively licensed.

    ## ⚠️ This row REVERSED with the move to bge-m3, and the reversal is the point

    shipping the base image as a tarball made us the
    **redistributor** of the embedding weights, because they were baked into the
    image we hand over. `bge-m3` takes them back out: the client's own Ollama
    pulls them, exactly as it already pulls the chat model. So:

    * we are **not** a distributor of these weights — the load-bearing
      sentence *"we never distribute weights"* is true again, for every model;
    * the obligations that fire on **Distribution** therefore attach to nobody
      here, and are not reported, which is the `use_` vs `distribution_` split
      the licence-verification gate established it;
    * **stating an obligation that does not exist is the same defect as omitting
      one that does**, so this row says what is true now rather than carrying
      the sentence that was true last week.

    What did *not* change: the Python dependency closure still ships inside the
    tarball, so `THIRD-PARTY-NOTICES.md` still exists and still matters. It just
    no longer has an embedding model in it.

    MIT's one condition — that the copyright notice travels with copies — is
    therefore not triggered by us at all. The attribution is carried anyway,
    because a bill of materials that omits an item the client is running is
    incomplete whether or not a licence compels the line.
    """
    name = DEFAULT_EMBEDDING_NAME
    return BomItem(
        name=f"{name} ({_EMBEDDING_UPSTREAM})",
        # ⚠️ Named by VERSION. "bge-m3" is a family; the tag is what
        # the config pins and what the client's Ollama actually pulls.
        version=f"Ollama tag {name}, pulled by you from {_EMBEDDING_SOURCE_URL}",
        licence=_EMBEDDING_LICENCE,
        modified="No.",
        incorporation=(
            "Turns your documents and your questions into vectors so that "
            "passages can be matched. It runs on your own Ollama, on your own "
            "machine, and it is the reason your documents can be searched "
            "without any of them leaving the building. It is NOT inside the "
            "image handed to you — your Ollama downloads it once."
        ),
        obligations=(
            "None fall on you from us. You obtain these weights directly from "
            "the model's own distribution channel, under the terms you accept "
            "there, exactly as you do for the chat model. The terms are MIT, "
            "which places no restriction on internal commercial use.",
        ),
        blocked_because=_UNARCHIVED.get(name, ""),
    )


def _distribution_licence(dist: metadata.Distribution) -> str:
    """The declared licence of an installed distribution.

    Modern wheels carry `License-Expression` (an SPDX string); older ones put a
    free-text blob in `License` or state it only through a classifier. Falling
    back through all three beats reporting "unknown" for a package that declared
    it perfectly well in the older way.
    """
    meta = dist.metadata

    expression = meta.get("License-Expression")
    if expression:
        return expression.strip()

    classifiers = [
        value.split("::")[-1].strip()
        for value in meta.get_all("Classifier") or []
        if value.startswith("License ::")
    ]
    if classifiers:
        return ", ".join(classifiers)

    declared = (meta.get("License") or "").strip()
    if declared and "\n" not in declared and len(declared) < 60:
        return declared

    return "UNDECLARED"


# ⛔ Licences that would breach §6.2 — the clause banning anything whose use or
# distribution creates obligations for the CLIENT to grant rights under THEIR
# intellectual property to a third party: source disclosure, derivative-works
# licensing, or free redistribution.
#
# Project-scoped ("strong") copyleft is the real hazard, because its reach
# extends to the work it is combined with. Nothing in the closure hits this
# today; the guard exists for the dependency somebody adds in a year, when this
# session's reasoning is long gone.
_STRONG_COPYLEFT = re.compile(r"\b(A?GPL|SSPL|CC-BY-SA|EUPL|OSL)\b", re.IGNORECASE)

# ✅ File-scoped ("weak") copyleft. Present and ACCEPTABLE — recorded rather than
# waved through, because "permissive throughout" was asserted about this closure
# on 2026-07-21 and was not quite true.
#
# `certifi`, `orjson` and `tqdm` are MPL-2.0. MPL's copyleft is **per-file and
# triggers on distributing a MODIFIED covered file**; §3.3 expressly permits
# combining it into a Larger Work under other terms. We modify none of them, and
# the client redistributes nothing. So no obligation reaches the client's own
# IP — which is the only thing §6.2 actually forbids. Named here so a future
# reader sees a decision instead of an oversight.
_WEAK_COPYLEFT = re.compile(r"\b(MPL|EPL|CDDL|MS-PL)\b", re.IGNORECASE)

# The optional-dependency groups the DELIVERED image installs — the Dockerfile
# runs `pip install .[ollama,docs]`. Their members (the Ollama client; the pypdf/
# python-docx/openpyxl document loaders) are redistributed in the tarball and so
# must be declared here and in the notices. `dev`/`test` extras are NOT shipped.
# ⚠️ Keep this in lockstep with the Dockerfile: a shipped extra added there but
# not here would ship undeclared — the exact defect that fix repaired.
_SHIPPED_EXTRAS = ("ollama", "docs")
_EXTRA_MARKER = re.compile(r"extra\s*==\s*[\"']([^\"']+)[\"']")


def dependency_distributions() -> dict[str, metadata.Distribution]:
    """The installed dependency closure of `stillroom`, keyed by normalised name.

    Read from installed metadata rather than `pyproject.toml`: the file states
    ranges and optional extras, and the client received exactly one resolution
    of them. Empty when the engine is not installed — the caller decides what
    that means, and both callers treat it as a blocking condition.

    Shared with `notices.py`, which needs the same closure for a different
    obligation: the bill of materials names each licence, the notices carry each
    licence's text. Two documents disagreeing about what is in the
    image is exactly the drift a hand-maintained list produces.
    """
    try:
        root = metadata.distribution("stillroom")
    except metadata.PackageNotFoundError:
        return {}

    seen: dict[str, metadata.Distribution] = {}
    queue = [root]
    while queue:
        dist = queue.pop()
        for requirement in dist.requires or []:
            # "chromadb>=1.0", or "pypdf>=4; extra == 'docs'". Keep a plain
            # requirement, and keep an extra ONLY if that extra is one the
            # delivered image ships (`_SHIPPED_EXTRAS`): the loaders in `docs`
            # and the Ollama client in `ollama` are installed and redistributed
            # in the tarball, so they belong here. A `dev`/`test` extra is not
            # shipped and must not be counted — and resolution alone cannot tell
            # them apart, because a dev machine has them installed too. The old
            # blanket `if "extra ==": continue` dropped every extra, so the
            # document loaders shipped undeclared for months.
            marker = _EXTRA_MARKER.search(requirement)
            if marker and marker.group(1) not in _SHIPPED_EXTRAS:
                continue
            name = (
                requirement.split(";")[0]
                .split("[")[0]
                .split("(")[0]
                .split(">")[0]
                .split("<")[0]
                .split("=")[0]
                .split("!")[0]
                .split("~")[0]
                .strip()
            )
            key = name.lower().replace("_", "-")
            if not name or key in seen:
                continue
            try:
                found = metadata.distribution(name)
            except metadata.PackageNotFoundError:
                continue
            seen[key] = found
            queue.append(found)

    return seen


def _dependency_items() -> list[BomItem]:
    """One §6.2 row per installed third-party library."""
    closure = dependency_distributions()
    if not closure:
        # Running from a source tree that was never installed. Returning an
        # empty list here would print a bill of materials with **no libraries in
        # it at all** and exit zero — a confident, complete-looking, wrong
        # document. Say so instead, and block.
        return [
            BomItem(
                name="(Python dependencies)",
                version="unknown",
                licence="unknown",
                modified="No.",
                incorporation="Third-party libraries the engine is built on.",
                blocked_because=(
                    "stillroom is not installed in this environment, so the "
                    "dependency versions cannot be read; generate the bill of "
                    "materials from the built container, not a source checkout"
                ),
            )
        ]

    items = []
    for _, dist in sorted(closure.items()):
        licence = _distribution_licence(dist)
        blocked = ""
        if _STRONG_COPYLEFT.search(licence):
            blocked = (
                f"licensed {licence}, which is project-scoped copyleft — §6.2 "
                "forbids delivering anything that creates obligations for the "
                "client to license or disclose their own intellectual property"
            )
        elif licence == "UNDECLARED":
            blocked = "declares no licence, so its terms are unknown"

        items.append(
            BomItem(
                name=dist.metadata["Name"],
                version=dist.version,
                licence=licence,
                modified="No.",
                incorporation=(
                    "Third-party Python library, used as published. Installed in "
                    "the base image you received; its licence text is reproduced "
                    "in THIRD-PARTY-NOTICES.md."
                ),
                blocked_because=blocked,
            )
        )
    return items


def build(config: ClientConfig) -> list[BomItem]:
    """Every item that goes in front of the client, engine first."""
    return [
        _engine_item(),
        _chat_model_item(config),
        _embedding_item(),
        *_dependency_items(),
    ]


def blocked(items: list[BomItem]) -> list[BomItem]:
    return [item for item in items if item.blocked_because]


def render(items: list[BomItem], client: str) -> str:
    """The delivered document. Markdown, because the client reads it."""
    lines = [
        f"# Bill of materials — {client}",
        "",
        "Everything incorporated into what was delivered, with its licence.",
        "Provided under section 6.2 of the service contract terms.",
        "",
        "## What you are licensed to do with the engine itself",
        "",
        f"`stillroom` is licensed to you under **{ENGINE_LICENCE}**. You may read",
        "it, audit it, run it, and modify it inside your organisation, for as long",
        "as you like. The one thing you may not do is use it to provide a product",
        "that competes with it.",
        "",
        "## Items",
        "",
        "| Item | Version | Licence | Modified | How it is used |",
        "|---|---|---|---|---|",
    ]

    for item in items:
        cells = [
            item.name,
            item.version,
            item.licence,
            item.modified,
            item.incorporation,
        ]
        lines.append("| " + " | ".join(c.replace("|", r"\|") for c in cells) + " |")

    with_obligations = [item for item in items if item.obligations]
    if with_obligations:
        lines += [
            "",
            "## Terms that apply to you",
            "",
            "These come from the publishers of the items above, not from us. They",
            "are reproduced so that nothing in this delivery obliges you to",
            "something you were not told about.",
            "",
        ]
        for item in with_obligations:
            lines.append(f"**{item.name}** — {item.licence}")
            lines += [f"- {ob}" for ob in item.obligations]
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
