# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""The delivery manifest: what was handed over, so drift from it is EVIDENCE.

⚠️ **This is tamper-EVIDENCE. It is not, and must never become, tamper-proofing.**

The engine ships every capability to every deployment — one engine, a deployment
is a config and not a fork — so the agreed limits are a line of TOML rather than
a locked door. That is a deliberate choice with a real consequence: a deployment
can quietly grant itself capacity, a scheduled refresh, or a larger prepared set
by editing a file and rebuilding.

**None of that can be prevented in software, and the attempt would be worse than
the exposure.** The container runs on hardware the client owns, with the source
public for them to audit — which is the product's central claim. Any check
written here is a check they can read and delete, so a lock published in a
source-available repository publishes its own bypass. A licence check that
phoned home would falsify the one claim that has been audited and is true.

So this module does the thing that *does* survive contact with reality: it
records what was delivered, and it lets `doctor` say — out loud, in a support
bundle, in front of whoever asks — **"this deployment no longer matches what was
agreed."** That is a sentence with contractual weight, and it costs nobody
anything if the deployment is honest.

Two rules that keep it from becoming the thing it must not be:

1. **It never blocks and never modifies.** `doctor` is read-only. A drifting
   deployment still starts, still answers, still serves the client.
2. **It never destroys anything.** Deleting data on suspicion of tampering would
   put the client's own indexed documents at risk on a product sold to protect
   exactly those documents, and false positives are certain rather than
   hypothetical.

A missing manifest is reported as *unverifiable*, never as *tampered*: the
honest reading of "no evidence" is no evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stillroom.config import ClientConfig

# Beside `client.toml` inside the image, written at delivery and not by the
# client's own rebuild -- which is exactly what makes a rebuild after an edit
# visible rather than self-certifying.
MANIFEST_NAME = "DELIVERY_MANIFEST.json"

SCHEMA = 1


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shape(config: ClientConfig) -> dict[str, Any]:
    """The commercially meaningful shape of a deployment.

    Deliberately NOT a hash of the whole config file. A bulk hash answers
    "something changed" and then makes a human diff two files to find out what;
    these fields answer *which* agreed limit moved, which is the only question
    anybody actually asks. Theme edits, a retitled page or a new curated
    question's wording are none of our business and do not show up here.
    """
    return {
        "max_documents": config.capacity.max_documents,
        "capacity_enforced": not config.capacity.warn_only,
        "curated_answers": len(config.answer_cache.curated),
        "ui_languages": list(config.ui.languages),
        "scheduled_refresh": config.refresh.enabled,
    }


@dataclass(frozen=True)
class Drift:
    """One agreed value, and what it is now."""

    field: str
    delivered: Any
    current: Any

    def __str__(self) -> str:
        return f"{self.field}: delivered {self.delivered!r}, now {self.current!r}"


def build(config: ClientConfig, build_id: str | None = None) -> dict[str, Any]:
    """The manifest written at delivery time."""
    shape = _shape(config)
    return {
        "schema": SCHEMA,
        "client": config.client,
        "build_id": build_id,
        "agreed": shape,
        # Over the shape, not the file: the file's formatting is not the
        # agreement, and re-indenting a TOML is not a breach.
        "fingerprint": _sha256(json.dumps(shape, sort_keys=True)),
    }


def render(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def load(directory: Path | str = ".") -> dict[str, Any] | None:
    """Read the manifest beside the config, or None when there is not one."""
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def compare(manifest: dict[str, Any], config: ClientConfig) -> list[Drift]:
    """Which agreed values this deployment no longer matches.

    An unreadable or unknown-schema manifest yields no drift rather than false
    drift: reporting a breach because we could not parse our own file would be
    the worst possible failure mode for a document meant to be used in a
    disagreement.
    """
    if manifest.get("schema") != SCHEMA:
        return []
    agreed = manifest.get("agreed")
    if not isinstance(agreed, dict):
        return []

    current = _shape(config)
    drifts: list[Drift] = []
    for field, delivered in sorted(agreed.items()):
        if field not in current:
            continue
        if current[field] != delivered:
            drifts.append(Drift(field, delivered, current[field]))
    return drifts
