# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""The third-party licence notices that travel with the base image.

**This exists because of leg 3, and it did not need to exist before it**. Until the base image became a `docker load`-able tarball, the client's
own machine fetched every Python library from PyPI and the embedding weights from
Chroma's bucket. We handed over source and a `Dockerfile`; we redistributed
nothing, and `hardware.ModelLicence` says so in as many words — *"we never
distribute model weights"*, with the distribution-triggered obligations recorded
but deliberately not reported, because they applied to nobody.

A tarball is a redistribution. MIT, BSD and Apache-2.0 all permit it and all
require the same thing in return: the notices travel with the copy. Apache-2.0
§4(a) requires giving recipients a copy of the Licence; §4(d) requires
reproducing any `NOTICE` file. MIT and BSD require the copyright notice and the
permission text. None of that is onerous — but none of it happens by itself, and
"permissively licensed" is not the same as "no conditions".

## ⚠️ The move to bge-m3 took the embedding weights back OUT, and this file had to follow

This document used to carry a section headed *"The embedding model"*, describing
weights that were inside the image. They are not any more: `bge-m3` is pulled by
the client's own Ollama, exactly like the chat model, so we are not its
distributor and MIT's notice condition is not triggered by us at all.

**Leaving that section in place would have been the failure this whole file
exists to prevent** — a licence document, handed over at delivery, describing a
model that is not in the thing it describes. It is replaced by a short section
saying what is *not* here and why, because silence would read as an omission to
anybody auditing the image against the bill of materials.

What has **not** changed: the Python dependency closure still ships inside the
tarball, so every obligation below is still live. The file got smaller, not
unnecessary.

⚠️ **Generate it inside the image it describes.** Licence texts are read from the
installed distributions, so a source checkout produces a document about nothing.
It refuses rather than writing a confident, empty one — the same posture as
`stillroom bom`, for the same reason.

    python -m stillroom.notices --output /home/app/THIRD-PARTY-NOTICES.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from stillroom.bom import (
    _EMBEDDING_LICENCE,
    _EMBEDDING_UPSTREAM,
    dependency_distributions,
)
from stillroom.index.embeddings import DEFAULT_EMBEDDING_NAME

# Filenames wheels use for the text we are obliged to carry. Matched case-
# insensitively, with or without an extension, and `NOTICE` is included because
# Apache-2.0 §4(d) makes reproducing it a separate obligation from §4(a)'s copy
# of the licence itself.
_LICENCE_STEMS = ("license", "licence", "copying", "notice", "authors", "copyright")


@dataclass(frozen=True)
class Notice:
    name: str
    version: str
    licence: str
    # One entry per licence-ish file found in the wheel: (filename, text).
    texts: tuple[tuple[str, str], ...]

    @property
    def has_text(self) -> bool:
        return bool(self.texts)


def _is_licence_file(name: str) -> bool:
    stem = name.split(".")[0].lower()
    return stem in _LICENCE_STEMS


def _texts_of(dist: metadata.Distribution) -> tuple[tuple[str, str], ...]:
    """Every licence file a wheel shipped in its `.dist-info`.

    Wheels put them either directly in `<name>.dist-info/` or, since PEP 639, in
    `<name>.dist-info/licenses/`. Both are read; anything unreadable is skipped
    rather than raised, because one malformed wheel must not stop the build that
    is producing the notices for all the others.
    """
    found: list[tuple[str, str]] = []
    for path in dist.files or []:
        if not any(part.endswith(".dist-info") for part in path.parts):
            continue
        if not _is_licence_file(path.name):
            continue
        # ⚠️ `locate()`, not `dist.read_text(path)`. `read_text` resolves names
        # *inside* the `.dist-info` directory, so handing it the full recorded
        # path silently finds nothing — and "silently finds nothing" is a
        # notices file that looks complete and carries no licence at all. Caught
        # by the build refusing, which is why it refuses.
        try:
            found.append((path.name, path.locate().read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    # Deduplicated by CONTENT: several wheels record the same file twice, once
    # in the `.dist-info` root and once under PEP 639's `licenses/`, and printing
    # one licence twice makes the document look padded rather than thorough.
    seen: set[str] = set()
    unique = []
    for name, text in sorted(found):
        body = text.strip()
        if body and body not in seen:
            seen.add(body)
            unique.append((name, body))
    return tuple(unique)


def collect() -> list[Notice]:
    """One notice per installed third-party distribution, engine excluded.

    The engine is ours and is licensed to the client separately (PolyForm Shield
    1.0.0, in `LICENSE.md`); putting it in a list headed "third party" would be
    both wrong and confusing.
    """
    notices = []
    for _, dist in sorted(dependency_distributions().items()):
        from stillroom.bom import _distribution_licence

        notices.append(
            Notice(
                name=dist.metadata["Name"],
                version=dist.version,
                licence=_distribution_licence(dist),
                texts=_texts_of(dist),
            )
        )
    return notices


def render(notices: list[Notice]) -> str:
    missing = [n for n in notices if not n.has_text]

    lines = [
        "# Third-party notices",
        "",
        "The assistant runs in a container image that was built for you. That",
        "image contains third-party software, and this file carries the licence",
        "notices those licences require to travel with it.",
        "",
        "Nothing here asks anything of you. These are the terms under which the",
        "software inside the image may be passed on, and reproducing them is our",
        "obligation, not yours. The engine itself is licensed to you separately —",
        "see `LICENSE.md`.",
        "",
        f"**{len(notices)} third-party libraries.**",
        "",
        "---",
        "",
        "## What is NOT in this image, and therefore not below",
        "",
        "Neither model this assistant uses is inside the image. Your own Ollama",
        f"downloads both — the one that writes the answers, and **{DEFAULT_EMBEDDING_NAME}**",
        f"(`{_EMBEDDING_UPSTREAM}`, {_EMBEDDING_LICENCE}), which turns your documents and",
        "your questions into vectors so that passages can be matched.",
        "",
        "They are named here because you are running them, not because anything",
        "is being passed on to you: you obtain them from their own distribution",
        "channel under the terms you accept there. Both appear in the bill of",
        "materials with their licences.",
        "",
    ]

    if missing:
        lines += [
            "---",
            "",
            "## Libraries whose packages carried no licence file",
            "",
            "These declare a licence in their package metadata but ship no copy of",
            "its text. The declared licence is authoritative and its full text is",
            "published by the project itself.",
            "",
            "| Library | Version | Declared licence |",
            "|---|---|---|",
        ]
        lines += [f"| {n.name} | {n.version} | {n.licence} |" for n in missing]
        lines.append("")

    lines += ["---", "", "## Licence texts", ""]

    for notice in notices:
        if not notice.has_text:
            continue
        lines += [f"### {notice.name} {notice.version} — {notice.licence}", ""]
        for filename, text in notice.texts:
            lines += [f"<!-- {notice.name}: {filename} -->", "", "```", text.strip(), "```", ""]

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the third-party licence notices.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    notices = collect()

    # ⚠️ Nothing found means this ran somewhere it cannot see the installed
    # dependencies — a source checkout, most likely. Writing an empty document
    # would be worse than writing none: it looks like a completed obligation.
    if not any(n.has_text for n in notices):
        print(
            "No licence text could be read from any installed distribution.\n"
            "Run this inside the built image, not in a source checkout.",
            file=sys.stderr,
        )
        return 1

    args.output.write_text(render(notices), encoding="utf-8")
    missing = sum(1 for n in notices if not n.has_text)
    print(
        f"wrote {args.output} — {len(notices)} libraries, "
        f"{len(notices) - missing} with licence text, {missing} declared only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
