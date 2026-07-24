# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""The operator CLI — ingest, bake, ask, eval, refresh, serve.

Two audiences, and they need different things from the same commands:

* **Delivery (me).** `ingest` then `bake` then `ask` is the build loop, and
  `bake` is where a corpus that cannot answer the client's own questions gets
  caught — before handover, not after.
* **The operator, later.** Operators re-ingest as documents change, so they
  can add documents themselves. That script is `stillroom ingest`; there is no
  second implementation to drift out of sync with this one. `refresh` is the
  same work on a timer, and `eval` is the accuracy suite they
  are shown at handover.

Output is plain text on purpose. A client reads this over a shoulder or in a
runbook screenshot, and a JSON blob is not a thing a non-technical person is
willing to act on.
"""

from __future__ import annotations

import argparse
import logging
import sys

from stillroom.config import ClientConfig
from stillroom.engine import Engine, NotIngested
from stillroom.pipeline import ingest as run_ingest
from stillroom.provider import model_label


def _load(path: str) -> ClientConfig:
    return ClientConfig.load(path)


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = _load(args.config)
    result = run_ingest(config, model_name=model_label(config))

    print(f"Indexed {result.documents} documents into {result.chunks} passages.")
    if result.pruned:
        print(f"Removed {result.pruned} passages no longer in your documents.")
    print(f"Corpus fingerprint: {result.fingerprint[:16]}")

    if result.skipped:
        print(f"\nSkipped {len(result.skipped)} file(s):")
        for reason in result.skipped:
            print(f"  - {reason}")
        # Each line above already says *why*. This one only says what it means.
        #
        # ⚠️ It stopped naming a single cause once skips could also be unsupported
        # types and links out of the folder — then immediately
        # hard-coded an explanation for *one* cause anyway, so a client who
        # renamed a file to `.bak` was told about scanned documents (leg B #16).
        # The scanning note is genuinely the one clients need most, so it is kept
        # — but only when a scan is actually among the reasons.
        print("\nNothing above is in the assistant's knowledge. Each line says why.")
        if any("scanned" in reason.lower() or "no text" in reason.lower() for reason in result.skipped):
            print(
                "A scanned document is a picture of a page, so there is no text "
                "in it to read."
            )
    return 0


def _cmd_bake(args: argparse.Namespace) -> int:
    config = _load(args.config)
    engine = Engine(config)

    questions = tuple(args.question) if args.question else None
    results = engine.bake_curated(questions)
    if not results:
        print("No curated questions configured; nothing to bake.")
        return 0

    baked = [q for q, ok in results if ok]
    missed = [q for q, ok in results if not ok]

    print(f"Baked {len(baked)} instant answer(s).")
    if missed:
        # Deliberately loud: this is a finding to take back to the client, not
        # a warning to scroll past.
        print(f"\n{len(missed)} question(s) found NOTHING in the documents:")
        for question in missed:
            print(f"  - {question}")
        print(
            "\nThese need resolving before handover: either the documents that "
            "answer them are missing from the corpus, or the question is asking "
            "for something the documents do not cover."
        )
        return 1
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    config = _load(args.config)
    engine = Engine(config)
    answer = engine.ask(args.question, use_cache=not args.no_cache)

    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for i, citation in enumerate(answer.citations, start=1):
            label = citation["source"]
            if citation.get("heading"):
                label = f"{label} — {citation['heading']}"
            print(f"  [{i}] {label}")

    origin = {
        "cache": "curated instant answer" if answer.curated else "cached answer",
        "model": "model",
        "no-match": "no relevant documents",
    }[answer.served_by]
    print(f"\n({origin})")
    return 0


def _cmd_refresh(args: argparse.Namespace) -> int:
    """One refresh cycle, on demand — what the scheduler runs on a timer."""
    from stillroom.refresh import refresh_once

    result = refresh_once(_load(args.config))
    if result.error:
        print(f"Refresh failed: {result.error}", file=sys.stderr)
        print("The assistant keeps serving the previous index.", file=sys.stderr)
        return 1
    if not result.changed:
        print(f"No change — {result.documents} documents, index already current.")
        return 0

    print(f"Corpus changed: {result.documents} documents indexed.")
    print(f"Rebaked {result.rebaked} instant answer(s).")
    if result.unanswerable:
        print(f"\n{len(result.unanswerable)} curated question(s) can no longer be answered:")
        for question in result.unanswerable:
            print(f"  - {question}")
        return 1
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from stillroom.evals import EvalSuite, format_report, run_suite

    config = _load(args.config)
    suite = EvalSuite.load(args.suite)
    result = run_suite(Engine(config), suite)

    print(format_report(result))
    # Non-zero on any failure: this is a delivery gate, and the client is shown
    # the report at handover.
    return 0 if result.ok else 1


def _cmd_manifest(args: argparse.Namespace) -> int:
    """Record what is being delivered, so later drift from it is evidence.

    Run by us at handover, never by the client — a manifest the client's own
    rebuild regenerated would certify whatever it found, which is the one thing
    it must not do.
    """
    from stillroom import integrity

    config = _load(args.config)
    manifest = integrity.build(config, build_id=args.build_id)
    document = integrity.render(manifest)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(document)
        print(f"Wrote the delivery manifest for {config.client} to {args.output}")
    else:
        print(document, end="")
    return 0


def _cmd_bom(args: argparse.Namespace) -> int:
    """The §6.2 bill of materials. A delivery gate, like `eval`."""
    from stillroom import bom

    config = _load(args.config)
    items = bom.build(config)
    blocked = bom.blocked(items)

    document = bom.render(items, config.client)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(document)
        print(f"Wrote {len(items)} items to {args.output}")
    else:
        print(document)

    if blocked:
        # Deliberately loud and non-zero: this document goes to the client as a
        # contractual statement, and an item nobody checked must not be asserted
        # in it. Handing over is the thing that is blocked, not the writing.
        print(
            f"\n⛔ {len(blocked)} item(s) cannot be certified — DO NOT DELIVER:",
            file=sys.stderr,
        )
        for item in blocked:
            print(f"  - {item.name}: {item.blocked_because}", file=sys.stderr)
        print(
            "\nEach needs its licence read from a document in docs/legal/ and "
            "recorded, before this delivery goes out.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Readiness verification. Reports; changes nothing."""
    from stillroom import doctor

    checks = doctor.run(_load(args.config))
    print(doctor.format_report(checks, share=getattr(args, "share", False)))
    # Non-zero only on a real failure: a warning is something to know, not
    # something that stops a launcher.
    return 1 if doctor.worst(checks) == "fail" else 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from stillroom.api import create_app

    config = _load(args.config)
    uvicorn.run(create_app(config), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stillroom", description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Show progress logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text, handler in (
        ("ingest", "Read and index the documents named in the config.", _cmd_ingest),
        ("bake", "Pre-compute the curated instant answers.", _cmd_bake),
        ("ask", "Ask one question and print the answer.", _cmd_ask),
        ("eval", "Run the client's accuracy suite against the built engine.", _cmd_eval),
        ("refresh", "Re-ingest and rebake if the documents changed.", _cmd_refresh),
        ("bom", "Write the bill of materials that ships with the delivery.", _cmd_bom),
        (
            "manifest",
            "Record the delivered configuration, so later drift is visible.",
            _cmd_manifest,
        ),
        ("doctor", "Check this build's setup and report; change nothing.", _cmd_doctor),
        ("serve", "Run the local HTTP service.", _cmd_serve),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="Path to the client config TOML.")
        p.set_defaults(func=handler)

        if name == "ask":
            p.add_argument("question")
            p.add_argument(
                "--no-cache",
                action="store_true",
                help="Force a live model call, ignoring cached answers.",
            )
        if name == "bake":
            p.add_argument(
                "--question",
                action="append",
                help="Bake this question instead of the configured set (repeatable).",
            )
        if name == "eval":
            p.add_argument(
                "--suite", required=True, help="Path to the client's eval suite TOML."
            )
        if name == "bom":
            p.add_argument(
                "--output",
                help="Write to this file instead of stdout (e.g. BILL_OF_MATERIALS.md).",
            )
        if name == "manifest":
            p.add_argument(
                "--output",
                help="Write to this file instead of stdout "
                "(the delivery expects DELIVERY_MANIFEST.json).",
            )
            p.add_argument(
                "--build-id",
                help="Provenance for this build, recorded in the manifest and "
                "in the image label.",
            )
        if name == "doctor":
            p.add_argument(
                "--share",
                action="store_true",
                help="Withhold filenames, headings and question text, so the "
                "report can be sent to whoever is supporting this build.",
            )
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        return args.func(args)
    except NotIngested as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
