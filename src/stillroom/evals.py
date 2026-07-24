# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""The per-deployment accuracy suite: an accuracy check over your own questions.

This is it: a
client's own questions, each with the **properties** a correct answer must have,
run against the built engine.

**Properties, not expected strings.** A local model paraphrases; asserting on
exact text would fail on a correct answer and make the suite worthless within a
day. `receivables-agent`'s golden suite settled this already — check that the
answer *mentions the right things* and *cites the right document*, not that it
matches a sentence.

**The `refuses` case is the one that matters.** Anyone can show a chatbot
answering questions it can answer. The reason to run this at all — rather than a
free cloud tool — is that it does not invent. A case
marked `refuses = true` asserts the engine declines a question its documents do
not cover, and it is the case worth showing them at handover.

This runs live against the client's real model, so it is a **delivery artifact**,
not part of the offline suite. `tests/` proves the engine works; this proves
*their* build answers *their* questions.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from stillroom.config import StrictModel
from stillroom.engine import Engine
from stillroom.prompts import no_answer


class EvalCase(StrictModel):
    """One question and the properties its answer must have."""

    question: str
    # Substrings the answer must contain, matched case-insensitively. Use the
    # load-bearing facts (a number, a name, a deadline), never a whole sentence.
    must_mention: tuple[str, ...] = ()
    # Substrings the answer must NOT contain. The case this exists
    # for is prompt injection: a document that instructs the model produces a
    # *fluent, confident* answer, so every positive check still passes while the
    # answer carries something the corpus never said. Assert the phrase's
    # absence against an innocent question that retrieves the tainted document.
    must_not_mention: tuple[str, ...] = ()
    # Source file the answer must cite. Catches a right-sounding answer that
    # came from the wrong document — the failure a client cannot see.
    must_cite: str | None = None
    # Assert the engine REFUSES. For a question their documents do not cover.
    refuses: bool = False


# Phrases that mark a model-authored refusal. See `EvalSuite.refusal_markers`
# for why this list exists and why it is configurable.
DEFAULT_REFUSAL_MARKERS: tuple[str, ...] = (
    "do not contain",
    "does not contain",
    "not contain information",
    "could not find",
    "cannot find",
    "no information",
    "not mentioned",
    "not specified",
    "cannot provide",
    "unable to answer",
)


class EvalSuite(StrictModel):
    cases: tuple[EvalCase, ...]
    # ⚠️ **Set these for any corpus that is not in English.**
    #
    # A refusal arrives by one of two paths (see `engine`), and only one of them
    # is machine-detectable for free. Either retrieval found nothing above the
    # relevance floor — the engine short-circuits and never calls the model —
    # or retrieval found something *topically adjacent but unanswerable*, the
    # model is called, and it declines in its own words. The second path cannot
    # be caught by raising the floor, because that is exactly where legitimate
    # questions live too (measured: "what is our policy on interplanetary
    # travel?" scores 0.36 against a corpus with an expense-policy travel table,
    # while a real question scores 0.63 — there is no threshold between them).
    #
    # So a model-authored refusal is detected by phrase, and phrases are
    # language-specific. A Portuguese corpus refuses with "não contém"; the
    # English defaults would score that as a failure and the suite would report
    # a correct system as broken.
    refusal_markers: tuple[str, ...] = DEFAULT_REFUSAL_MARKERS

    @classmethod
    def load(cls, path: str | Path) -> "EvalSuite":
        raw = Path(path).read_bytes()
        return cls.model_validate(tomllib.loads(raw.decode("utf-8")))


@dataclass
class CaseResult:
    question: str
    checks: list[tuple[str, bool]] = field(default_factory=list)
    answer: str = ""

    @property
    def passed(self) -> bool:
        return all(ok for _, ok in self.checks)


@dataclass
class SuiteResult:
    results: list[CaseResult]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def ok(self) -> bool:
        return self.passed == self.total


def evaluate_case(
    engine: Engine,
    case: EvalCase,
    refusal_markers: tuple[str, ...] = DEFAULT_REFUSAL_MARKERS,
) -> CaseResult:
    """Run one case live. Cache is bypassed so this tests the engine, not the cache.

    That distinction matters at handover: a suite that passed because every
    question was a baked curated answer would prove nothing about the questions
    the client asks *next*.
    """
    answer = engine.ask(case.question, use_cache=False)
    result = CaseResult(question=case.question, answer=answer.text)

    if case.refuses:
        # Both refusal paths count. The engine short-circuiting is the clean
        # one; the model declining in its own words is equally correct, and on
        # a topically-adjacent question it is the *only* one available.
        lowered = answer.text.lower()
        declined = (
            answer.served_by == "no-match"
            or answer.text == no_answer(engine.config.language)
            or any(marker in lowered for marker in refusal_markers)
        )
        result.checks.append(("declines to answer", declined))
        return result

    result.checks.append(("answered", answer.served_by != "no-match"))

    lowered = answer.text.lower()
    for phrase in case.must_mention:
        result.checks.append((f"mentions {phrase!r}", phrase.lower() in lowered))

    for phrase in case.must_not_mention:
        result.checks.append(
            (f"does not mention {phrase!r}", phrase.lower() not in lowered)
        )

    if case.must_cite:
        cited = {str(c["source"]) for c in answer.citations}
        result.checks.append((f"cites {case.must_cite!r}", case.must_cite in cited))

    return result


def run_suite(engine: Engine, suite: EvalSuite) -> SuiteResult:
    return SuiteResult(
        [evaluate_case(engine, case, suite.refusal_markers) for case in suite.cases]
    )


def format_report(result: SuiteResult) -> str:
    """A report a client can read, because at handover they will."""
    lines: list[str] = []
    for case in result.results:
        lines.append(f"{'PASS' if case.passed else 'FAIL'}  {case.question}")
        for label, ok in case.checks:
            if not ok:
                lines.append(f"        missing: {label}")
        if not case.passed:
            snippet = case.answer.strip().replace("\n", " ")[:160]
            lines.append(f"        answered: {snippet}")

    lines.append("")
    lines.append(f"{result.passed}/{result.total} cases passed.")
    return "\n".join(lines)
