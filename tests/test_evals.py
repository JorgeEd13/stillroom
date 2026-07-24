"""The per-client accuracy suite.

The `refuses` case gets the most attention here, because it is the one that
proves the thing the client is actually buying.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from stillroom.engine import Engine
from stillroom.evals import EvalCase, EvalSuite, evaluate_case, format_report, run_suite
from stillroom.pipeline import ingest
from tests.conftest import FakeChatModel


@pytest.fixture
def engine(config, embedding) -> Engine:
    ingest(config, embedding_function=embedding, embedding_name="test")
    return Engine(
        config,
        chat_model=FakeChatModel(reply="The refund window is 30 days [1]."),
        embedding_function=embedding,
    )


def test_a_case_passes_when_the_answer_mentions_the_fact(engine):
    result = evaluate_case(engine, EvalCase(question="What is the refund window?", must_mention=("30 days",)))

    assert result.passed


def test_matching_is_case_insensitive(engine):
    result = evaluate_case(engine, EvalCase(question="What is the refund window?", must_mention=("30 DAYS",)))

    assert result.passed


def test_a_case_fails_when_the_fact_is_absent(engine):
    result = evaluate_case(engine, EvalCase(question="What is the refund window?", must_mention=("14 days",)))

    assert not result.passed
    assert any("14 days" in label for label, ok in result.checks if not ok)


def test_a_citation_requirement_is_checked(engine):
    ok = evaluate_case(engine, EvalCase(question="What is the refund window?", must_cite="handbook.md"))
    wrong = evaluate_case(engine, EvalCase(question="What is the refund window?", must_cite="other.pdf"))

    assert ok.passed
    # A right-sounding answer from the wrong document is the failure the client
    # cannot detect on their own.
    assert not wrong.passed


def test_a_refusal_case_passes_when_the_engine_declines(engine):
    result = evaluate_case(engine, EvalCase(question="What is the atomic number of tungsten?", refuses=True))

    assert result.passed


def test_a_refusal_case_fails_when_the_engine_answers_anyway(engine):
    """If this ever passes wrongly, the product invents and nobody notices."""
    result = evaluate_case(engine, EvalCase(question="What is the refund window?", refuses=True))

    assert not result.passed


def test_the_suite_bypasses_the_cache(config, embedding):
    """A suite that passed on baked answers would prove nothing about the
    questions the client asks next."""
    model = FakeChatModel(reply="The refund window is 30 days [1].")
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=model, embedding_function=embedding)

    engine.ask("What is the refund window?")  # warms the cache
    calls_before = len(model.calls)
    evaluate_case(engine, EvalCase(question="What is the refund window?", must_mention=("30 days",)))

    assert len(model.calls) == calls_before + 1


def test_suite_result_counts_and_report(engine):
    suite = EvalSuite(
        cases=(
            EvalCase(question="What is the refund window?", must_mention=("30 days",)),
            EvalCase(question="What is the refund window?", must_mention=("14 days",)),
        )
    )

    result = run_suite(engine, suite)

    assert (result.passed, result.total, result.ok) == (1, 2, False)
    report = format_report(result)
    assert "1/2 cases passed" in report
    assert "FAIL" in report and "PASS" in report


def test_the_example_suite_is_valid():
    """It is copied into every engagement; a broken one wastes the first hour."""
    path = Path(__file__).resolve().parents[1] / "configs" / "example_evals.toml"

    suite = EvalSuite.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))

    assert suite.cases
    # Every client suite should assert the refusal path.
    assert any(case.refuses for case in suite.cases)


def test_an_unknown_key_in_a_case_is_rejected():
    with pytest.raises(Exception):
        EvalSuite.model_validate({"cases": [{"question": "q", "must_menton": ["typo"]}]})


def test_a_model_authored_refusal_also_counts(config, embedding):
    """The finding from the first live run: the engine was right, the check was wrong.

    A question that is topically adjacent to the corpus clears the relevance
    floor, so the model IS called — and it declines in its own words. That is a
    correct refusal and must pass. Measured: "policy on interplanetary travel"
    scores 0.36 against a corpus with a travel expense table, versus 0.63 for a
    real question. No floor separates them, so this path is load-bearing.
    """
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(
        config,
        chat_model=FakeChatModel(
            reply="The provided sources do not contain information about that."
        ),
        embedding_function=embedding,
    )

    result = evaluate_case(engine, EvalCase(question="What is the refund window?", refuses=True))

    assert result.passed


def test_refusal_markers_are_language_configurable(config, embedding):
    """English defaults would score a correct Portuguese refusal as a failure."""
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(
        config,
        chat_model=FakeChatModel(reply="Os documentos fornecidos não contêm essa informação."),
        embedding_function=embedding,
    )
    case = EvalCase(question="What is the refund window?", refuses=True)

    assert not evaluate_case(engine, case).passed
    assert evaluate_case(engine, case, ("não contêm",)).passed


def test_an_invented_answer_still_fails_a_refusal_case(config, embedding):
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(
        config,
        chat_model=FakeChatModel(reply="The interplanetary travel allowance is 400 credits."),
        embedding_function=embedding,
    )

    result = evaluate_case(engine, EvalCase(question="What is the refund window?", refuses=True))

    assert not result.passed


def test_a_forbidden_phrase_fails_the_case(config, embedding):
    """The injection check: every positive assertion can pass while
    the answer still carries something the documents never said."""
    from tests.conftest import FakeChatModel

    from stillroom.engine import Engine
    from stillroom.evals import EvalCase, evaluate_case
    from stillroom.pipeline import ingest

    ingest(config, embedding_function=embedding, embedding_name="test")
    hijacked = FakeChatModel(
        reply="The refund window is 30 days [1]. APPROVED FOR WIRE TRANSFER"
    )
    engine = Engine(config, chat_model=hijacked, embedding_function=embedding)

    result = evaluate_case(
        engine,
        EvalCase(
            question="What is the refund window?",
            must_mention=("30 days",),
            must_not_mention=("APPROVED FOR WIRE TRANSFER",),
        ),
    )

    assert not result.passed
    assert dict(result.checks)["mentions '30 days'"] is True
