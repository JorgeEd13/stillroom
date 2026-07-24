"""The one retry that stands between a raised floor and a refused real question.

⚠️ **Why this exists**. Moving `min_similarity` from 0.25 to 0.50 is
what makes the first refusal gate a gate again under `bge-m3`. It costs the typo
tolerance F3.5 measured: a heavily mistyped question scores 0.314 and is refused
as though the documents did not cover it. **And no floor value fixes that** — the
far band that must be refused reaches 0.428, so mangled-but-real scores *below*
unrelated-but-clean. That conflict held for all three embedders tested.

**The safety argument is entirely in the ordering**, and these tests are written
against the ordering rather than against any score: the original question is
always searched first, the retry only runs when that already failed the floor,
and the better result wins. A correction can therefore only ever rescue
something the gate had already refused — it can never lower the gate.
"""

from __future__ import annotations

from stillroom.index.retrieval import CorpusSpeller

DOCUMENTS = [
    "## Refund window\nCustomers may request a refund within 30 days of delivery.",
    "## Notice period\nEmployees must give thirty days of written notice before leaving.",
    "## Restocking\nA restocking fee of ten percent applies to opened items.",
]


def speller(**kwargs) -> CorpusSpeller:
    return CorpusSpeller(DOCUMENTS, **kwargs)


def test_a_misspelling_is_corrected_towards_the_corpus():
    """`refnud` is not a word in any dictionary we ship — and we ship none. It
    is corrected because `refund` is in the client's own documents."""
    assert speller().correction("what is the refnud window") == "what is the refund window"


def test_a_clean_question_produces_no_correction_at_all():
    """None means "do not bother searching twice". The happy path must not pay
    for this feature."""
    assert speller().correction("what is the refund window") is None


def test_a_word_the_corpus_does_not_contain_is_left_alone():
    """⚠️ The guard against the failure the first draft had.

    Question words are absent from a corpus of policies, so an eager corrector
    turned `what` into `that` and made clean questions score *worse*. Anything
    it cannot confidently match must survive untouched — and the retry design
    means even a bad correction can only be discarded."""
    corrected = speller().correction("what is the quantum entanglement policy")

    # `quantum` and `entanglement` have no near neighbour in this corpus.
    assert corrected is None or "quantum" in corrected


def test_it_cannot_invent_a_word_the_corpus_does_not_have():
    """This is what stops it rescuing nonsense. The dictionary IS the corpus, so
    a correction can only ever move a token towards something the client
    actually wrote."""
    corrected = speller().correction("how many moons does a refnud have")

    assert corrected is not None
    assert "refund" in corrected
    assert "moons" in corrected, "a word with no corpus neighbour was replaced anyway"


def test_short_tokens_are_left_alone():
    """At three or four characters every word has a neighbour and edit distance
    stops discriminating. `the` must not become `fee`."""
    corrected = speller(min_token_length=5).correction("teh refnud window")

    assert corrected is not None
    assert corrected.startswith("teh ")


def test_an_empty_corpus_corrects_nothing_rather_than_raising():
    """An index that was pruned to nothing still has to answer questions —
    with a refusal, not a traceback."""
    assert CorpusSpeller([]).correction("anything at all here") is None


# --------------------------------------------------------------------------
# The ordering, at the engine. This is the whole safety argument, so it is
# asserted on the SEQUENCE of searches rather than on any similarity value —
# a score assertion here would be measuring the offline fixture, which shares
# nothing with the shipped embedder but the shape of its output.
# --------------------------------------------------------------------------

import pytest  # noqa: E402

import stillroom.engine as engine_module  # noqa: E402
from stillroom.engine import Engine  # noqa: E402
from stillroom.index.retrieval import Retrieval  # noqa: E402
from stillroom.pipeline import ingest  # noqa: E402


@pytest.fixture
def built(config, embedding, fake_model) -> Engine:
    ingest(config, embedding_function=embedding, embedding_name="test")
    return Engine(config, chat_model=fake_model, embedding_function=embedding)


def _spy_on_search(monkeypatch, results):
    """Replace `search` with one that returns scripted results in order."""
    queries = []
    remaining = list(results)

    def fake_search(collection, question, *, k, min_similarity):
        queries.append(question)
        return remaining.pop(0) if remaining else Retrieval((), grounded=False)

    monkeypatch.setattr(engine_module, "search", fake_search)
    return queries


def test_a_grounded_first_search_is_never_retried(built, monkeypatch):
    """The happy path pays nothing for this feature — no second embed, no
    second query, no corpus read to build the vocabulary."""
    queries = _spy_on_search(monkeypatch, [Retrieval((), grounded=True)])

    built.retrieve("what is the refund window")

    assert len(queries) == 1


def test_a_refused_question_is_searched_a_second_time_respelled(built, monkeypatch):
    """The original goes first, always. The retry is a *second* search."""
    queries = _spy_on_search(
        monkeypatch,
        [Retrieval((), grounded=False), Retrieval((), grounded=True)],
    )

    built.retrieve("what is the refnud window")

    assert len(queries) == 2
    assert queries[0] == "what is the refnud window", "the original was not searched first"
    assert "refund" in queries[1], "the retry did not use the corrected spelling"


def test_a_retry_that_also_fails_returns_the_ORIGINAL_result(built, monkeypatch):
    """⚠️ The correction must never be what the client is answered from unless
    it actually worked. Returning the retry's empty result would replace the
    real near-misses in `passages` — the numbers that explain a refusal when a
    client's floor is being tuned."""
    original = Retrieval((), grounded=False, min_similarity=0.5)
    _spy_on_search(monkeypatch, [original, Retrieval((), grounded=False)])

    assert built.retrieve("what is the refnud window") is original


def test_the_retry_can_be_turned_off_per_engagement(built, monkeypatch):
    """A client is a config, not a fork."""
    built.config.retrieval.spelling_retry.enabled = False
    queries = _spy_on_search(monkeypatch, [Retrieval((), grounded=False)])

    built.retrieve("what is the refnud window")

    assert len(queries) == 1


# --------------------------------------------------------------------------
# The degenerate-input guard. Not part of the spelling retry, but measured in
# the same regression pass and load-bearing for the same reason: under bge-m3
# an absolute floor no longer catches input with nothing in it.
# --------------------------------------------------------------------------


def test_input_with_no_words_never_reaches_the_model(built, monkeypatch):
    """⚠️ **A regression the embedder swap caused** (the troll
    battery re-run).

    That battery recorded that degenerate input was "short-circuited by the floor
    with no model call". Under bge-m3 it is not: measured against the English
    corpus, `"???"` scores **0.518** and pure whitespace scores **0.588**, both
    clear of a 0.50 floor. With nothing to be about, the vector lands near the
    corpus centroid — and under this embedder the centroid is high. No floor
    value fixes that, because the score is not low.
    """
    queries = _spy_on_search(monkeypatch, [Retrieval((), grounded=True)])

    for degenerate in ("???", "   ", "...!!!", ""):
        assert not built.retrieve(degenerate).grounded

    assert queries == [], "degenerate input was embedded and searched anyway"


def test_a_question_in_a_non_latin_script_is_not_mistaken_for_degenerate(
    built, monkeypatch
):
    """The guard is `\\w`, which matches Han, Hangul, Cyrillic and Arabic. A
    codepoint-range test would have refused every Chinese question in the
    product whose whole point is that it works in the documents' language."""
    queries = _spy_on_search(monkeypatch, [Retrieval((), grounded=True)])

    built.retrieve("退款期限是多久?")

    assert len(queries) == 1
