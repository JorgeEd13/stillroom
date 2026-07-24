"""The engine: grounding, citations, caching, and the invalidation boundary.

`test_a_reingest_invalidates_a_cached_answer` is the most important test in this
repo. The immutable-corpus rule permits caching whole answers only because a re-ingest busts them;
if that stops being true, the product confidently serves answers from documents
the client has already changed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from stillroom.engine import Answer, Engine, NotIngested
from stillroom.pipeline import ingest
from stillroom.prompts import NO_ANSWER


def drain(agen) -> list[dict]:
    """Collect an async generator into a list.

    A stdlib helper rather than pytest-asyncio: four async tests do not justify
    a plugin dependency in a deliverable that ships to client machines.
    """

    async def _collect() -> list[dict]:
        return [event async for event in agen]

    return asyncio.run(_collect())


@pytest.fixture
def engine(config, embedding, fake_model) -> Engine:
    ingest(config, embedding_function=embedding, embedding_name="test")
    return Engine(config, chat_model=fake_model, embedding_function=embedding)


def test_asking_before_ingest_fails_clearly(config, embedding, fake_model):
    with pytest.raises(NotIngested):
        Engine(config, chat_model=fake_model, embedding_function=embedding)


def test_a_grounded_question_calls_the_model_and_cites_sources(engine, fake_model):
    answer = engine.ask("What is the refund window?")

    assert answer.served_by == "model"
    assert answer.text == fake_model.reply
    assert answer.citations
    assert answer.citations[0]["source"] == "handbook.md"


def test_citations_are_built_from_retrieval_not_from_the_model(config, embedding):
    """A model that cites a document it never saw must not be believed."""
    from tests.conftest import FakeChatModel

    liar = FakeChatModel(reply="According to invented-document.pdf, the answer is 12.")
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=liar, embedding_function=embedding)

    answer = engine.ask("What is the refund window?")

    sources = {c["source"] for c in answer.citations}
    assert "invented-document.pdf" not in sources
    assert sources <= {"handbook.md", "shipping.txt"}


def test_an_unrelated_question_says_so_without_calling_the_model(engine, fake_model):
    answer = engine.ask("What is the atomic number of tungsten?")

    assert answer.served_by == "no-match"
    assert answer.text == NO_ANSWER
    assert answer.citations == []
    # The model was never asked to improvise over unrelated passages.
    assert fake_model.calls == []


def test_a_repeated_question_is_served_from_cache(engine, fake_model):
    engine.ask("What is the refund window?")
    again = engine.ask("What is the refund window?")

    assert again.served_by == "cache"
    assert len(fake_model.calls) == 1


def test_no_cache_forces_a_live_call(engine, fake_model):
    engine.ask("What is the refund window?")
    engine.ask("What is the refund window?", use_cache=False)

    assert len(fake_model.calls) == 2


def test_a_reingest_invalidates_a_cached_answer(config, embedding, fake_model):
    """The immutable-corpus honesty boundary. If this breaks, the product lies."""
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=fake_model, embedding_function=embedding)
    engine.ask("What is the refund window?")
    assert engine.ask("What is the refund window?").served_by == "cache"

    # The client changes the policy and re-ingests.
    handbook = Path(config.corpus.path) / "handbook.md"
    handbook.write_text(
        handbook.read_text(encoding="utf-8").replace("30 days", "14 days"),
        encoding="utf-8",
    )
    ingest(config, embedding_function=embedding, embedding_name="test")

    rebuilt = Engine(config, chat_model=fake_model, embedding_function=embedding)
    answer = rebuilt.ask("What is the refund window?")

    assert answer.served_by == "model", "a changed corpus must not serve a stale answer"


def test_a_reingest_purges_the_stale_entries_rather_than_leaving_them(
    config, embedding, fake_model
):
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=fake_model, embedding_function=embedding)
    engine.ask("What is the refund window?")
    assert engine.cache.count() == 1

    handbook = Path(config.corpus.path) / "handbook.md"
    handbook.write_text(
        handbook.read_text(encoding="utf-8").replace("30 days", "14 days"),
        encoding="utf-8",
    )
    ingest(config, embedding_function=embedding, embedding_name="test")

    rebuilt = Engine(config, chat_model=fake_model, embedding_function=embedding)
    # A dead entry that still counts would make the runbook's "N instant
    # answers" number a fiction.
    assert rebuilt.cache.count() == 0


def test_baking_marks_curated_answers_as_curated(engine):
    results = engine.bake_curated()

    assert results == [("What is the refund window?", True)]
    answer = engine.ask("What is the refund window?")
    assert answer.served_by == "cache"
    assert answer.curated is True


def test_baking_reports_a_question_the_documents_cannot_answer(engine):
    """A finding to raise with the client before handover, not a silent skip."""
    results = engine.bake_curated(("What is our policy on space travel?",))

    assert results == [("What is our policy on space travel?", False)]


def test_streaming_emits_sources_then_tokens_then_the_answer(engine):
    events = drain(engine.astream("What is the refund window?"))
    kinds = [e["type"] for e in events]

    assert kinds[0] == "sources"
    assert "token" in kinds
    assert kinds[-1] == "answer"
    assert events[-1]["reply"].strip()


def test_streaming_a_cached_answer_short_circuits(engine):
    engine.ask("What is the refund window?")

    events = drain(engine.astream("What is the refund window?"))

    assert [e["type"] for e in events] == ["cached"]


def test_a_model_failure_surfaces_as_an_error_event(config, embedding):
    class BrokenModel:
        def invoke(self, messages):
            raise RuntimeError("ollama is not running")

        async def astream(self, messages):
            raise RuntimeError("ollama is not running")
            yield  # pragma: no cover

    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=BrokenModel(), embedding_function=embedding)

    events = drain(engine.astream("What is the refund window?"))

    # It fails loudly and locally. There is nowhere for it to fall back to,
    # which is the no-cloud-fallback guarantee behaving as designed.
    assert events[-1]["type"] == "error"
    assert "ollama" in events[-1]["message"]


def test_answers_carry_no_citations_when_nothing_matched(engine):
    assert Answer(text=NO_ANSWER).citations == []


# ------------------------------------------- the relevance floor ---
#
# The floor used to decide only *whether* to call the model. Everything Chroma
# returned then went into the prompt and out to the client as a source — found
# in a delivered container, where a correct answer about shipping arrived with
# four unrelated "sources" under it, two of them at negative similarity.


def test_only_passages_above_the_floor_are_cited(config, embedding, fake_model):
    """The fixture corpus scores 0.24-0.31 on topic and 0.09-0.13 off it, so a
    floor of 0.20 admits some of what `k=3` returns and rejects the rest."""
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=fake_model, embedding_function=embedding)

    answer = engine.ask("What is the notice period?")

    assert answer.citations, "a question the corpus answers should cite something"
    floor = config.retrieval.min_similarity
    assert all(c["similarity"] >= floor for c in answer.citations)


def test_the_model_is_never_shown_a_passage_below_the_floor(
    config, embedding, fake_model
):
    """The half that the client cannot see, and the one that changes answers.

    Unrelated text in the prompt is what a model improvises from — the failure
    the floor exists to prevent, which the floor was not being used to prevent.
    """
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=fake_model, embedding_function=embedding)

    retrieval = engine.retrieve("What is the notice period?")
    rejected = [p for p in retrieval.passages if p not in retrieval.relevant]
    assert rejected, "this fixture must retrieve something below the floor to be a test"

    engine.ask("What is the notice period?")

    prompt = str(fake_model.calls[-1][-1].content)
    for passage in rejected:
        assert passage.text not in prompt


def test_a_refusal_still_cites_nothing_at_all(config, embedding, fake_model):
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=fake_model, embedding_function=embedding)

    answer = engine.ask("What is our policy on interplanetary travel?")

    assert answer.served_by == "no-match"
    assert answer.citations == []
