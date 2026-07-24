"""Conversation memory, and the three rules that keep it from costing anything.

The engine was stateless until 2026-07-21. What forced the change was not a bug
report but a measurement: *"And after that period?"* was refused, and *"How long
does it take?"* — asked in a conversation about refunds — came back as a
confident, cited answer about **shipping**.

Memory fixes that. It also introduces three ways to break the product, and every
test here is about one of them:

1. history becoming a **source** of facts it never earned a citation for,
2. history **poisoning the shared cache**, whose key has no idea which
   conversation an answer came from,
3. history **crowding the retrieved passages** out of a small context window.
"""

from __future__ import annotations

import pytest

from stillroom.conversation import Turn, needs_context, retrieval_query, trim
from stillroom.engine import Engine
from stillroom.pipeline import ingest

REFUND = Turn("What is the refund window?", "Refunds are within 30 days [1].")


# --------------------------------------------------------- the silent counter ---


def test_only_the_most_recent_turns_survive():
    history = tuple(Turn(f"q{i}", f"a{i}") for i in range(10))

    kept = trim(history, max_turns=3, max_chars=100)

    assert [turn.question for turn in kept] == ["q7", "q8", "q9"]


def test_each_turn_is_truncated_rather_than_dropped():
    """A follow-up needs an earlier answer's topic, which is at the front."""
    history = (Turn("q", "A" * 5000),)

    kept = trim(history, max_turns=6, max_chars=600)

    assert len(kept[0].answer) == 600


def test_memory_can_be_switched_off_entirely(config, embedding, fake_model):
    """A build configured without it must not carry it — and the client's own
    documents must not be pushed out of the window by a feature they did not
    buy."""
    config.conversation.enabled = False
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=fake_model, embedding_function=embedding)

    engine.ask("What is the notice period?", history=(REFUND,))

    assert "refund window" not in str(fake_model.calls[-1][-1].content).lower()


# ------------------------------------------------------------- retrieval ---
#
# The half that history alone does not fix. The model can understand "how long
# does it take?" from a transcript; the *search* still runs on six words.


@pytest.mark.parametrize(
    "question",
    ["And after that?", "How long does it take?", "What about it?", "E depois?"],
)
def test_a_fragment_borrows_the_previous_question(question):
    assert needs_context(question)
    assert retrieval_query(question, (REFUND,)).startswith(REFUND.question)


def test_a_complete_new_question_searches_on_its_own_terms():
    question = "What is the bicycle allowance for employees who commute daily?"

    assert not needs_context(question)
    assert retrieval_query(question, (REFUND,)) == question


def test_the_previous_ANSWER_is_never_searched():
    """It is long and full of incidental nouns; it would drag the search towards
    whatever it happened to mention in passing."""
    query = retrieval_query("How long does it take?", (REFUND,))

    assert "30 days" not in query


# ------------------------------------------------------------- the cache ---


def test_a_follow_up_never_reads_the_shared_cache(config, embedding, fake_model):
    """The killer case. The cache is keyed on question text, and a follow-up's
    text does not identify it: "how long does it take?" means one thing after a
    question about refunds and another after one about shipping."""
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=fake_model, embedding_function=embedding)

    # Warm the cache the ordinary way, with no conversation.
    engine.ask("What is the refund window?")
    assert engine.ask("What is the refund window?").served_by == "cache"

    # The same question, now as a follow-up, must go to the model instead.
    answer = engine.ask("What is the refund window?", history=(REFUND,))

    assert answer.served_by == "model"


def test_a_follow_up_never_writes_to_the_shared_cache(config, embedding, fake_model):
    """Worse than reading it: a conversational answer stored under an ambiguous
    question is served to everybody afterwards, until the next ingest."""
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=fake_model, embedding_function=embedding)

    engine.ask("What is the notice period?", history=(REFUND,))

    assert engine.cache is not None
    hit = engine.cache.lookup("What is the notice period?")
    assert hit is None


# --------------------------------------------------- history is not a source ---


def test_the_history_reaches_the_model_as_conversation_not_as_a_source(
    config, embedding, fake_model
):
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=fake_model, embedding_function=embedding)

    engine.ask("What is the notice period?", history=(REFUND,))
    prompt = str(fake_model.calls[-1][-1].content)

    assert "NOT a source" in prompt
    # It is above the sources, so the most authoritative block is nearest the
    # question — see `build_user_prompt`.
    assert prompt.index("Conversation so far") < prompt.index("Sources —")


def test_citations_still_come_only_from_retrieval(config, embedding, fake_model):
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=fake_model, embedding_function=embedding)

    answer = engine.ask("What is the notice period?", history=(REFUND,))

    assert all(c["source"] in {"handbook.md", "shipping.txt"} for c in answer.citations)


# ------------------------------------------------ the standing context doc ---


def test_the_standing_context_is_background_and_is_never_cited(
    config, embedding, fake_model, tmp_path
):
    notes = tmp_path / "context.md"
    notes.write_text("Acme Ltd sells bicycles. FD means finance director.", "utf-8")
    config.context.path = str(notes)
    ingest(config, embedding_function=embedding, embedding_name="test")
    engine = Engine(config, chat_model=fake_model, embedding_function=embedding)

    answer = engine.ask("What is the notice period?")
    prompt = str(fake_model.calls[-1][-1].content)

    assert "FD means finance director" in prompt
    assert "never cite these" in prompt
    assert all(c["source"] != str(notes) for c in answer.citations)


def test_a_missing_context_file_is_a_hard_error(config, embedding, fake_model):
    """Silently dropping the client's source of truth produces answers that are
    plausible and slightly wrong, forever."""
    config.context.path = "/nowhere/context.md"
    ingest(config, embedding_function=embedding, embedding_name="test")

    with pytest.raises(FileNotFoundError):
        Engine(config, chat_model=fake_model, embedding_function=embedding)


def test_editing_the_standing_context_invalidates_cached_answers(
    config, embedding, fake_model, tmp_path
):
    """It changes answers without changing a single document, so the corpus
    fingerprint cannot see it."""
    from stillroom.answers.cache import answer_key

    before = answer_key(model_name="m", prompt_version="4", k=5, standing_context="a")
    after = answer_key(model_name="m", prompt_version="4", k=5, standing_context="b")

    assert before != after
