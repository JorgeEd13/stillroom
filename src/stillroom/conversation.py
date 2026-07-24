# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""Conversation state: what the engine is allowed to remember, and its limits.

**Why this exists at all.** The engine was stateless, and that was defensible
right up until somebody used it: a client who has just been given an answer asks
*"and after that?"*. Measured before this module existed, against a real corpus
and a live model:

| turn | what happened |
|---|---|
| *"What is the refund window?"* | correct |
| *"And after that period?"* | **refused** — nothing retrieved above the floor |
| *"How long does it take?"* | **answered about international shipping**, while the conversation was about refunds |

The refusals are merely annoying. **The third row is the dangerous one**: a
short follow-up matched a different document and produced a confident, cited,
wrong-topic answer, with nothing on screen saying the topic had moved. And the
page has always had a *New chat* button, so the product was already promising a
continuity it did not have.

**Three rules govern everything here, and they are what keep the product's claim
intact while adding memory:**

1. **History disambiguates; it never informs.** It exists so the engine can tell
   what *"it"* refers to. Facts still come only from retrieved passages, and
   citations still come only from retrieval — a model must never be able to
   answer from something it said earlier, because that answer would carry a
   citation it did not earn.
2. **History is untrusted text.** It arrives from the browser, which means the
   user composed it. It is length-capped, turn-capped and marker-stripped like
   any other input.
3. **A conversation turn is never cached.** See `Engine.ask` — a cached answer to
   *"how long does it take?"* is correct in one conversation and wrong in the
   next, and the cache key has no idea which conversation it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Words that make a question depend on the one before it. English and Brazilian
# Portuguese, matching `prompts._LANGUAGE_NAMES` — the languages this ships in.
#
# ⚠️ This list decides when the *retrieval query* borrows the previous question,
# and being wrong is cheap in one direction only: borrowing when it was not
# needed adds a little noise to a search that would have worked, while not
# borrowing when it was needed is the wrong-topic answer above. So it is
# deliberately generous.
_ANAPHORA = frozenset(
    {
        "it", "its", "that", "this", "these", "those", "they", "them", "their",
        "there", "then", "he", "she", "him", "her", "one", "ones", "same",
        "isso", "isto", "aquilo", "esse", "essa", "este", "esta", "aquele",
        "aquela", "ele", "ela", "eles", "elas", "dele", "dela", "lá", "ali",
        "mesmo", "mesma",
    }
)

# A question this short is almost certainly a fragment of the previous one
# ("and after that?", "how long?", "e depois?") whether or not it contains a
# pronoun.
_SHORT_QUESTION_WORDS = 6


@dataclass(frozen=True)
class Turn:
    """One exchange that already happened."""

    question: str
    answer: str


def trim(
    history: tuple[Turn, ...], *, max_turns: int, max_chars: int
) -> tuple[Turn, ...]:
    """Keep the conversation inside what the model can actually hold.

    **This is the silent counter.** A local model has a fixed context window, and
    the sources plus the system prompt are the part that must not be squeezed —
    an answer is worthless if the passage it should have quoted fell out of the
    window to make room for small talk. So history is trimmed **first and
    silently**: the client is never shown a "conversation too long" error,
    because that is our budget problem and not something they did wrong.

    Oldest turns go first, and each surviving answer is truncated: what a
    follow-up needs from an earlier answer is its *topic*, which is at the
    beginning, not its full text.
    """
    kept = history[-max_turns:] if max_turns > 0 else ()
    return tuple(
        Turn(
            question=turn.question[:max_chars],
            answer=turn.answer[:max_chars],
        )
        for turn in kept
    )


def needs_context(question: str) -> bool:
    """Does this question only make sense against the one before it?

    Used to decide whether the **retrieval query** borrows the previous
    question. It is not used to decide what the model sees — the model always
    gets the history it was given, because judging what a model needs to
    understand a sentence is exactly the judgement it is better at than a
    regular expression.
    """
    words = re.findall(r"[\w']+", question.lower())
    if len(words) <= _SHORT_QUESTION_WORDS:
        return True
    return bool(set(words) & _ANAPHORA)


def retrieval_query(question: str, history: tuple[Turn, ...]) -> str:
    """The text actually searched against the index.

    ⚠️ **Retrieval is the half that history alone does not fix, and it is the
    half that produced the dangerous failure.** Handing the model a transcript
    lets it *understand* "how long does it take?", but the search still runs on
    those six words, and six words about nothing in particular land wherever the
    embedding space happens to put them — which is how a refund conversation
    returned shipping times.

    So an anaphoric question searches with the **previous question** prepended.
    The previous *answer* is deliberately not included: it is long, it is full
    of incidental nouns, and it would drag the search towards whatever that
    answer mentioned in passing rather than what was being asked about.
    """
    if not history or not needs_context(question):
        return question
    return f"{history[-1].question} {question}"
