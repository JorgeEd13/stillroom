"""The answer cache, unit-level.

`test_engine` covers invalidation end to end through a real re-ingest. These
tests pin the cache's own behaviour, including the cases the engine tests cannot
easily reach — a model swap, and a threshold that is doing its job.
"""

from __future__ import annotations

import pytest

from stillroom.answers.cache import answer_key, get_answer_cache
from stillroom.index.store import open_client

FINGERPRINT = "corpus-v1"
KEY = answer_key(model_name="ollama:qwen2.5:7b", prompt_version="1", k=5)


@pytest.fixture
def cache_factory(tmp_path, embedding):
    client = open_client(str(tmp_path / "index"))

    def make(*, fingerprint: str = FINGERPRINT, key: str = KEY):
        return get_answer_cache(
            client, "answers", embedding, threshold=0.90, fingerprint=fingerprint, key=key
        )

    return make


def test_a_warmed_answer_comes_back(cache_factory):
    cache = cache_factory()
    cache.warm("What is the refund window?", "30 days.", [{"source": "h.md"}])

    hit = cache.lookup("What is the refund window?")

    assert hit is not None
    assert hit.answer == "30 days."
    assert hit.citations == [{"source": "h.md"}]
    assert hit.curated is False


def test_curated_entries_are_marked_as_such(cache_factory):
    """The UI distinguishes them, because they earn different trust."""
    cache = cache_factory()
    cache.warm("What is the refund window?", "30 days.", [], curated=True)

    assert cache.lookup("What is the refund window?").curated is True


def test_an_unrelated_question_misses(cache_factory):
    cache = cache_factory()
    cache.warm("What is the refund window?", "30 days.", [])

    assert cache.lookup("How do I reset my password?") is None


def test_a_new_corpus_fingerprint_misses(cache_factory):
    """The immutable-corpus boundary: the documents changed, so the answer is dead."""
    cache_factory().warm("What is the refund window?", "30 days.", [])

    assert cache_factory(fingerprint="corpus-v2").lookup("What is the refund window?") is None


def test_a_stale_entry_is_deleted_rather_than_skipped(cache_factory):
    """Skipping would leave dead entries that still count towards the total."""
    cache_factory().warm("What is the refund window?", "30 days.", [])
    moved_on = cache_factory(fingerprint="corpus-v2")

    moved_on.lookup("What is the refund window?")

    assert moved_on.count() == 0


def test_swapping_the_model_invalidates_cached_answers(cache_factory):
    """The fingerprint covers the documents, not the model that read them."""
    cache_factory().warm("What is the refund window?", "30 days.", [])
    other_model = answer_key(model_name="ollama:llama3.2:3b", prompt_version="1", k=5)

    assert cache_factory(key=other_model).lookup("What is the refund window?") is None


def test_changing_the_prompt_version_invalidates_cached_answers(cache_factory):
    cache_factory().warm("What is the refund window?", "30 days.", [])
    new_prompt = answer_key(model_name="ollama:qwen2.5:7b", prompt_version="2", k=5)

    assert cache_factory(key=new_prompt).lookup("What is the refund window?") is None


def test_purge_stale_drops_only_the_invalidated_entries(cache_factory):
    old = cache_factory()
    old.warm("What is the refund window?", "30 days.", [])

    current = cache_factory(fingerprint="corpus-v2")
    current.warm("What is the notice period?", "Four weeks.", [])

    assert current.purge_stale() == 1
    assert current.count() == 1
    assert current.lookup("What is the notice period?") is not None


def test_warming_the_same_question_twice_updates_in_place(cache_factory):
    cache = cache_factory()
    cache.warm("What is the refund window?", "30 days.", [])
    cache.warm("What is the refund window?", "14 days.", [])

    assert cache.count() == 1
    assert cache.lookup("What is the refund window?").answer == "14 days."


def test_answer_key_is_stable_for_the_same_inputs():
    a = answer_key(model_name="m", prompt_version="1", k=5)
    b = answer_key(model_name="m", prompt_version="1", k=5)

    assert a == b


def test_answer_key_changes_with_k():
    """A different k means the model was shown a different amount of context."""
    a = answer_key(model_name="m", prompt_version="1", k=5)
    b = answer_key(model_name="m", prompt_version="1", k=8)

    assert a != b
