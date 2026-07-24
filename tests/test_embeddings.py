"""The offline fixture must be able to tell documents apart in any script.

⚠️ **This file exists because the fixture silently could not, and the failure
was invisible**. `_TOKEN_RE` was `[a-z0-9]+`, so Chinese,
Japanese, Korean, Arabic and Russian text produced **no tokens at all** and
embedded to the **zero vector**. Chroma indexes a zero vector without complaint
and it retrieves at similarity 0.0 against everything — so a multilingual
retrieval test written against this fixture would have passed while proving
nothing, in the phase whose whole point is making multilingual retrieval real.

The guard is `test_no_script_embeds_to_the_zero_vector`: it asserts the
*precondition* rather than an outcome, because an outcome test is exactly what
the old fixture would have passed.
"""

from __future__ import annotations

import io
import json
from unittest import mock

import pytest

from stillroom.index.embeddings import DeterministicEmbeddingFunction

# One document, one question that should find it, one that should not — in five
# scripts, chosen for how they break a naive tokeniser: Han and Japanese have no
# spaces between words at all, Korean has spaces but non-Latin syllables, and
# Cyrillic and Arabic are spaced but outside `[a-z]`.
CASES = {
    "zh": (
        "## 退款期限\n\n客户可在交付后30天内申请退款。",
        "退款期限是多久?",
        "## 补货费\n\n已开封商品需支付10%的补货费。",
    ),
    "ja": (
        "## 返金期限\n\nお客様は配達後30日以内に返金を請求できます。",
        "返金期限はどのくらいですか?",
        "## 在庫補充手数料\n\n開封済み商品には10%の手数料がかかります。",
    ),
    "ko": (
        "## 환불 기간\n\n고객은 배송 후 30일 이내에 환불을 요청할 수 있습니다.",
        "환불 기간은 얼마나 됩니까?",
        "## 재입고 수수료\n\n개봉한 상품에는 10%의 수수료가 부과됩니다.",
    ),
    "ru": (
        "## Срок возврата\n\nКлиент может запросить возврат в течение 30 дней.",
        "Каков срок возврата?",
        "## Плата за пополнение\n\nЗа вскрытые товары взимается 10%.",
    ),
    "ar": (
        "## فترة الاسترداد\n\nيمكن للعميل طلب استرداد خلال 30 يوما من التسليم.",
        "ما هي فترة الاسترداد؟",
        "## رسوم إعادة التخزين\n\nتطبق رسوم عشرة بالمئة على السلع المفتوحة.",
    ),
}


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@pytest.mark.parametrize("script", sorted(CASES))
def test_no_script_embeds_to_the_zero_vector(script: str) -> None:
    """A zero vector is what the old tokeniser produced, and it looks like data.

    Asserted on the norm rather than on retrieval, because a corpus of zero
    vectors retrieves *consistently* — every score is 0.0, the first result is
    whichever Chroma returns first, and a one-document assertion passes.
    """
    embed = DeterministicEmbeddingFunction()
    document, question, _ = CASES[script]
    for text in (document, question):
        vector = embed([text])[0]
        assert sum(v * v for v in vector) ** 0.5 == pytest.approx(1.0), (
            f"{script} embedded to the zero vector — the tokeniser cannot see this script"
        )


@pytest.mark.parametrize("script", sorted(CASES))
def test_the_right_passage_outranks_the_wrong_one(script: str) -> None:
    """Non-zero is not enough; the vectors have to carry which words were there."""
    embed = DeterministicEmbeddingFunction()
    document, question, other = CASES[script]
    right, wrong, asked = embed([document, other, question])

    assert cosine(asked, right) > cosine(asked, wrong)


def test_a_scriptless_run_is_not_one_token() -> None:
    """Han text has no spaces, so a word-run regex reads a sentence as a word.

    Two Chinese sentences that share most of their characters would then share
    no token at all, and score 0.0 against each other — non-zero vectors, and
    still useless.
    """
    embed = DeterministicEmbeddingFunction()
    a, b = embed(["客户可在交付后30天内申请退款", "客户可在交付后申请退款吗"])

    assert cosine(a, b) > 0.5


def test_latin_text_still_tokenises_by_word() -> None:
    """The widened regex must not have turned English into character soup."""
    embed = DeterministicEmbeddingFunction()
    refund, notice, question = embed(
        [
            "Customers may request a refund within 30 days of delivery.",
            "Employees must give four weeks of notice before leaving.",
            "What is the refund window?",
        ]
    )

    assert cosine(question, refund) > cosine(question, notice)


def test_accents_are_part_of_the_word() -> None:
    """`[a-z0-9]+` split `reposição` into `reposi` + `o`.

    Portuguese survived that by accident — enough of each word is unaccented
    that the fragments still matched. It is not a property to keep relying on.
    """
    embed = DeterministicEmbeddingFunction()
    fee, notice, question = embed(
        [
            "Uma taxa de reposição de 10% é cobrada sobre itens abertos.",
            "O colaborador deve dar aviso prévio de 30 dias.",
            "Qual é a taxa de reposição?",
        ]
    )

    assert cosine(question, fee) > cosine(question, notice)


# --------------------------------------------------------------------------
# The Ollama-backed embedder. Offline throughout: `conftest` refuses sockets,
# so these drive `_embed_batch` directly rather than standing up a server.
# --------------------------------------------------------------------------

import urllib.error  # noqa: E402

from stillroom.index import embeddings  # noqa: E402
from stillroom.index.embeddings import (  # noqa: E402
    EmbeddingError,
    OllamaEmbeddingFunction,
    _mean_unit_vector,
    _split_in_half,
)


class FakeOllama:
    """Stands in for `_embed_batch`, refusing whatever it is told to refuse.

    Raises with `passage_rejected=True`, because that is what the real HTTP 500
    NaN failure carries — and it is the *only* failure the split-and-retry may
    react to. A double that raised the generic error would have let the
    deployment/passage confusion back in through the test suite.
    """

    def __init__(self, refuse: set[str] = frozenset(), dim: int = 4):
        self.refuse = set(refuse)
        self.dim = dim
        self.seen: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.seen.append(list(texts))
        bad = [t for t in texts if t in self.refuse]
        if bad:
            raise EmbeddingError(
                f"refused {len(bad)} of {len(texts)}", passage_rejected=True
            )
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]


def test_an_unreachable_ollama_is_NOT_retried_passage_by_passage():
    """⚠️ The distinction the other tests in this file caught by failing.

    Splitting exists for one failure — the model produced NaN for this text.
    An unreachable Ollama is not going to become reachable one half-passage at
    a time, and treating it as a passage problem tells the client their document
    is at fault while grinding through their whole corpus to say so.
    """
    calls = []

    def unreachable(texts):
        calls.append(texts)
        raise EmbeddingError("nothing is answering", passage_rejected=False)

    embed = OllamaEmbeddingFunction(base_url="http://x", model="m")
    embed._embed_batch = unreachable

    with pytest.raises(EmbeddingError):
        embed(["one passage here that is quite long", "another passage here"])

    assert len(calls) == 1, "it retried a deployment failure as if it were a passage"


def test_a_passage_the_model_refuses_is_split_rather_than_dropped():
    """⚠️ Dropping it would make the assistant deny a document it was given.

    Ollama's bge-m3 returns NaN for some ordinary passages (ollama#16625), and
    NaN cannot be serialised, so the request 500s. A dropped passage produces
    "that is not in your documents" about a document the client supplied — the
    silent-wrong-answer class this engine refuses everywhere else.
    """
    passage = "Orders placed before 14:00 are dispatched the same working day and arrive soon."
    fake = FakeOllama(refuse={passage})
    embed = OllamaEmbeddingFunction(base_url="http://x", model="m")
    embed._embed_batch = fake

    vectors = embed([passage])

    assert len(vectors) == 1
    # It was tried whole, refused, then tried in halves.
    assert fake.seen[0] == [passage]
    assert len(fake.seen) > 1


def test_one_bad_passage_does_not_take_the_whole_batch_with_it():
    """Ollama fails the entire request for one unembeddable text, so a batch
    failure says nothing about which text caused it. Every good passage in the
    batch must still be indexed."""
    bad = "b" * 100 + " tail"
    texts = ["first passage here", bad, "third passage here"]
    fake = FakeOllama(refuse={bad})
    embed = OllamaEmbeddingFunction(base_url="http://x", model="m")
    embed._embed_batch = fake

    vectors = embed(texts)

    assert len(vectors) == 3
    assert all(len(v) == 4 for v in vectors)


def test_a_passage_too_short_to_split_raises_and_names_the_cause():
    """The floor of the recursion. It must not loop, and the message must say
    this is an upstream defect rather than something the client did wrong."""
    tiny = "short text"
    embed = OllamaEmbeddingFunction(base_url="http://x", model="m")
    embed._embed_batch = FakeOllama(refuse={tiny})

    with pytest.raises(EmbeddingError) as caught:
        embed([tiny])

    assert "upstream defect" in str(caught.value)
    assert "ollama" in str(caught.value).lower()


def test_the_refusal_path_never_emits_the_passage_text(caplog):
    """⚠️ Security audit (2026-07-23): a debug convenience must not be the one
    path that copies a client document out of the passage.

    The refusal message and the split-retry log used to print the first 90
    characters of the passage — client document text landing in the container
    log and in any support bundle the client later sends us, on a product whose
    whole promise is that the documents stay in the building. It is replaced by a
    content-free reference (length + a short SHA-256 prefix), which still
    identifies *which* chunk hit ollama#16625 without revealing what it says.
    """
    import logging

    secret = "The 2027 acquisition of Northwind closes at a valuation of 4.2 million pounds."

    # 1. The split-retry log line (passage long enough to split) carries no text.
    embed = OllamaEmbeddingFunction(base_url="http://x", model="m")
    embed._embed_batch = FakeOllama(refuse={secret})
    with caplog.at_level(logging.WARNING):
        embed([secret])
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert secret not in logged
    assert "Northwind" not in logged and "4.2 million" not in logged
    # It still names which passage, reproducibly, without the content.
    assert "sha256:" in logged and f"{len(secret)} chars" in logged

    # 2. The too-short-to-split exception message carries no text either.
    tiny = "Project Zephyr"
    embed2 = OllamaEmbeddingFunction(base_url="http://x", model="m")
    embed2._embed_batch = FakeOllama(refuse={tiny})
    with pytest.raises(EmbeddingError) as caught:
        embed2([tiny])
    assert tiny not in str(caught.value)
    assert "Zephyr" not in str(caught.value)
    assert "sha256:" in str(caught.value)


def test_a_missing_model_says_which_command_pulls_it():
    """⚠️ The failure a client actually hits: they pulled the CHAT
    model because the runbook said so, and this is a *second* model. Ollama
    answers 404, and leg B #13 is the standing lesson that the message must name
    the real world it is in."""
    embed = OllamaEmbeddingFunction(base_url="http://host:11434", model="bge-m3:567m")

    def raise_404(request, timeout):
        raise urllib.error.HTTPError("u", 404, "not found", {}, io.BytesIO(b"no such model"))

    with mock.patch.object(embeddings.urllib.request, "urlopen", raise_404):
        with pytest.raises(EmbeddingError) as caught:
            embed(["anything"])

    message = str(caught.value)
    assert "ollama pull bge-m3:567m" in message
    assert "your documents cannot be read" in message.lower()


def test_an_unreachable_ollama_explains_the_container_case():
    """`ingest` gains this failure mode with the move to bge-m3 — it never needed Ollama
    before. The container/loopback explanation is the one that was measured to
    matter twice already."""
    embed = OllamaEmbeddingFunction(base_url="http://host:11434", model="m")

    def refuse(request, timeout):
        raise urllib.error.URLError("connection refused")

    with mock.patch.object(embeddings.urllib.request, "urlopen", refuse):
        with pytest.raises(EmbeddingError) as caught:
            embed(["anything"])

    message = str(caught.value)
    assert "http://host:11434" in message
    assert "container" in message


def test_a_short_reply_is_refused_rather_than_padded_with_blanks():
    """A missing vector would become a zero vector downstream, and a zero vector
    indexes and retrieves silently — the same failure the offline fixture had."""
    embed = OllamaEmbeddingFunction(base_url="http://x", model="m")

    class ShortReply:
        def __enter__(self):
            return io.BytesIO(json.dumps({"embeddings": [[1.0, 0.0]]}).encode())

        def __exit__(self, *a):
            return False

    with mock.patch.object(embeddings.urllib.request, "urlopen", lambda *a, **k: ShortReply()):
        with pytest.raises(EmbeddingError) as caught:
            embed(["one", "two"])

    assert "partly blank" in str(caught.value)


def test_splitting_lands_on_a_word_boundary():
    halves = _split_in_half("the quick brown fox jumps over the lazy dog again and again")

    assert halves is not None
    assert not halves[0].endswith(" ")
    assert " ".join(halves).replace("  ", " ").split() == (
        "the quick brown fox jumps over the lazy dog again and again".split()
    )


def test_the_mean_of_two_halves_is_a_unit_vector():
    """Chroma's cosine space assumes it, and a non-unit vector would score
    differently for reasons unrelated to meaning."""
    merged = _mean_unit_vector([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert sum(v * v for v in merged) ** 0.5 == pytest.approx(1.0)
