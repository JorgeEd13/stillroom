#!/usr/bin/env python3
"""Re-measure every F3.5 number that the embedder swap invalidated.

**Why this is a committed artifact.** The requirement was *"a bunch
of tests to make sure it didn't break anything we already tested"*, and that is
larger than it sounds: **every adversarial battery in F3.5 was measured with the
instrument this phase replaced.** The typo margin (0.297), the nonsense score
(0.341), the refusal gradient, the negation-cache guard, the eval suite's
thresholds — each is a number produced by MiniLM. A green unit suite proves
nothing about any of them, because the unit suite runs on a hashing fixture that
shares nothing with either embedder.

So this runs both models over the same batteries and prints them side by side.
Anything that moved has to be *explained*, not merely noticed.

    python benchmarks/regression_f35.py
    python benchmarks/regression_f35.py --model bge-m3:567m --floor 0.50

⚠️ **What this does NOT cover, and must not be described as covering.** These are
retrieval measurements. The batteries that depend on the *model's* behaviour —
prompt injection, the roleplay refusals, the poem, the contradiction handling,
forecasting — are unchanged by an embedder swap in their prompt path, but their
*retrieval* inputs move, which is what is measured here. The generation half is
re-run live in the session log, not by this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval_floor import EN_DOCS, PT_DOCS, build_chunks, cosine  # noqa: E402

from stillroom.index.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_NAME,
    EmbeddingError,
    OllamaEmbeddingFunction,
)
from stillroom.index.retrieval import CorpusSpeller  # noqa: E402

OLD_FLOOR = 0.25
OLD_MODEL = "onnx-minilm-l6-v2 (shipped before the move to bge-m3)"

# Every case carries the F3.5 finding it comes from, and what the number MEANT.
# A battery whose provenance cannot be named is a battery nobody has to explain.
BATTERIES: dict[str, list[tuple[str, str, str]]] = {
    # (language, label, question)
    "typos. 0.297 was the worst English case, and that was called "
    "thin margin 'an argument against ever raising min_similarity'": [
        ("en", "clean", "What is the refund window?"),
        ("en", "light", "What is the refund windo?"),
        ("en", "HEAVY", "wat is teh refnud windwo"),
        ("en", "heavy", "hwo lnog is teh refund peroid"),
        ("en", "caps", "WHAT IS THE REFUND WINDOW"),
        ("en", "no punctuation", "what is the refund window"),
        ("pt", "clean", "Qual é o prazo de reembolso?"),
        ("pt", "unaccented", "Qual e o prazo de reembolso?"),
        ("pt", "heavy", "qal e o prazu de rembolso"),
    ],
    "the refusal gradient. 'semantic similarity is not aboutness': "
    "nonsense scored 0.341, ABOVE the sensible-but-absent 0.220": [
        ("en", "sensible, present", "What is the refund window?"),
        ("en", "sensible, ABSENT", "What are the office opening hours?"),
        ("en", "NONSENSE", "How many moons does a refund have?"),
        ("en", "far", "What is the boiling point of water?"),
    ],
    "cross-language. 0.076–0.220 against a 0.25 floor, 0 of 5 "
    "relevant, every question refused. This is what the swap was FOR": [
        ("en->pt", "EN question, PT corpus", "What is the refund window?"),
        ("en->pt", "EN question, PT corpus", "How long does international shipping take?"),
        ("pt->en", "PT question, EN corpus", "Qual é o prazo de reembolso?"),
        ("pt->en", "PT question, EN corpus", "Qual é o aviso prévio?"),
    ],
    "the negation guard. 'What is NOT the refund window?' was served "
    "the cached positive answer, instantly, with NO model call": [
        ("en", "positive", "What is the refund window?"),
        ("en", "NEGATED", "What is NOT the refund window?"),
        ("en", "negated", "Which items are not eligible for a refund?"),
    ],
    "the troll battery. Degenerate input was short-circuited by the "
    "floor with no model call at all; it must still be": [
        ("en", "empty-ish", "???"),
        ("en", "SQL", "'; DROP TABLE documents; --"),
        ("en", "repetition", "refund " * 40),
        ("en", "single word", "refund"),
    ],
}

CORPORA = {"en": EN_DOCS, "pt": PT_DOCS}


def chroma_embedder():
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    fn = DefaultEmbeddingFunction()
    return lambda texts: [list(v) for v in fn(texts)]


def indexed(embed, docs):
    """Embed a corpus, tolerating a model that refuses individual passages."""
    vectors, sources = [], []
    for chunk in build_chunks(docs):
        try:
            vectors.append(embed([chunk.text])[0])
            sources.append(chunk.source)
        except EmbeddingError:
            pass
    return vectors, sources


def top_score(embed, vectors, question: str) -> float:
    return max(cosine(embed([question])[0], v) for v in vectors)


def corpus_for(language: str) -> dict:
    if language in CORPORA:
        return CORPORA[language]
    return CORPORA[language.split("->")[1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_NAME)
    parser.add_argument("--floor", type=float, default=0.50)
    args = parser.parse_args()

    new = OllamaEmbeddingFunction(base_url=args.ollama, model=args.model)
    old = chroma_embedder()

    print(f"\n{'=' * 92}")
    print("F3.5 REGRESSION PASS — every battery, both embedders, same corpora")
    print(f"  before: {OLD_MODEL}   floor {OLD_FLOOR}")
    print(f"  after : {args.model}   floor {args.floor}")
    print(f"{'=' * 92}")

    old_index = {lang: indexed(old, docs) for lang, docs in CORPORA.items()}
    new_index = {lang: indexed(new, docs) for lang, docs in CORPORA.items()}
    spellers = {
        lang: CorpusSpeller([c.text for c in build_chunks(docs)])
        for lang, docs in CORPORA.items()
    }

    for battery, cases in BATTERIES.items():
        print(f"\n--- {battery}")
        print(
            f"    {'case':22} {'OLD':>7} {'@0.25':>8} | {'NEW':>7} {'@floor':>8} "
            f"{'retry':>7}  question"
        )
        for language, label, question in cases:
            corpus_language = language.split("->")[-1] if "->" in language else language
            old_vectors, _ = old_index[corpus_language]
            new_vectors, _ = new_index[corpus_language]

            before = top_score(old, old_vectors, question)
            after = top_score(new, new_vectors, question)

            retry = ""
            if after < args.floor:
                corrected = spellers[corpus_language].correction(question)
                if corrected:
                    rescued = top_score(new, new_vectors, corrected)
                    retry = f"{rescued:.3f}" + ("*" if rescued >= args.floor else "")

            print(
                f"    {label:22} {before:7.3f} {'pass' if before >= OLD_FLOOR else 'REFUSED':>8} | "
                f"{after:7.3f} {'pass' if after >= args.floor else 'REFUSED':>8} "
                f"{retry:>7}  {question[:44]!r}"
            )

    print("\n    * = the spelling retry lifted it back over the floor.")
    print("      A blank retry column means no correction was available or none was needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
