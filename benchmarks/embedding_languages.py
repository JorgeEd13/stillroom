#!/usr/bin/env python3
"""Compare embedding models on the two things the relevance floor depends on.

**Why this is a committed artifact and not a paragraph in an ADR.** Every
similarity number this product's design rests on — the 0.25 floor, the typo
margin, the nonsense score, the cross-language gap — was measured once, by hand,
and then quoted in prose. Quoted numbers rot silently and cannot be re-derived
without redoing the work. Run this instead.

It answers two questions that pull in opposite directions:

1. **Can a question in language A find documents in language B?**  The product's
   language rule says the assistant works in the documents' language,
   and that is only true if the *embedder* can do it. The shipped default cannot.
2. **Can the floor still tell a real question from an unrelated one?**  This is
   the cost side, and it is the one that surprises. A model trained for
   cross-lingual alignment compresses the whole similarity range upward, so a
   floor calibrated for one embedder is meaningless for another — and the
   distance between a sensible question and a nonsense one shrinks with it.

Usage — no dependencies beyond the engine's own:

    python benchmarks/embedding_languages.py                    # shipped default only
    python benchmarks/embedding_languages.py --ollama http://localhost:11434 --model bge-m3

Ollama models are compared through `/api/embed`, which is how the engine would
reach them if the embedder moved there: the client already runs Ollama, so a
multilingual embedder costs no new Python dependency, no image size and no
download from Chroma's S3 — but it does make `ingest` depend on Ollama, which
today it does not.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Synthetic throughout: the engine repo stays clean-room of client material
#, and a benchmark that needed a real corpus could not be committed.
DOCS = {
    "en": [
        "## Refund window\n\nCustomers may request a refund within 30 days of delivery.",
        "## Restocking\n\nA restocking fee of 10% applies to opened items.",
        "## Notice period\n\nEmployees must give 30 days notice before leaving.",
        "## Delivery times\n\nInternational shipping takes ten to fifteen working days.",
    ],
    "pt": [
        "## Prazo de reembolso\n\nO cliente pode solicitar reembolso em até 30 dias corridos após a entrega.",
        "## Taxa de reposição\n\nUma taxa de reposição de 10% é cobrada sobre itens abertos.",
        "## Aviso prévio\n\nO colaborador deve dar aviso prévio de 30 dias antes do desligamento.",
        "## Prazo de entrega\n\nA entrega internacional leva de dez a quinze dias úteis.",
    ],
    "zh": [
        "## 退款期限\n\n客户可在交付后30天内申请退款。",
        "## 补货费\n\n已开封商品需支付10%的补货费。",
        "## 通知期\n\n员工离职前必须提前30天通知。",
        "## 交货时间\n\n国际运输需要十到十五个工作日。",
    ],
}

# Bands, not just questions. A single score means nothing; what the floor needs
# is the DISTANCE between the lowest thing that must pass and the highest thing
# that must not.
QUESTIONS = [
    ("on-topic", "en", "What is the refund window?"),
    ("on-topic", "en", "How long does international shipping take?"),
    ("on-topic", "pt", "Qual é o prazo de reembolso?"),
    ("on-topic", "zh", "退款期限是多久?"),
    # Reasonable, and genuinely not in the documents. The floor SHOULD stop it.
    ("adjacent", "en", "What are the office opening hours?"),
    # ⚠️ Lexically related nonsense — leg A's "semantic similarity is not
    # aboutness". This is the band that leaks, in every embedder tested.
    ("nonsense", "en", "How many moons does a refund have?"),
    ("far", "en", "What is the boiling point of water?"),
    ("far", "en", "Who won the 1998 World Cup?"),
    ("far", "en", "Explain quantum entanglement."),
]

FLOOR = 0.25  # what the product ships with today


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def chroma_embedder():
    from stillroom.index.embeddings import default_embedding_function

    fn = default_embedding_function()
    return lambda texts: [list(v) for v in fn(texts)]


def ollama_embedder(base_url: str, model: str):
    def embed(texts):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/embed",
            data=json.dumps({"model": model, "input": texts}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as fh:
            return json.load(fh)["embeddings"]

    return embed


def report(label: str, embed) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"{'corpus':8} {'band':9} {'q-lang':7} {'top sim':>8}  {'@0.25':>7}  question")

    indexed = {lang: embed(docs) for lang, docs in DOCS.items()}
    rows = []
    for corpus_lang, vectors in indexed.items():
        for band, q_lang, question in QUESTIONS:
            top = max(cosine(embed([question])[0], v) for v in vectors)
            rows.append((corpus_lang, band, q_lang, top))
            passes = "pass" if top >= FLOOR else "REFUSED"
            print(
                f"{corpus_lang:8} {band:9} {q_lang:7} {top:8.3f}  {passes:>7}  {question}"
            )

    same = [r[3] for r in rows if r[1] == "on-topic" and r[0] == r[2]]
    cross = [r[3] for r in rows if r[1] == "on-topic" and r[0] != r[2]]
    far = [r[3] for r in rows if r[1] == "far"]
    noise = [r[3] for r in rows if r[1] == "nonsense"]

    print(f"\n  same-language on-topic : {min(same):.3f} … {max(same):.3f}")
    print(f"  CROSS-language on-topic: {min(cross):.3f} … {max(cross):.3f}")
    print(f"  far band (must refuse) : {min(far):.3f} … {max(far):.3f}")
    print(f"  lexical nonsense       : {min(noise):.3f} … {max(noise):.3f}")
    print(f"\n  SEPARATION (lowest on-topic - highest far) : {min(same + cross) - max(far):+.3f}")
    print(f"  A usable floor sits between {max(far):.3f} and {min(same + cross):.3f}.")
    if max(noise) >= min(same + cross):
        print("  ⚠️  Lexical nonsense outranks a real question. No floor separates them;")
        print("      the model's own refusal is the only gate left for that band.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", help="Base URL of an Ollama serving an embedding model.")
    parser.add_argument("--model", default="bge-m3", help="Embedding model tag on that Ollama.")
    parser.add_argument(
        "--skip-default", action="store_true", help="Do not measure the shipped embedder."
    )
    args = parser.parse_args()

    if not args.skip_default:
        report("SHIPPED DEFAULT — onnx-minilm-l6-v2 (Chroma bundled)", chroma_embedder())
    if args.ollama:
        report(f"{args.model} via Ollama at {args.ollama}", ollama_embedder(args.ollama, args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
