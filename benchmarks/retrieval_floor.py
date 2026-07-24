#!/usr/bin/env python3
"""Derive the relevance floor, and test whether anything can compensate for it.

**Why this is a second benchmark and not a bigger first one.**
`embedding_languages.py` answers *"can a question in language A find documents in
language B"* over four synthetic paragraphs, which is the right size for that
question. It is the wrong size for this one. The embedder decision said the floor *"must be
derived on a realistic corpus, not on the four synthetic paragraphs in the
benchmark"*, and the difference is not cosmetic: with four passages, the top hit
has three competitors and every question looks decisive. A real corpus has
dozens of passages, several of them about roughly the right topic, and the top
score is drawn from a distribution rather than a shortlist.

**The two questions here:**

1. **Where does `min_similarity` go now?** `bge-m3` compresses the whole range
   upward, so 0.25 stopped separating anything — all 27 cells of the language
   benchmark passed it.
2. **Can a relative signal catch what no constant can?** Lexically-related
   nonsense scores within 0.012 of the correct question. If the answer is no,
   that has to be *stated*, not left implied — the model's own grounding refusal
   is then load-bearing alone for that band.

Run it:

    python benchmarks/retrieval_floor.py --ollama http://localhost:11434
    python benchmarks/retrieval_floor.py --model bge-m3:567m --sweep

⚠️ **The corpus below is synthetic and always will be.** The engine repo is
clean-room of client material, so "realistic" here means *shaped like*
a real corpus — seven documents, overlapping topics, sections that compete with
each other — not *taken from* one.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stillroom.index.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_NAME,
    EmbeddingError,
    OllamaEmbeddingFunction,
)
from stillroom.ingest.chunking import chunk_corpus  # noqa: E402
from stillroom.ingest.loaders import RawDocument  # noqa: E402

# --------------------------------------------------------------------------
# A corpus with the shape of a small company's document set: policies that
# overlap, sections that compete, and more than one place a plausible question
# could land. Mirrored EN/PT so cross-language retrieval is measured against
# the SAME content rather than against a different corpus.
# --------------------------------------------------------------------------

EN_DOCS = {
    "returns-policy.md": """# Returns and refunds

## Refund window
Customers may request a refund within 30 days of delivery. The request must
come from the account that placed the order. Refunds are issued to the original
payment method and take five to seven working days to appear.

## Restocking fee
A restocking fee of 10% applies to items that have been opened. Unopened items
in their original packaging are refunded in full.

## Damaged goods
Goods damaged in transit are replaced at no cost. Report damage within 48 hours
of delivery with photographs of the packaging.

## Exchanges
An exchange is treated as a return followed by a new order. The refund window
for the replacement item starts again from the date the replacement arrives.
""",
    "shipping.md": """# Shipping

## Domestic delivery
Orders placed before 14:00 are dispatched the same working day and arrive
within two to three working days.

## International shipping
International shipping takes ten to fifteen working days. Customs charges are
the responsibility of the recipient.

## Tracking
A tracking number is emailed when the parcel leaves the warehouse. Tracking
updates can take 24 hours to appear.

## Delivery failures
After two failed delivery attempts the parcel returns to the warehouse and is
held for 14 days before being refunded.
""",
    "handbook.md": """# Staff handbook

## Notice period
Employees must give 30 days of written notice before leaving. Notice runs from
the date the written notice is received, not the date it is written.

## Annual leave
Full-time employees receive 25 days of annual leave per year plus public
holidays. Leave does not carry over into the next year.

## Probation
New employees are on probation for three months. During probation the notice
period is one week on either side.

## Remote work
Employees may work remotely up to two days per week with their manager's
agreement. Remote days are recorded in the shared calendar.
""",
    "expenses.md": """# Expenses

## Approval
Any expense above 500 must be approved in advance by the finance director.
Expenses below that threshold are approved by the line manager.

## Travel
Standard-class rail and economy-class air travel are reimbursed in full. Taxi
fares are reimbursed only when public transport is not available.

## Receipts
Every claim needs an itemised receipt. A card statement is not a receipt.
Claims without receipts are rejected.

## Payment
Approved expenses are paid with the next payroll run, which is the 25th of each
month.
""",
    "pricing.md": """# Pricing

## Bulk discount
Orders above 500 units receive a 12% discount. The discount is applied at
checkout and cannot be combined with a promotional code.

## Payment terms
Invoices are payable within 30 days. Late payment carries interest at 2% per
month.

## Price changes
Prices are reviewed each quarter. Customers on an annual agreement keep their
agreed price for the length of the agreement.
""",
    "security.md": """# Information security

## Passwords
Passwords must be at least twelve characters and are changed every 12 months.
Password reuse across systems is prohibited.

## Devices
Company laptops are encrypted. A lost or stolen device must be reported to IT
within one hour of being noticed.

## Data handling
Customer data is never copied to personal storage. Files leave the company
network only through the approved transfer service.
""",
    "support.md": """# Customer support

## Response times
Support requests are answered within one working day. Priority customers are
answered within four working hours.

## Escalation
An unresolved request is escalated to the team lead after three working days.
The customer is told when this happens.

## Out of hours
There is no out-of-hours support. Requests received outside working hours are
queued for the next working day.
""",
}

PT_DOCS = {
    "politica-devolucoes.md": """# Devoluções e reembolsos

## Prazo de reembolso
O cliente pode solicitar reembolso em até 30 dias corridos após a entrega. O
pedido deve partir da conta que realizou a compra. O reembolso é devolvido ao
meio de pagamento original e leva de cinco a sete dias úteis para aparecer.

## Taxa de reposição
Uma taxa de reposição de 10% é cobrada sobre itens abertos. Itens lacrados na
embalagem original são reembolsados integralmente.

## Produtos danificados
Produtos danificados no transporte são substituídos sem custo. Comunique o dano
em até 48 horas após a entrega, com fotografias da embalagem.

## Trocas
Uma troca é tratada como devolução seguida de novo pedido. O prazo de reembolso
do item substituto recomeça na data em que o substituto chega.
""",
    "entregas.md": """# Entregas

## Entrega nacional
Pedidos feitos antes das 14:00 são despachados no mesmo dia útil e chegam em
dois a três dias úteis.

## Entrega internacional
A entrega internacional leva de dez a quinze dias úteis. As taxas alfandegárias
são de responsabilidade do destinatário.

## Rastreamento
O código de rastreamento é enviado por e-mail quando a encomenda sai do
armazém. As atualizações podem levar 24 horas para aparecer.

## Falhas na entrega
Após duas tentativas de entrega sem sucesso a encomenda retorna ao armazém e
fica retida por 14 dias antes de ser reembolsada.
""",
    "manual-do-colaborador.md": """# Manual do colaborador

## Aviso prévio
O colaborador deve dar aviso prévio por escrito de 30 dias antes do
desligamento. O prazo conta a partir da data em que o aviso é recebido, não da
data em que foi escrito.

## Férias
Colaboradores em tempo integral têm 25 dias de férias por ano além dos
feriados. As férias não são acumuladas para o ano seguinte.

## Experiência
Novos colaboradores cumprem três meses de experiência. Durante esse período o
aviso prévio é de uma semana para ambos os lados.

## Trabalho remoto
O colaborador pode trabalhar remotamente até dois dias por semana mediante
acordo com o gestor. Os dias remotos são registrados no calendário compartilhado.
""",
    "despesas.md": """# Despesas

## Aprovação
Qualquer despesa acima de 500 deve ser aprovada previamente pelo diretor
financeiro. Despesas abaixo desse valor são aprovadas pelo gestor direto.

## Viagens
Passagens de trem em classe padrão e passagens aéreas em classe econômica são
reembolsadas integralmente. Táxis são reembolsados apenas quando não há
transporte público disponível.

## Comprovantes
Toda solicitação precisa de comprovante detalhado. Extrato de cartão não é
comprovante. Solicitações sem comprovante são recusadas.

## Pagamento
As despesas aprovadas são pagas na folha seguinte, no dia 25 de cada mês.
""",
    "precos.md": """# Preços

## Desconto por volume
Pedidos acima de 500 unidades recebem 12% de desconto. O desconto é aplicado na
finalização da compra e não pode ser combinado com código promocional.

## Condições de pagamento
As faturas vencem em 30 dias. O atraso no pagamento gera juros de 2% ao mês.

## Reajustes
Os preços são revisados a cada trimestre. Clientes com contrato anual mantêm o
preço acordado durante a vigência do contrato.
""",
    "seguranca.md": """# Segurança da informação

## Senhas
As senhas devem ter no mínimo doze caracteres e são trocadas a cada 12 meses. É
proibido reutilizar a mesma senha em sistemas diferentes.

## Dispositivos
Os notebooks da empresa são criptografados. A perda ou o roubo de um
dispositivo deve ser comunicado ao setor de TI em até uma hora.

## Tratamento de dados
Dados de clientes nunca são copiados para armazenamento pessoal. Arquivos saem
da rede da empresa apenas pelo serviço de transferência aprovado.
""",
    "suporte.md": """# Suporte ao cliente

## Prazos de resposta
As solicitações de suporte são respondidas em até um dia útil. Clientes
prioritários são respondidos em até quatro horas úteis.

## Escalonamento
Uma solicitação não resolvida é escalada para o líder da equipe após três dias
úteis. O cliente é avisado quando isso acontece.

## Fora do horário
Não há suporte fora do horário comercial. As solicitações recebidas fora do
horário entram na fila do próximo dia útil.
""",
}

# --------------------------------------------------------------------------
# Questions, in four bands. `expect` names the file that should answer it, so
# the benchmark can tell "scored high" apart from "scored high on the RIGHT
# passage" — the two come apart exactly where the design is weakest.
# --------------------------------------------------------------------------

# ⚠️ **`expect` is a TOPIC, not a filename, and that is a correction.** The
# first version named the expected source file, which scored every *cross*-
# language question as a failure: an English question against the Portuguese
# corpus correctly lands on `politica-devolucoes.md`, and a filename check
# called that wrong. It would have reported the one capability this whole phase
# is buying as broken in all three candidates.
TOPIC_FILES = {
    "en": {
        "refund": "returns-policy.md", "shipping": "shipping.md",
        "staff": "handbook.md", "expenses": "expenses.md",
        "pricing": "pricing.md", "security": "security.md", "support": "support.md",
    },
    "pt": {
        "refund": "politica-devolucoes.md", "shipping": "entregas.md",
        "staff": "manual-do-colaborador.md", "expenses": "despesas.md",
        "pricing": "precos.md", "security": "seguranca.md", "support": "suporte.md",
    },
}

# (band, question language, question, expected TOPIC or None)
QUESTIONS: list[tuple[str, str, str, str | None]] = [
    # ---- must pass: the corpus genuinely answers these ----
    ("on-topic", "en", "What is the refund window?", "refund"),
    ("on-topic", "en", "How long does international shipping take?", "shipping"),
    ("on-topic", "en", "How much notice do I have to give before leaving?", "staff"),
    ("on-topic", "en", "Who approves an expense over 500?", "expenses"),
    ("on-topic", "en", "What is the bulk discount for large orders?", "pricing"),
    ("on-topic", "en", "How long must a password be?", "security"),
    ("on-topic", "en", "When is a support request escalated?", "support"),
    ("on-topic", "en", "Is there a fee for returning an opened item?", "refund"),
    ("on-topic", "en", "How many days of annual leave do I get?", "staff"),
    ("on-topic", "pt", "Qual é o prazo de reembolso?", "refund"),
    ("on-topic", "pt", "Quanto tempo leva a entrega internacional?", "shipping"),
    ("on-topic", "pt", "Qual é o aviso prévio para pedir demissão?", "staff"),
    ("on-topic", "pt", "Quem aprova uma despesa acima de 500?", "expenses"),
    ("on-topic", "pt", "Qual é o desconto para pedidos grandes?", "pricing"),
    ("on-topic", "pt", "Qual é o tamanho mínimo da senha?", "security"),
    ("on-topic", "pt", "Quando uma solicitação de suporte é escalada?", "support"),
    ("on-topic", "pt", "Há taxa para devolver um item aberto?", "refund"),
    ("on-topic", "pt", "Quantos dias de férias eu tenho?", "staff"),
    # ---- should be refused: plausible, related, genuinely absent ----
    ("adjacent", "en", "What are the office opening hours?", None),
    ("adjacent", "en", "Do you offer a student discount?", None),
    ("adjacent", "en", "Can I pay by cryptocurrency?", None),
    ("adjacent", "en", "How much is paternity leave?", None),
    ("adjacent", "pt", "Qual é o horário de funcionamento do escritório?", None),
    ("adjacent", "pt", "Vocês oferecem desconto para estudantes?", None),
    ("adjacent", "pt", "Qual é a licença-paternidade?", None),
    # ---- the known leak: lexically overlapping, semantically absurd ----
    ("nonsense", "en", "How many moons does a refund have?", None),
    ("nonsense", "en", "What colour is the notice period?", None),
    ("nonsense", "en", "Does the restocking fee taste of shipping?", None),
    ("nonsense", "pt", "Quantas luas tem um reembolso?", None),
    ("nonsense", "pt", "De que cor é o aviso prévio?", None),
    # ---- must be refused, with room to spare ----
    ("far", "en", "What is the boiling point of water?", None),
    ("far", "en", "Who won the 1998 World Cup?", None),
    ("far", "en", "Explain quantum entanglement.", None),
    ("far", "en", "Write me a recipe for bread.", None),
    ("far", "pt", "Qual é a capital da Austrália?", None),
    ("far", "pt", "Como se faz um bolo de cenoura?", None),
]

BANDS = ("on-topic", "adjacent", "nonsense", "far")
MUST_PASS = ("on-topic",)
MUST_REFUSE = ("far",)


class Rejected(NamedTuple):
    """A text the model would not embed at all.

    ⚠️ **This class exists because a candidate model did exactly that**, and it
    is a result rather than an error path: `bge-m3:567m` on Ollama 0.24.0
    returns `NaN` components for some ordinary English passages, which Ollama
    cannot serialise, so the request fails with `HTTP 500 … json: unsupported
    value: NaN`. Deterministic per text, and **one bad text fails the whole
    batch**. A benchmark that crashed here would have reported nothing; a
    benchmark that skipped silently would have reported a corpus it did not
    measure.
    """

    text: str
    reason: str


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def build_chunks(docs: dict[str, str]) -> list:
    """Chunk with the engine's own chunker, at the shipped defaults."""
    raw = [RawDocument(source=name, text=text) for name, text in docs.items()]
    return chunk_corpus(raw, chunk_chars=1200, chunk_overlap=200)


class Measurement:
    """One question against one corpus, with the whole score distribution kept."""

    def __init__(self, band, language, question, expect, scores, sources, k, corpus_language):
        ranked = sorted(zip(scores, sources), key=lambda p: -p[0])
        self.band = band
        self.language = language
        self.question = question
        self.expect = expect
        self.corpus_language = corpus_language
        self.all_scores = scores
        self.top = ranked[0][0]
        self.top_source = ranked[0][1]
        self.second = ranked[1][0] if len(ranked) > 1 else 0.0
        self.top_k = [s for s, _ in ranked[:k]]
        self.corpus_mean = statistics.fmean(scores)
        self.corpus_sd = statistics.pstdev(scores) or 1e-9

    @property
    def found_the_right_document(self) -> bool:
        if self.expect is None:
            return True
        return TOPIC_FILES[self.corpus_language][self.expect] == self.top_source

    @property
    def crosses_languages(self) -> bool:
        return self.language != self.corpus_language

    # ---- candidate signals, all computable at retrieval time and free ----
    @property
    def margin_over_runner_up(self) -> float:
        return self.top - self.second

    @property
    def margin_over_field(self) -> float:
        """Top hit against the mean of the rest of the retrieved page."""
        rest = self.top_k[1:]
        return self.top - statistics.fmean(rest) if rest else 0.0

    @property
    def margin_over_corpus(self) -> float:
        """Top hit against the mean score over the WHOLE corpus.

        The intuition being tested: a question the documents really answer
        should stand out from the corpus, while a question that merely shares
        vocabulary with it scores middlingly against everything.
        """
        return self.top - self.corpus_mean

    @property
    def z_score(self) -> float:
        return (self.top - self.corpus_mean) / self.corpus_sd


def embed_each(embed, texts: list[str]) -> tuple[list, list[Rejected]]:
    """Embed one text at a time, recording the ones the model refuses.

    One at a time on purpose. Batching is right in the product and wrong here:
    a model that fails the whole batch for one bad passage would hide how many
    passages are bad, and that count is the measurement.
    """
    vectors, rejected = [], []
    for text in texts:
        try:
            vectors.append((text, embed([text])[0]))
        except EmbeddingError as exc:
            rejected.append(Rejected(text, str(exc).split(":")[-1].strip()[:60]))
    return vectors, rejected


def measure(
    embed, docs: dict[str, str], corpus_language: str, k: int
) -> tuple[list[Measurement], list[Rejected], int]:
    chunks = build_chunks(docs)
    by_text = {c.text: c.source for c in chunks}
    embedded, rejected = embed_each(embed, [c.text for c in chunks])
    vectors = [v for _, v in embedded]
    sources = [by_text[t] for t, _ in embedded]
    if not vectors:
        return [], rejected, len(chunks)

    q_embedded, q_rejected = embed_each(embed, [q for _, _, q, _ in QUESTIONS])
    rejected += q_rejected
    q_by_text = {q: (b, l, e) for b, l, q, e in QUESTIONS}

    out = []
    for question, qv in q_embedded:
        band, language, expect = q_by_text[question]
        scores = [cosine(qv, dv) for dv in vectors]
        out.append(
            Measurement(band, language, question, expect, scores, sources, k, corpus_language)
        )
    return out, rejected, len(chunks)


def band_table(title: str, measurements: list[Measurement]) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"{'band':10} {'n':>3} {'top: min':>9} {'max':>7} | "
          f"{'runner-up':>10} {'field':>7} {'corpus':>7} {'z':>6}")
    for band in BANDS:
        rows = [m for m in measurements if m.band == band]
        if not rows:
            continue
        tops = [m.top for m in rows]
        print(
            f"{band:10} {len(rows):>3} {min(tops):>9.3f} {max(tops):>7.3f} | "
            f"{statistics.fmean([m.margin_over_runner_up for m in rows]):>10.3f} "
            f"{statistics.fmean([m.margin_over_field for m in rows]):>7.3f} "
            f"{statistics.fmean([m.margin_over_corpus for m in rows]):>7.3f} "
            f"{statistics.fmean([m.z_score for m in rows]):>6.2f}"
        )

    wrong = [m for m in measurements if not m.found_the_right_document]
    if wrong:
        print(f"\n  ⚠️  {len(wrong)} on-topic question(s) whose top hit is the wrong document:")
        for m in wrong:
            print(f"      {m.question!r} -> {m.top_source} ({m.top:.3f})")
    else:
        print("\n  ✅ Every on-topic question's top hit is in the expected document.")


def separation(measurements: list[Measurement], signal: str) -> tuple[float, float, float]:
    """(lowest must-pass, highest must-refuse, gap) for one candidate signal."""
    value = lambda m: getattr(m, signal) if signal != "top" else m.top  # noqa: E731
    passes = [value(m) for m in measurements if m.band in MUST_PASS]
    refuses = [value(m) for m in measurements if m.band in MUST_REFUSE]
    return min(passes), max(refuses), min(passes) - max(refuses)


def signal_report(measurements: list[Measurement]) -> None:
    print(f"\n{'-' * 78}\nCANDIDATE SIGNALS — can any of them gate what a constant cannot?"
          f"\n{'-' * 78}")
    print(f"{'signal':22} {'lowest on-topic':>16} {'highest far':>12} {'gap':>8} "
          f"{'highest nonsense':>17}")
    for signal in ("top", "margin_over_runner_up", "margin_over_field",
                   "margin_over_corpus", "z_score"):
        low, high, gap = separation(measurements, signal)
        value = lambda m: getattr(m, signal) if signal != "top" else m.top  # noqa: E731
        noise = max(value(m) for m in measurements if m.band == "nonsense")
        flag = "  <-- separates nonsense" if noise < low else ""
        print(f"{signal:22} {low:>16.3f} {high:>12.3f} {gap:>8.3f} {noise:>17.3f}{flag}")


def sweep(measurements: list[Measurement], margin_signal: str | None = None) -> None:
    print(f"\n{'-' * 78}\nFLOOR SWEEP — what each constant admits, by band\n{'-' * 78}")
    print(f"{'floor':>6} | " + " ".join(f"{b:>10}" for b in BANDS) + "    verdict")
    for floor in [round(0.20 + 0.025 * i, 3) for i in range(21)]:
        cells = []
        for band in BANDS:
            rows = [m for m in measurements if m.band == band]
            admitted = sum(1 for m in rows if m.top >= floor)
            cells.append(f"{admitted:>4}/{len(rows):<5}")
        on_topic = [m for m in measurements if m.band == "on-topic"]
        far = [m for m in measurements if m.band == "far"]
        ok_pass = all(m.top >= floor for m in on_topic)
        ok_refuse = all(m.top < floor for m in far)
        verdict = "OK" if ok_pass and ok_refuse else (
            "loses real questions" if not ok_pass else "admits the far band"
        )
        print(f"{floor:>6.3f} | " + " ".join(cells) + f"    {verdict}")


def run_model(model: str, base_url: str, k: int, do_sweep: bool) -> None:
    embed = OllamaEmbeddingFunction(base_url=base_url, model=model)

    everything: list[Measurement] = []
    total_rejected = 0
    for label, docs, language in (
        ("ENGLISH CORPUS", EN_DOCS, "en"),
        ("PORTUGUESE CORPUS", PT_DOCS, "pt"),
    ):
        measurements, rejected, n_chunks = measure(embed, docs, language, k)
        band_table(f"{model} — {label}, {n_chunks} chunks", measurements)
        total_rejected += len(rejected)
        if rejected:
            print(f"\n  ⛔ {len(rejected)} of {n_chunks} passages: THE MODEL WOULD NOT "
                  f"EMBED THEM AT ALL.")
            for r in rejected[:6]:
                print(f"      {r.text.splitlines()[0][:64]!r} … -> {r.reason}")
            print("      An index missing these silently cannot answer about them, and")
            print("      the refusal is indistinguishable from 'not in your documents'.")
        everything.extend(measurements)

    if not everything:
        print(f"\n⛔ {model}: nothing measurable.")
        return

    band_table(f"{model} — BOTH CORPORA", everything)
    signal_report(everything)
    if do_sweep:
        sweep(everything)
    if total_rejected:
        print(f"\n⛔ VERDICT for {model}: {total_rejected} passage(s) unembeddable. "
              "Every number above is measured on the corpus that SURVIVED, so the "
              "quality figures flatter it.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_NAME,
        help="Comma-separated: candidates are compared on one corpus, one run.",
    )
    parser.add_argument("--k", type=int, default=5, help="Retrieval k the product ships with.")
    parser.add_argument("--sweep", action="store_true", help="Print the full floor sweep.")
    args = parser.parse_args()

    for model in [m.strip() for m in args.model.split(",") if m.strip()]:
        run_model(model, args.ollama, args.k, args.sweep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
