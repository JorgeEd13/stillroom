#!/usr/bin/env python3
"""Which languages may this product promise? Measured, and only measured.

**Why this is a committed benchmark and not a sentence in a doc.** The documents'-language rule set
the rule *"do not advertise a language nobody has measured"*, and until this file
existed the measured set was **English and Portuguese** — not because the others
failed, but because nobody had asked. The supported-language list is a **public
public claim about what this product supports**, which is the class of statement that gets
edited under re-review if it turns out to be wrong.

Run it before changing that list:

    python benchmarks/supported_languages.py

## What it measures, and what it does NOT

It asks four questions the corpus genuinely answers, translated into fifteen
languages, against the **realistic** EN and PT corpora from
`retrieval_floor.py` — so it measures **cross-language retrieval into a real
corpus**, which is the shape of the actual engagement: the client's documents are
in one language and their staff ask in another.

⚠️ **Three things it does not measure, and no public claim may imply them:**

1. **Same-language retrieval in a non-Latin corpus at product scale.** There is
   no 32-chunk Chinese or Arabic corpus here; `embedding_languages.py` covers
   Chinese on four paragraphs only. A client whose *documents* are in Chinese is
   an unmeasured configuration.
2. **Generation.** This is the embedder. Whether the chat model *answers* well in
   a language is a different question with a different model behind it, and
   An earlier measurement already found `qwen2.5:7b` refusing in the wrong language.
3. **`refusal_markers`.** The eval suite detects layer-2 refusals **by phrase**,
   so any non-English client needs them configured or a working build
   reports as broken.

`FLOOR` must track `RetrievalConfig.min_similarity`. A language that clears a
floor the product does not ship is not supported.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_floor import EN_DOCS, PT_DOCS, build_chunks, cosine
from stillroom.index.embeddings import OllamaEmbeddingFunction, EmbeddingError

FLOOR = 0.50
# Four on-topic questions the corpus genuinely answers, translated. `expect` is
# the file that must come top.
Q = {
 "en": [("What is the refund window?","refund"),("How long does international shipping take?","ship"),
        ("How much notice must I give before leaving?","staff"),("Who approves an expense over 500?","exp")],
 "pt": [("Qual é o prazo de reembolso?","refund"),("Quanto tempo leva a entrega internacional?","ship"),
        ("Qual é o aviso prévio antes do desligamento?","staff"),("Quem aprova uma despesa acima de 500?","exp")],
 "es": [("¿Cuál es el plazo de reembolso?","refund"),("¿Cuánto tarda el envío internacional?","ship"),
        ("¿Cuánto preaviso debo dar antes de irme?","staff"),("¿Quién aprueba un gasto superior a 500?","exp")],
 "fr": [("Quel est le délai de remboursement ?","refund"),("Combien de temps prend la livraison internationale ?","ship"),
        ("Quel préavis dois-je donner avant de partir ?","staff"),("Qui approuve une dépense supérieure à 500 ?","exp")],
 "de": [("Wie lange ist die Rückgabefrist?","refund"),("Wie lange dauert der internationale Versand?","ship"),
        ("Welche Kündigungsfrist muss ich einhalten?","staff"),("Wer genehmigt eine Ausgabe über 500?","exp")],
 "it": [("Qual è il termine per il rimborso?","refund"),("Quanto tempo richiede la spedizione internazionale?","ship"),
        ("Quanto preavviso devo dare prima di andarmene?","staff"),("Chi approva una spesa superiore a 500?","exp")],
 "nl": [("Wat is de retourtermijn?","refund"),("Hoe lang duurt internationale verzending?","ship"),
        ("Welke opzegtermijn moet ik aanhouden?","staff"),("Wie keurt een uitgave boven 500 goed?","exp")],
 "pl": [("Jaki jest termin zwrotu?","refund"),("Ile trwa wysyłka międzynarodowa?","ship"),
        ("Jaki jest okres wypowiedzenia?","staff"),("Kto zatwierdza wydatek powyżej 500?","exp")],
 "tr": [("İade süresi nedir?","refund"),("Uluslararası kargo ne kadar sürer?","ship"),
        ("Ayrılmadan önce ne kadar önceden haber vermeliyim?","staff"),("500'ün üzerindeki gideri kim onaylar?","exp")],
 "ru": [("Каков срок возврата?","refund"),("Сколько времени занимает международная доставка?","ship"),
        ("За сколько дней нужно предупредить об увольнении?","staff"),("Кто утверждает расход свыше 500?","exp")],
 "zh": [("退款期限是多久?","refund"),("国际运输需要多长时间?","ship"),
        ("离职需要提前多久通知?","staff"),("谁批准超过500的支出?","exp")],
 "ja": [("返金期限はどのくらいですか?","refund"),("国際配送にはどのくらいかかりますか?","ship"),
        ("退職前にどのくらい前に通知が必要ですか?","staff"),("500を超える経費は誰が承認しますか?","exp")],
 "ko": [("환불 기간은 얼마나 됩니까?","refund"),("국제 배송은 얼마나 걸립니까?","ship"),
        ("퇴사 전에 얼마나 미리 통보해야 합니까?","staff"),("500이 넘는 지출은 누가 승인합니까?","exp")],
 "ar": [("ما هي فترة الاسترداد؟","refund"),("كم تستغرق الشحنة الدولية؟","ship"),
        ("ما هي مدة الإشعار قبل المغادرة؟","staff"),("من يوافق على نفقة تزيد عن 500؟","exp")],
 "hi": [("वापसी की अवधि क्या है?","refund"),("अंतर्राष्ट्रीय शिपिंग में कितना समय लगता है?","ship"),
        ("नौकरी छोड़ने से पहले कितने दिन की सूचना देनी होगी?","staff"),("500 से अधिक के खर्च को कौन मंजूरी देता है?","exp")],
}
FILES = {"en":{"refund":"returns-policy.md","ship":"shipping.md","staff":"handbook.md","exp":"expenses.md"},
         "pt":{"refund":"politica-devolucoes.md","ship":"entregas.md","staff":"manual-do-colaborador.md","exp":"despesas.md"}}
FAR = {"en":"Explain quantum entanglement.","zh":"解释量子纠缠","ru":"Объясните квантовую запутанность","ar":"اشرح التشابك الكمي"}

e = OllamaEmbeddingFunction(base_url="http://localhost:11434", model="bge-m3:567m")
for corpus_lang, docs in (("en",EN_DOCS),("pt",PT_DOCS)):
    chunks = build_chunks(docs); vecs=[]; srcs=[]
    for c in chunks:
        try: vecs.append(e([c.text])[0]); srcs.append(c.source)
        except EmbeddingError: pass
    print(f"\n{'='*74}\nCORPUS LANGUAGE: {corpus_lang.upper()}   (floor {FLOOR}, {len(vecs)} chunks)\n{'='*74}")
    print(f"{'asked in':10} {'min':>6} {'max':>6}  {'pass':>5}  {'right doc':>9}   verdict")
    for lang, pairs in Q.items():
        tops=[]; right=0
        for q, expect in pairs:
            qv=e([q])[0]
            best=max(zip((cosine(qv,v) for v in vecs), srcs), key=lambda p:p[0])
            tops.append(best[0])
            if best[1]==FILES[corpus_lang][expect]: right+=1
        npass=sum(1 for t in tops if t>=FLOOR)
        verdict = "OK" if npass==4 and right==4 else ("PARTIAL" if npass>=2 else "NO")
        star = " <- same language" if lang==corpus_lang else ""
        print(f"{lang:10} {min(tops):6.3f} {max(tops):6.3f}  {npass:>3}/4  {right:>7}/4   {verdict}{star}")
    fars=[max(cosine(e([q])[0],v) for v in vecs) for q in FAR.values()]
    print(f"{'FAR band':10} {min(fars):6.3f} {max(fars):6.3f}  {sum(1 for f in fars if f>=FLOOR):>3}/{len(fars)}         must be 0/{len(fars)}")
