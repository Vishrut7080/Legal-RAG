# Legal-RAG — presentation

4 slides · 5-minute demo · every figure below is measured, with the command beside it

---

## Slide 1 — What it is, and the one failure it exists to prevent

**A retrieval service over Indian statutes that treats the legal status of an answer as part
of the answer.**

Indian criminal law was recodified on 1 July 2024. The IPC, CrPC and Evidence Act were
repealed and replaced by the BNS, BNSS and BSA. A RAG system over that corpus retrieves
**IPC s.302** for a question about murder — near-perfect lexical and semantic match for one —
and quoting it is a wrong answer today. The citation looks authoritative.

> That failure is worse than being wrong, because nothing marks it.

**Both generations are indexed on purpose.** Answering *"what does the law currently say"*
requires being able to retrieve the repealed text in order to recognise and exclude it.

**Design rule, and the decision it forced:** *a repeal warning that overstates its own
confidence is how a user cites the wrong successor as settled.* So 41 of 2,162 weak
recodification links are kept but marked tentative — never suppressed, never stated as fact.

```
948 files pinned by SHA-256 → 930 provisions → 996 chunks → hybrid retrieval + hard current-law filter
```

---

## Slide 2 — Measured behaviour

`python -m src.gate` — **17 checks, all green, 2.9 s, no network, no API key.** 102 golden cases.

| | measured | floor |
| --- | --- | --- |
| hit@1 · hit@10 | 0.707 · 0.939 | 0.65 · 0.90 |
| MRR · nDCG@10 | 0.790 · 0.884 | 0.74 · 0.84 |
| **stale@10 on current-law questions** | **0.000** | ≤ 0.08 |
| refusal precision / recall | 1.000 / 1.000 | 1.0 / 1.0 |
| **unannounced repealed citations** | **0** | 0 |
| per-case failures | 0 | 0 |

**The retriever comparison is the argument for the design.** Every retriever without the
current-law filter quotes 1860 law in 20–23% of cases (`stale@10`: 0.72–0.77). The filtered
one quotes it **zero** times. Its hit@10 is *lower* than plain hybrid — because the filter
**removes** repealed chunks. That is the intended trade.

| | hit@1 | hit@10 | stale@10 |
| --- | --- | --- | --- |
| hybrid | 0.586 | 0.929 | 0.737 |
| **hybrid+current** (default) | **0.667** | 0.869 | **0.000** |

**Not measured, and printed on every gate run:** answer quality, factual correctness,
faithfulness to sources. No judge model runs offline, so those are recorded `UNMEASURED`
rather than given a number nobody can defend.

*Read the table as "it finds the right provision", not "it answers the question right."*

---

## Slide 3 — The semantic cache trap, and cold start

**A semantic cache returns a stored answer when a new question is close enough to an old one.
It looks like free performance, and it is the most dangerous feature here** — cosine
similarity does not know what a question is *about*.

> "When may a **temporary** injunction be granted?"  cosine 0.82
> "When is a **perpetual** injunction granted?"  ← different answer (CPC O.39 r.1 vs Specific Relief Act s.38)

Below the threshold: a confident, well-cited, fast answer to a question nobody asked. Latency
improves, cost improves, correctness quietly falls. **Nothing errors.**

### `python -m src.cache_sweep` — 9,702 ordered question pairs

| cosine | would serve the answer for… | gold clash |
| --- | --- | --- |
| 0.8699 | *electronic record admissible?* | BSA s.63 vs s.62 |
| 0.8387 | *warrant remain in force?* | BNSS s.72 vs s.41 |
| 0.8222 | *temporary* injunction | CPC O.39 r.1 vs Specific Relief s.38 |

**There is no safe similarity band on this corpus.** Max off-diagonal = 0.8699; at every
threshold with any hits, precision is **0.000**; above 0.87, zero hits. So the threshold ships
at **0.95 — above the measured maximum, where the layer is inert.** Zero semantic hits on
`/metrics` is the correct state, not a misconfiguration. The scope guard blocks BNS-vs-IPC but
caught *none* of these: "temporary" vs "perpetual" is the axis, and no lexical guard sees it.

### Cold start was a real defect the benchmark found

`python -m src.benchmark`, embedding cache **off** — every query a genuinely new question:

| | p50 | p95 | max |
| --- | --- | --- | --- |
| uncached (new question) | 20.99 ms | 25.00 ms | 155.94 ms |
| **first request, before the fix** | 20 ms | — | **18,370 ms** |
| cached, exact replay (40/40 confirmed) | 2.47 ms | 3.01 ms | 3.66 ms |
| with embedding cache on (repeat question) | 7.45 ms | 9.65 ms | 101.76 ms |

Lazy model load made the first request 18 s against a 1,500 ms target. **The p95 never showed
it — only the max did.** Fixed by warming at startup: worst case 18,370 ms → 156 ms.

---

## Slide 4 — Cost, boundaries, and what the gate proves

**$0.00 per query, and that is measured, not absent.** Every model is local; nobody is billed.
Reported as `priced: true`, because a `$0.00` meaning *we did not look* is a different claim
from one meaning *nobody is billed*. Projected onto a priced tier (12.4 counted tokens/query):
**$0.0019 per 1,000 · $6.82 per year at 10,000/day.**

### What it is not safe for

1. **Not unsupervised legal advice.** The generator quotes rank-1 verbatim, so accuracy is
   bounded by hit@1 = 0.707 — and on the worst question it confidently quotes the **wrong**
   provision. It retrieves text; it does not interpret or apply.
2. **Not for anything depending on which provision is cited.** `enforce_citations` checks the
   pointer is *in range*, not that the section supports the claim. Research tool with a human
   reading every citation — not a tool where the citation is the answer.
3. **Not beyond English or this corpus.** 930 provisions, 15 documents, one embedding model
   trained on web text rather than statute prose. No auth, no isolation — loopback only.
4. **May run unsupervised for:** internal statute lookup on the pinned corpus, human reading the
   provision, wrong hits visible, refusal acceptable.

### `python -m src.break_it` — a gate never seen fail is not a gate

| break | result |
| --- | --- |
| retrieval window `k=1` | **6 checks red**, exit 1 |
| empty a corpus act | **`corpus is reproducible from the snapshot` red**, exit 1 |
| corrupted index checksum | **`index matches chunks.jsonl` red**, exit 1 |

**Row 2 was a real hole, found by running the experiment.** Deleting 32 provisions left *every*
pre-existing check green: the gate measured the committed index, and nothing tied
`data/corpus/` back to `data/raw/`. Fixed by rebuilding the corpus from the snapshot and
comparing byte for byte. CI runs the gate **and** a second job that proves it can still fail.

### Next three, ranked

1. **Replace the embedding model** — the worst failure is exactly its limitation, and the
   cross-encoder already fixes that case. *High EV, one day.*
2. **Add an offline judge** — converts three `UNMEASURED` fields into measured ones and catches
   citation *precision*, which nothing currently checks. *High EV, 1–2 days.*
3. **`--historical` mode** filtering *to* repealed provisions — real correctness gap, narrow blast
   radius. *Medium EV, half a day.*

---

## Demo script (5 min)

| time | item |
| --- | --- |
| 0:00–0:30 | A question it answers well — *"What is cheating by personation?"* → **BNS s.319** (in force), which in the answer carries `replaces: IPC s.416`. Retrieval **and** the supersession link in one answer. Show `citations_valid: true`, latency, cost, trace id |
| 0:30–1:00 | A question it **refuses** — *"rate of corporate tax"*. **HTTP 200**, `refused: true`, reason given. Refusal is correct: the corpus has no tax law, and answering from the nearest statute is the failure this project is about |
| 1:00–2:00 | `GET /metrics` — real cost ($0.00, priced), p50/p95 per stage, and **`semantic_hits: 0`**, which is the correct state and the point of slide 3 |
| 2:00–3:00 | `python -m src.break_it --mode corpus` — the red build, then auto-restored |
| 3:00–4:00 | Worst remaining failure: *"punishment for murder"* → **BNS s.109** (culpable homicide, 10 yrs) instead of **BNS s.103** (murder, life). s.103 is at **rank 4**, so hit@10 is unaffected. An embedding-model limitation |
| 4:00–5:00 | Questions |

**Worst remaining failure, stated plainly:** the top-1 answer to the most obvious question in
this corpus is the wrong provision. `hybrid+current+rerank` returns BNS s.103 for it; the
default does not, because the cross-encoder does not beat the filter in aggregate (§2). The
fix is measured, affordable, and not yet made.