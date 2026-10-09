# Legal-RAG — concept note

**Track:** Legal · **Sequence:** corpus → index → retrieval → generation → guards → evaluation → gate → service

A retrieval service over Indian statutes that treats *the legal status of an answer* as
part of the answer. Indian criminal law was recodified on 1 July 2024: the IPC, CrPC and
Indian Evidence Act were repealed and replaced by the BNS, BNSS and BSA. A RAG system over
that corpus will happily retrieve IPC s.302 for a question about murder — it is a near-perfect
lexical and semantic match for one — and quoting it is a wrong answer today. The citation
looks authoritative. That failure is what this project is built around.

Everything runs locally with no API key. Every number below is measured and every measurement
has a command next to it.

---

## 1 · The problem, stated precisely

A statutory retrieval tool has one failure mode that is worse than being wrong, because nothing
marks it: **answering a current-law question from repealed law.** A wrong answer that says
"wrong" is recoverable. A wrong answer with a valid-looking citation is cited in a brief.

So the design constraint that drove every decision below is: *a repeal warning that
overstates its own confidence is how a user ends up citing the wrong successor as settled.*

## 2 · Corpus and provenance

`data/raw/` is a pinned snapshot of the IndiaCode API, not a live crawl.

- **948 files, 930 provisions**, 14 Acts, SHA-256 per response in `FETCHED.json`, pinning
  upstream build `7c930314ed6e` and corpus date `2026-10-01`.
- `data/manifest.json` declares what to fetch and why, per Act, using upstream's own
  old→new mapping at graded similarity thresholds (0.90 for BNS/IPC, 0.96 for BNSS/CrPC,
  0.75 for BSA/IEA) — a similarity join, not a legal cross-reference table.
- Both generations are indexed **on purpose**. Answering "what does the law *currently* say"
  requires being able to retrieve the repealed text in order to recognise and exclude it.

### Licensing: an engineering decision, enforced rather than documented

The API ships 2,732 eCourts judgment records carrying `applied_to_this_section` summaries
that IndiaCode's own licence line calls "the ratio decidendi as extracted from the order by
eCourts India" — 53% of the snapshot's bytes, and editorial matter that is not ours to
redistribute. `fetch.py` keeps `judgments_count` and drops the records, so the built corpus
is byte-identical.

A storage policy that is a paragraph in a README is a policy that gets reverted by whoever
wants the raw data, so `make gate` **fails** if any stored payload carries a `judgments` key.
The reasoning and the question to put to a lawyer are in `LICENSING.md`.

## 3 · Index and retrieval

996 chunks over 15 documents (930 provisions; 302 repealed), `all-MiniLM-L6-v2`, dim 384,
2,600-char chunks with a 600-char floor.

**Two mechanisms guard the supersession chain, and they work differently.**

1. **Status is in the embedded text, not just metadata.** Each provision is rendered with an
   explicit status block, so the retriever can *match* on it. Metadata is invisible to an
   embedding model; this text is not.
2. **`hybrid+current` is a hard filter**, not a penalty — it excludes every chunk whose
   provision is repealed. A penalty would leave "probably fine" as an option, and that is how
   1860 ends up in a 2024 answer.

Historical questions deliberately bypass the filter, because "what did the old CrPC say about
X" is answerable and should be. Those answers are *required* to announce the repeal and name
the replacement, and `max_unannounced_repealed: 0` gates exactly that.

### Retriever comparison, measured on 99 scored cases

| retriever | hit@1 | hit@10 | MRR | nDCG@10 | stale@1 | stale@10 |
| --- | --- | --- | --- | --- | --- | --- |
| dense | 0.576 | 0.919 | 0.698 | 0.803 | 0.232 | 0.717 |
| bm25 | 0.475 | 0.838 | 0.606 | 0.684 | 0.202 | 0.768 |
| hybrid | 0.586 | 0.929 | 0.716 | 0.812 | 0.212 | 0.737 |
| hybrid+current | 0.667 | 0.869 | **0.742** | 0.824 | **0.000** | **0.000** |
| hybrid+rerank | 0.616 | 0.919 | 0.729 | 0.831 | 0.172 | 0.677 |
| hybrid+current+rerank | 0.667 | 0.879 | 0.738 | **0.830** | **0.000** | **0.000** |

`hybrid+current` is the default. Its hit@10 is *lower* than plain hybrid because the filter
**removes** repealed chunks — that is the intended behaviour, and the reason its staleness is
zero while every other row quotes 1860 law in 20–23% of cases.

**Reranking does not earn its place in aggregate.** It matches on hit@1, is marginally worse
on MRR, and costs about a second per query. It is not the default. Worth recording though: for
the single worst failure of this project (§8), `hybrid+current+rerank` returns BNS s.103
(murder) where `hybrid+current` returns s.109 (culpable homicide). The reranker fixes that
case and costs the corpus elsewhere.

### Weak correspondence links, kept and hedged

The recodification maps are many-to-one by nature — BNS folds several 1860 offences into one
section — and **41 of 2,162** resolved links come back graded `weak` upstream. Those are kept,
but marked, in both the corpus and the embedded text:

> Replaced by BNS s.1 (weak, score 0.56 — tentative, confirm against the commencement notification)

Suppressing them would discard real information, and overstating confidence is the failure in
§1. `tests/test_pipeline.py` asserts the caveat cannot be dropped by a refactor.

## 4 · Generation and the answer contract

`python -m src.answer -q "..."`. Every answer is a `RagAnswer` carrying `question`, `answer`,
`hits`, `citations_valid`, `invalid_citations`, `refused`, `stages`, plus `trace_id`,
`latency_ms` and `latency_by_stage`. Five rules hold regardless of backend:

1. Every factual claim carries a numbered citation into the retrieved set.
2. No citation points outside that set. An unresolvable citation is a bug, so it fails rather
   than being printed.
3. A repealed citation must be **announced as repealed**, with its replacement.
4. Out-of-scope questions are refused, not answered from the nearest statute.
5. A prompt injection is refused *before retrieval runs*.

Default generation is **deterministic and extractive**: no model, no network, same input same
output. That is what makes the gate reproducible and the CI keyless. The LLM path (`--llm`) is
opt-in, keeps every rule, and falls back to extraction on any error — and is **untested in this
repository**, so no claim is made about what it does to answer quality.

## 5 · Guards, wired with the pipeline rather than after it

From `aip/guards.py`, mapped to OWASP LLM Top 10: `redact_pii` (LLM06) and `detect_injection`
(LLM01) run before retrieval; `enforce_citations` (LLM09) runs on every answer; retrieved text
is wrapped by `delimit_untrusted` on the LLM path.

Wiring guards last is the classic cut: under deadline pressure, the component whose value is
measured in things that *did not happen* is the one that gets removed. Here they are inside
`answer()`, so removing them is a visible diff rather than a silent omission — and
`test_diagnosis_stops_at_the_guards_for_an_injection` fails if retrieval ever runs on a
flagged request.

## 6 · Evaluation, and what it refuses to measure

102 golden cases, deliberately shaped: 58 single-hop, 21 supersession traps, 8 historical,
6 multi-hop, 3 paraphrase, 3 aggregation, 3 unanswerable. **43 carry forbidden repealed
provisions**, so a system that ignores supersession fails loudly rather than scoring well.

| metric | measured | floor | cases |
| --- | --- | --- | --- |
| hit@1 | 0.707 | 0.65 | 99 |
| hit@5 | 0.889 | 0.84 | 99 |
| hit@10 | 0.939 | 0.90 | 99 |
| MRR | 0.790 | 0.74 | 99 |
| nDCG@10 | 0.884 | 0.84 | 99 |
| **stale@10 (current-law only)** | **0.000** | ≤ 0.08 | 91 |
| refusal precision / recall | 1.000 / 1.000 | 1.0 / 1.0 | 102 |
| citations resolving to the index | all | all | 102 |
| **unannounced repealed citations** | **0** | 0 | 102 |
| per-case failures | 0 | 0 | 102 |

By category at k=10: supersession trap 1.000, multi-hop 1.000, paraphrase 1.000,
single-hop 0.931, historical 0.875, aggregation 0.667.

`stale@k` counts **current-law questions only** (91 cases). On a historical question a
repealed provision in the top-k *is* the correct answer, and averaging the two together made
the figure move for the wrong reason — adding good historical cases made it look worse. The
historical-only rate is reported separately as `stale@k_historical_rate` (1.0 over 8 cases),
with its case count beside it.

### Not measured, and stated at the top of every run

**answer_quality · factual_correctness · faithfulness_to_sources.**

No judge model runs offline. These are recorded as `UNMEASURED` in `results.json` and printed
by the gate on every run, rather than given a number nobody can defend. `enforce_citations`
verifies that a citation pointer is *in range*; it cannot verify the cited section supports
the claim. Those are different numbers and neither stands in for the other.

Read the retrieval table as **"it finds the right provision"**, not "it answers the question
right". The extractive generator quotes rank-1 verbatim, so its accuracy is bounded above by
hit@1 = 0.707 — and on the worst question it is confidently quoting the wrong provision (§8).

## 7 · The regression gate, and watching it fail

17 checks in `src/gate.py`: provenance, referential integrity, corpus reproducibility, storage
policy, per-case failures, five retrieval floors, supersession, refusal, citations. Exit code
non-zero on any failure; 2.9 s, no network, no key.

### The part that matters: a gate you have never seen fail is a gate you do not have

`python -m src.break_it` breaks the pipeline three ways and asserts the gate catches each.

| break | red checks |
| --- | --- |
| retrieval window `k=1` | 6 metric checks, exit 1 |
| empty `data/corpus/limitation-act.md` | `corpus is reproducible from the snapshot`, exit 1 |
| corrupted `chunks_sha256` | `index matches chunks.jsonl`, exit 1 |

**The second one was a real hole, found by running the experiment rather than by reading the
code.** Emptying a corpus act — deleting 32 provisions — left *every* pre-existing check green.
The gate measured the committed index, and nothing tied `data/corpus/*.md` back to
`data/raw/`. The README's own warning ("the service answers beautifully right up until
somebody edits a corpus document") was entirely uncaught.

The fix rebuilds the corpus from the pinned snapshot via the same `build()` the pipeline
calls, and compares byte for byte. It costs ~3 s and it is the check that makes the storage
and provenance claims true rather than aspirational.

Transcript committed at `reports/gate_failure_demo.txt`; `.github/workflows/gate.yml` runs the
gate on every push **and** a second job that runs `break_it` on the runner, because a gate
never seen to fail there is not known to fail there either.

## 8 · The service

`POST /ask` returns answer, citations, hits, `trace_id`, `latency_ms`, `latency_by_stage`, cost
and a cache block. `POST /ask/stream` streams it. `GET /metrics` aggregates. No authentication;
loopback by default, warns otherwise.

Four decisions that are load-bearing:

- **`current_only` defaults to true on every route.** A caller who reads no documentation still
  gets in-force law.
- **A refusal is HTTP 200** with `refused: true` and a reason. "I will not answer that" is a
  successful, useful response; a 4xx would force every client to parse prose to learn it
  happened.
- **Observability travels in the body**, not headers. Headers are for caches and proxies; a
  client debugging one slow answer needs the numbers next to the answer.
- **Index and embedding model are built at startup.** See §9.

### Measured latency — `python -m src.benchmark`

Wall clock around the HTTP call, over 40 golden questions. Not read from the response body: a
replay replays the original computation's timings, so quoting those would describe work the
caller did not wait for.

| | p50 | p95 | max |
| --- | --- | --- | --- |
| uncached, embedding cache off (a genuinely new question) | 20.99 ms | 25.00 ms | 155.94 ms |
| uncached, embedding cache on (a repeated question) | 7.45 ms | 9.65 ms | 101.76 ms |
| cached (exact layer replay, 40/40 confirmed) | 2.47 ms | 3.01 ms | 3.66 ms |
| refusal | 22.13 ms | 22.91 ms | 22.91 ms |

Retrieval is the system; every other stage is sub-millisecond:

| stage | p50 | p95 |
| --- | --- | --- |
| retrieval | 18.14 ms | 22.34 ms |
| sufficiency | 0.50 ms | 0.92 ms |
| guards | 0.04 ms | 0.07 ms |
| generation | 0.02 ms | 0.02 ms |
| citations | 0.01 ms | 0.02 ms |

### Cold start was a real defect, and the benchmark is how it surfaced

With the embedding model loaded lazily, the first `/ask` took **18,370 ms** against a 1,500 ms
TTFT target, while every later request took ~20 ms. **The p95 never showed it; only the max
did.** A service that is fast except for the one request that matters is not fast. The lifespan
handler now warms the model and reports `index_load_ms` and `model_warm_ms` separately, and
the worst case is 156 ms — a ~118× improvement on the number that mattered.

### Measured cost

**$0.00 per query**, reported as `priced: true`. That is a measured fact, not an absent
measurement: every model is local under the default profile. `aip/cost.py` draws a hard line
between "free, you own the hardware" and "we do not know the price", and a `$0.00` that means
*we did not look* is a different claim from one that means *nobody is billed*.

Because a real zero makes "what does this cost" unanswerable, `src/benchmark.py` also projects
the same workload onto a priced tier from the project's own price table:

| | value |
| --- | --- |
| model / price | `gemini/gemini-embedding-001`, $0.15 per Mtok input |
| measured tokens per question | 12.45 (this machine's tokenizer, counted not estimated) |
| per 1,000 queries | $0.0019 |
| per day at 10,000 queries | $0.019 |
| **per year at 10,000 queries** | **$6.82** |

That covers one embedding call per uncached query. The generator is extractive and free; a
hosted generative model would dominate it by orders of magnitude, and the default path does not
ship one. The assumption is written into `benchmark.json` so nobody has to guess.

## 9 · The semantic cache: the threshold is the deliverable

A semantic cache returns a stored answer when a new question is close enough to an old one, and
it looks like free performance. It is also the most dangerous feature in this system, because
cosine similarity does not know what a question is *about*:

> "When may a **temporary** injunction be granted?"  cosine 0.82
> "When is a **perpetual** injunction granted?"   ← different answer, CPC O.39 r.1 vs Specific Relief Act s.38

Below the threshold the service returns a confident, well-cited, fast answer to a question
nobody asked. Latency improves, cost improves, correctness quietly falls, nothing errors.

### What the sweep found — `python -m src.cache_sweep`

9,702 ordered pairs of golden questions, threshold swept 0.80–0.995:

- maximum off-diagonal similarity across all pairs: **0.8699**
- at every threshold with any hits at all, precision is **0.000**
- above 0.87 there are **no hits whatsoever**

| cosine | question | would serve the answer to | gold clash |
| --- | --- | --- | --- |
| 0.8699 | *When is an electronic record admissible?* | *…special provisions…electronic records?* | BSA s.63 vs BSA s.62 |
| 0.8387 | *How long may a warrant remain in force?* | *When may an officer arrest without a warrant?* | BNSS s.72 vs BNSS s.41 |
| 0.8222 | *temporary* injunction | *perpetual* injunction | CPC O.39 r.1 vs Specific Relief Act s.38 |

**There is no safe similarity band on this corpus.** The shipped threshold is **0.95**, above
the measured maximum, where the layer is inert; `/metrics` reporting zero semantic hits is the
correct state, not a misconfiguration. The intuition-driven 0.95 is the right ballpark and the
wrong *reason* — it would still have been unsafe on a corpus whose questions collide lower.
The number is an experiment result and the experiment ships beside it in
`data/eval/cache_sweep.json`.

The scope guard (Act identity, section number) blocks BNS-vs-IPC and s.138-vs-s.139 collisions
and costs nothing, but it caught **none** of the pairs above: "temporary" versus "perpetual"
is the dangerous axis, and no lexical guard sees it. That is why the answer was "measure the
band, find it empty, ship where it cannot fire" rather than "tune the guard harder".

Four further properties, each a bug that shipped and was then fixed:

- A cached **refusal** is never served to a similar question. A refusal is a decision about one
  question; propagating it converts a correct refusal into a wrong answer.
- The **retrieval config gates the semantic layer**, not just the exact key. Without it,
  `current_only=false` was served the answer recorded against the filtered request —
  reintroducing the exact defect §3 exists to prevent.
- An answer failing its own citation contract is **never cached**; a transient bug would
  otherwise become permanent.
- A replay carries a **fresh trace id** and names its origin. The stored payload describes the
  request that created it, so returning it verbatim made every cache hit report
  `cache.hit: false` — the answer was right and the instrumentation was wrong, which no test of
  the answer alone would have caught.

## 10 · Streaming, and the problem it creates

You cannot validate citations until the answer is complete, and streaming means you have
already sent it. The honest answer differs by generator, and the system implements both:

- **Extractive (default).** The cited provision is chosen by *retrieval*, not by a model, and
  the generator is deterministic — so the citations are known and valid before the first byte.
  The stream sends `citations` (with `validated: true`) *before* any token. There is no window
  in which an unvalidated citation is on the wire.
- **LLM (`"llm": true`).** The citation is written by the model and can be wrong, so it cannot
  be validated before the tokens go out. Every `token` event carries `unvalidated: true`, and if
  `enforce_citations` fails at the end the stream closes with `final {retracted: true}` and the
  client is told to discard what it rendered.

You cannot un-send bytes. The only honest moves are to label them or to retract them after the
fact, and the choice of which depends on which generator produced them.

## 11 · What this is not safe for

Not a disclaimer — four boundaries, each with the reason attached.

1. **Not for unsupervised legal advice, or any case where a human does not read the provision.**
   The extractive generator quotes rank-1 verbatim, so its accuracy is bounded by hit@1 = 0.707
   and on the §8 question it confidently quotes the *wrong* provision. It retrieves text; it
   does not interpret, weigh or apply.

2. **Not for anything depending on which provision is cited.** §6 lists the failure plainly.
   That would be a research tool where a human reads every citation, not a tool where the
   citation is the answer. The `enforce_citations` check verifies the pointer is *in range*;
   it cannot verify the cited section supports the claim, and the two fail independently.

3. **Not beyond English, and not beyond this corpus.** 15 documents, 930 provisions, one
   embedding model trained on English web text rather than on statute prose. `data/manifest.json`
   selects a subset of each Act — the full upstream corpus is 271,673 sections.

4. **Not exposed to a network, and not unsupervised on untrusted questions.** No authentication,
   no per-user isolation, no rate limiting. The injection detector is a filter that raises the
   cost of an attack and makes a successful one visible; a bypassed detector costs one bad
   answer, which `enforce_citations` then catches. That is a layered argument, not a guarantee.

**What may run unsupervised:** internal statute lookup over the pinned corpus, with a human
reading the returned provision, where a wrong hit is visible and a refusal is an acceptable
answer. That is the boundary the measurements actually support.

## 12 · What I would do next, ranked

**1 · Replace `all-MiniLM-L6-v2`.** Expected value is concentrated here. §8's worst failure —
"punishment for murder" → culpable homicide — is an embedding-model limitation, and the
cross-encoder already fixes that one case. A statute-trained embedding model, or
`hybrid+current+rerank` promoted to default after a re-measure, is the cheapest large gain.
*EV: high. Cost: one day plus a re-measure.*

**2 · Add an offline judge so §6 stops being a gap.** A small NLI model scoring
"is this proposition entailed by the cited provision" would convert three `UNMEASURED` fields
into measured ones and would catch citation *precision*, which `enforce_citations` cannot.
It stays offline and keyless; `aip/evals.py` already has the rubric scaffolding unused.
*EV: high — it is the number a reader most wants. Cost: one to two days.*

**3 · Fix historical retrieval with an explicit `--historical` mode** that filters *to* repealed
provisions. §3 measures historical hit@10 at 0.875, and "as it stood before 2024" is currently
answered from the current provision because BNSS s.72 is a near-identical match to CrPC s.70.
It would change measured metrics, so it belongs behind a flag and a re-baseline.
*EV: medium — a real correctness gap, narrow in blast radius. Cost: half a day.*

Not on the list, deliberately: a bigger corpus, a hosted model, or a semantic cache with a lower
threshold. Each trades a measured property for an unmeasured one.

---

## Reproducing every number

```bash
make setup && make check      # environment
make gate                     # 17 checks, the headline table in §6
make cache-sweep              # the threshold in §9
make bench                    # latency and cost in §8
make gate-record              # the red build in §7
python -m src.retrieve --compare --rerank   # the table in §3
```

Committed evidence: `data/eval/results.json` · `data/eval/cache_sweep.json` ·
`data/eval/benchmark.json` · `data/eval/benchmark_embed_cache_on.json` ·
`reports/gate_failure_demo.txt`.