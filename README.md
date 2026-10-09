# Legal-RAG

Retrieval over Indian statutes, built so that the answer's legal status is part
of the answer.

The corpus is the recodified criminal law in force since 1 July 2024 (BNS,
BNSS, BSA) alongside the repealed 1860 codes (IPC, CrPC, Evidence Act) and
nine general Acts. Both sides are indexed on purpose: the interesting question
is usually not "what does the law say" but "what does the law *currently* say",
and answering that requires being able to retrieve the repealed text in order to
recognise and exclude it.

Everything runs locally with no API key. The regression gate measures what it
can defend and refuses to guess about the rest.

---

## Quick start

```bash
make setup     # venv + dependencies (~1.8 GB, mostly PyTorch)
make env       # .env from the template
make check     # what this machine can actually do
make fetch     # network: download the pinned snapshot from IndiaCode
make corpus    # local: data/raw/  -> data/corpus/
make index     # local: data/corpus/ -> data/index/
make gate      # the point of the whole thing
```

Then `make service` or `make ui` to actually use it.

`make fetch` is the only step that needs the network. `make corpus` and
`make index` are pure functions of the committed snapshot, so the corpus is
reproducible on a machine that has never seen the IndiaCode API.

On Windows, `make` targets work under Git Bash or WSL; every target is also a
plain command (`python -m src.gate`), so nothing requires `make`.

---

## Measured results

102 golden cases in `data/eval/golden.jsonl`, scored by `python -m src.gate`:

| metric | measured | floor |
| --- | --- | --- |
| hit@1 | 0.707 | 0.65 |
| hit@5 | 0.889 | 0.84 |
| hit@10 | 0.939 | 0.90 |
| MRR | 0.790 | 0.74 |
| nDCG@10 | 0.884 | 0.84 |
| stale@k on current-law questions | 0.000 | ≤ 0.08 |
| refusal precision / recall | 1.000 / 1.000 | 1.0 / 1.0 |
| citations resolving to the index | all | all |
| unannounced repealed citations | 0 | 0 |

**Not measured: answer quality, factual correctness, faithfulness to sources.**
There is no judge model in the offline path, so `src/evaluate.py` records these
as `UNMEASURED` in `data/eval/results.json` rather than producing a number
nobody can defend. The gate prints that gap on every run. Read the retrieval
table as "it finds the right provision", not "it answers the question right".

Retriever comparison (`make compare`) on the same cases, 99 scored (the 3
refusal cases have no gold provision to score against):

| retriever | hit@1 | hit@10 | MRR | nDCG@10 | stale@1 |
| --- | --- | --- | --- | --- | --- |
| dense | 0.576 | 0.919 | 0.698 | 0.803 | 0.232 |
| bm25 | 0.475 | 0.838 | 0.606 | 0.684 | 0.202 |
| hybrid | 0.586 | 0.929 | 0.716 | 0.812 | 0.212 |
| hybrid+rerank | 0.616 | 0.919 | 0.729 | 0.831 | 0.172 |
| hybrid+current | **0.667** | 0.869 | **0.742** | 0.824 | **0.000** |
| hybrid+current+rerank | **0.667** | 0.879 | 0.738 | **0.830** | **0.000** |

`hybrid+current` is the configured default. Its hit@10 is lower than plain
hybrid because the filter *removes* repealed chunks — that is the intended
behaviour, and the reason its staleness is zero.

Reranking does not earn its place here. `hybrid+current+rerank` matches
`hybrid+current` on hit@1 and is marginally *worse* on MRR, for roughly a
second per query, so it is not the default. The cross-encoder cannot rescue
what the current-law filter already handles, and on the paraphrase cases where
a reranker should help it does not.

---

## The failure this project is built around

A legal RAG that quotes repealed law with a straight face is worse than useless,
because the citation looks authoritative. Two mechanisms guard against it.

**1. Status is in the embedded text, not just metadata.** Each provision is
rendered with an explicit status block, so the retriever can match on it:

```
BNS s.109 - Punishment for culpable homicide not amounting to murder

- Provision: BNS s.109
- In force: yes
- Replaced: IPC s.307 (near-identical, score 0.96)
```

Metadata is invisible to an embedding model. This text is not, which is what
makes `hybrid+current` able to filter on status rather than on a post-hoc
guess.

**2. The current-law filter.** `--current` excludes every chunk whose
provision is repealed. It is a hard filter, not a penalty, because "probably
fine" is how 1860 ends up in a 2024 answer.

Historical questions deliberately bypass it (`src/evaluate.py` routes the
`historical` category to unrestricted `hybrid`), because "what did the old CrPC
say about X" is answerable and should be. Those answers are required to
announce that the provision is repealed, and `max_unannounced_repealed: 0` gates
exactly that.

### Weak correspondence links

The recodification maps are a similarity join, not a legal cross-reference
table. They are many-to-one by nature — BNS folds several 1860 offences into
one section — and 41 of the 2162 resolved links come back graded `weak` by the
upstream map.

Those are kept but marked, in both the corpus and the embedded text:

> Replaced by BNS s.1 (weak, score 0.56 — tentative, confirm against the
> commencement notification)

Suppressing them instead would have thrown away real information, and a repeal
warning that overstates its own confidence is how a user ends up citing the
wrong successor as settled. `tests/test_pipeline.py` asserts the caveat cannot
be dropped by a refactor.

---

## Layout

```
data/manifest.json        what gets fetched, and why
data/raw/                 IndiaCode API responses + FETCHED.json checksums
data/corpus/              normalised markdown, one file per Act
data/index/               chunks.jsonl, matrix.npy, index.json (with checksums)
data/eval/golden.jsonl    102 cases: question, category, gold, forbidden refs
data/eval/results.json    the last full evaluation run
data/eval/cache_sweep.json  measured semantic-cache thresholds
data/eval/benchmark.json  measured latency, cost and stage breakdown
thresholds.yml            every gate floor, with the measured value beside it
LICENSING.md              what may be redistributed, and why
reports/                  gate_failure_demo.txt: the gate failing on purpose
src/fetch.py              keyless API client, resumable, checksummed
src/build.py              raw -> corpus, plus supersession resolution
src/ingest.py             corpus -> chunks + embeddings + index metadata
src/retrieve.py           dense / bm25 / hybrid / current / rerank
src/answer.py             the answer contract (below)
src/evaluate.py           retrieval, staleness, refusal, citation metrics
src/gate.py               regression gate over thresholds.yml
src/diagnose.py           explains one query stage by stage
src/telemetry.py          per-stage latency, per-request cost, aggregates
src/cache.py              exact + semantic answer cache, with the scope guard
src/cache_sweep.py        measures the semantic threshold instead of asserting it
src/benchmark.py          latency and cost, cached and uncached
src/break_it.py           breaks the pipeline on purpose, to prove the gate bites
src/record_gate_demo.py   writes reports/gate_failure_demo.txt
src/service.py            FastAPI: /health /ask /ask/stream /retrieve /metrics
src/ui.py                 Streamlit UI over the same functions
src/validate_golden.py    checks the golden set against the built index
src/check.py              environment probe; never raises
aip/                      vendored library: config, embedding, caching, guards
.github/workflows/        the gate, plus a job that proves the gate can fail
```

`src/check.py` is deliberately the only module that reports rather than raises.
Every other module failing on a missing dependency tells you nothing; this one
tells you exactly which.

---

## Serving it

```bash
make service   # FastAPI, http://127.0.0.1:8000/docs
make ui        # Streamlit, http://127.0.0.1:8501
```

Both call the same functions the gate measures — the service layer holds no
judgement of its own, because an API that could answer something the CLI cannot
means the gate is measuring a different system than the one people use.

| route | does |
| --- | --- |
| `GET /health` | index loaded, chunk count, embedding model, index checksum, startup timings |
| `POST /ask` | question → answer, citations, refusal reason, latency, cost, trace id |
| `POST /ask/stream` | the same answer as server-sent events, citations before tokens |
| `POST /retrieve` | question → ranked provisions, no generation |
| `GET /provision/{ref}` | one provision by label, e.g. `BNS s.103` |
| `GET /metrics` | p50/p95/p99, per-stage latency, cost, cache hit rate |
| `POST /cache/clear` | empty the answer cache |

Five decisions worth stating:

- **`current_only` defaults to true on every route.** A caller who reads no
  documentation still gets in-force law.
- **A refusal is HTTP 200** with `refused: true` and a reason, not a 4xx. "I
  will not answer that" is a successful, useful response; turning it into an
  error would force every client to parse prose to learn it happened.
- **`citation_marker` travels with each hit**, so a client rendering `[1]` as a
  footnote knows which provision that is without re-deriving the link.
- **Latency, cost and the trace id are in the body**, not in headers. Headers
  are for caches and proxies; a client debugging one slow answer needs the
  numbers next to the answer.
- **The index and the embedding model are built at startup.** Loading them
  lazily cost the first request ~18 seconds — see [Measured latency](#measured-latency).

The service has **no authentication**. It binds to loopback by default and
warns if you bind it anywhere else. Put an authenticating proxy in front of it
before exposing it.

---

## Measured latency

`python -m src.benchmark`, over 40 golden questions, driving the HTTP surface:

| | p50 | p95 | max |
| --- | --- | --- | --- |
| uncached, embedding cache off (a genuinely new question) | 20.99 ms | 25.00 ms | 155.94 ms |
| uncached, embedding cache on (a repeated question) | 7.45 ms | 9.65 ms | 101.76 ms |
| cached (exact layer replay, 40/40 confirmed) | 2.47 ms | 3.01 ms | 3.66 ms |
| refusal | 22.13 ms | 22.91 ms | 22.91 ms |

Uncached, by stage — retrieval is the system:

| stage | p50 | p95 |
| --- | --- | --- |
| retrieval | 18.14 ms | 22.34 ms |
| sufficiency | 0.50 ms | 0.92 ms |
| guards | 0.04 ms | 0.07 ms |
| generation | 0.02 ms | 0.02 ms |
| citations | 0.01 ms | 0.02 ms |

Latency is wall clock around the HTTP call, not the number in the response body:
a replay replays the original computation's timings, so reading them there would
quote work the caller did not wait for. Both runs are committed —
`data/eval/benchmark.json` and `data/eval/benchmark_embed_cache_on.json`.

**Cold start was a real defect here.** With the embedding model loaded lazily,
the first `/ask` took **18,370 ms** against a 1,500 ms TTFT target while every
later request took ~20 ms. The p95 never showed it; only the max did. The
service now warms the model in its lifespan handler, and the worst case is
156 ms.

---

## Measured cost

$0.00 per query, and that is a measured fact rather than a missing measurement:
every model runs locally under the default profile, so nobody is billed. The
meter reports `priced: true` alongside the zero, because a `$0.00` that means
"we did not look" is a different claim from one that means "you own the
hardware".

For a figure a budget can be built on, `src/benchmark.py` also projects the
same workload onto a priced tier from `aip.config.PRICES_PER_MTOK`:

| | value |
| --- | --- |
| model | `gemini/gemini-embedding-001` |
| price | $0.15 / Mtok input |
| measured tokens per question | 12.45 |
| per query | $0.0000019 |
| per 1,000 queries | $0.0019 |
| per day at 10,000 queries | $0.019 |
| **per year at 10,000 queries** | **$6.82** |

That projection covers one embedding call per uncached query and nothing else.
The generator is extractive and free on this profile; a hosted generative model
would dominate the figure by orders of magnitude, and the default path does not
ship one.

---

## The semantic cache, and why its threshold is 0.95

`python -m src.cache_sweep` sweeps the similarity threshold over 9,702 ordered
pairs of golden questions. The result is the single most useful thing this
project measured:

- the maximum off-diagonal similarity across all pairs is **0.8699**
- at every threshold with any hits at all, precision is **0.000** — every
  similar pair of questions on this corpus is a pair about *different*
  provisions
- above 0.87 there are no hits whatsoever

The offending pairs are the statute-shaped version of the Gold/Silver problem:

| cosine | question | would serve the answer to | gold clash |
| --- | --- | --- | --- |
| 0.8699 | *When is an electronic record admissible?* | *…special provisions…electronic records?* | BSA s.63 vs BSA s.62 |
| 0.8387 | *How long may a warrant remain in force?* | *When may an officer arrest without a warrant?* | BNSS s.72 vs BNSS s.41 |
| 0.8222 | *When may a **temporary** injunction be granted?* | *When is a **perpetual** injunction granted?* | CPC O.39 r.1 vs Specific Relief Act s.38 |

So there is no safe similarity band on this corpus, and the layer ships above
the measured maximum, where it is inert. The exact layer carries the hit rate;
`/metrics` shows the semantic count is zero and that is the correct state.

The scope guard blocks Act-identity and section-number mismatches (BNS vs IPC,
s.138 vs s.139) and costs nothing, but it caught **none** of the pairs above —
"temporary" versus "perpetual" is the dangerous axis and no lexical guard sees
it. That residual is the reason the answer was "measure the band, find it
empty, ship where it cannot fire" rather than "tune the guard harder".

---

## When an answer is wrong

```bash
make diagnose Q="What is the punishment for murder?"
```

`src/diagnose.py` runs each stage explicitly and reports what it decided, then
names the stage most likely responsible:

```
guards     : pii=none  injection=False
retrieval  : 8 hits  (0 repealed)
sufficiency: 2/2 terms, ratio 1.0 (need 0.34)
outcome    : answered

    rank  score   status    provision          counterpart
  *1    0.031   current   BNS s.109          <- IPC s.307
   2    0.031   current   BNS s.105          <- IPC s.304
   3    0.029   current   BNS s.104          <- IPC s.303
   4    0.029   current   BNS s.103          <- IPC s.302

likely cause: none
  No single stage stands out; the pipeline behaved as configured. If the answer is wrong,
  it is wrong because the provision was retrieved and quoted correctly and the corpus does
  not say what you expected.
```

That last verdict is the honest one for this query. BNS s.109 is a correct
retrieval of "murder" in 2026 — it is culpable homicide not amounting to
murder, and BNS s.103 is three ranks down. The system is not broken; the
question was ambiguous and the corpus resolved it a defensible way.

The UI embeds this same output behind a "why this answer" panel, because when
an answer is wrong the useful question is which stage got it wrong.

---

## The answer contract

```bash
python -m src.answer -q "What is the punishment for murder under the law currently in force?"
```

Every answer is a `RagAnswer` carrying `question`, `answer`, `hits`,
`citations_valid`, `invalid_citations`, `refused`, and `stages` (which records
*why* a refusal happened, so a refusal is explainable rather than a shrug). Four
rules hold regardless of which backend produced the text:

1. **Citations must resolve to the index.** An unresolvable citation is a bug,
   not a formatting problem, so it fails rather than being printed.
2. **A repealed citation must be announced as repealed**, with its replacement.
3. **Out-of-scope questions are refused**, not answered from the nearest
   statute. Patent filings, corporate tax rates, and case holdings are refused;
   so is anything matching a prompt-injection pattern.
4. **PII in retrieved text is redacted** before it reaches the answer.

Default generation is deterministic and extractive: no model, no network, same
input same output. That is what makes the gate reproducible.

The sufficiency check behind refusal is a lexical overlap threshold over the
retrieved chunks (`_chunks_have_answer` in `src/answer.py`). It is a heuristic
and it is the weakest part of the pipeline — a question whose wording shares no
words with its answer will be refused rather than answered. That fails safe, but
it is a limitation, not a design.

### Generation with a chat model

```bash
python -m src.answer -q "..." --llm     # needs `ollama serve`
```

The LLM path is opt-in, keeps every citation rule above, and falls back to the
extractive path on any model error. It is **untested in this repository** —
there is no judge model available offline, so no claim is made about what it
does to answer quality.

---

## Evaluating

```bash
python -m src.validate_golden   # do the citations in the golden set exist?
python -m src.evaluate          # full run -> data/eval/results.json
python -m src.gate              # pass/fail against thresholds.yml
python -m src.gate --fast       # structure only, skips the golden run
python -m src.break_it          # break it on purpose; prove the gate goes red
python -m src.cache_sweep       # measure the semantic threshold
python -m src.benchmark         # latency and cost, cached and uncached
```

17 checks, and the gate has been watched failing. `python -m src.break_it`
breaks the pipeline three ways and asserts the gate catches each:

| break | what it trips | check that goes red |
| --- | --- | --- |
| retrieval window `k=1` | retrieval quality | 6 metric checks |
| empty `data/corpus/limitation-act.md` | corpus integrity | `corpus is reproducible from the snapshot` |
| corrupted `chunks_sha256` | index provenance | `index matches chunks.jsonl` |

The corpus check was added *because* the second break went undetected. Deleting
a corpus act left every other check green: the gate measured the committed
index, and nothing tied `data/corpus/*.md` back to `data/raw/`. The gate now
rebuilds the corpus from the snapshot and compares byte for byte, which costs
about 3 seconds per run. A transcript of all three is committed at
`reports/gate_failure_demo.txt`; regenerate it with `make gate-record`.

`.github/workflows/gate.yml` runs the gate on every push with no API key, and a
second job runs `break_it` on the runner — because a gate that has never been
seen to fail there is not known to fail there either.

`stale@k` counts current-law questions only. On a historical question a
repealed provision in the top-k *is* the correct answer, and averaging the two
together makes the metric move for the wrong reason — adding good historical
cases made the figure look worse. The historical-only figure is reported
separately as `stale@k_historical_rate`, with its case count beside it.

The golden set is deliberately shaped: 58 single-hop, 21 supersession traps, 8
historical, 6 multi-hop, 3 paraphrase, 3 aggregation, 3 unanswerable. 43 cases
carry forbidden repealed provisions, so a system that ignores supersession
fails loudly rather than quietly scoring well.

---

## Tests

```bash
make test    # 95 tests
make lint    # ruff
```

The tests target invariants that broke during development rather than coverage:
supersession resolution against known repeal pairs, heading-match ranking,
golden-set integrity, coverage of every Act in the corpus, citation-marker
strictness, the refusal lexicon's blast radius, the service's defaults and error
codes, and the storage policy.

Two of them exist because the obvious version would have passed anyway:
`test_corpus_still_reports_judgment_counts` (dropping third-party records must
not silently drop the number the corpus shows) and
`test_ask_defaults_to_current_law` (the safe default must hold for a caller who
read no documentation).

---

## Known limitations

- **Answer quality is unmeasured.** Stated at the top because it is the one
  number a reader would most want. Retrieval is measured; correctness is not.
- **Rank-1 misses on paraphrased questions.** "What is the punishment for
  murder?" retrieves BNS s.109 (culpable homicide not amounting to murder,
  10 years) rather than BNS s.103 (murder, life imprisonment and fine). The
  gold provision is in the top 10, so hit@10 is unaffected. This is what the
  embedding model is: all-MiniLM-L6-v2 is trained on English web text, not
  statute prose with its enumeration markers and defined terms.
- **Historical questions do not reliably retrieve historical provisions.** "As
  it stood before 2024" is answered from the *current* provision, because BNSS
  s.72 is a near-identical match to CrPC s.70 and hybrid ranks it first. The
  repealed text does appear by rank 10 (historical hit@10 is 0.875), and a
  user who wants the old text has to turn the current-law filter off. An
  explicit `--historical` mode that filters *to* repealed provisions would be
  the principled fix; it is not built, and it would change measured metrics.
  Diagnose shows exactly this when it happens.
- **Lexical collisions mislead historical questions.** "What did the IPC say
  about cheating by personation?" retrieves Contract Act s.15, because that
  section mentions "Indian Penal Code" and "coercion". Same root cause as the
  line above.
- **The sufficiency check is the weakest stage.** Refusal is a lexical overlap
  threshold over the top 3 chunks, so a question whose wording shares no words
  with its answer is refused rather than answered. It fails safe, and it is a
  limitation rather than a design.
- **Reranking does not beat the current filter** on these cases (identical
  hit@1, worse MRR) and is not the default.
- **The corpus is a selected subset of each Act**, not the whole of each Act.
  `data/manifest.json` says which sections, and `also_include` exists for
  additions.
- **One model, one language, English.** No regional language coverage.
- **The HF Hub warning on every run.** Set `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` once the models are cached — but not before the
  first `make index`, which needs to download them.

---

## Licensing and data provenance

Statutory text is a legislative work and nobody owns it. The judgment records
that come with the API responses are a different matter, and this repository
does not store them: 2732 eCourts records, 53% of the snapshot's bytes, each
carrying an `applied_to_this_section` summary that the API's own note calls
"the ratio decidendi as extracted from the order by eCourts India". That is
editorial matter under the licence line IndiaCode publishes, and it is not ours
to redistribute.

The corpus only ever needed the count, so `fetch.py` keeps `judgments_count`
and drops the records — the built corpus is byte-identical. The policy is
enforced rather than documented: `make gate` fails if any stored payload
contains a `judgments` key.

Full reasoning, the counts, and the escape hatch are in
[LICENSING.md](LICENSING.md). It is an engineering decision, not a legal
opinion, and it names the question to put to a lawyer.