# Licensing and data provenance

Short version: the statutory text in this repository is fine to redistribute.
The judgment summaries that the IndiaCode API also returns are not, and this
repository does not store them.

## What is stored, and why it is fine

`data/raw/` holds verbatim responses from the IndiaCode e-Bhoomi API
(`https://indiacode.ecourtsindia.com/api/v1`), reduced as described below.
Every file is checksummed in `data/raw/FETCHED.json`, and `python -m src.fetch
--check` verifies the whole snapshot offline.

Statutory text is a legislative work. It is not a copyright subject in India,
and nobody holds copyright in the bare text of an Act. The API's own `licence`
field, present on every payload and recorded in `FETCHED.json`, reads:

> Published with commentary and other original matter under s.52(1)(q)(ii),
> Copyright Act 1957.

That clause permits publication of such matter, but it does not transfer
copyright. Whatever the clause permits a publisher to do, it does not make the
publisher the copyright owner, and it does not make us one.

## What is not stored, and why

Each section payload from the API also carries a `judgments` array. Measured
across this snapshot:

- 903 section payloads, **2732 judgment records**
- **53% of the bytes** in `data/raw/sections/`

Each record contains, among other fields, `applied_to_this_section` — a summary
of the holding. The API's own `source_note` states what that field is:

> Judgment holdings are the ratio decidendi as extracted from the order by
> eCourts India, reproduced unaltered.

So it is not the judgment text (Indian judgments are outside copyright under
s.5(1)(o) of the Copyright Act) and it is not the statute. It is a summary
written by eCourts India. That is "commentary and other original matter" in the
licence line, and it belongs to whoever wrote it.

Keeping it would mean this repository redistributing 2732 third-party editorial
records, under a licence we cannot grant.

The pipeline does not need them. `build.py` only ever rendered
`len(judgments)`, so `fetch.py` keeps that number as `judgments_count` and
drops the records.

**The built corpus is unchanged.** Every `Reported judgments citing it: N` line
still carries its original N, verified across all 110 BNS provisions and the
2732-record total.

## How to get the full payloads anyway

```bash
python -m src.fetch --force --keep-judgments
```

This is off by default. If you use it, you are taking on the licensing
question for your own copy, and `python -m src.gate` will fail on
`raw snapshot excludes judgment records` until you strip them again.

## This is enforced, not documented

A storage policy that lives only in prose gets reverted by the next person who
wants the raw data for debugging. So:

- `src/fetch.py::sanitise_payload` strips the records at fetch time.
- `src/fetch.py --sanitise` migrates an existing snapshot offline and refreshes
  the lockfile.
- `src/gate.py::check_raw_storage_policy` fails the gate if any stored payload
  still contains a `judgments` key.
- `tests/test_pipeline.py` covers the sanitiser and the corpus invariance.

## Not a legal opinion

This is an engineering decision about what to commit, written by someone who is
not a lawyer, based on the licence line the API publishes and the provenance
note it attaches to its own data. It is deliberately the conservative choice:
keep what the pipeline uses, drop what cannot be licensed.

If this is going in front of real users or a compliance reviewer, that is the
question to ask a lawyer, and the answer they will want is about s.52(1)(q)(ii)
of the Copyright Act 1957 and eCourts India's terms of use.

## Upstream

- Source of record: India Code (indiacode.gov.in)
- API build: recorded in `data/raw/FETCHED.json` under `upstream.build`
- Corpus date: recorded under `upstream.corpus_date`

Nothing here was scraped from a commercial provider.