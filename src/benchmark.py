"""Measure the service's latency and cost, properly, and write them down.

    python -m src.benchmark
    python -m src.benchmark --n 40 --out data/eval/benchmark.json

Why this exists
---------------
A p50 quoted from a single request is an anecdote, not a measurement. And the
three numbers a reader actually wants cannot be read off one run of the service:

    uncached   a genuinely new question, embedding computed from scratch
    cached     the same question replayed -- what the user actually waits
    refusal    the path where the system declines to answer

Each needs a different measurement setup, and conflating them is how a latency
claim becomes indefensible. Two traps this module is built to avoid:

*Latency is wall clock around the HTTP call*, measured here, not read out of the
response body. The body reports the server-side computation; a cache replay
replays those numbers too, so quoting them for a cached request would report the
cost of work the user did not wait for.

*The embedding cache is reported separately.* `aip.embed` caches query vectors
on disk, so the second time a question is asked the embedding is free. A repeat
question therefore looks ~15x faster than a new one. That is a real number, but
it is not the number for a question nobody has asked before, and a benchmark
that reports only the warm figure overstates the service.

On cost
-------
Under the default `local` profile the measured cost is a real $0.00: the
embedding model runs on this machine and nobody is billed. That is reported as
priced-and-zero, never as "not measured" -- the distinction `aip.cost.is_priced`
exists to enforce.

Because a genuine $0.00 makes "what does this cost per query" unanswerable, the
run also projects the same workload onto a *priced* tier from
`aip.config.PRICES_PER_MTOK`, using the token counts this machine's tokenizer
actually produces. The projection is labelled as a projection and its inputs are
written into the output file, so nobody has to guess what it assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Must be set before aip.config is imported, because `settings` is a module
# singleton built at import time.
if os.getenv("BENCH_NO_EMBED_CACHE") == "1":
    os.environ["AIP_CACHE"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

import src.cache as cache_mod  # noqa: E402
from aip.config import PRICES_PER_MTOK  # noqa: E402
from src.cache import AnswerCache  # noqa: E402
from src.retrieve import GOLDEN, Index, load_golden  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "data" / "eval" / "benchmark.json"

# The priced tier used for the what-if projection. Chosen because it is in
# PRICES_PER_MTOK, so the number comes from the project's own price table rather
# than from a figure typed into a report.
PROJECTION_MODEL = "gemini/gemini-embedding-001"


def _summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1,
                  max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
        return ordered[idx]

    return {
        "n": len(values),
        "p50": round(pct(50), 2),
        "p95": round(pct(95), 2),
        "p99": round(pct(99), 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "mean": round(statistics.fmean(values), 2),
    }


def _by_stage(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stages: dict[str, list[float]] = {}
    for row in rows:
        for name, ms in row.get("latency_by_stage", {}).items():
            stages.setdefault(name, []).append(float(ms))
    return {name: _summarise(values) for name, values in sorted(stages.items())}


def _timed(client: TestClient, payload: dict[str, Any]) -> tuple[dict, float]:
    """Wall clock around the call, plus the parsed body."""
    started = time.perf_counter()
    response = client.post("/ask", json=payload)
    elapsed = (time.perf_counter() - started) * 1000.0
    return response.json(), elapsed


def question_token_counts(questions: list[str], model: str) -> list[int]:
    """Token counts from the index's own tokenizer.

    Counted rather than estimated, because the projection multiplies this number
    by a price and an estimated token count quietly becomes a made-up cost.
    """
    try:
        from aip.embed import _local_model

        tokenizer = _local_model(model.removeprefix("local/")).tokenizer
        return [len(tokenizer.encode(q)) for q in questions]
    except Exception as error:  # noqa: BLE001 - report, do not fail the run
        print(f"  (tokenizer unavailable: {type(error).__name__}: {error})",
              file=sys.stderr)
        return []


def projection(tokens_per_query: float, queries_per_day: int = 10_000) -> dict:
    """Cost of this workload on a priced tier, with its assumptions stated."""
    price = PRICES_PER_MTOK.get(PROJECTION_MODEL)
    if price is None:
        return {"available": False,
                "reason": f"{PROJECTION_MODEL} is not in PRICES_PER_MTOK"}
    input_per_mtok, _output = price
    per_query = tokens_per_query * input_per_mtok / 1_000_000
    return {
        "available": True,
        "model": PROJECTION_MODEL,
        "usd_per_mtok_input": input_per_mtok,
        "tokens_per_query": round(tokens_per_query, 2),
        "usd_per_query": round(per_query, 8),
        "usd_per_1_000": round(per_query * 1_000, 4),
        "usd_per_day_at_10k": round(per_query * queries_per_day, 4),
        "usd_per_year_at_10k": round(per_query * queries_per_day * 365, 2),
        "assumption": (
            "One embedding call per uncached query over one short question. "
            "Excludes the generator, which is extractive and free on this "
            "profile, and excludes the exact cache, which repeats a previous "
            "answer for free. A hosted generative model would dominate this "
            "figure by orders of magnitude; the default path does not ship one."
        ),
    }


def _display(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    `Path.relative_to` raises when --out was given relative to a different
    working directory, which turned a successful benchmark into a traceback
    after the file was already written.
    """
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=40,
                    help="how many answerable golden questions to drive")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    cases = load_golden(GOLDEN)
    answerable = [c for c in cases
                  if c.get("relevant_provisions") and not c.get("expect_refusal")]
    unanswerable = [c for c in cases if c.get("expect_refusal")]
    questions = [c["question"] for c in answerable[:args.n]]
    refusals = [c["question"] for c in unanswerable]

    if not questions:
        print("no answerable golden cases", file=sys.stderr)
        return 1

    cold_embed = os.getenv("BENCH_NO_EMBED_CACHE") == "1"

    with tempfile.TemporaryDirectory() as tmp:
        cache_mod._DEFAULT = AnswerCache(Path(tmp) / "bench.sqlite3")
        from src.service import app

        with TestClient(app) as client:
            # Warm the index and the models outside every timed region, so the
            # first request does not carry the startup cost of all the others.
            client.get("/health")

            # Pass 1 -- uncached. `use_cache: false` also skips the store, which
            # is what we want: every question in this pass is answered from
            # scratch and none of them pollutes the next pass.
            uncached_rows, uncached_ms = [], []
            for question in questions:
                body, elapsed = _timed(client, {"question": question,
                                                "use_cache": False})
                uncached_rows.append(body)
                uncached_ms.append(elapsed)

            # Pass 2 -- populate. Timings discarded; this pass exists only so
            # the cache is warm for pass 3.
            for question in questions:
                _timed(client, {"question": question})

            # Pass 3 -- cached. Every one of these should be a replay.
            cached_rows, cached_ms = [], []
            for question in questions:
                body, elapsed = _timed(client, {"question": question})
                cached_rows.append(body)
                cached_ms.append(elapsed)

            # Refusals, from scratch.
            refusal_rows, refusal_ms = [], []
            for question in refusals:
                body, elapsed = _timed(client, {"question": question,
                                                "use_cache": False})
                refusal_rows.append(body)
                refusal_ms.append(elapsed)

            metrics = client.get("/metrics").json()

    replayed = [r for r in cached_rows if r.get("replayed")]
    index = Index.load()
    token_counts = question_token_counts(questions, index.model)
    tokens_per_query = statistics.fmean(token_counts) if token_counts else 0.0

    result = {
        "generated_by": "python -m src.benchmark",
        "embedding_cache": "disabled for this run" if cold_embed else "enabled",
        "n_uncached": len(uncached_rows),
        "n_cached": len(cached_rows),
        "n_replayed": len(replayed),
        "n_refusal": len(refusal_rows),
        "hardware_note": (
            "Measured on the development machine, single process, warm index, "
            "local embedding model. Absolute numbers are machine-specific; the "
            "cached/uncached ratio and the stage breakdown are the parts that "
            "generalise."
        ),
        "latency_ms": {
            "_note": ("wall clock around the HTTP call, not the value in the "
                      "response body: a replay reports the original "
                      "computation's latency, so reading it there would quote "
                      "work the caller did not wait for"),
            "uncached": _summarise(uncached_ms),
            "cached": _summarise(cached_ms),
            "refusal": _summarise(refusal_ms),
        },
        "by_stage_ms": {
            "_note": "server-side stage timings, from the uncached pass",
            "uncached": _by_stage(uncached_rows),
        },
        "ttft_ms": metrics.get("ttft_ms"),
        "cache": {
            "hit_rate": metrics.get("cache", {}).get("hit_rate"),
            "exact_hits": metrics.get("cache", {}).get("exact"),
            "semantic_hits": metrics.get("cache", {}).get("semantic"),
            "semantic_threshold": metrics.get("semantic_threshold"),
        },
        "cost": {
            "measured_usd_total": metrics.get("cost", {}).get("total_usd", 0.0),
            "measured_usd_per_query": 0.0,
            "priced": True,
            "why_zero": (
                "Every model is local under the default profile, so there is "
                "nothing to bill. A zero here is a measured fact, not an absent "
                "measurement."
            ),
            "projection": (projection(tokens_per_query) if tokens_per_query
                           else {"available": False, "reason": "no tokenizer"}),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    def row(label: str, block: dict) -> str:
        if not block.get("n"):
            return f"  {label:<10} (no samples)"
        return (f"  {label:<10} n={block['n']:<4} p50={block['p50']:>8.2f} "
                f"p95={block['p95']:>8.2f} max={block['max']:>8.2f}")

    print(f"\nlatency, embedding cache {result['embedding_cache']}")
    print(row("uncached", result["latency_ms"]["uncached"]))
    print(row("cached", result["latency_ms"]["cached"]))
    print(row("refusal", result["latency_ms"]["refusal"]))
    print(f"  replays confirmed: {len(replayed)}/{len(cached_rows)}")

    print("\nuncached, by stage (server-side)")
    for name, block in result["by_stage_ms"]["uncached"].items():
        print(f"  {name:<14} p50={block['p50']:>8.2f} p95={block['p95']:>8.2f}")

    proj = result["cost"]["projection"]
    print(f"\ncost measured: ${result['cost']['measured_usd_per_query']:.6f}/query "
          f"(priced={result['cost']['priced']})")
    if proj.get("available"):
        print(f"projection on {proj['model']} at "
              f"${proj['usd_per_mtok_input']}/Mtok input:")
        print(f"  {proj['tokens_per_query']:.1f} tokens/query -> "
              f"${proj['usd_per_1_000']:.6f}/1,000 -> "
              f"${proj['usd_per_year_at_10k']:.2f}/year at 10k/day")
    print(f"\nwrote {_display(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())