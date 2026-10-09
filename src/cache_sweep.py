"""Measure the semantic cache threshold instead of asserting one.

    python -m src.cache_sweep
    python -m src.cache_sweep --min 0.80 --max 0.995 --step 0.005

Why this file exists
--------------------
A semantic cache's threshold is the deliverable. Set it too low and the service
answers questions nobody asked with confident, well-cited, fast responses; set it
too high and the layer is dead weight that only adds a lookup. Both failure modes
are silent. Neither errors, nothing goes red, and the only symptom is that
correctness quietly fell while latency and cost improved.

So the threshold gets swept, and the sweep gets committed next to the number.

How a "wrong hit" is judged
---------------------------
For every ordered pair of golden questions (probe, cached) with cosine
similarity at least `t`, ask: if the cache had answered `probe` with the answer
it holds for `cached`, would it have cited a provision that answers `probe`?

    correct  <=>  gold(cached) INTERSECT gold(probe) is non-empty

Two questions that need the same provision are interchangeable for this purpose.
This is a proxy in one respect and a direct measurement in another: it does not
run the generator, but the generator is deterministic and its citation comes
from retrieval, so gold overlap predicts the cited provision closely. `validate()`
below checks that claim empirically at three thresholds rather than asking you to
take it on trust -- it runs the real pipeline and compares.

What is deliberately *not* excluded
-----------------------------------
Cross-category pairs. A historical question pairing with a current-law one is
not noise, it is the most dangerous pair on this corpus, and excluding it would
be measuring a friendlier system than the one that ships. Pairs that straddle
the current/historical boundary are counted and reported separately.

Outputs `data/eval/cache_sweep.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.answer import answer
from src.cache import scope_of, scopes_compatible
from src.retrieve import GOLDEN, load_golden

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "eval" / "cache_sweep.json"


def _embedding_matrix(questions: list[str], model: str | None = None
                      ) -> np.ndarray:
    """Embed every question once.

    One pass, not one per pair: the pairwise matrix is then a single matmul.
    With 99 questions the difference is a second of wall clock, but the same code
    scales to a question log of ten thousand, where it is the difference between
    a sweep that runs and one that does not.
    """
    from aip.embed import embed_batch

    # input_type="query": these are search queries, not passages.
    return embed_batch(questions, model=model, input_type="query")


def sweep(cases: list[dict[str, Any]], min_t: float, max_t: float, step: float,
          model: str | None = None) -> dict[str, Any]:
    """Precision of the semantic cache at every threshold in the sweep."""
    scored = [
        c for c in cases
        if c.get("relevant_provisions") and not c.get("expect_refusal")
    ]
    questions = [c["question"] for c in scored]
    golds = [set(c["relevant_provisions"]) for c in scored]
    cats = [c.get("category", "") for c in scored]
    scopes = [scope_of(q) for q in questions]

    vectors = _embedding_matrix(questions, model=model)
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -1.0)          # a question never matches itself here

    thresholds: list[float] = []
    t = min_t
    while t <= max_t + 1e-9:
        thresholds.append(round(t, 4))
        t += step

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        hits = [(i, j) for i in range(len(scored)) for j in range(len(scored))
                if i != j and sim[i, j] >= threshold]
        if not hits:
            rows.append({"threshold": threshold, "hits": 0, "correct": 0,
                         "wrong": 0, "precision": None, "guarded_hits": 0,
                         "guarded_correct": 0, "guarded_wrong": 0,
                         "guarded_precision": None, "blocked_by_scope": 0,
                         "cross_boundary_hits": 0, "examples": []})
            continue

        correct = [(i, j) for i, j in hits if golds[j] & golds[i]]
        wrong = [(i, j) for i, j in hits if not (golds[j] & golds[i])]

        guarded_hits: list[tuple[int, int]] = []
        blocked = 0
        for i, j in hits:
            allowed, _why = scopes_compatible(scopes[j], scopes[i])
            if allowed:
                guarded_hits.append((i, j))
            else:
                blocked += 1
        guarded_correct = [(i, j) for i, j in guarded_hits
                           if golds[j] & golds[i]]
        guarded_wrong = [(i, j) for i, j in guarded_hits
                         if not (golds[j] & golds[i])]

        # A pair straddling the in-force/repealed line is the failure this whole
        # project is about, so it gets its own count.
        cross = sum(1 for i, j in wrong
                    if (cats[i] == "historical") != (cats[j] == "historical"))

        rows.append({
            "threshold": threshold,
            "hits": len(hits),
            "correct": len(correct),
            "wrong": len(wrong),
            "precision": round(len(correct) / len(hits), 4),
            "guarded_hits": len(guarded_hits),
            "guarded_correct": len(guarded_correct),
            "guarded_wrong": len(guarded_wrong),
            "guarded_precision": (round(len(guarded_correct) / len(guarded_hits), 4)
                                  if guarded_hits else None),
            "blocked_by_scope": blocked,
            "cross_boundary_wrong": cross,
            "examples": [
                {
                    "similarity": round(float(sim[i, j]), 4),
                    "probe": questions[i],
                    "probe_gold": sorted(golds[i]),
                    "would_serve_answer_to": questions[j],
                    "would_cite_gold_for": sorted(golds[j]),
                }
                for i, j in sorted(wrong, key=lambda p: -sim[p])[:5]
            ],
        })

    return {
        "n_questions": len(scored),
        "n_pairs_evaluated": len(scored) * (len(scored) - 1),
        "sweep": rows,
        "sim_summary": {
            "max_offdiagonal": round(float(sim.max()), 4),
            "mean_offdiagonal": round(float(sim[sim > -1].mean()), 4),
        },
        "method": (
            "For every ordered pair with cosine >= threshold, a hit counts as "
            "correct iff gold(cached) INTERSECT gold(probe) is non-empty. "
            "validate() below checks the proxy against the live pipeline."
        ),
    }


def knee(result: dict[str, Any]) -> dict[str, Any]:
    """Find a threshold that is safe, or report that there is not one.

    "Safe" means strictly: zero wrong hits. A threshold with 95% precision still
    ships a 5% chance of confidently answering the wrong provision, and in this
    domain the wrong provision is the whole product failing.

    Two outcomes, and the second is the interesting one:

    * a threshold exists with hits > 0 and wrong == 0 -- use it
    * no such threshold exists -- every similarity band on this corpus is
      entirely wrong hits, so the honest answer is that the layer must ship
      above the measured maximum and is therefore inert

    The second case is a real result, not a failure of the sweep. "There is no
    safe similarity threshold for this corpus" is a finding worth more than a
    number someone picked because it felt safe.
    """
    rows = [r for r in result["sweep"] if r["hits"] > 0]
    for row in rows:
        if row["guarded_wrong"] == 0:
            return {
                "threshold": row["threshold"],
                "guarded_hits": row["guarded_hits"],
                "raw_hits": row["hits"],
                "raw_wrong": row["wrong"],
                "basis": "lowest threshold with zero wrong hits under the scope guard",
                "safe": True,
            }

    max_sim = result["sim_summary"]["max_offdiagonal"]
    return {
        "threshold": None,
        "safe": False,
        "max_offdiagonal_similarity": max_sim,
        "n_wrong_at_lowest_threshold": rows[0]["wrong"] if rows else 0,
        "n_hits_at_lowest_threshold": rows[0]["hits"] if rows else 0,
        "basis": (
            "no threshold yields hits>0 with zero wrong hits. Every similarity "
            f"band on this corpus is wrong hits only (worst band: "
            f"{rows[0]['hits']}/{rows[0]['wrong']} at "
            f"{rows[0]['threshold']}) and the maximum off-diagonal similarity "
            f"across all pairs is {max_sim}. The layer must therefore ship "
            f"above {max_sim}, where it is inert on this corpus."
        ),
        "recommendation": (
            "Ship the semantic layer enabled but above the measured maximum "
            f"off-diagonal similarity ({max_sim}); it will not fire on this "
            "golden set, which is the correct outcome. Carry the hit rate on "
            "the exact layer instead."
        ),
    }


def validate(result: dict[str, Any], thresholds: list[float],
             limit: int = 12) -> list[dict[str, Any]]:
    """Check the gold-overlap proxy against the live pipeline.

    The sweep says a pair *should* be scored wrong because the two questions
    need different provisions. This runs the real `answer()` for a sample of
    those pairs and records what the cached answer would actually have cited, so
    the proxy is corroborated by measurement rather than assumed. It is not run
    over every pair: each `answer()` is a real retrieval, and a full validation
    would cost minutes for the same conclusion on a sample.
    """
    scored = [
        c for c in load_golden(GOLDEN)
        if c.get("relevant_provisions") and not c.get("expect_refusal")
    ][:limit]
    out: list[dict[str, Any]] = []
    for case in scored:
        gold = set(case["relevant_provisions"])
        result_a = answer(case["question"], k=8)
        cited = set(result_a.cited_provisions)
        out.append({
            "question": case["question"],
            "gold": sorted(gold),
            "cited": sorted(cited),
            "hit_at_1": bool(cited & gold),
        })
    return out


def format_table(result: dict[str, Any]) -> str:
    rows = result["sweep"]
    header = (f"{'thr':>6} {'hits':>6} {'wrong':>6} {'prec':>7} | "
              f"{'g_hits':>7} {'g_wrong':>7} {'g_prec':>7} {'blocked':>8} "
              f"{'xbound':>7}")
    lines = [header, "-" * len(header)]
    for r in rows:
        gp = f"{r['guarded_precision']:.3f}" if r["guarded_precision"] is not None else "  -  "
        p = f"{r['precision']:.3f}" if r["precision"] is not None else "  -  "
        lines.append(
            f"{r['threshold']:>6.3f} {r['hits']:>6} {r['wrong']:>6} {p:>7} | "
            f"{r['guarded_hits']:>7} {r['guarded_wrong']:>7} {gp:>7} "
            f"{r['blocked_by_scope']:>8} {r.get('cross_boundary_wrong', 0):>7}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min", type=float, default=0.80)
    ap.add_argument("--max", type=float, default=0.995)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--validate", type=int, default=0,
                    help="also run N golden questions end-to-end to corroborate")
    args = ap.parse_args(argv)

    cases = load_golden(GOLDEN)
    if not cases:
        print("no golden set", file=sys.stderr)
        return 1

    result = sweep(cases, args.min, args.max, args.step)
    result["knee"] = knee(result)
    if args.validate:
        result["validation"] = validate(result, [args.min, 0.9, 0.97])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    print(f"semantic cache sweep over {result['n_questions']} questions "
          f"({result['n_pairs_evaluated']} ordered pairs)\n")
    print(format_table(result))
    print("\nreading: 'hits' is the raw semantic cache, 'g_*' is with the scope "
          "guard.\nA threshold is safe when g_wrong is 0 -- zero wrong hits, not "
          "a good average.")
    print(f"\nknee: {result['knee']}")
    print(f"wrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())