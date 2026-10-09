"""Score the pipeline end to end against data/eval/golden.jsonl.

    python -m src.evaluate
    python -m src.evaluate --retriever hybrid      # compare against the default
    python -m src.evaluate --out data/eval/results.json

What this measures, and what it refuses to measure
-------------------------------------------------
Everything here is decidable on a cold machine with no model pulled:

  retrieval    hit_rate@k, mrr, ndcg@k, recall@k per retriever
  supersession stale@k -- did the top-k contain a repealed provision
  refusal      did an unanswerable question get refused, and did an answerable
               one get refused by mistake
  citations    did every citation resolve, and did any resolve to repealed text

It does not measure answer *quality*. There is no judge model in the offline
path, so "is this answer legally correct" is unmeasured and is reported as
`unmeasured: [answer_quality]`, not as a passing grade. Recording a number we
cannot defend is worse than recording no number.

The extractive generator quotes the rank-1 provision verbatim, so its accuracy
is bounded above by hit_rate@1 -- 0.707 on this corpus. That is why the
extractive mode is a harness for the contract, not a QA system, and why the two
scores are reported together rather than separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from aip.evals import retrieval_metrics
from src.answer import answer
from src.retrieve import load_golden, search

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "data" / "eval" / "golden.jsonl"

UNMEASURED = ["answer_quality", "factual_correctness", "faithfulness_to_sources"]


def evaluate(retriever: str, k: int, path: Path) -> dict[str, Any]:
    cases = load_golden(path)
    if not cases:
        raise FileNotFoundError(f"no golden set at {path}")

    retrieval_rows: list[dict[str, Any]] = []
    refusal_rows: list[dict[str, Any]] = []
    citation_rows: list[dict[str, Any]] = []
    by_category: defaultdict[str, list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []

    for case in cases:
        question = case["question"]
        gold = list(case.get("relevant_provisions", []))
        expect_refusal = bool(case.get("expect_refusal"))
        category = case.get("category", "")

        # A historical question asks about repealed text, so the current-law
        # filter must not apply to it -- otherwise the gold provision is
        # unreachable by construction and the category scores a guaranteed 0.
        # Scoring it that way would be measuring the filter, not the retriever.
        #
        # Falling back to the plain retriever rather than passing
        # current_only=False is deliberate: `hybrid+current` *is* the filtered
        # retriever, so relaxing the flag does not relax it.
        current_only = category != "historical"
        which = retriever if current_only else "hybrid"

        hits = search(question, k=k, which=which, current_only=current_only)
        got = [h.chunk.meta.get("provision", "") for h in hits]
        stale = float(any(str(h.chunk.meta.get("in_force")) == "False" for h in hits))

        result = answer(question, k=k, retriever=which, current_only=current_only)

        record = {
            "id": case["id"],
            "category": category,
            "question": question,
            "gold": gold,
            "top_hits": got[:5],
        }

        if gold:
            metrics = retrieval_metrics(got, gold, ks=(1, 5, k))
            record["retrieval"] = metrics
            record["stale@k"] = stale
            record["current_only"] = current_only
            retrieval_rows.append({"metrics": metrics, "stale": stale,
                                   "current_only": current_only})
            by_category[category].append(metrics[f"hit_rate@{k}"])

        refusal_rows.append({
            "id": case["id"],
            "expect_refusal": expect_refusal,
            "refused": result.refused,
        })
        record["refused"] = result.refused

        # Two different things, and only one is a defect. Quoting repealed law
        # in answer to "what did the old IPC say?" is the whole point. Quoting
        # it without saying so is the failure. So the announcement is what gets
        # counted, not the citation.
        announced = "repealed" in result.answer.lower()

        citation_rows.append({
            "id": case["id"],
            "citations_valid": result.citations_valid,
            "cited_repealed": result.cited_repealed,
            "cited_provisions": result.cited_provisions,
            "unannounced": bool(result.cited_repealed and not announced),
        })
        record["citations_valid"] = result.citations_valid
        record["cited_repealed"] = result.cited_repealed
        record["repealed_cited_but_not_announced"] = bool(
            result.cited_repealed and not announced
        )

        # Failures are what makes this file worth having: a metric that moves
        # without anyone noticing is not the same as a metric that holds.
        if expect_refusal and not result.refused:
            failures.append({**record, "why": "answerable-looking refusal case was not refused"})
        if not expect_refusal and result.refused:
            failures.append({
                **record,
                "why": result.stages.get("refusal_reason", "refused unexpectedly"),
            })
        if not result.citations_valid:
            failures.append({**record, "why": f"invalid citations {result.invalid_citations}"})
        if result.cited_repealed and not announced:
            failures.append({**record, "why": "quoted a repealed provision without flagging it"})
        if stale and category != "historical":
            failures.append({**record, "why": f"stale provision in top-{k}: {got}"})

    return _summarise(retriever, k, retrieval_rows, refusal_rows, citation_rows,
                      by_category, failures)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(r[key] for r in rows) / len(rows)


def _summarise(retriever: str, k: int, retrieval_rows, refusal_rows,
               citation_rows, by_category, failures) -> dict[str, Any]:
    metric_keys = sorted({key for row in retrieval_rows for key in row["metrics"]})

    retrieval = {
        f"hit_rate@{cutoff}": _mean(
            [{"v": row["metrics"].get(f"hit_rate@{cutoff}", 0.0)} for row in retrieval_rows],
            "v",
        )
        for cutoff in (1, 5, k)
    }
    retrieval["mrr"] = _mean([{"v": row["metrics"].get("mrr", 0.0)} for row in retrieval_rows], "v")
    retrieval[f"ndcg@{k}"] = _mean(
        [{"v": row["metrics"].get(f"ndcg@{k}", 0.0)} for row in retrieval_rows], "v"
    )
    # Staleness is only a defect on current-law questions. On a historical one
    # a repealed provision in the top-k is the correct answer, so averaging the
    # two together produces a number that moves for the wrong reason -- adding
    # good historical cases made the staleness figure look worse.
    current_rows = [r for r in retrieval_rows if r["current_only"]]
    historical_rows = [r for r in retrieval_rows if not r["current_only"]]
    retrieval["stale@k"] = _mean(current_rows, "stale")
    retrieval["stale@k_n_current_law_cases"] = len(current_rows)
    # A rate, and named as one. `stale@k_historical_cases` read like a count and
    # was a rate, which is the kind of naming mistake that makes a metric
    # unreadable to the person deciding whether to trust it.
    retrieval["stale@k_historical_rate"] = _mean(historical_rows, "stale")
    retrieval["stale@k_n_historical_cases"] = len(historical_rows)
    retrieval["n_cases"] = len(retrieval_rows)

    tp = sum(1 for r in refusal_rows if r["expect_refusal"] and r["refused"])
    fp = sum(1 for r in refusal_rows if not r["expect_refusal"] and r["refused"])
    fn = sum(1 for r in refusal_rows if r["expect_refusal"] and not r["refused"])
    refusal = {
        "n_cases": len(refusal_rows),
        "expected_refusals": sum(1 for r in refusal_rows if r["expect_refusal"]),
        "refused": sum(1 for r in refusal_rows if r["refused"]),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
    }

    citations = {
        "n_cases": len(citation_rows),
        "all_valid": all(r["citations_valid"] for r in citation_rows),
        "cases_with_repealed_citation": sum(1 for r in citation_rows if r["cited_repealed"]),
        "cases_with_unannounced_repealed": sum(
            1 for r in citation_rows if r["unannounced"]
        ),
    }

    return {
        "retriever": retriever,
        "k": k,
        "generator": "extractive",
        "retrieval": retrieval,
        "refusal": refusal,
        "citations": citations,
        "hit_rate_by_category": {
            name: sum(values) / len(values)
            for name, values in sorted(by_category.items())
        },
        "unmeasured": UNMEASURED,
        "failures": failures,
        "metric_keys": metric_keys,
    }


def format_report(summary: dict[str, Any]) -> str:
    lines = []
    lines.append(f"retriever  : {summary['retriever']}  (k={summary['k']}, "
                 f"current-only, {summary['generator']})")
    lines.append("")
    lines.append("retrieval")
    for key, value in summary["retrieval"].items():
        if key.endswith("n_cases") or key == "n_cases" or value is None:
            continue
        lines.append(f"  {key:<34}{value:>8.3f}")
    lines.append(f"  {'n_cases':<34}{summary['retrieval']['n_cases']:>8d}")
    lines.append("")
    r = summary["refusal"]
    lines.append("refusal")
    for key in ("n_cases", "expected_refusals", "refused", "true_positive",
                "false_positive", "false_negative", "precision", "recall"):
        value = r[key]
        shown = f"{value:.3f}" if isinstance(value, float) else str(value)
        lines.append(f"  {key:<20}{shown:>10}")
    lines.append("")
    c = summary["citations"]
    lines.append("citations")
    for key, value in c.items():
        lines.append(f"  {key:<34}{value}")
    lines.append("")
    lines.append(f"hit_rate by category (@{summary['k']})")
    for name, value in summary["hit_rate_by_category"].items():
        lines.append(f"  {name:<22}{value:>8.3f}")
    lines.append("")
    lines.append("NOT MEASURED (needs a judge model): " + ", ".join(summary["unmeasured"]))
    lines.append("")
    if summary["failures"]:
        lines.append(f"failures: {len(summary['failures'])}")
        for failure in summary["failures"][:25]:
            lines.append(f"  [{failure['category']}] {failure['id']}: {failure['why']}")
        if len(summary["failures"]) > 25:
            lines.append(f"  ... and {len(summary['failures']) - 25} more")
    else:
        lines.append("failures: 0")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--retriever", default="hybrid+current",
                    choices=["dense", "bm25", "hybrid", "hybrid+current"])
    ap.add_argument("--out", type=Path, default=None,
                    help="write the summary as JSON (default: data/eval/results.json)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    summary = evaluate(args.retriever, args.k, args.golden)
    out = args.out or (REPO / "data" / "eval" / "results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(format_report(summary))
        print(f"\nwrote {out.relative_to(REPO)}")
    # A failure is a non-zero exit so CI can gate on it. Unmeasured quality is
    # not a failure, because pretending to know it is the failure.
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())