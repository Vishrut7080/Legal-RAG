"""Score the pipeline end to end against data/eval/golden.jsonl, on `aip.evals`.

    python -m src.evaluate                      # offline: retrieval + contract
    python -m src.evaluate --judge              # + LLM-judge correctness/faithfulness
    python -m src.evaluate --mode agent         # evaluate the tool-calling path
    python -m src.evaluate --compare            # retriever/mode comparison table

The harness is the AI-in-Practice Module 1 one (`aip.evals.run_eval` /
`EvalReport` / `compare` / `retrieval_metrics` / `llm_judge`), not a second
implementation. What this module adds is the *domain* metrics the module does
not have: supersession staleness, unannounced repealed citations, forbidden
provisions, and the refusal contract.

What is measured, and what is refused to be
------------------------------------------
Everything here is decidable on a cold machine with no model pulled:

  retrieval    hit_rate@k, recall@k, mrr, ndcg@k, stale@k
  refusal      did an unanswerable question get refused, and did an answerable
               one get refused by mistake
  citations    did every citation resolve, and did any resolve to repealed text
  generation   LLM-judge correctness and faithfulness, when a judge is asked for

`answer_quality`, `factual_correctness` and `faithfulness_to_sources` are the
three the module would normally reach for, and they need a judge. With no judge
model in the offline path they are recorded as `UNMEASURED`, not given a number
nobody can defend -- the module's "no number, no claim" rule applied to itself.
Pass `--judge` (with a reachable model) and they become measured, with the
calibration status stated beside them rather than assumed.

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

from aip.evals import (
    JUDGE_RUBRIC_CORRECTNESS,
    JUDGE_RUBRIC_FAITHFULNESS,
    Case,
    compare,
    llm_judge,
    retrieval_metrics,
    run_eval,
)
from aip.retrieval import format_context
from src.answer import answer
from src.retrieve import load_golden, search

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "data" / "eval" / "golden.jsonl"
INDEX = REPO / "data" / "index" / "chunks.jsonl"

DEFAULT_K = 10
UNMEASURED = ["answer_quality", "factual_correctness", "faithfulness_to_sources"]

_INDEX_META: dict[str, dict] | None = None


def index_meta() -> dict[str, dict]:
    """provision ref -> chunk meta, loaded once."""
    global _INDEX_META
    if _INDEX_META is None:
        meta: dict[str, dict] = {}
        for line in INDEX.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)["meta"]
                meta.setdefault(entry["provision"], entry)
        _INDEX_META = meta
    return _INDEX_META


def _route(case: dict[str, Any]) -> tuple[str, bool]:
    """(retriever, current_only) for a case.

    A historical question asks about repealed text, so the current-law filter
    must not apply to it -- otherwise the gold provision is unreachable by
    construction and the category scores a guaranteed 0. Use the same
    question-aware routing as the answer pipeline.
    """
    current_only = case.get("category") != "historical"
    if current_only:
        return "hybrid+current", True
    return "dense", False


# --------------------------------------------------------------------------
# Systems under test
# --------------------------------------------------------------------------
def _retrieve(case: dict[str, Any]) -> list[str]:
    """Retrieval only: the ranked provision refs."""
    which, current_only = _route(case)
    hits = search(case["question"], k=DEFAULT_K, which=which,
                  current_only=current_only)
    return [h.chunk.meta.get("provision", "") for h in hits]


def _answer_case(case: dict[str, Any], *, mode: str) -> Any:
    """End-to-end answer, honouring the case's routing."""
    which, current_only = _route(case)
    return answer(case["question"], k=DEFAULT_K, retriever=which,
                  current_only=current_only, mode=mode)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _retrieval_metric(got: list[str], case: dict[str, Any]) -> dict[str, float]:
    """IR metrics plus supersession staleness for one case."""
    out = retrieval_metrics(got, case.get("relevant_provisions", []),
                            ks=(1, 5, DEFAULT_K))
    meta = index_meta()
    out["stale"] = float(any(
        str(meta.get(ref, {}).get("in_force")) == "False" for ref in got
    ))
    return out


def _contract_metric(result: Any, case: dict[str, Any]) -> dict[str, float]:
    """The answer contract, as booleans, for one case."""
    announced = "repealed" in result.answer.lower()
    cited = list(result.cited_provisions)
    forbidden = list(case.get("forbidden_provisions", []))
    return {
        "citations_valid": float(result.citations_valid),
        "grounded": float(bool(cited) or result.refused),
        "refused": float(result.refused),
        "refusal_correct": float(result.refused == bool(case.get("expect_refusal"))),
        "cited_forbidden": float(any(ref in cited for ref in forbidden)),
        "unannounced_repealed": float(bool(result.cited_repealed) and not announced),
    }


def _judge_metric(result: Any, case: dict[str, Any]) -> dict[str, float]:
    """LLM-judge correctness and faithfulness, or a deterministic refusal check.

    Correctness against the reference answer is only meaningful for answerable
    questions; an unanswerable case is scored deterministically on whether it
    refused, so the judge is not asked to grade a refusal against prose.
    """
    if case.get("expect_refusal"):
        return {"correctness": float(result.refused), "faithfulness": 1.0}

    if result.refused:
        return {"correctness": 0.0, "faithfulness": 1.0}

    correctness = llm_judge(JUDGE_RUBRIC_CORRECTNESS.format(
        question=case["question"], reference=case.get("reference_answer", ""),
        candidate=result.answer))
    faithfulness = llm_judge(JUDGE_RUBRIC_FAITHFULNESS.format(
        context=format_context(result.hits), answer=result.answer))

    if correctness.get("parse_error") or faithfulness.get("parse_error"):
        return {"judge_error": 1.0}
    return {
        # The rubric is 0/1/2; normalise to 1 for "same substantive content".
        "correctness": float(correctness.get("score", 0) == 2),
        "faithfulness": float(faithfulness.get("score", 0)),
    }


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def _mean(report: Any, key: str) -> float | None:
    vals = [r.metrics[key] for r in report.results if key in r.metrics]
    return sum(vals) / len(vals) if vals else None


def _summarise_retrieval(report: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {c["id"]: c for c in cases}
    # Only answerable cases have a gold provision to score against. The three
    # unanswerable cases are scored on refusal, not on retrieval -- including
    # them here would divide by 102 and report a retrieval quality the system
    # never claimed.
    scored = [r for r in report.results
              if by_id.get(r.id, {}).get("relevant_provisions")]
    current = [r for r in scored
               if by_id.get(r.id, {}).get("category") != "historical"]
    historical = [r for r in scored
                  if by_id.get(r.id, {}).get("category") == "historical"]

    def m(key: str) -> float | None:
        vals = [r.metrics[key] for r in scored if key in r.metrics]
        return sum(vals) / len(vals) if vals else None

    by_category: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        if f"hit_rate@{DEFAULT_K}" in r.metrics:
            by_category[by_id.get(r.id, {}).get("category", "?")].append(
                r.metrics[f"hit_rate@{DEFAULT_K}"])

    return {
        "hit_rate@1": m("hit_rate@1"),
        "hit_rate@5": m("hit_rate@5"),
        f"hit_rate@{DEFAULT_K}": m(f"hit_rate@{DEFAULT_K}"),
        "recall@5": m("recall@5"),
        "mrr": m("mrr"),
        f"ndcg@{DEFAULT_K}": m(f"ndcg@{DEFAULT_K}"),
        # Staleness is a defect on current-law questions only. On a historical
        # one a repealed provision in the top-k is the correct answer, so the
        # two are reported separately and the current-law figure is the one the
        # gate holds tight.
        "stale@k": _mean_rows(current, "stale"),
        "stale@k_n_current_law_cases": len(current),
        "stale@k_historical_rate": _mean_rows(historical, "stale"),
        "stale@k_n_historical_cases": len(historical),
        "n_cases": len(scored),
        "hit_rate_by_category": {
            name: sum(values) / len(values)
            for name, values in sorted(by_category.items())
        },
    }


def _mean_rows(rows: list[Any], key: str) -> float | None:
    vals = [r.metrics[key] for r in rows if key in r.metrics]
    return sum(vals) / len(vals) if vals else None


def _summarise_refusal(report: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {c["id"]: c for c in cases}
    tp = fp = fn = 0
    for r in report.results:
        expect = bool(by_id.get(r.id, {}).get("expect_refusal"))
        refused = bool(r.metrics.get("refused"))
        if expect and refused:
            tp += 1
        elif not expect and refused:
            fp += 1
        elif expect and not refused:
            fn += 1
    return {
        "n_cases": len(report.results),
        "expected_refusals": sum(1 for c in cases if c.get("expect_refusal")),
        "refused": sum(1 for r in report.results if r.metrics.get("refused")),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
    }


def _summarise_citations(report: Any) -> dict[str, Any]:
    rows = list(report.results)
    return {
        "n_cases": len(rows),
        "all_valid": all(r.metrics.get("citations_valid") for r in rows),
        "cases_with_unannounced_repealed": sum(
            1 for r in rows if r.metrics.get("unannounced_repealed")),
        "cited_forbidden": sum(1 for r in rows if r.metrics.get("cited_forbidden")),
        "grounded_rate": _mean(report, "grounded"),
    }


def _summarise_latency(report: Any) -> dict[str, float]:
    """p50/p95/p99 from the per-case wall clock, not from model-call latency.

    `aip.cost.Budget` only records *model* call latencies. The extractive path
    makes none, so its budget latency is 0 -- which would make a latency gate
    permanently green. `run_eval` times every case, so that is the number to
    gate on.
    """
    vals = sorted(r.latency_ms for r in report.results)
    if not vals:
        return {}

    def pct(p: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round((p / 100.0) * (len(vals) - 1)))))
        return vals[idx]

    return {
        "p50": round(pct(50), 2),
        "p95": round(pct(95), 2),
        "p99": round(pct(99), 2),
        "mean": round(sum(vals) / len(vals), 2),
        "max": round(vals[-1], 2),
        "n": float(len(vals)),
    }


def _summarise_generation(report: Any | None) -> dict[str, Any]:
    if report is None:
        return {"measured": False, "unmeasured": list(UNMEASURED)}
    return {
        "measured": True,
        "correctness": _mean(report, "correctness"),
        "faithfulness": _mean(report, "faithfulness"),
        "judge_errors": sum(1 for r in report.results if "judge_error" in r.metrics),
        "judge_calibration": "not_calibrated",
        "unmeasured": ["judge_calibration"],
    }


def _failures(answer_report: Any, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-case failures. An aggregate can sit above its floor while a case fails."""
    by_id = {c["id"]: c for c in cases}
    out: list[dict[str, Any]] = []
    for r in answer_report.results:
        case = by_id.get(r.id, {})
        m = r.metrics
        if r.error:
            out.append({"id": r.id, "category": case.get("category", ""),
                        "why": f"error: {r.error}"})
        if m.get("cited_forbidden"):
            out.append({"id": r.id, "category": case.get("category", ""),
                        "why": "cited a forbidden repealed provision"})
        if m.get("unannounced_repealed"):
            out.append({"id": r.id, "category": case.get("category", ""),
                        "why": "quoted a repealed provision without flagging it"})
        if not m.get("citations_valid") and not m.get("refused"):
            out.append({"id": r.id, "category": case.get("category", ""),
                        "why": "answer is neither cited nor a refusal"})
        if not m.get("refusal_correct"):
            out.append({"id": r.id, "category": case.get("category", ""),
                        "why": "refusal decision did not match the golden label"})
    return out


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------
def evaluate(mode: str = "extractive", k: int = DEFAULT_K,
             path: Path = GOLDEN, judge: bool = False,
             progress: bool = True) -> dict[str, Any]:
    global DEFAULT_K
    DEFAULT_K = k
    cases = load_golden(path)
    if not cases:
        raise FileNotFoundError(f"no golden set at {path}")

    case_objs = [
        Case(id=c["id"], input=c, expected=c, meta={"category": c.get("category", "")})
        for c in cases
    ]

    # 1. Retrieval: what did the retriever return, and was any of it repealed?
    retrieval_report = run_eval(
        f"retrieval[{mode}]", case_objs, _retrieve, _retrieval_metric,
        workers=1, progress=progress,
    )

    # 2. End-to-end answer: the contract, per case.
    answer_report = run_eval(
        f"answer[{mode}]", case_objs,
        lambda c: _answer_case(c, mode=mode), _contract_metric,
        workers=1, progress=progress,
    )

    # 3. Generation quality: only when a judge is explicitly asked for.
    generation_report = None
    if judge:
        generation_report = run_eval(
            f"generation[{mode}]", case_objs,
            lambda c: _answer_case(c, mode=mode), _judge_metric,
            workers=1, progress=progress,
        )

    retrieval = _summarise_retrieval(retrieval_report, cases)
    refusal = _summarise_refusal(answer_report, cases)
    citations = _summarise_citations(answer_report)
    generation = _summarise_generation(generation_report)
    failures = _failures(answer_report, cases)

    summary: dict[str, Any] = {
        "retriever": retrieval_report.name,
        "mode": mode,
        "k": k,
        "generator": mode,
        "retrieval": retrieval,
        "refusal": refusal,
        "citations": citations,
        "generation": generation,
        "latency_ms": _summarise_latency(answer_report),
        "unmeasured": generation.get("unmeasured", list(UNMEASURED)),
        "failures": failures,
        "budget": {
            "retrieval": retrieval_report.budget,
            "answer": answer_report.budget,
            "generation": generation_report.budget if generation_report else None,
        },
    }
    summary["metric_keys"] = sorted(
        {key for r in retrieval_report.results for key in r.metrics}
        | {key for r in answer_report.results for key in r.metrics}
    )
    return summary


# --------------------------------------------------------------------------
# Comparison, on aip.evals.compare
# --------------------------------------------------------------------------
def _run_mode(path: Path, mode: str) -> Any:
    """One mode, as a single EvalReport carrying retrieval + contract metrics.

    The two runs are merged by case id so `aip.evals.compare` sees one row per
    case with both families of metric, and can tabulate them side by side.
    """
    from aip.evals import CaseResult, EvalReport

    cases = load_golden(path)
    case_objs = [Case(id=c["id"], input=c, expected=c,
                      meta={"category": c.get("category", "")}) for c in cases]
    retrieval_report = run_eval(f"{mode}", case_objs, _retrieve,
                                _retrieval_metric, workers=1, progress=False)
    answer_report = run_eval(f"{mode}", case_objs,
                             lambda c: _answer_case(c, mode=mode),
                             _contract_metric, workers=1, progress=False)
    answer_by_id = {r.id: r for r in answer_report.results}
    merged = [
        CaseResult(
            r.id, r.output,
            {**r.metrics, **({} if answer_by_id.get(r.id) is None
                              else answer_by_id[r.id].metrics)},
            r.error, r.latency_ms,
        )
        for r in retrieval_report.results
    ]
    return EvalReport(mode, merged, answer_report.budget, "")


def compare_modes(path: Path = GOLDEN, modes: list[str] | None = None) -> str:
    """Side-by-side table of modes. This is the experiment log.

    Defaults to `extractive` because that is the only mode that runs offline
    with no model. Pass `llm`/`agent` when a chat model is reachable -- they are
    the modes the tool-calling work is about, and they cannot be measured
    without one.
    """
    modes = modes or ["extractive"]
    reports = [_run_mode(path, mode) for mode in modes]
    return compare(*reports)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def format_report(summary: dict[str, Any]) -> str:
    lines = []
    lines.append(f"retriever  : {summary['retriever']}  "
                 f"(mode={summary['mode']}, k={summary['k']})")
    lines.append("")
    lines.append("retrieval")
    for key, value in summary["retrieval"].items():
        if key.endswith("n_cases") or key == "n_cases" or value is None:
            continue
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            for name, v in value.items():
                lines.append(f"    {name:<22}{v:>8.3f}")
        else:
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
    lines.append(f"  all_valid{'':<28}{c['all_valid']}")
    lines.append(f"  cases_with_unannounced_repealed"
                 f"{'':<11}{c['cases_with_unannounced_repealed']}")
    lines.append(f"  cited_forbidden{'':<23}{c['cited_forbidden']}")
    lines.append("")

    g = summary["generation"]
    lines.append("generation")
    if g.get("measured"):
        lines.append(f"  correctness{'':<25}{g['correctness']:>8.3f}")
        lines.append(f"  faithfulness{'':<24}{g['faithfulness']:>8.3f}")
        lines.append(f"  judge_errors{'':<25}{g['judge_errors']:>8d}")
        lines.append(f"  judge_calibration{'':<21}{g['judge_calibration']}")
    else:
        lines.append("  UNMEASURED (needs a judge model): " + ", ".join(g["unmeasured"]))
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
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--mode", default="extractive",
                    choices=["extractive", "llm", "agent"])
    ap.add_argument("--judge", action="store_true",
                    help="also run the LLM judge for correctness/faithfulness")
    ap.add_argument("--compare", action="store_true",
                    help="print a comparison table instead of a single run")
    ap.add_argument("--modes", nargs="+", default=None,
                    help="modes to compare (default: extractive; llm/agent need a model)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the summary as JSON (default: data/eval/results.json)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.compare:
        print(compare_modes(args.golden, modes=args.modes))
        return 0

    summary = evaluate(mode=args.mode, k=args.k, path=args.golden, judge=args.judge)
    out = args.out or (REPO / "data" / "eval" / "results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")

    if not args.quiet:
        print(format_report(summary))
        print(f"\nwrote {out.relative_to(REPO)}")
    # A failure is a non-zero exit so CI can gate on it. Unmeasured quality is
    # not a failure, because pretending to know it is the failure.
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
