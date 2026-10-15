"""Tests for the evaluation harness: the metric families, the comparison table,
and the gate checks that read them.

The harness runs on a two-case golden file so these tests are fast. The full
102-case run is the gate's job; what is asserted here is that the harness
produces the families the gate and the report depend on, and that the gate
refuses to manufacture a green tick when quality is unmeasured.
"""
from __future__ import annotations

import json
from pathlib import Path

from aip.evals import CaseResult, EvalReport
from src.evaluate import (
    _summarise_latency,
    compare_modes,
    evaluate,
)
from src.gate import check_cost_latency, check_generation

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "data" / "eval" / "golden.jsonl"


def _tiny_golden(tmp_path: Path) -> Path:
    """Two answerable cases, so the harness runs in milliseconds."""
    lines = [line for line in GOLDEN.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    cases = [json.loads(line) for line in lines]
    answerable = [c for c in cases if c.get("relevant_provisions")][:2]
    out = tmp_path / "golden.jsonl"
    out.write_text("\n".join(json.dumps(c) for c in answerable) + "\n",
                   encoding="utf-8")
    return out


def test_evaluate_produces_every_metric_family(tmp_path: Path) -> None:
    summary = evaluate(path=_tiny_golden(tmp_path), progress=False)
    assert summary["retrieval"]["n_cases"] == 2
    assert summary["retrieval"]["hit_rate@1"] is not None
    assert summary["retrieval"]["mrr"] is not None
    assert summary["refusal"]["n_cases"] == 2
    # With no expected refusals in the subset, precision is undefined -- None,
    # not a confident 0 or 1. That distinction is the whole point.
    assert summary["refusal"]["precision"] is None
    assert summary["citations"]["all_valid"] is True
    assert summary["generation"]["measured"] is False
    assert summary["unmeasured"] == [
        "answer_quality", "factual_correctness", "faithfulness_to_sources"]
    assert summary["latency_ms"]["p95"] >= summary["latency_ms"]["p50"]
    assert summary["failures"] == []


def test_evaluate_can_be_asked_for_a_judge(tmp_path: Path,
                                           monkeypatch) -> None:
    """The judge path is opt-in; the summary records that it is calibrated or not.

    The judge is stubbed because the thing under test is the harness, not the
    model -- and a live judge call would make this a test of the provider.
    """
    def fake_judge(prompt, **kwargs):
        return {"score": 2} if "REFERENCE" in prompt else {"score": 1}

    monkeypatch.setattr("src.evaluate.llm_judge", fake_judge)
    summary = evaluate(path=_tiny_golden(tmp_path), progress=False, judge=True)
    assert summary["generation"]["measured"] is True
    assert summary["generation"]["correctness"] == 1.0
    assert summary["generation"]["faithfulness"] == 1.0
    assert summary["generation"]["judge_calibration"] == "not_calibrated"


def test_compare_modes_returns_a_table() -> None:
    table = compare_modes(GOLDEN, modes=["extractive"])
    assert "extractive" in table
    assert "hit_rate@1" in table


def test_latency_summary_is_nearest_rank() -> None:
    report = EvalReport("x", [
        CaseResult("a", [], {}, None, 10.0),
        CaseResult("b", [], {}, None, 20.0),
        CaseResult("c", [], {}, None, 100.0),
    ], {}, "")
    summary = _summarise_latency(report)
    assert summary["p50"] == 20.0
    assert summary["max"] == 100.0
    assert summary["p95"] >= summary["p50"]


def test_latency_summary_of_nothing_is_empty() -> None:
    assert _summarise_latency(EvalReport("x", [], {}, "")) == {}


# --------------------------------------------------------------------------
# Gate checks
# --------------------------------------------------------------------------
def test_generation_check_is_silent_when_quality_is_unmeasured() -> None:
    """A gate must not manufacture a green tick nobody earned."""
    checks = check_generation(
        {"generation": {"measured": False}},
        {"generation": {"min_correctness": 0.70}},
    )
    assert checks == []


def test_generation_check_fires_below_the_floor() -> None:
    checks = check_generation(
        {"generation": {"measured": True, "correctness": 0.50, "faithfulness": 0.95}},
        {"generation": {"min_correctness": 0.70, "min_faithfulness": 0.90}},
    )
    assert len(checks) == 2
    assert checks[0].ok is False
    assert checks[1].ok is True


def test_cost_latency_check_reads_cost_per_query_and_p95() -> None:
    summary = {
        "latency_ms": {"p95": 100.0},
        "refusal": {"n_cases": 10},
        "budget": {"answer": {"cost_usd": 0.50}},
    }
    checks = check_cost_latency(
        summary,
        {"cost_latency": {"max_cost_per_query_usd": 0.01, "max_p95_latency_ms": 6000}},
    )
    assert checks[0].ok is False   # 0.50 / 10 = 0.05 > 0.01
    assert checks[1].ok is True    # 100 ms < 6000 ms
