"""Regression gate. Runs the checks that are decidable offline, and only those.

    python -m src.gate
    python -m src.gate --thresholds thresholds.yml
    python -m src.gate --fast        # skip the golden-set evaluation

Exit code 0 means every assertion held. Non-zero means one did not, and the
failing assertion is printed with its measured value and its threshold.

What this gate asserts, and what it refuses to
---------------------------------------------
It asserts provenance (the raw snapshot matches its checksums), referential
integrity (every golden citation resolves, the index matches its chunks), and
measured retrieval behaviour (hit rates, staleness, refusal, citations).

It does not assert anything about answer correctness, because no judge model
runs offline. A gate that graded generated prose it cannot check would be
worse than no gate: it would manufacture a green tick nobody earned. The gap is
stated explicitly in the report rather than left implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
THRESHOLDS = REPO / "thresholds.yml"


class Check:
    """One assertion, with enough context to act on a failure."""

    def __init__(self, name: str, ok: bool, detail: str = "",
                 measured: Any = None, threshold: Any = None) -> None:
        self.name = name
        self.ok = ok
        self.detail = detail
        self.measured = measured
        self.threshold = threshold

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        text = f"[{mark}] {self.name}"
        if self.measured is not None or self.threshold is not None:
            text += f"  (measured {self.measured}, threshold {self.threshold})"
        elif self.detail:
            text += f"  ({self.detail})"
        return text


def _run(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", *args], cwd=REPO, capture_output=True, text=True
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def check_snapshot() -> Check:
    code, out = _run(["src.fetch", "--check"])
    tail = out.splitlines()[-1] if out else ""
    return Check("raw snapshot matches FETCHED.json", code == 0, detail=tail)


def check_golden() -> Check:
    code, out = _run(["src.validate_golden"])
    last = out.splitlines()[-1] if out else ""
    return Check("golden set references resolve", code == 0, detail=last)


def check_index() -> Check:
    index_path = REPO / "data" / "index" / "index.json"
    chunks_path = REPO / "data" / "index" / "chunks.jsonl"
    if not index_path.exists() or not chunks_path.exists():
        return Check("index matches chunks.jsonl", False, detail="index missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    recorded = index.get("chunks_sha256")
    if not recorded:
        return Check("index matches chunks.jsonl", False,
                     detail="index.json has no chunks_sha256")
    digest = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    return Check(
        "index matches chunks.jsonl", digest == recorded,
        detail="" if digest == recorded else f"{digest[:16]} != {recorded[:16]}",
    )


def check_raw_storage_policy() -> Check:
    """No stored payload may carry third-party judgment summaries.

    A storage policy that is only a paragraph in a README is a policy that gets
    undone by the next person to want the raw data. IndiaCode's payloads include
    eCourts-written ratio-decidendi summaries -- original matter under its
    licence line -- and committing those would redistribute material whose
    copyright is not ours. The count is kept; the records are not. See
    LICENSING.md.
    """
    offenders: list[str] = []
    for path in sorted((REPO / "data" / "raw").glob("*/*.json")):
        if path.parent.name == "mappings":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "judgments" in payload:
            offenders.append(path.name)
    return Check(
        "raw snapshot excludes judgment records",
        not offenders,
        detail=(
            f"{len(offenders)} payload(s) still store judgments "
            f"(e.g. {offenders[0]}) - run `python -m src.fetch --sanitise`"
        ) if offenders else "",
        measured=len(offenders), threshold=0,
    )


def check_golden_count(golden: Path) -> Check:
    # Takes the path from thresholds.yml rather than hardcoding it, so this
    # check cannot end up counting a different file than the one just
    # evaluated -- a gate that counts one set and scores another is theatre.
    n_cases = sum(1 for line in golden.read_text(encoding="utf-8").splitlines() if line.strip())
    return Check(
        "golden set is non-trivial",
        n_cases >= 40,
        detail=f"{n_cases} cases",
        measured=n_cases,
        threshold=40,
    )


def check_case_failures(summary: dict[str, Any]) -> Check:
    """Assert the per-case failure count, not just the aggregate metrics.

    Aggregate metrics can sit above their floors while individual cases fail --
    an aggregate is an average, and averaging hides a broken case. `evaluate`
    already computes per-case pass/fail, so the gate has to honour it; printing
    the number and exiting 0 is the version of this check that does nothing.
    """
    failures = summary.get("failures", [])
    return Check("no failing golden cases", not failures,
                 detail=f"{len(failures)} case(s) failed",
                 measured=len(failures), threshold=0)


def check_retrieval(summary: dict[str, Any], cfg: dict[str, Any]) -> list[Check]:
    retrieval = summary["retrieval"]
    k = cfg["config"]["k"]
    limits = cfg["retrieval"]
    pairs = [
        ("hit_rate@1", limits["min_hit_rate_at_1"]),
        ("hit_rate@5", limits["min_hit_rate_at_5"]),
        (f"hit_rate@{k}", limits["min_hit_rate_at_k"]),
        ("mrr", limits["min_mrr"]),
        (f"ndcg@{k}", limits["min_ndcg_at_k"]),
    ]
    return [
        Check(f"retrieval {name}", retrieval.get(name) is not None
              and retrieval[name] >= floor,
              measured=round(retrieval.get(name) or 0.0, 3), threshold=floor)
        for name, floor in pairs
    ]


def check_supersession(summary: dict[str, Any], cfg: dict[str, Any]) -> list[Check]:
    ceiling = cfg["supersession"]["max_stale_at_k"]
    measured = summary["retrieval"].get("stale@k") or 0.0
    return [Check("no stale provisions in current-law answers", measured <= ceiling,
                  measured=round(measured, 3), threshold=f"<= {ceiling}")]


def check_refusal(summary: dict[str, Any], cfg: dict[str, Any]) -> list[Check]:
    limits = cfg["refusal"]
    refusal = summary["refusal"]
    checks = []
    for name in ("precision", "recall"):
        floor = limits[f"min_{name}"]
        measured = refusal.get(name)
        checks.append(Check(
            f"refusal {name}", measured is not None and measured >= floor,
            measured=measured, threshold=f">= {floor}",
        ))
    return checks


def check_citations(summary: dict[str, Any], cfg: dict[str, Any]) -> list[Check]:
    limits = cfg["citations"]
    citations = summary["citations"]
    checks = []
    if limits.get("require_all_valid", True):
        checks.append(Check("every citation resolves", bool(citations["all_valid"])))
    ceiling = limits.get("max_unannounced_repealed")
    if ceiling is not None:
        measured = citations.get("cases_with_unannounced_repealed", 0)
        checks.append(Check(
            "no unannounced repealed citations", measured <= ceiling,
            measured=measured, threshold=f"<= {ceiling}",
            detail=(
                f"{citations.get('cases_with_repealed_citation', 0)} repealed "
                f"citation(s), all announced as repealed"
            ) if measured == 0 else "",
        ))
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    ap.add_argument("--fast", action="store_true",
                    help="skip the golden-set evaluation (structure only)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))
    checks: list[Check] = []

    integrity = cfg.get("integrity", {})
    if integrity.get("snapshot_checksums", True):
        checks.append(check_snapshot())
    if integrity.get("golden_references_resolve", True):
        checks.append(check_golden())
        checks.append(check_golden_count(REPO / cfg["config"]["golden"]))
    if integrity.get("index_matches_chunks", True):
        checks.append(check_index())
    if integrity.get("raw_excludes_judgments", True):
        checks.append(check_raw_storage_policy())

    summary = None
    if not args.fast:
        from src.evaluate import evaluate

        run_cfg = cfg["config"]
        summary = evaluate(run_cfg["retriever"], run_cfg["k"], REPO / run_cfg["golden"])
        checks.append(check_case_failures(summary))
        checks += check_retrieval(summary, cfg)
        checks += check_supersession(summary, cfg)
        checks += check_refusal(summary, cfg)
        checks += check_citations(summary, cfg)

    print(f"gate: {args.thresholds.name}"
          f"{' (fast)' if args.fast else ''}\n")
    for check in checks:
        print(f"  {check.line()}")

    failed = [c for c in checks if not c.ok]
    print()
    if summary is not None:
        unmeasured = ", ".join(summary["unmeasured"])
        print(f"NOT MEASURED (no judge model offline): {unmeasured}")
        print()
    if failed:
        print(f"FAILED: {len(failed)} of {len(checks)} checks")
        return 1
    print(f"PASSED: {len(checks)} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())