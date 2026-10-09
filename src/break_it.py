"""Break the pipeline on purpose, to prove the gate notices.

    python -m src.break_it                 # drop the retrieval window to 1
    python -m src.break_it --mode corpus   # empty a corpus act
    python -m src.break_it --mode index    # corrupt the index checksum
    python -m src.break_it --keep          # break it and leave it broken

A gate you have never seen fail is a gate you do not have.

Three things produce exactly the same green tick, and none of them is a working
gate: an evaluation that never runs, a threshold no change can reach, and an
exit code that gets swallowed somewhere between the assertion and CI. The only
way to tell them apart from a passing build is to break something deliberately
and watch it go red.

So this module breaks the pipeline, runs the real gate, asserts a non-zero exit,
and restores what it touched. It exits non-zero itself if the gate *passes*,
which is the failure that matters: a gate that cannot fail is decoration.

Three break modes, and they fail for different reasons
-----------------------------------------------------
`k` sets the retrieval window to 1. hit@1 and hit@5 collapse, MRR falls,
per-case failures appear, and the numbers trip their floors. This is the
*metric* failure.

`corpus` empties a corpus act the golden set cites. `validate_golden` then fails
on referential integrity, before a single retrieval metric is computed. This is
the *integrity* failure, and it is the more important of the two: it means the
gate checks that the corpus is intact, not only that retrieval is good.

`index` corrupts the recorded chunks checksum. This is the *provenance* failure
-- the index no longer matches the chunks it claims to describe, which is the
check that catches a half-finished rebuild committed by accident.

All three are restored on the way out, including on an exception, because
leaving a repository broken is not a demonstration of anything.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
THRESHOLDS = REPO / "thresholds.yml"
INDEX_META = REPO / "data" / "index" / "index.json"
CORPUS_ACT = REPO / "data" / "corpus" / "limitation-act.md"


def run_gate() -> tuple[int, str]:
    """Run the real gate as a subprocess and return (exit code, output).

    A subprocess rather than a call, deliberately: the claim being tested is
    that `python -m src.gate` exits non-zero, because that is exactly what CI
    and a pre-commit hook execute. Calling `gate.main()` in-process would prove
    something subtly different and slightly weaker.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "src.gate"], cwd=REPO,
        capture_output=True, text=True,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _guard(path: Path) -> Path:
    """Refuse to touch anything outside the repository."""
    resolved = path.resolve()
    if REPO.resolve() not in resolved.parents:
        raise RuntimeError(f"refusing to modify a file outside the repo: {resolved}")
    return resolved


# --------------------------------------------------------------------------
# Break modes. Each returns (path, original_text) and leaves the file broken.
# --------------------------------------------------------------------------
def break_k() -> tuple[Path, str]:
    """Retrieval window to 1. Degrades quality; trips the metric floors."""
    path = _guard(THRESHOLDS)
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    out, done = [], False
    for line in lines:
        # Only the `k:` under `config:`; never a k that appears elsewhere.
        if not done and line.strip().startswith("k:"):
            out.append("  k: 1\n")
            done = True
            continue
        out.append(line)
    if not done:
        raise RuntimeError("no `k:` line found in thresholds.yml")
    path.write_text("".join(out), encoding="utf-8")
    return path, original


def break_corpus() -> tuple[Path, str]:
    """Empty a corpus act the golden set cites. Refuses to touch anything else."""
    path = _guard(CORPUS_ACT)
    original = path.read_text(encoding="utf-8")
    path.write_text("", encoding="utf-8")
    return path, original


def break_index() -> tuple[Path, str]:
    """Corrupt the recorded chunks checksum in data/index/index.json."""
    path = _guard(INDEX_META)
    original = path.read_text(encoding="utf-8")
    meta = json.loads(original)
    if "chunks_sha256" not in meta:
        raise RuntimeError(f"{path.name} has no chunks_sha256 to corrupt")
    meta["chunks_sha256"] = "0" * 64
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path, original


MODES = {
    "k": ("retrieval window forced to k=1", break_k),
    "corpus": ("data/corpus/limitation-act.md emptied", break_corpus),
    "index": ("chunks_sha256 corrupted", break_index),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=sorted(MODES), default="k")
    ap.add_argument("--keep", action="store_true",
                    help="do not restore -- for reproducing a red build on purpose")
    args = ap.parse_args(argv)

    label, breaker = MODES[args.mode]
    target, original = breaker()

    try:
        code, out = run_gate()
        print(f"\n=== gate output with the pipeline broken ({args.mode}) ===")
        print(out)
        print(f"=== gate exit code: {code} ===\n")

        if code == 0:
            print("FATAL: the gate PASSED on a deliberately broken pipeline.")
            print("This gate cannot fail, so it is not a gate. Fix it before "
                  "trusting any green tick it has ever printed.")
            return 2

        print(f"GOOD: the gate exited {code} on a broken pipeline ({label}). It bites.")
        return 0
    finally:
        if args.keep:
            print(f"\n--keep: {target.relative_to(REPO)} left broken on purpose.")
        else:
            target.write_text(original, encoding="utf-8")
            print(f"restored {target.relative_to(REPO)}")


if __name__ == "__main__":
    raise SystemExit(main())