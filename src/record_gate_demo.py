"""Record the gate-failure demonstration to reports/, as committed evidence.

    python -m src.break_it --record reports/gate_failure_demo.txt

The point is that "the gate is green" and "the gate still bites" are two
different claims, and only the second one is worth anything. A transcript of the
gate going red on a deliberately broken pipeline is the artifact that supports
it, and it belongs in the repository rather than in somebody's memory of having
watched it once.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=REPO / "reports" / "gate_failure_demo.txt")
    ap.add_argument("--modes", nargs="*", default=["k", "corpus", "index"])
    args = ap.parse_args(argv)

    chunks: list[str] = [
        "Gate failure demonstration",
        "=" * 72,
        f"generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "Each section below breaks the pipeline on purpose, runs the real gate,",
        "and records its exit code. Every one of these MUST exit non-zero. A gate",
        "that cannot fail is not a gate.",
        "",
    ]

    overall = 0
    for mode in args.modes:
        proc = subprocess.run(
            [sys.executable, "-m", "src.break_it", "--mode", mode],
            cwd=REPO, capture_output=True, text=True,
        )
        chunks.append("=" * 72)
        chunks.append(f"break mode: {mode}")
        chunks.append("=" * 72)
        chunks.append(proc.stdout.rstrip())
        if proc.stderr.strip():
            chunks.append("--- stderr ---")
            chunks.append(proc.stderr.rstrip())
        chunks.append("")
        chunks.append(f"src.break_it exit code: {proc.returncode}")
        chunks.append("")
        if proc.returncode != 0:
            overall = 1

    chunks.append("=" * 72)
    verdict = ("ALL BREAK MODES WERE CAUGHT -- the gate fails when it should."
               if overall == 0 else
               "AT LEAST ONE BREAK MODE WENT UNDETECTED -- the gate is not trusted.")
    chunks.append(verdict)
    chunks.append("=" * 72)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print(f"wrote {args.out.relative_to(REPO)}")
    print(verdict)
    return overall


if __name__ == "__main__":
    raise SystemExit(main())