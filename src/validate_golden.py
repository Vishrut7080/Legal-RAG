"""Validate data/eval/golden.jsonl against the built index.

Run: python -m src.validate_golden
Exits non-zero when any structural or referential check fails.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "index" / "chunks.jsonl"
GOLDEN = ROOT / "data" / "eval" / "golden.jsonl"

REQUIRED = (
    "id",
    "question",
    "category",
    "relevant_provisions",
    "forbidden_provisions",
    "expect_refusal",
    "reference_answer",
)

CATEGORIES = {
    "single_hop",
    "multi_hop",
    "paraphrase",
    "aggregation",
    "supersession_trap",
    "historical",
    "unanswerable",
}


def load_index() -> dict[str, dict]:
    provisions: dict[str, dict] = {}
    with INDEX.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)["meta"]
            provisions.setdefault(meta["provision"], meta)
    return provisions


def load_golden() -> list[dict]:
    cases = []
    with GOLDEN.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{GOLDEN.name}:{number}: invalid JSON ({error})") from error
    return cases


def main() -> int:
    provisions = load_index()
    cases = load_golden()
    errors: list[str] = []
    warnings: list[str] = []
    categories: Counter[str] = Counter()
    seen_ids: set[str] = set()
    trap_count = 0
    refusal_count = 0

    for number, case in enumerate(cases, start=1):
        where = f"case #{number} ({case.get('id', '?')})"

        missing = [field for field in REQUIRED if field not in case]
        if missing:
            errors.append(f"{where}: missing field(s) {missing}")
            continue

        case_id = case["id"]
        if case_id in seen_ids:
            errors.append(f"{where}: duplicate id")
        seen_ids.add(case_id)

        extra = sorted(set(case) - set(REQUIRED))
        if extra:
            errors.append(f"{where}: unexpected field(s) {extra}")

        category = case["category"]
        if category not in CATEGORIES:
            errors.append(f"{where}: unknown category {category!r}")
        categories[category] += 1

        gold = case["relevant_provisions"]
        forbidden = case["forbidden_provisions"]
        refusal = case["expect_refusal"]

        if not isinstance(refusal, bool):
            errors.append(f"{where}: expect_refusal must be a boolean")
            refusal = bool(refusal)

        if refusal:
            refusal_count += 1
            if gold or forbidden:
                errors.append(f"{where}: unanswerable case must not cite provisions")
        else:
            if not gold:
                errors.append(f"{where}: answerable case must cite at least one provision")

        if forbidden:
            trap_count += 1

        for ref in gold:
            meta = provisions.get(ref)
            if meta is None:
                errors.append(f"{where}: relevant provision {ref} is not in the index")
                continue
            in_force = str(meta.get("in_force", "")).lower() != "false"
            # A historical case is *supposed* to cite repealed text, so the
            # current-law requirement does not apply to it.
            if not in_force and category != "historical":
                errors.append(
                    f"{where}: relevant provision {ref} is repealed but cited as current law"
                )
            elif category == "historical" and in_force:
                errors.append(
                    f"{where}: historical case cites {ref}, which is still in force"
                )

        # A forbidden provision must be repealed, and must be recognisably the
        # same provision as one of the gold ones. "Recognisably" is deliberately
        # loose: the corpus records one counterpart per section, but a question
        # about punishment has two genuine traps (the repealed definition and
        # the repealed punishment), and the BNS folded those two into one
        # section, so the recorded counterpart is only one of them.
        recorded = {
            provisions[ref].get("replaces")
            for ref in gold
            if provisions.get(ref, {}).get("replaces")
        }
        gold_headings = [
            provisions[ref].get("heading", "") for ref in gold if ref in provisions
        ]
        for ref in forbidden:
            meta = provisions.get(ref)
            if meta is None:
                errors.append(f"{where}: forbidden provision {ref} is not in the index")
                continue
            if str(meta.get("in_force", "")).lower() != "false":
                errors.append(
                    f"{where}: forbidden provision {ref} is still in force but marked forbidden"
                )
            if ref in recorded:
                continue
            heading = meta.get("heading", "").lower()
            similar = any(
                heading and (heading in other or other in heading)
                for other in (h.lower() for h in gold_headings)
            )
            if similar:
                warnings.append(
                    f"{where}: forbidden provision {ref} is a near-duplicate heading "
                    f"for {gold} but is not the recorded counterpart"
                )
            else:
                errors.append(
                    f"{where}: forbidden provision {ref} has no relationship to {gold} "
                    f"(heading {meta.get('heading')!r}); it is not a supersession trap"
                )

    print(f"cases            : {len(cases)}")
    print(f"provisions indexed: {len(provisions)}")
    print(f"supersession traps: {trap_count}")
    print(f"refusal cases     : {refusal_count}")
    print("categories        :")
    for name, count in sorted(categories.items(), key=lambda item: -item[1]):
        print(f"  {name:<20} {count}")

    coverage: defaultdict[str, int] = defaultdict(int)
    for case in cases:
        for ref in case["relevant_provisions"]:
            coverage[ref.split()[0]] += 1
    print("cited acts        :")
    for name, count in sorted(coverage.items()):
        print(f"  {name:<20} {count}")

    for line in warnings:
        print(f"WARN  {line}")
    for line in errors:
        print(f"ERROR {line}")

    if errors:
        print(f"\n{len(errors)} error(s) found.")
        return 1
    print(f"\nOK: {len(cases)} cases, {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())