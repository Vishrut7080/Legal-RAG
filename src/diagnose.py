"""Explain one query, stage by stage.

A retrieval system that answers badly is easy to build and hard to fix, because
the wrong answer looks exactly like a right one. `answer()` can only report
that it refused and why; this module reports what it *saw*, so a wrong answer
can be traced to the stage that caused it:

    question -> guards -> retrieval -> status filter -> sufficiency -> citation

Two things this deliberately does not do. It does not second-guess the answer
against a gold reference, because there is no judge model offline and a
self-assessed correctness score would be noise. And it does not hide the
sufficiency heuristic's output, because that heuristic is the weakest link in
the chain -- see `src/answer.py::_chunks_have_answer`.

Usage:

    python -m src.diagnose -q "what is the punishment for murder?"
    python -m src.diagnose -q "..." --retriever hybrid --all --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field

from aip.guards import detect_injection, redact_pii
from src.answer import answer
from src.retrieve import search

# Overlap threshold mirrored from src/answer.py so the diagnosis can show the
# arithmetic that produced a refusal. Duplicated rather than imported because
# the threshold is a private constant there; if it moves, the diagnosis is the
# first thing that should start disagreeing with the answer path.
SUFFICIENCY_THRESHOLD = 0.34


@dataclass
class HitView:
    rank: int
    provision: str
    heading: str
    in_force: bool
    score: float
    superseded_by: str = ""
    replaces: str = ""
    source: str = ""
    cited: bool = False
    preview: str = ""


@dataclass
class Diagnosis:
    """One query, every stage, and what each stage decided."""

    question: str
    retriever: str
    current_only: bool
    k: int

    # guards
    pii_redacted: list[str] = field(default_factory=list)
    injection_flagged: bool = False

    # retrieval
    n_hits: int = 0
    hits: list[HitView] = field(default_factory=list)

    # status filter
    n_repealed_returned: int = 0
    stale_hits: list[str] = field(default_factory=list)

    # sufficiency
    question_terms: list[str] = field(default_factory=list)
    shared_terms: list[str] = field(default_factory=list)
    overlap_ratio: float = 0.0
    sufficient: bool = False

    # answer
    refused: bool = False
    refusal_reason: str = ""
    answer_text: str = ""
    cited_provisions: list[str] = field(default_factory=list)
    citations_valid: bool = True

    # agent mode: did the model actually drive retrieval?
    mode: str = "extractive"
    n_searches: int = 0
    stopped_because: str = ""

    # provenance, so a diagnosis can be reproduced
    embedding_model: str = ""
    n_chunks_in_index: int = 0

    # The stage that most likely explains a wrong answer. Derived rather than
    # guessed at read time, because a diagnosis that lists everything and
    # concludes nothing is a log file.
    likely_cause: str = ""
    note: str = ""


def _stem(word: str) -> str:
    """Crude suffix stripper, mirroring src/answer.py so the two agree."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _question_terms(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "what", "which", "when", "where", "who", "whom",
        "whose", "does", "did", "do", "is", "are", "was", "were", "be",
        "been", "being", "of", "in", "on", "at", "to", "by", "with", "from",
        "as", "that", "this", "these", "those", "it", "its", "under", "law",
        "act", "code", "indian", "any", "how", "can", "has", "have", "had",
    }
    return {_stem(w) for w in re.findall(r"[a-z]{3,}", text.lower())} - stop


def diagnose(question: str, *, k: int = 8, retriever: str = "hybrid+current",
             current_only: bool = True, mode: str = "extractive") -> Diagnosis:
    """Run every stage explicitly and report what each one decided."""
    out = Diagnosis(
        question=question, retriever=retriever, current_only=current_only,
        k=k, mode=mode,
    )

    # -- stage 1: guards --------------------------------------------------
    clean, pii = redact_pii(question)
    out.pii_redacted = list(pii)
    out.injection_flagged = detect_injection(clean).flagged
    if out.injection_flagged:
        out.refused = True
        out.refusal_reason = "prompt injection pattern in the question"
        out.likely_cause = "guards"
        out.note = "Refused before retrieval, so the numbers below are empty."
        return out

    # -- stage 2: retrieval ------------------------------------------------
    hits = search(clean, k=k, which=retriever, current_only=current_only)
    out.n_hits = len(hits)

    # -- stage 3: status filter -------------------------------------------
    for position, hit in enumerate(hits, start=1):
        meta = hit.chunk.meta
        in_force = meta.get("in_force") != "False"
        if not in_force:
            out.n_repealed_returned += 1
            out.stale_hits.append(meta.get("provision", f"hit {position}"))
        out.hits.append(HitView(
            rank=position,
            provision=meta.get("provision", ""),
            heading=meta.get("heading", ""),
            in_force=in_force,
            score=round(float(hit.score), 4),
            superseded_by=meta.get("superseded_by", ""),
            replaces=meta.get("replaces", ""),
            source=hit.source or meta.get("strategy", ""),
            preview=" ".join(hit.chunk.text.split())[:180],
        ))

    # -- stage 4: sufficiency ---------------------------------------------
    terms = _question_terms(clean)
    pool = " ".join(h.chunk.text.lower() for h in hits[:3])
    pool_terms = {_stem(w) for w in re.findall(r"[a-z]{3,}", pool)}
    shared = terms & pool_terms
    out.question_terms = sorted(terms)
    out.shared_terms = sorted(shared)
    out.overlap_ratio = round(len(shared) / len(terms), 3) if terms else 0.0
    out.sufficient = bool(terms) and out.overlap_ratio >= SUFFICIENCY_THRESHOLD

    # -- stage 5: the answer path, for its citation decisions --------------
    result = answer(question, k=k, retriever=retriever, current_only=current_only,
                    mode=mode)
    out.refused = result.refused
    out.refusal_reason = str(result.stages.get("refusal_reason", ""))
    out.answer_text = result.answer
    out.cited_provisions = list(result.cited_provisions)
    out.citations_valid = result.citations_valid
    out.n_searches = int(getattr(result, "n_searches", 0))
    out.stopped_because = str(getattr(result, "stopped_because", ""))
    for view, provision in zip(out.hits, out.cited_provisions):
        if view.provision == provision:
            view.cited = True

    out.likely_cause, out.note = _explain(out)
    return out


def _explain(d: Diagnosis) -> tuple[str, str]:
    """Name the stage most likely responsible. Evidence, not a verdict."""
    if d.injection_flagged:
        return "guards", "Refused before retrieval ran."
    if d.n_hits == 0:
        return (
            "retrieval",
            "Nothing came back at all. Check the retriever name and that "
            "data/index/ exists.",
        )
    if d.current_only and d.n_repealed_returned:
        return (
            "status filter",
            f"The filter let {d.n_repealed_returned} repealed provision(s) "
            "through. That is a bug in CurrentOnly, not a retrieval problem.",
        )
    if not d.current_only and d.stale_hits:
        return (
            "expected",
            f"Repealed text is reachable here by design: {', '.join(d.stale_hits[:4])}.",
        )
    if d.refused and d.overlap_ratio < SUFFICIENCY_THRESHOLD:
        return (
            "sufficiency",
            f"Only {len(d.shared_terms)}/{len(d.question_terms)} question terms "
            f"appear in the top 3 hits (ratio {d.overlap_ratio} < "
            f"{SUFFICIENCY_THRESHOLD}). The provisions may well contain the "
            "answer; the lexical overlap test cannot see that. This is the "
            "known weak stage.",
        )
    if d.refused:
        return "answer", d.refusal_reason or "Refused with no recorded reason."
    if d.cited_provisions and all(
        not h.in_force for h in d.hits if h.cited
    ):
        return (
            "expected",
            "Answered from repealed text, which means the question was read as "
            "historical.",
        )
    top = d.hits[0] if d.hits else None
    if top and not top.cited:
        return (
            "generation",
            f"Retrieved {top.provision} first but cited something else, so the "
            "extractor judged it a poor fit.",
        )
    return (
        "none",
        "No single stage stands out; the pipeline behaved as configured. If the "
        "answer is wrong, it is wrong because the provision was retrieved and "
        "quoted correctly and the corpus does not say what you expected.",
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render(d: Diagnosis) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"question   : {d.question}")
    add(f"retriever  : {d.retriever}   k={d.k}   current_only={d.current_only}")
    if d.embedding_model:
        add(f"index      : {d.n_chunks_in_index} chunks, {d.embedding_model}")
    add("")

    add(f"guards     : pii={d.pii_redacted or 'none'}  "
        f"injection={d.injection_flagged}")
    add(f"retrieval  : {d.n_hits} hits  "
        f"({d.n_repealed_returned} repealed)")
    add(f"sufficiency: {len(d.shared_terms)}/{len(d.question_terms)} terms, "
        f"ratio {d.overlap_ratio} (need {SUFFICIENCY_THRESHOLD})")
    add(f"outcome    : {'REFUSED' if d.refused else 'answered'}"
        + (f" - {d.refusal_reason}" if d.refusal_reason else ""))
    if d.mode == "agent":
        add(f"agent      : {d.n_searches} search(es), stopped={d.stopped_because}")
    add("")

    if d.hits:
        add("    rank  score   status    provision          counterpart")
        for hit in d.hits:
            mark = "*" if hit.cited else " "
            status = "current " if hit.in_force else "REPEALED"
            if hit.superseded_by:
                other = f"-> {hit.superseded_by}"
            elif hit.replaces:
                other = f"<- {hit.replaces}"
            else:
                other = "-"
            add(f"  {mark}{hit.rank:<4} {hit.score:<7.3f} {status}  "
                f"{hit.provision:<18} {other}")
        add("")
        add("  * = cited by the answer;  -> = repealed in favour of, "
            "<- = replaces")
        add("")

    if d.answer_text:
        add("answer     :")
        for row in _wrap(d.answer_text, 92):
            add(f"  {row}")
        add("")
        add(f"citations  : {', '.join(d.cited_provisions) or 'none'}  "
            f"(valid={d.citations_valid})")

    add("")
    add(f"likely cause: {d.likely_cause}")
    if d.note:
        for row in _wrap(d.note, 88):
            add(f"  {row}")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, rows, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            rows.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        rows.append(current)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--question", "-q", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--retriever", default="hybrid+current",
                    choices=["dense", "bm25", "hybrid", "hybrid+current",
                             "hybrid+rerank"])
    ap.add_argument("--all", action="store_true",
                    help="do not filter repealed provisions")
    ap.add_argument("--mode", default="extractive",
                    choices=["extractive", "llm", "agent"])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    d = diagnose(args.question, k=args.k, retriever=args.retriever,
                 current_only=not args.all, mode=args.mode)

    # Attach index provenance so a diagnosis pins the index it describes.
    try:
        from src.retrieve import Index

        loaded = Index.load()
        d.embedding_model = loaded.meta.get("embedding_model", "")
        d.n_chunks_in_index = len(loaded.chunks)
    except Exception:  # noqa: BLE001 - provenance is a nicety, not the point
        pass

    if args.json:
        print(json.dumps(asdict(d), indent=2, ensure_ascii=False))
    else:
        print(render(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())