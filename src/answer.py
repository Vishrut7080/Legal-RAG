"""Answer a legal question from the pinned corpus, with citations.

    python -m src.answer -q "What is the punishment for murder?"
    python -m src.answer -q "..." --retriever hybrid --k 8
    python -m src.answer -q "..." --llm             # opt in to generation

The contract this module enforces
--------------------------------
An answer is acceptable only if it does all five of these things:

1. Every factual claim carries a numbered citation into the retrieved set.
2. No citation points at a provision that is not in that set.
3. No cited provision is repealed, unless the question was historical and the
   answer says so out loud.
4. A question the corpus cannot answer is refused, not guessed at.
5. A question carrying a prompt injection is refused before retrieval runs.

Points 1-4 are checked here rather than trusted, because they are the ones a
generative model gets wrong silently. `aip.rag.RagAnswer` carries the verdict;
`src.evaluate` scores it.

Why there is an extractive mode
-------------------------------
The interesting failures in this project -- quoting 1860 law, citing a source
number that does not exist, answering a question about patents from a corpus of
criminal statutes -- are all decidable without a language model. So the answer
contract is implemented over a deterministic extractive generator first, and
the LLM path is layered on top of the same checks.

That ordering is deliberate. A gate that can only run when Ollama is up cannot
gate a commit. `make check` has to work on a cold machine with no model pulled,
and the offline path is what makes that possible.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from aip.guards import detect_injection, redact_pii
from aip.rag import (
    ANSWER_SYSTEM,
    REFUSAL,
    RagAnswer,
    delimit_untrusted,
    enforce_citations,
    format_context,
)
from src.retrieve import search

REPO = Path(__file__).resolve().parent.parent

# A citation marker is a bracketed integer. Kept deliberately strict: the
# generator must produce "[1]", not "[1]" with a footnote or "[see 1]".
_CITE = re.compile(r"\[(\d+)\]")

# Signals that a question is asking about the past. Used to allow a repealed
# citation, but only if the answer announces it.
_HISTORICAL = re.compile(
    r"\b(repealed|old law|earlier law|before the (?:2024 )?recodification|"
    r"former|previously|historical|indian penal code|code of criminal procedure|"
    r"evidence act)\b",
    re.IGNORECASE,
)


def _extract(question: str, hits: list[Any]) -> str:
    """Deterministic answer: quote the best provision, cite it, flag its status.

    Not a language model and does not pretend to be one. It exists so the
    citation, refusal and supersession checks have something to run against when
    no model is available.
    """
    if not hits:
        return REFUSAL

    top = hits[0].chunk
    meta = top.meta
    body = _body_of(top.text)
    provision = meta.get("provision", "the retrieved provision")

    parts = [f"{provision} provides:\n\n{body} [1]"]

    link = meta.get("replaces") or meta.get("superseded_by")
    if meta.get("in_force") == "False":
        parts.append(
            f"\n\nNote: {provision} has been repealed"
            + (f" and replaced by {link}." if link else ".")
        )
    elif link:
        parts.append(f"\n\nThis provision replaced {link}.")

    return "".join(parts)


def _body_of(text: str) -> str:
    """The operative prose of a chunk, minus its status header lines."""
    body = text
    for line in list(text.splitlines()):
        if line.startswith(("Current law.", "REPEALED", "Replaces ", "Replaced by ")):
            body = body.replace(line, "", 1)
    # Drop the leading "BNS s.103 - heading" echo; it is metadata, not answer.
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if lines:
        lines = lines[1:]
    text = "\n".join(lines).strip()
    # Trim at the amendment history block, which is not operative text.
    marker = text.find("Amendment history:")
    if marker > 0:
        text = text[:marker].strip()
    return text[:1800]


# Subject matter the corpus provably does not contain. This is a curated
# backstop, not a classifier, and it is here because lexical overlap alone is
# demonstrably not enough: "penalty for late filing of a patent application"
# overlaps Arbitration Act s.34 twice, because that section says "patent
# illegality" and "application", and the first implementation answered a
# patent question from the Arbitration Act with a valid citation. Curating the
# known gaps is honest about what it is; a real sufficiency judgement needs the
# LLM judge in `src.evaluate`, which is the primary defence, not this.
_OUT_OF_SCOPE = re.compile(
    r"\b(patent|patents|trademark|trade mark|copyright|"
    r"income[-\s]?tax|corporate[-\s]?tax|corporation[-\s]?tax|"
    r"gst|goods and services tax|"
    r"customs duty|excise duty|"
    r"succession|intestate|testamentary|"
    r"articles of association|"
    r"case law|held by the (?:supreme|high) court|"
    r"kesavananda|maneka gandhi|golak nath|vishaka|"
    r"electricity (?:act|supply)|railways act|"
    r"labour (?:act|union|court)|"
    r"election commission)\b",
    re.IGNORECASE,
)

# Words too common in any legal question to carry topical signal.
_STOP = {
    "the", "a", "an", "of", "in", "to", "is", "are", "what", "which", "who",
    "how", "does", "do", "did", "and", "or", "for", "on", "by", "with", "that",
    "this", "it", "as", "be", "under", "when", "was", "were", "has", "have",
    "can", "may", "must", "any", "my", "i", "you", "law", "act", "section",
    "provision", "provide", "provided", "gives", "give", "say", "says", "mean",
    "means", "current", "currently", "force", "india", "indian", "please",
    "tell", "explain", "describe", "according",
}

_SUFFIXES = ("ment", "ingly", "edly", "ing", "ed", "ly", "es", "s")


def _stem(word: str) -> str:
    """Crude suffix stripper, enough to align 'punishment' with 'punished'.

    A real stemmer is not worth the dependency here; the only thing this has to
    survive is the question using a different inflection than the statute.
    """
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _content(question: str) -> set[str]:
    return {
        _stem(w) for w in re.findall(r"[a-z]{3,}", question.lower())
        if w not in _STOP
    }


def _chunks_have_answer(hits: list[Any], question: str) -> tuple[bool, str]:
    """Decide whether the retrieved provisions can plausibly answer `question`.

    Two checks, in order, because each catches what the other misses:

    1. Out-of-scope subject matter. Decisive, and cheap.
    2. Stemmed content-word overlap against the top hits. A weak but real floor.

    Returns (ok, reason) so the refusal is explainable rather than a shrug.
    """
    if _OUT_OF_SCOPE.search(question):
        return False, "question is about subject matter the corpus does not cover"

    words = _content(question)
    if not words:
        return True, ""

    pool = " ".join(h.chunk.text.lower() for h in hits[:3])
    pool_words = {_stem(w) for w in re.findall(r"[a-z]{3,}", pool)}
    shared = words & pool_words
    if not shared:
        return False, f"no question term appears in the top hits ({sorted(words)[:6]})"
    ratio = len(shared) / len(words)
    if ratio < 0.34:
        return False, (
            f"only {len(shared)}/{len(words)} question terms match the top hits "
            f"(ratio {ratio:.2f} < 0.34)"
        )
    return True, ""


def answer(
    question: str,
    k: int = 8,
    retriever: str = "hybrid+current",
    current_only: bool = True,
    llm: bool = False,
) -> RagAnswer:
    """Answer `question`, enforcing the citation contract either way."""
    stages: dict[str, Any] = {}

    clean, pii = redact_pii(question)
    stages["pii"] = pii

    verdict = detect_injection(clean)
    stages["injection"] = verdict.flagged
    if verdict.flagged:
        return RagAnswer(
            question=question,
            answer="I can't process that request.",
            refused=True,
            stages=stages,
        )

    hits = search(clean, k=k, which=retriever, current_only=current_only)
    stages["n_hits"] = len(hits)
    stages["retriever"] = retriever
    stages["provisions"] = [h.chunk.meta.get("provision", "") for h in hits]

    if not hits:
        stages["refusal_reason"] = "retrieval returned nothing"
        return RagAnswer(question=question, answer=REFUSAL, hits=hits,
                         refused=True, stages=stages)

    sufficient, why = _chunks_have_answer(hits, clean)
    if not sufficient:
        stages["refusal_reason"] = why
        return RagAnswer(question=question, answer=REFUSAL, hits=hits,
                         refused=True, stages=stages)

    if not llm:
        stages["generator"] = "extractive"
        text = _extract(clean, hits)
    else:
        stages["generator"] = "llm"
        text = _generate(clean, hits, stages)

    valid, invalid = enforce_citations(text, len(hits))
    stages["citations_valid"] = valid

    cited_repealed = [
        hits[i - 1].chunk.meta.get("provision", f"source {i}")
        for i in sorted({int(m) for m in _CITE.findall(text)})
        if 1 <= i <= len(hits) and hits[i - 1].chunk.meta.get("in_force") == "False"
    ]
    stages["cited_repealed"] = cited_repealed

    # A repealed citation is only acceptable if the question was about the past
    # *and* the answer told the reader. Otherwise it is the failure this corpus
    # exists to catch.
    if cited_repealed and not (_HISTORICAL.search(clean) and "repealed" in text.lower()):
        stages["refusal_reason"] = f"quoted repealed provision(s): {cited_repealed}"
        return RagAnswer(question=question, answer=REFUSAL, hits=hits,
                         refused=True, stages=stages)

    return RagAnswer(
        question=question,
        answer=text,
        hits=hits,
        citations_valid=valid,
        invalid_citations=invalid,
        refused=False,
        stages=stages,
    )


def _generate(question: str, hits: list[Any], stages: dict[str, Any]) -> str:
    """LLM path. Falls back to extractive if no chat model is reachable."""
    from aip.config import resolve_model
    from aip.llm import chat

    tier = resolve_model("MAIN")
    stages["chat_tier"] = tier
    context = format_context(hits)
    prompt = (
        f"{delimit_untrusted(context)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the provisions above. Cite every provision you rely on "
        "as [1], [2] in order of first use. If the provisions do not answer the "
        "question, reply exactly: "
        f'"{REFUSAL}"'
    )
    try:
        return chat(prompt, system=ANSWER_SYSTEM)
    except Exception as error:  # noqa: BLE001 - any failure means no model
        stages["llm_error"] = f"{type(error).__name__}: {error}"
        stages["generator"] = "extractive (llm fallback)"
        return _extract(question, hits)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--question", "-q", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--retriever", default="hybrid+current",
                    choices=["dense", "bm25", "hybrid", "hybrid+current"])
    ap.add_argument("--no-current-only", action="store_true",
                    help="allow repealed provisions into the context")
    ap.add_argument("--llm", action="store_true",
                    help="generate with a chat model instead of extracting")
    ap.add_argument("--json", action="store_true", help="emit the full result as JSON")
    args = ap.parse_args(argv)

    result = answer(
        args.question,
        k=args.k,
        retriever=args.retriever,
        current_only=not args.no_current_only,
        llm=args.llm,
    )

    if args.json:
        print(json.dumps({
            "question": result.question,
            "answer": result.answer,
            "refused": result.refused,
            "citations_valid": result.citations_valid,
            "invalid_citations": result.invalid_citations,
            "cited_provisions": result.cited_provisions,
            "cited_repealed": result.cited_repealed,
            "stages": result.stages,
        }, ensure_ascii=False, indent=2))
        return 0 if result.citations_valid else 1

    print(f"Q: {result.question}")
    if result.refused:
        print("\nREFUSED.")
    print(f"\n{result.answer}\n")
    print(f"cited      : {', '.join(result.cited_provisions) or '(none)'}")
    print(f"citations  : {'valid' if result.citations_valid else 'INVALID'} "
          f"{result.invalid_citations or ''}")
    if result.cited_repealed:
        print(f"repealed   : {', '.join(result.cited_repealed)}")
    return 0 if result.citations_valid else 1


if __name__ == "__main__":
    sys.exit(main())