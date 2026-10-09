"""Retrieval experiments. Measure the choices; do not argue about them.

    python -m src.retrieve --question "punishment for murder"
    python -m src.retrieve --compare                      # dense vs bm25 vs hybrid
    python -m src.retrieve --compare --rerank            # and with a cross-encoder
    python -m src.retrieve --question "..." --current-only

Three claims this module is built to test rather than assert.

1. **BM25 is not a strawman on statutory text.** Lawyers query by citation.
   A question that says "section 138" must find NI Act s.138, and a dense
   bi-encoder has no particular reason to. If hybrid is not beating dense on
   this corpus, the corpus is not doing its job.

2. **The hard cases are supersession, not semantics.** "What is the
   punishment for murder?" retrieves BNS s.103 and IPC s.302 as a near-tie,
   because the two provisions *are* near-identical. That is not a retrieval
   failure, and no embedding model fixes it. `--current-only` exists to make
   the distinction visible and measurable.

3. **A cross-encoder is worth its latency here.** It reads the query and the
   provision together, including the status block, so it can prefer BNS s.103
   over IPC s.302 for a question about *current* law in a way that neither
   bi-encoder nor BM25 can express.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from aip.chunking import Chunk
from aip.config import resolve_model
from aip.embed import embed
from aip.evals import retrieval_metrics
from aip.retrieval import (
    Bm25Retriever,
    CrossEncoderReranker,
    Hit,
    HybridRetriever,
    Retriever,
)

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / "data" / "index"
GOLDEN = REPO / "data" / "eval" / "golden.jsonl"


@dataclass
class Index:
    """The committed index: chunks plus their embedding matrix."""

    chunks: list[Chunk]
    matrix: np.ndarray
    model: str
    meta: dict[str, Any]

    @classmethod
    def load(cls) -> Index:
        meta_path, chunks_path = INDEX / "index.json", INDEX / "chunks.jsonl"
        matrix_path = INDEX / "matrix.npy"
        for p in (meta_path, chunks_path, matrix_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} is missing. Run `python -m src.ingest` first."
                )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = [
            Chunk(r["text"], r["doc_id"], r["chunk_id"], r["meta"])
            for r in (json.loads(line) for line in
                     chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())
        ]
        matrix = np.load(matrix_path)
        if len(chunks) != matrix.shape[0]:
            raise ValueError(
                f"index is inconsistent: {len(chunks)} chunks but matrix has "
                f"{matrix.shape[0]} rows. Run `python -m src.ingest --rebuild`."
            )
        return cls(chunks, matrix, meta["embedding_model"], meta)

    def provision_ids(self) -> list[str]:
        """chunk_id -> provision ref, so metrics can be scored per provision."""
        return [c.meta.get("provision", c.chunk_id) for c in self.chunks]


class CurrentOnly(Retriever):
    """Wrap a retriever and drop repealed provisions from its results.

    This is the domain control, and it is worth being precise about why it
    exists. When someone asks "what is the punishment for murder", IPC s.302 is
    a *correct* retrieval: it is near-identical to BNS s.103 and its heading
    is literally "Punishment for murder". It is the wrong *answer* in 2026.

    So the fix is not a better model. It is deciding, as a product decision,
    that a statute reference tool shows in-force law by default and makes the
    repealed text reachable behind an explicit flag. `--all` is the other half
    of that: historical questions ("what did the IPC say about X?") are
    legitimate and this corpus can answer them, but they are a different
    request and should not be the default.

    `aip.rag.ANSWER_SYSTEM` carries the same rule, so the generator is told
    which is which even when retrieval returns a repealed provision.
    """

    name = "current_only"

    def __init__(self, inner: Retriever):
        self.inner = inner

    def search(self, query: str, k: int = 8) -> list[Hit]:
        hits = self.inner.search(query, k=max(k * 6, 40))
        kept = [h for h in hits if h.chunk.meta.get("in_force") == "True"]
        return [
            Hit(h.chunk, h.score, f"{self.inner.name}+current_only", rank)
            for rank, h in enumerate(kept[:k])
        ]


class DenseIndexRetriever(Retriever):
    """Dense search over the committed matrix, without re-embedding the corpus.

    `aip.retrieval.DenseRetriever` embeds passages in `__init__`. That is the
    right behaviour for a fresh run and the wrong behaviour for every run
    afterwards, because the whole point of committing `matrix.npy` is that the
    corpus is never embedded twice.
    """

    name = "dense"

    def __init__(self, index: Index):
        self.index = index

    def search(self, query: str, k: int = 8) -> list[Hit]:
        q = embed(query, model=self.index.model, input_type="query")
        scores = self.index.matrix @ q
        top = np.argsort(-scores)[:k]
        return [
            Hit(self.index.chunks[i], float(scores[i]), "dense", rank)
            for rank, i in enumerate(top)
        ]


class Reranked(Retriever):
    """Wrap a retriever with a cross-encoder second stage.

    Retrieve wide and cheap, rerank narrow and accurate. The two stages are not
    independent and the order is a real choice: reranking first means the
    cross-encoder sees near-identical old/new pairs and has to guess which the
    question meant, while filtering first means it only ever scores in-force
    text. `compare()` measures both orders rather than assuming one.

    The reranker is loaded on first use, not here. `build_retrievers` runs at
    process startup and loading a cross-encoder costs about a second; charging
    that to every service that merely *lists* the retriever names would be a
    real regression against the "build once, at startup" rule.
    """

    def __init__(self, inner: Retriever, name: str, pool_k: int = 30):
        self.inner = inner
        self._name = name
        self.pool_k = pool_k

    @property
    def name(self) -> str:  # type: ignore[override]
        # The registered name, not a derivation of the inner one: the inner
        # retriever calls the filter "current_only", and a hit labelled
        # "current_only+rerank" is not a name any caller asked for.
        return self._name

    def search(self, query: str, k: int = 8) -> list[Hit]:
        pool = self.inner.search(query, k=max(k * 3, self.pool_k))
        rr = _reranker()
        if rr is None:
            # No cross-encoder reachable. Degrading to the first stage is better
            # than erroring: the caller asked for better ranking, and unreranked
            # results are still a correct answer to that.
            return list(pool)[:k]
        return rr.rerank(query, pool, k=k)


def build_retrievers(index: Index) -> dict[str, Retriever]:
    dense = DenseIndexRetriever(index)
    bm25 = Bm25Retriever(index.chunks)
    hybrid = HybridRetriever([dense, bm25])
    current = CurrentOnly(hybrid)
    # Every name offered in the UI, the service and `diagnose` has to exist
    # here. Offering a retriever that `search()` cannot resolve is a 500 waiting
    # to happen on the one route where somebody picks it.
    return {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
        "hybrid+current": current,
        "hybrid+rerank": Reranked(hybrid, "hybrid+rerank"),
        "hybrid+current+rerank": Reranked(current, "hybrid+current+rerank"),
    }


@lru_cache(maxsize=1)
def _shared() -> tuple[Index, dict[str, Retriever]]:
    """Load the index and build retrievers once per process.

    Loading means reading ~1000 chunks and a 1.5 MB matrix; BM25 indexing is
    the slower half. Every entry point here goes through this so a `--compare`
    run does not rebuild the BM25 index per retriever.
    """
    idx = Index.load()
    return idx, build_retrievers(idx)


def search(question: str, *, k: int = 5, which: str = "hybrid",
           current_only: bool = False) -> list[Hit]:
    _, retrievers = _shared()
    retriever: Retriever = retrievers[which]
    if current_only and which != "hybrid+current":
        retriever = CurrentOnly(retriever)
    return retriever.search(question, k=k)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
def load_golden(path: Path = GOLDEN) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoderReranker | None:
    """The cross-encoder, or None when this profile cannot supply one.

    Returns None rather than raising, because reranking is an enhancement: a
    caller who asked for it still deserves a correct answer without it. Callers
    degrade to the first stage.

    The `local/` check matters more than it looks. `resolve_model` passes any
    unknown tier name through unchanged, and only the `local` profile declares a
    RERANK tier -- so under AIP_PROFILE=gemini it would hand back the literal
    string "RERANK", and `CrossEncoderReranker` would ask the HuggingFace Hub for
    a model repository called RERANK. That is a confusing 404 in a path that has
    nothing to do with the user's question.
    """
    model = resolve_model("RERANK")
    if not model or not model.startswith("local/"):
        return None
    try:
        return CrossEncoderReranker(model.split("local/", 1)[-1])
    except Exception:  # noqa: BLE001 - an unavailable model is not a crash
        return None


def compare(k: int = 10, rerank: bool = False,
            path: Path = GOLDEN) -> dict[str, dict[str, float]]:
    """Score every retriever against the golden set.

    Reports two families deliberately kept apart:

      hit_rate@k / mrr / recall@k -- did the right provision get retrieved?
      stale@1 / stale@k           -- did the top hits include REPEALED provisions?

    The second is not in `aip.evals.retrieval_metrics` because it only means
    anything on a corpus with a supersession chain. A retriever with a perfect
    hit_rate@1 and a stale@1 of 0.5 is not doing well; it is confidently quoting
    1860 law.
    """
    index, retrievers = _shared()
    cases = load_golden(path)
    if not cases:
        raise FileNotFoundError(
            f"no golden set at {path}. Write data/eval/golden.jsonl, or pass "
            f"--golden to point at another file."
        )

    provisions = set(index.provision_ids())
    ks = sorted({1, 3, 5, k})

    def score(get_hits: Any) -> dict[str, float] | None:
        rows = []
        for case in cases:
            gold = [g for g in case.get("relevant_provisions", []) if g in provisions]
            if not gold:
                continue
            hits = get_hits(case["question"])
            got = [h.chunk.meta.get("provision", "") for h in hits]
            # stale@1 is about rank 1 specifically. Filtering the hit list down
            # to the repealed ones and asking whether that list is non-empty
            # answers "is any hit stale", which is stale@k, not stale@1.
            stale_top1 = float(bool(hits) and str(hits[0].chunk.meta.get("in_force")) == "False")
            stale_any = float(any(
                str(h.chunk.meta.get("in_force")) == "False" for h in hits
            ))
            rows.append({
                **retrieval_metrics(got, gold, ks=ks),
                "stale@1": stale_top1,
                f"stale@{k}": stale_any,
            })
        if not rows:
            return None
        mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        mean["n_cases"] = len(rows)
        return mean

    out: dict[str, dict[str, float]] = {}
    for name, retriever in retrievers.items():
        # The reranked variants are registered unconditionally so that a caller
        # who picks one over the API cannot get a KeyError. That does not mean
        # every experiment should pay for a cross-encoder, so they are skipped
        # here unless --rerank asked for them.
        if not rerank and isinstance(retriever, Reranked):
            continue
        result = score(lambda q, r=retriever: r.search(q, k=k))
        if result:
            out[name] = result
    return out


def format_report(results: dict[str, dict[str, float]], k: int = 10) -> str:
    cols = ["hit_rate@1", "mrr", f"hit_rate@{k}", f"recall@{k}",
            f"precision@{k}", f"ndcg@{k}", "stale@1", f"stale@{k}", "n_cases"]
    cols = [c for c in cols if any(c in v for v in results.values())]
    header = f"{'retriever':<18}" + "".join(f"{c:>13}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, vals in results.items():
        row = f"{name:<18}"
        for c in cols:
            v = vals.get(c)
            if isinstance(v, float):
                row += f"{v:>13.3f}"
            elif isinstance(v, int):
                row += f"{v:>13d}"
            else:
                row += f"{'-':>13}"
        lines.append(row)
    return "\n".join(lines)


def show(question: str, hits: Sequence[Hit], k: int) -> str:
    lines = [f"q: {question}", ""]
    for h in hits[:k]:
        m = h.chunk.meta
        flag = "REPEALED" if m.get("in_force") == "False" else "in force"
        link = (m.get("superseded_by") or m.get("replaces") or "")
        lines.append(
            f"  [{h.rank + 1}] {h.score:+.3f}  {m.get('provision', '?'):<26} "
            f"{flag:<9} {m.get('heading', '')[:52]}"
        )
        if link:
            lines.append(f"        {link}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--question", "-q", help="ask one question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--retriever", default="hybrid",
                    choices=["dense", "bm25", "hybrid", "hybrid+current"])
    ap.add_argument("--current-only", action="store_true",
help="drop repealed provisions from the results")
    ap.add_argument("--compare", action="store_true",
                    help="score every retriever against the golden set")
    ap.add_argument("--rerank", action="store_true",
                    help="with --compare, add a cross-encoder reranking stage")
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    args = ap.parse_args(argv)

    if args.compare:
        results = compare(k=10, rerank=args.rerank, path=args.golden)
        print(format_report(results, k=10))
        print("\nhit_rate@k and stale@k are at k=10. stale@k counts questions "
              "whose top-10 includes a repealed provision.")
        return 0

    if not args.question:
        ap.error("pass --question/-q, or --compare")
    hits = search(args.question, k=args.k, which=args.retriever,
                  current_only=args.current_only)
    print(show(args.question, hits, args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
