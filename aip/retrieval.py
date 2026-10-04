"""Retrievers: dense, lexical, hybrid, reranked.

Deliberately implemented over plain NumPy + rank_bm25 rather than a framework,
because the point of this project is that you can see and change every step of
retrieval. Persistence is handled separately, by exporting the matrix to
`data/index/matrix.npy` (see `src/ingest.py`) — at this corpus size that is a
few MB, needs no server, and keeps `AIP_OFFLINE=1` working with no network.

Interface
---------
    r = DenseRetriever(chunks)              # or Bm25Retriever / HybridRetriever
    hits = r.search("punishment for murder", k=8)  # -> list[Hit]

A Hit carries the chunk, a score, and the retriever that produced it, so a
hybrid result set stays explainable.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from aip import tracing
from aip.chunking import Chunk
from aip.embed import embed, embed_batch


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str = ""
    rank: int = 0

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def text(self) -> str:
        return self.chunk.text


class Retriever:
    name = "base"

    def search(self, query: str, k: int = 8) -> list[Hit]:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
class DenseRetriever(Retriever):
    """Exact cosine search over an in-memory matrix.

    Exact search is O(n*d). For the ~800 chunks in this corpus that is well
    under a millisecond, and it removes ANN recall as a confounder while you
    are trying to work out *why* a question retrieved the wrong provision.

    That last point matters more here than in most corpora. Statutes are full
    of near-duplicate section numbers across codes that repealed each other —
    IPC s.302 and BNS s.103 are both "Punishment for murder" — so a retrieval
    miss is easy to mistake for a retrieval-quality problem when it is actually
    a supersession problem. See `src/diagnose.py`.
    """

    name = "dense"

    def __init__(self, chunks: Sequence[Chunk], model: str | None = None,
                 show_progress: bool = True):
        self.chunks = list(chunks)
        self.model = model
        # input_type="passage": these are documents being indexed. Asymmetric
        # embedding models (NVIDIA NIM) return a different vector for a passage
        # than for a query, and mixing the two silently ruins retrieval.
        self.matrix = embed_batch(
            [c.text for c in self.chunks], model=model,
            show_progress=show_progress, input_type="passage",
        )

    def search(self, query: str, k: int = 8) -> list[Hit]:
        with tracing.trace("retrieve.dense", k=k):
            q = embed(query, model=self.model, input_type="query")
            scores = self.matrix @ q
            top = np.argsort(-scores)[:k]
            return [
                Hit(self.chunks[i], float(scores[i]), "dense", rank)
                for rank, i in enumerate(top)
            ]


_TOKEN = re.compile(r"[a-z0-9]+")

# Statutory citations carry letters and hyphens that a bare [a-z0-9]+ tokeniser
# throws away: "124A", "43-B", "6A", "NI Act s.138". Splitting on those loses
# exactly the tokens BM25 is best at, so keep alphanumerics plus the connector.
_CITE_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[a-z]-\d+[a-z]?")


def tokenize(text: str) -> list[str]:
    return _CITE_TOKEN.findall(text.lower())


class Bm25Retriever(Retriever):
    """Lexical BM25.

    Keep this baseline in every experiment. It is not a strawman on statutory
    text — dense retrieval loses to BM25 on exact provision numbers, Act
    short titles, defined terms of art, and negation, and the way lawyers
    actually query a statute book is by citation far more often than by
    paraphrase. `src/retrieve.py` measures this rather than assuming it.
    """

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def search(self, query: str, k: int = 8) -> list[Hit]:
        with tracing.trace("retrieve.bm25", k=k):
            scores = self.bm25.get_scores(tokenize(query))
            top = np.argsort(-scores)[:k]
            return [
                Hit(self.chunks[i], float(scores[i]), "bm25", rank)
                for rank, i in enumerate(top)
            ]


class HybridRetriever(Retriever):
    """Reciprocal Rank Fusion of any number of retrievers.

        score(d) = sum_r  1 / (rrf_k + rank_r(d))

    RRF is the right default for fusion because it needs no score
    normalisation — BM25 scores and cosine similarities are not on a
    comparable scale, and every attempt to make them comparable by min-max
    scaling is fragile. Ranks are comparable by construction.

    Cormack et al. (2009), "Reciprocal Rank Fusion outperforms Condorcet".
    """

    name = "hybrid"

    def __init__(self, retrievers: Iterable[Retriever], rrf_k: int = 60,
                 weights: Sequence[float] | None = None):
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.weights = list(weights) if weights else [1.0] * len(self.retrievers)

    def search(self, query: str, k: int = 8) -> list[Hit]:
        with tracing.trace("retrieve.hybrid", k=k, n_retrievers=len(self.retrievers)):
            pool = max(k * 4, 20)
            fused: dict[str, float] = {}
            best: dict[str, Hit] = {}
            for w, r in zip(self.weights, self.retrievers):
                for hit in r.search(query, k=pool):
                    cid = hit.chunk.chunk_id
                    fused[cid] = fused.get(cid, 0.0) + w / (self.rrf_k + hit.rank + 1)
                    if cid not in best or hit.rank < best[cid].rank:
                        best[cid] = hit
            ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
            return [
                Hit(best[cid].chunk, score, "hybrid", rank)
                for rank, (cid, score) in enumerate(ordered)
            ]


class CrossEncoderReranker:
    """Second-stage reranking with a cross-encoder.

    Retrieve wide (k=30) and cheap, then rerank narrow (k=5) and accurate.
    A cross-encoder reads the query and the passage *together*, so it can
    judge relevance in a way a bi-encoder — which never sees them at the same
    time — structurally cannot. It is far too slow to run over the whole
    corpus, which is why it is a second stage.
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)

    def rerank(self, query: str, hits: Sequence[Hit], k: int = 5) -> list[Hit]:
        if not hits:
            return []
        with tracing.trace("rerank.cross_encoder", n_in=len(hits), k=k):
            scores = self.model.predict([(query, h.text) for h in hits])
            order = np.argsort(-np.asarray(scores))[:k]
            return [
                Hit(hits[i].chunk, float(scores[i]), "reranked", rank)
                for rank, i in enumerate(order)
            ]


class LLMReranker:
    """Reranking by asking a small LLM to score relevance 0-10.

    Slower and more expensive than a cross-encoder, but needs no extra model
    download and handles domain-specific notions of relevance you can describe
    in words. Lab 3 asks you to compare the two on quality, cost and latency.
    """

    PROMPT = (
        "Rate how well the passage answers the query, 0 (irrelevant) to 10 "
        "(fully answers it). Reply with the number only.\n\n"
        "Query: {q}\n\nPassage: {p}\n\nScore:"
    )

    def __init__(self, tier: str = "SMALL"):
        self.tier = tier

    def rerank(self, query: str, hits: Sequence[Hit], k: int = 5) -> list[Hit]:
        from aip.llm import chat

        scored: list[tuple[float, Hit]] = []
        with tracing.trace("rerank.llm", n_in=len(hits), k=k):
            for h in hits:
                out = chat(
                    self.PROMPT.format(q=query, p=h.text[:1500]),
                    tier=self.tier, max_tokens=8, temperature=0.0,
                )
                m = re.search(r"\d+(?:\.\d+)?", out)
                scored.append((float(m.group()) if m else 0.0, h))
        scored.sort(key=lambda t: -t[0])
        return [Hit(h.chunk, s, "llm_reranked", i) for i, (s, h) in enumerate(scored[:k])]


def format_context(hits: Sequence[Hit], max_chars: int = 8000) -> str:
    """Render hits as a numbered context block the generator can cite.

    Numbering is what makes `[n]`-style citations checkable: the generator can
    only cite a number you gave it, so a hallucinated citation index is a
    detectable error rather than an invisible one.

    Each block is labelled with the doc_id *and* the provision reference that
    `src/build.py` put in the chunk metadata, because "BNS s.103" is what a
    lawyer can act on — "bns.md" is not. The provision string is passed in as
    part of the heading line already present in the chunk text; doc_id is the
    stable key for scoring.
    """
    parts, total = [], 0
    for i, h in enumerate(hits, start=1):
        block = f"[{i}] (source: {h.doc_id})\n{h.text.strip()}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)
