"""A reference RAG pipeline, written as seven named stages.

`src/answer.py` builds on this. This file exists so there is a working baseline
to improve on, and so the diagnosis in `src/diagnose.py` has something to
diagnose.

The reason it is structured as seven explicit stages is the reason the whole
diagnosis step exists: when a RAG system gives a wrong answer, "RAG is bad" is
not a diagnosis. Exactly one of these stages usually failed, and each has a
different fix:

    1. ingest    the provision was never in the corpus        -> data
    2. chunk     the answer was split across two chunks      -> chunking
    3. embed     the query and passage words do not overlap  -> hybrid/HyDE
    4. retrieve  the right chunk existed but ranked 14th     -> k, rerank
    5. rerank    the right chunk was in the pool but dropped -> reranker
    6. generate  the right chunk was in context, ignored     -> prompt
    7. present   correct answer, wrong or missing citation  -> output contract

Trace every stage (this file does) and the diagnosis takes a minute instead of
an afternoon.

On this corpus stage 6 has a domain-specific failure that stages 1-5 cannot
catch, which is why the system prompt below is longer than a generic one. The
Indian criminal and evidence law was recodified on 1 July 2024: the IPC, CrPC
and Indian Evidence Act 1872 were repealed and replaced by the Bharatiya Nyaya
Sanhita, Bharatiya Nagarik Suraksha Sanhita and Bharatiya Sakshya Adhiniyam.
The corpus deliberately holds both generations. Retrieval will happily return
IPC s.302 ("Punishment for murder") for a question about murder, it is a
near-perfect lexical and semantic match for it, and quoting it is a wrong
answer in July 2026. No retriever can be held responsible for that. The rule
belongs in the output contract.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from aip import tracing
from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted, enforce_citations
from aip.llm import chat
from aip.retrieval import Hit, Retriever, format_context

ANSWER_SYSTEM = f"""\
You answer questions about Indian statutes and court procedure using ONLY the
numbered sources provided.

Rules, in priority order:
1. If the sources do not contain the answer, reply exactly:
   "I don't have enough information in the provided sources to answer that."
   Do not guess, and do not fall back on your own knowledge of the law.
2. Every factual sentence must end with a citation of the source(s) that
   support it, in the form [1] or [2][5].
3. Never cite a provision number that was not given to you. If a source is
   marked REPEALED, say so, and prefer the provision of the Act that replaced
   it when one is also present.
4. If two sources state different rules, say they differ and cite both.
5. Answer the question asked. If it names an Act, that Act's numbering applies.
6. Be concise: two or three sentences unless the question needs more.
7. This is reference information about the text of a law, not legal advice.
   Do not say what someone should do in their case.

{UNTRUSTED_SYSTEM_CLAUSE}
"""

REFUSAL = "I don't have enough information in the provided sources to answer that."


@dataclass
class RagAnswer:
    question: str
    answer: str
    hits: list[Hit] = field(default_factory=list)
    citations_valid: bool = True
    invalid_citations: list[int] = field(default_factory=list)
    refused: bool = False
    stages: dict[str, Any] = field(default_factory=dict)

    @property
    def cited_provisions(self) -> list[str]:
        """The provision refs the model actually cited, in order of first use."""
        idx = sorted({int(m) for m in re.findall(r"\[(\d+)\]", self.answer)})
        out: list[str] = []
        for i in idx:
            if 1 <= i <= len(self.hits):
                ref = self.hits[i - 1].chunk.meta.get("provision", "")
                if ref and ref not in out:
                    out.append(ref)
        return out

    @property
    def cited_repealed(self) -> list[str]:
        """Cited sources whose Act is repealed. Should always be empty.

        Non-empty means the generator quoted a repealed provision without
        flagging it, which is the failure this corpus exists to surface.
        """
        idx = sorted({int(m) for m in re.findall(r"\[(\d+)\]", self.answer)})
        return [
            self.hits[i - 1].chunk.meta.get("provision", f"source {i}")
            for i in idx
            if 1 <= i <= len(self.hits) and self.hits[i - 1].chunk.meta.get("in_force") == "False"
        ]


class RagPipeline:
    """Compose a retriever, an optional reranker, and a generator.

        pipe = RagPipeline(retriever, reranker=reranker, k=20, final_k=5)
        out  = pipe.answer("What is the punishment for murder under current law?")
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        reranker: Any | None = None,
        k: int = 12,
        final_k: int = 5,
        tier: str = "MAIN",
        max_context_chars: int = 8000,
        query_transform: Callable[[str], str] | None = None,
        system: str = ANSWER_SYSTEM,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.k = k
        self.final_k = final_k
        self.tier = tier
        self.max_context_chars = max_context_chars
        self.query_transform = query_transform
        self.system = system

    # -- stage 3/4 ---------------------------------------------------------
    def retrieve(self, question: str) -> list[Hit]:
        q = self.query_transform(question) if self.query_transform else question
        with tracing.trace("rag.retrieve", k=self.k, transformed=q != question) as s:
            hits = self.retriever.search(q, k=self.k)
            s["n_hits"] = len(hits)
            s["top_provision"] = hits[0].chunk.meta.get("provision") if hits else None
        return hits

    # -- stage 5 -----------------------------------------------------------
    def rerank(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        if not self.reranker:
            return list(hits)[: self.final_k]
        with tracing.trace("rag.rerank", n_in=len(hits), n_out=self.final_k):
            return self.reranker.rerank(question, hits, k=self.final_k)

    # -- stage 6/7 ---------------------------------------------------------
    def generate(self, question: str, hits: Sequence[Hit]) -> str:
        context = delimit_untrusted(
            format_context(hits, max_chars=self.max_context_chars)
        )
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer with citations:"
        with tracing.trace("rag.generate", n_sources=len(hits), tier=self.tier):
            return chat(prompt, system=self.system, tier=self.tier,
                        temperature=0.0, max_tokens=600).strip()

    def answer(self, question: str) -> RagAnswer:
        with tracing.trace("rag.answer", question=question[:120]):
            hits = self.retrieve(question)
            final = self.rerank(question, hits)
            text = self.generate(question, final)
            ok, invalid = enforce_citations(text, len(final))
            refused = text.strip().startswith(REFUSAL[:40])
            return RagAnswer(
                question=question,
                answer=text,
                hits=list(final),
                citations_valid=ok or refused,
                invalid_citations=invalid,
                refused=refused,
                stages={
                    "retrieved": [h.chunk.meta.get("provision", h.doc_id) for h in hits],
                    "final": [h.chunk.meta.get("provision", h.doc_id) for h in final],
                    "retrieve_k": self.k,
                    "final_k": self.final_k,
                },
            )


# --------------------------------------------------------------------------
# Query transforms
# --------------------------------------------------------------------------
def hyde(question: str, tier: str = "SMALL") -> str:
    """Hypothetical Document Embeddings.

    Ask the model to *write the provision that would answer the question*, then
    embed that instead of the question. It works because questions and statutory
    text live in different regions of embedding space: "when do I have to file
    a cheque-dishonour complaint?" shares almost no vocabulary with "Where any
    cheque drawn by a person... is returned by the banker for insufficiency of
    funds shall be punished..." A hypothetical provision does.

    Gao et al. (2022), arxiv.org/abs/2212.10496. Costs one extra small-model
    call per query — `src/retrieve.py` measures whether it earns it.
    """
    draft = chat(
        f"Write a short factual passage in the style of an Indian statute that "
        f"would answer this question. Invent plausible specifics; accuracy does "
        f"not matter, only phrasing.\n\nQuestion: {question}",
        tier=tier, max_tokens=180, temperature=0.0,
    )
    return f"{question}\n{draft}"


def multi_query(question: str, n: int = 3, tier: str = "SMALL") -> list[str]:
    """Generate paraphrases; retrieve for each; fuse with RRF.

    Buys recall when users phrase things unpredictably. Costs one small call
    plus n retrievals per query.
    """
    from aip.llm import chat as _chat
    from aip.llm import extract_json

    out = _chat(
        f"Rewrite this question in {n} different ways that a statute-search "
        f'engine would match differently. Reply as a JSON list of strings.\n\n{question}',
        tier=tier, max_tokens=250, temperature=0.3,
    )
    try:
        variants = [str(v) for v in extract_json(out)][:n]
    except ValueError:
        variants = []
    return [question, *variants]
