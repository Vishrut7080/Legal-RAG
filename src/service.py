"""HTTP service over the retrieval and answer pipeline.

Deliberately thin. Every judgement call -- what counts as a citation, when to
refuse, whether repealed text is reachable -- already lives in `src/answer.py`
and `src/retrieve.py`, and this module does not get a second opinion. That is
the reason to have a service layer at all: if the API could answer something the
CLI cannot, the gate would be measuring a different system than the one people
use.

Endpoints:

    GET  /health              is the index loaded, and which one
    POST /ask                 question -> answer with citations and status
    POST /retrieve            question -> ranked provisions, no generation
    GET  /provision/{ref}     one provision, by label

Two properties are load-bearing rather than decorative. `current_only` defaults
to true on every route, so calling the API without reading the docs still gets
in-force law. And a refusal is a 200 with `refused: true` plus a reason, not a
4xx: "I will not answer that" is a successful, useful response, and turning it
into an error would force every client to parse prose to find out.

Run it with `make service` (localhost only). It has no authentication -- do not
expose it. See the warning in `main()`.
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieve import Index, search

RETRIEVERS = ("dense", "bm25", "hybrid", "hybrid+current", "hybrid+rerank")

app = FastAPI(
    title="Legal-RAG",
    version="1.0",
    description=(
        "Retrieval over Indian statutes. Returns in-force law by default; "
        "repealed provisions (IPC, CrPC, Evidence Act) are reachable only via "
        "an explicit current_only=false."
    ),
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=8, ge=1, le=50)
    retriever: str = "hybrid+current"
    current_only: bool = True
    llm: bool = False


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=8, ge=1, le=50)
    retriever: str = "hybrid+current"
    current_only: bool = True


class HitModel(BaseModel):
    rank: int
    provision: str
    heading: str = ""
    in_force: bool
    score: float
    superseded_by: str = ""
    replaces: str = ""
    citation_marker: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_reason: str = ""
    citations_valid: bool = True
    cited_provisions: list[str] = Field(default_factory=list)
    hits: list[HitModel] = Field(default_factory=list)
    retriever: str
    current_only: bool
    stages: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _require_known_retriever(name: str) -> str:
    if name not in RETRIEVERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown retriever {name!r}; choose one of {list(RETRIEVERS)}",
        )
    return name


def _hit_models(hits: list[Any], cited: set[str]) -> list[HitModel]:
    """Serialise hits, tagging each with the marker that cites it.

    The marker matters more than it looks. A client rendering `[1]` as a
    footnote needs to know *which provision that is*, and an API that returns
    scores and a bare citation list leaves every caller to re-derive the link
    between them.
    """
    out: list[HitModel] = []
    for position, hit in enumerate(hits, start=1):
        meta = hit.chunk.meta
        ref = meta.get("provision", "")
        marker = f"[{position}]" if ref in cited and ref else None
        out.append(HitModel(
            rank=position,
            provision=ref,
            heading=meta.get("heading", ""),
            in_force=meta.get("in_force") != "False",
            score=round(float(hit.score), 4),
            superseded_by=meta.get("superseded_by", ""),
            replaces=meta.get("replaces", ""),
            citation_marker=marker,
        ))
    return out


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    try:
        index = Index.load()
    except Exception as error:  # noqa: BLE001 - report, do not crash the probe
        return {"status": "degraded", "index_loaded": False,
                "error": f"{type(error).__name__}: {error}"}
    meta = index.meta
    return {
        "status": "ok",
        "index_loaded": True,
        "n_chunks": meta.get("n_chunks"),
        "n_docs": meta.get("n_docs"),
        "n_repealed": meta.get("n_repealed"),
        "embedding_model": meta.get("embedding_model"),
        "dim": meta.get("dim"),
        "chunks_sha256": meta.get("chunks_sha256"),
        "answer_quality": "not measured - see README",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    from src.answer import answer as run_answer

    _require_known_retriever(request.retriever)
    result = run_answer(
        request.question,
        k=request.k,
        retriever=request.retriever,
        current_only=request.current_only,
        llm=request.llm,
    )
    cited = set(result.cited_provisions)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        refused=result.refused,
        refusal_reason=str(result.stages.get("refusal_reason", "")),
        citations_valid=result.citations_valid,
        cited_provisions=list(result.cited_provisions),
        hits=_hit_models(result.hits, cited),
        retriever=request.retriever,
        current_only=request.current_only,
        stages=dict(result.stages),
    )


@app.post("/retrieve")
def retrieve(request: RetrieveRequest) -> dict[str, Any]:
    _require_known_retriever(request.retriever)
    hits = search(
        request.question,
        k=request.k,
        which=request.retriever,
        current_only=request.current_only,
    )
    return {
        "question": request.question,
        "retriever": request.retriever,
        "current_only": request.current_only,
        "hits": [m.model_dump() for m in _hit_models(hits, set())],
    }


@app.get("/provision/{ref}")
def provision(ref: str) -> dict[str, Any]:
    """One provision by label, e.g. `BNS s.103`.

    Resolves from the committed index rather than the corpus files so the answer
    matches what retrieval would have returned, including the status block and
    the supersession link.
    """
    pattern = re.compile(r"^(?P<act>.+?)\s+s\.\s*(?P<number>[0-9A-Za-z]+)$")
    match = pattern.match(ref.strip())
    if not match:
        raise HTTPException(
            status_code=400,
            detail=f"cannot parse {ref!r}; expected a label like 'BNS s.103'",
        )
    want = f"{match.group('act').strip()} s.{match.group('number').strip()}"

    index = Index.load()
    matches = [c for c in index.chunks if c.meta.get("provision") == want]
    if not matches:
        raise HTTPException(status_code=404, detail=f"no provision labelled {want!r}")
    first = matches[0]
    return {
        "provision": want,
        "heading": first.meta.get("heading", ""),
        "in_force": first.meta.get("in_force") != "False",
        "role": first.meta.get("role", ""),
        "superseded_by": first.meta.get("superseded_by", ""),
        "replaces": first.meta.get("replaces", ""),
        "relation": first.meta.get("relation", ""),
        "n_parts": len(matches),
        "text": "\n\n".join(c.text for c in matches),
        "source": first.meta.get("url", ""),
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1",
                    help="default is loopback; this service has no auth")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args(argv)

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"WARNING: binding to {args.host} exposes an unauthenticated legal "
            "retrieval service to the network. Put it behind an authenticating "
            "proxy first.",
            file=sys.stderr,
        )
    uvicorn.run("src.service:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())