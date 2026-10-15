"""HTTP service over the retrieval and answer pipeline.

Deliberately thin in judgement, deliberate in instrumentation. Every judgement
call -- what counts as a citation, when to refuse, whether repealed text is
reachable -- already lives in `src/answer.py` and `src/retrieve.py`, and this
module does not get a second opinion. That is the reason to have a service layer
at all: if the API could answer something the CLI cannot, the gate would be
measuring a different system than the one people use.

What it *does* own is everything about being a service rather than a script:
caching, streaming, latency and cost accounting, and a metrics endpoint.

Endpoints
---------
    GET  /health              is the index loaded, and which one
    POST /ask                 question -> answer, citations, latency, cost, trace id
    POST /ask/stream          the same answer as server-sent events
    POST /retrieve            question -> ranked provisions, no generation
    GET  /provision/{ref}     one provision, by label
    GET  /metrics             p50/p95/p99, per-stage latency, cost, cache hit rate

Four properties are load-bearing rather than decorative:

1. `current_only` defaults to true on every route, so calling the API without
   reading the docs still gets in-force law.
2. A refusal is a 200 with `refused: true` plus a reason, not a 4xx. "I will
   not answer that" is a successful, useful response; turning it into an error
   would force every client to parse prose to find out.
3. Latency, cost and the trace id travel *in the response body*. Headers are for
   caches and proxies; a client debugging one slow answer needs the numbers next
   to the answer.
4. The index is built once at startup. Building it per request is the single
   most common way to make a RAG service slow, and it is invisible until you
   measure -- which is what /metrics is for.

Streaming, and the problem it creates
--------------------------------------
You cannot validate citations until the answer is complete, and streaming means
you have already sent it. The honest answer depends on the generator:

* **Extractive (default).** The cited provision is chosen by *retrieval*, not by
  a model, and the generator is deterministic. So the citations are known and
  valid before the first token is emitted: the stream sends `citations` first,
  then tokens. There is no window in which an unvalidated citation is on the
  wire.

* **LLM (`"llm": true`).** The citation is written by the model and can be wrong,
  so it cannot be validated before the tokens go out. Every `token` event is
  therefore marked `unvalidated: true`, and if `enforce_citations` fails at the
  end the stream closes with `final {retracted: true}`. The client is told to
  discard what it rendered. This is the correct behaviour, not a workaround:
  you cannot un-send bytes, so the only honest moves are to label them or to
  retract them after the fact.

Run with `make service` (localhost only). It has no authentication -- do not
expose it. See the warning in `main()`.
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src import telemetry
from src.cache import AnswerCache, Config, get_cache
from src.retrieve import Index, search

RETRIEVERS = ("dense", "bm25", "hybrid", "hybrid+current",
              "hybrid+rerank", "hybrid+current+rerank")

# How the answer text is split into stream tokens. Sentence-ish rather than
# character-wise: a client rendering tokens into a paragraph wants text, and
# per-character events would triple the event count for no benefit.
_TOKEN_SPLIT = re.compile(r"(\s+)")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the index, retrievers, cache and embedding model once, at startup.

    `Index.load()` reads ~1000 chunks and a 1.5 MB matrix and builds a BM25
    index over all of it. Doing that inside the first request means the first
    user pays for it, and doing it on every request means every user pays. Both
    show up only in the p95, which is why it is here rather than inline.

    The embedding model is warmed here too, and that one is not optional: it
    costs ~18 seconds to load, and leaving it lazy meant the first `/ask` after
    a deploy returned after 18 seconds -- against a 1,500 ms time-to-first-token
    target. Nothing in the request path warms it, so the benchmark measured
    exactly that and it showed up as the p99. A service that is fast except for
    the one request that matters is not fast.
    """
    started = time.perf_counter()
    try:
        index = Index.load()
        app.state.index = index
        app.state.index_error = None
    except Exception as error:  # noqa: BLE001 - report, do not crash the boot
        app.state.index = None
        app.state.index_error = f"{type(error).__name__}: {error}"
    app.state.index_ms = round((time.perf_counter() - started) * 1000, 2)

    model_started = time.perf_counter()
    warm_error: str | None = None
    try:
        if app.state.index is not None:
            # Touch the retriever's embedding path once, off the request path.
            from src.cache import AnswerCache as _AC
            from src.retrieve import DenseIndexRetriever  # noqa: F401
            _AC.embed_question("warmup probe",
                               model=app.state.index.model)
    except Exception as error:  # noqa: BLE001 - a cold model is not a dead service
        warm_error = f"{type(error).__name__}: {error}"
    app.state.model_warm_ms = round((time.perf_counter() - model_started) * 1000, 2)
    app.state.model_warm_error = warm_error
    app.state.startup_ms = app.state.index_ms + app.state.model_warm_ms

    try:
        # Exact cache hits are valuable; semantic hits are not safe on this
        # measured corpus and used to make a Gemini embedding call per new
        # question even though the threshold was deliberately inert.
        app.state.cache = get_cache(enable_semantic=False)
    except Exception:  # noqa: BLE001 - a cache failure must not stop serving
        app.state.cache = AnswerCache(enable_semantic=False)
    yield


app = FastAPI(
    title="Legal-RAG",
    version="2.0",
    description=(
        "Retrieval over Indian statutes. Returns in-force law by default; "
        "repealed provisions (IPC, CrPC, Evidence Act) are reachable only via "
        "an explicit current_only=false. Includes latency, cost and cache "
        "metadata in every answer."
    ),
    lifespan=lifespan,
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
    mode: str | None = Field(default=None, pattern="^(extractive|llm|agent)$")
    use_cache: bool = True


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


class CacheModel(BaseModel):
    """What the cache did, and why -- including when it declined to fire.

    A cache that silently declines is impossible to tune, so the reason travels
    with the answer.
    """

    hit: bool
    layer: str = "none"
    similarity: float | None = None
    matched_question: str | None = None
    reason: str = ""
    # On a replay, the trace id of the request that actually computed the
    # answer. Without it a cache hit is untraceable back to the run that
    # produced it, which is the one thing a cache makes harder.
    served_from_trace_id: str | None = None


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
    mode: str = "extractive"
    n_searches: int = 0
    stopped_because: str = ""
    stages: dict[str, Any] = Field(default_factory=dict)
    # Observability, in the body on purpose.
    trace_id: str = ""
    latency_ms: float = 0.0
    latency_by_stage: dict[str, float] = Field(default_factory=dict)
    cost: dict[str, Any] = Field(default_factory=dict)
    cache: CacheModel = Field(default_factory=CacheModel)
    # True when this answer was replayed rather than recomputed. Callers that
    # measure end-to-end user-perceived time need this to discount the replay:
    # the latency reported is the original computation's, not the read.
    replayed: bool = False


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


def _cache_for(request: Request) -> AnswerCache:
    return getattr(request.app.state, "cache", None) or get_cache()


def _mode_of(body: AskRequest) -> str:
    """Resolve explicit mode, retaining `llm: true` for older API clients."""
    if body.mode is not None:
        return body.mode
    return "llm" if body.llm else "extractive"


def _config_of(body: AskRequest) -> Config:
    return Config(retriever=body.retriever, current_only=body.current_only,
                  k=body.k, mode=_mode_of(body))


def _build_response(body: AskRequest, result: Any, cost: telemetry.CostDelta,
                    cache_info: CacheModel) -> AskResponse:
    """Turn an answer result into the response model.

    One builder for both `/ask` and `/ask/stream`, because the streaming route
    also populates the cache and a cache entry that the other route cannot
    replay is worse than no entry: the first `/ask` after an `/ask/stream` would
    fail to validate. That bug shipped once and this function is the fix.
    """
    cited = set(result.cited_provisions)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        refused=result.refused,
        refusal_reason=str(result.stages.get("refusal_reason", "")),
        citations_valid=result.citations_valid,
        cited_provisions=list(result.cited_provisions),
        hits=_hit_models(result.hits, cited),
        retriever=body.retriever,
        current_only=body.current_only,
        mode=_mode_of(body),
        n_searches=int(getattr(result, "n_searches", 0)),
        stopped_because=str(getattr(result, "stopped_because", "")),
        stages=dict(result.stages),
        trace_id=str(result.stages.get("trace_id", "")),
        latency_ms=float(result.stages.get("latency_ms", 0.0)),
        latency_by_stage=dict(result.stages.get("latency_by_stage", {})),
        cost=cost.as_dict(),
        cache=cache_info,
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    error = getattr(request.app.state, "index_error", None)
    if error:
        return {"status": "degraded", "index_loaded": False, "error": error}
    try:
        index = Index.load()
    except Exception as error:  # noqa: BLE001 - report, do not crash the probe
        return {"status": "degraded", "index_loaded": False,
                "error": f"{type(error).__name__}: {error}"}
    meta = index.meta
    cache = _cache_for(request)
    return {
        "status": "ok",
        "index_loaded": True,
        "n_chunks": meta.get("n_chunks"),
        "n_docs": meta.get("n_docs"),
        "n_repealed": meta.get("n_repealed"),
        "embedding_model": meta.get("embedding_model"),
        "dim": meta.get("dim"),
        "chunks_sha256": meta.get("chunks_sha256"),
        "startup_ms": getattr(request.app.state, "startup_ms", None),
        "index_load_ms": getattr(request.app.state, "index_ms", None),
        "model_warm_ms": getattr(request.app.state, "model_warm_ms", None),
        "model_warm_error": getattr(request.app.state, "model_warm_error", None),
        "cache": {
            "size": cache.size(),
            "semantic_threshold": cache.semantic_threshold,
            "semantic_enabled": cache.enable_semantic,
        },
        "answer_quality": "not measured - see README",
    }


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    from src.answer import answer as run_answer

    _require_known_retriever(body.retriever)
    config = _config_of(body)
    mode = _mode_of(body)
    cache = _cache_for(request)
    cache_info = CacheModel(hit=False, reason="cache bypassed by this request")

    with telemetry.CostScope() as cost:
        if body.use_cache:
            lookup = cache.lookup(body.question, config)
            cache_info = CacheModel(
                hit=lookup.hit, layer=lookup.layer,
                similarity=lookup.similarity,
                matched_question=lookup.matched_question,
                reason=lookup.reason,
            )
            if lookup.hit and lookup.payload is not None:
                # A hit still has to carry live measurements, or /metrics drifts
                # from reality: a cached answer did not re-run retrieval, and
                # reporting it as if it did is how a latency dashboard starts
                # lying.
                #
                # The stored payload cannot be returned verbatim, because it
                # holds the metadata of the request that *created* it -- most
                # visibly `cache.hit: false`. Serving that would make the
                # service report a cache hit as a miss, in the response body and
                # in the metrics the response feeds. So the answer is replayed
                # and the observation about *this* request is overwritten onto it.
                cached = dict(lookup.payload)
                origin_trace = str(cached.get("trace_id", ""))
                replay_latency = float(cached.get("latency_ms", 0.0))
                cached["trace_id"] = telemetry.new_trace_id()
                cached["cache"] = cache_info.model_dump()
                cached["cache"]["served_from_trace_id"] = origin_trace
                cached["replayed"] = True

                telemetry.observe(telemetry.Observation(
                    trace_id=str(cached["trace_id"]), route="/ask",
                    latency_ms=replay_latency,
                    latency_by_stage=cached.get("latency_by_stage", {}),
                    cost=cost.delta, cache_layer=lookup.layer,
                    cache_similarity=lookup.similarity,
                    refused=bool(cached.get("refused")),
                ))
                return AskResponse(**cached)

        result = run_answer(
            body.question,
            k=body.k,
            retriever=body.retriever,
            current_only=body.current_only,
            llm=body.llm,
            mode=mode,
        )

    latency_ms = float(result.stages.get("latency_ms", 0.0))
    by_stage = dict(result.stages.get("latency_by_stage", {}))
    trace_id = str(result.stages.get("trace_id", ""))

    response = _build_response(body, result, cost.delta, cache_info)

    if body.use_cache:
        cache.store(body.question, config, response.model_dump())

    telemetry.observe(telemetry.Observation(
        trace_id=trace_id, route="/ask", latency_ms=latency_ms,
        latency_by_stage=by_stage, cost=cost.delta,
        cache_layer=cache_info.layer, cache_similarity=cache_info.similarity,
        refused=result.refused, tool_calls=int(getattr(result, "n_searches", 0)),
    ))
    return response


def _sse(event: str, data: dict[str, Any]) -> str:
    """One server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Stream the answer as server-sent events.

    Event order is the design, and it differs by generator:

      extractive: citations -> token* -> final
      llm:        started -> token*(unvalidated) -> citations(validated) -> final

    For the extractive path the citations are already valid before the first
    token goes out, because the cited provision is chosen by retrieval and the
    generator is deterministic. For the LLM path, tokens are provisional until
    generation completes and the answer contract validates its citations; a
    validation failure retracts the draft in the final event.
    """
    _require_known_retriever(body.retriever)
    mode = _mode_of(body)
    config = _config_of(body)
    cache = _cache_for(request)

    def generate() -> Any:
        from src.answer import answer as run_answer

        started = time.perf_counter()
        first_token_ms: float | None = None
        yield _sse("started", {"trace_id": telemetry.new_trace_id(),
                               "unvalidated": mode == "llm",
                               "mode": mode})

        lookup = None
        if body.use_cache:
            lookup = cache.lookup(body.question, config)
            if lookup.hit and lookup.payload is not None:
                cached = dict(lookup.payload)
                # Same reason as the non-streaming path: the stored payload
                # describes the request that created it, not this one.
                origin_trace = str(cached.get("trace_id", ""))
                replay_trace = telemetry.new_trace_id()
                cached["trace_id"] = replay_trace
                cached["replayed"] = True
                yield _sse("citations", {
                    "cited_provisions": cached.get("cited_provisions", []),
                    "hits": cached.get("hits", []),
                    "validated": True,
                })
                for token in _TOKEN_SPLIT.split(str(cached.get("answer", ""))):
                    if token:
                        if first_token_ms is None:
                            first_token_ms = round(
                                (time.perf_counter() - started) * 1000, 2
                            )
                        yield _sse("token", {"text": token, "unvalidated": False})
                yield _sse("final", {
                    **cached,
                    "cache": {"hit": True, "layer": lookup.layer,
                              "similarity": lookup.similarity,
                              "reason": lookup.reason,
                              "served_from_trace_id": origin_trace},
                    # Same final-event shape whether or not this was a replay.
                    # A client should not have to branch on the cache to know
                    # what it has to handle.
                    "retracted": bool(not cached.get("citations_valid", True)),
                    "cost": cached.get("cost", {}),
                    "ttft_ms": first_token_ms or round(
                        (time.perf_counter() - started) * 1000, 2
                    ),
                })
                telemetry.observe(telemetry.Observation(
                    trace_id=replay_trace, route="/ask/stream",
                    latency_ms=float(cached.get("latency_ms", 0.0)),
                    latency_by_stage=cached.get("latency_by_stage", {}),
                    cache_layer=lookup.layer,
                    cache_similarity=lookup.similarity,
                    refused=bool(cached.get("refused")),
                    ttft_ms=round((time.perf_counter() - started) * 1000, 2),
                ))
                return

        if mode == "llm":
            # `answer()` and LiteLLM are synchronous. Run the pipeline in a
            # worker while the response generator drains token pieces from a
            # queue, allowing FastAPI to flush each SSE event as it arrives.
            pieces: queue.Queue[tuple[str, Any] | None] = queue.Queue()
            completed: dict[str, Any] = {}

            def run_streaming_answer() -> None:
                try:
                    with telemetry.CostScope() as cost_scope:
                        completed["result"] = run_answer(
                            body.question,
                            k=body.k,
                            retriever=body.retriever,
                            current_only=body.current_only,
                            mode=mode,
                            on_token=lambda text: pieces.put(("token", text)),
                        )
                    completed["cost"] = cost_scope.delta
                except Exception as error:  # sent after any already-flushed tokens
                    completed["error"] = error
                finally:
                    pieces.put(None)

            worker = threading.Thread(target=run_streaming_answer, daemon=True)
            worker.start()
            while True:
                item = pieces.get()
                if item is None:
                    break
                kind, text = item
                if kind == "token":
                    if first_token_ms is None:
                        first_token_ms = round(
                            (time.perf_counter() - started) * 1000, 2
                        )
                    yield _sse("token", {"text": text, "unvalidated": True})
            worker.join()
            if "error" in completed:
                error = completed["error"]
                yield _sse("error", {
                    "status": 503,
                    "detail": f"generation failed: {type(error).__name__}: {error}",
                })
                return
            result = completed["result"]
            cost_delta = completed["cost"]
        else:
            with telemetry.CostScope() as cost_scope:
                result = run_answer(
                    body.question, k=body.k, retriever=body.retriever,
                    current_only=body.current_only, mode=mode,
                )
            cost_delta = cost_scope.delta

        cited = set(result.cited_provisions)
        hits = [m.model_dump() for m in _hit_models(result.hits, cited)]
        validated = bool(result.citations_valid)

        yield _sse("citations", {
            "cited_provisions": list(result.cited_provisions),
            "hits": hits,
            # The whole point. True means the citations are known good and the
            # tokens below are safe to render. False means the model chose these
            # numbers and they have not been checked yet.
            "validated": validated,
            "note": ("citations were checked against retrieved provisions" if validated
                     else "citation validation failed; discard provisional prose"),
        })

        # TTFT is the time to the *first token*, so it is captured here and not
        # after the loop. Measuring it at the end would report total generation
        # time under a name that promises something much smaller, and the whole
        # reason to stream is that this number is better than the total.
        ttft_ms = first_token_ms
        # LLM tokens have already been flushed provisionally from the worker.
        # Extractive/agent answers are complete before they reach this point.
        if (mode != "llm"
                or result.stages.get("generator") != "llm"):
            for token in _TOKEN_SPLIT.split(result.answer):
                if token:
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - started) * 1000, 2)
                    yield _sse("token", {"text": token, "unvalidated": False})
        else:
            # TTFT was measured in the UI from the first provisional delta; the
            # server also exposes end-to-end stream time below.
            ttft_ms = round((time.perf_counter() - started) * 1000, 2)
        if ttft_ms is None:
            ttft_ms = round((time.perf_counter() - started) * 1000, 2)

        retracted = bool(
            not result.citations_valid or result.stages.get("llm_error")
        )
        trace_id = str(result.stages.get("trace_id", ""))
        stream_cache = CacheModel(
            hit=False, layer="none",
            reason=lookup.reason if lookup else "cache bypassed by this request",
            similarity=lookup.similarity if lookup else None,
        )
        # Same builder as /ask, so the cache entry is a complete AskResponse
        # either route can replay. A partial entry here is not a harmless
        # optimisation: it is an entry that breaks the other route.
        response = _build_response(body, result, cost_delta, stream_cache)

        if body.use_cache:
            cache.store(body.question, config, response.model_dump())

        telemetry.observe(telemetry.Observation(
            trace_id=trace_id,
            route="/ask/stream",
            latency_ms=float(result.stages.get("latency_ms", 0.0)),
            latency_by_stage=dict(result.stages.get("latency_by_stage", {})),
            cost=cost_delta,
            cache_layer=(lookup.layer if lookup and lookup.hit else "none"),
            cache_similarity=(lookup.similarity if lookup and lookup.hit else None),
            refused=result.refused, ttft_ms=ttft_ms,
            tool_calls=int(getattr(result, "n_searches", 0)),
        ))

        payload = response.model_dump()
        yield _sse("final", {
            **payload,
            "retracted": retracted,
            "ttft_ms": ttft_ms,
            "cache": stream_cache.model_dump(),
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/retrieve")
def retrieve(body: RetrieveRequest) -> dict[str, Any]:
    _require_known_retriever(body.retriever)
    hits = search(body.question, k=body.k, which=body.retriever,
                  current_only=body.current_only)
    return {
        "question": body.question,
        "retriever": body.retriever,
        "current_only": body.current_only,
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


@app.get("/metrics")
def metrics(request: Request) -> dict[str, Any]:
    """Aggregates for the cost and latency dashboard.

    Per-process and in-memory: it is a live view of this service instance, not a
    time series. That is a real limitation and it is the reason the numbers are
    also written into the response body of every request, where a caller can keep
    them after the process that produced them is gone.
    """
    cache = _cache_for(request)
    snapshot = telemetry.snapshot()
    snapshot["cache_stats"] = cache.stats.as_dict()
    snapshot["semantic_threshold"] = cache.semantic_threshold
    snapshot["semantic_enabled"] = cache.enable_semantic
    snapshot["scope"] = "process-local, since process start"
    return snapshot


@app.post("/cache/clear")
def cache_clear(request: Request) -> dict[str, Any]:
    """Empty the answer cache. Destructive and unauthenticated, hence a POST."""
    cache = _cache_for(request)
    return {"removed": cache.clear()}


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
