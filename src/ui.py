"""Browser UI over the same pipeline the gate measures.

It calls `src.answer` and `src.diagnose` in-process rather than going through
`src/service.py`, so the UI works without a server running. The two paths must
agree, and importing the same functions is what makes that true -- a UI that
reimplemented retrieval would drift silently from the CLI.

The one thing this adds on top of the answer is the **why** panel. When an
answer is wrong, the useful question is which stage got it wrong, and
`src/diagnose.py` answers that. A UI that only renders prose hides the most
interesting thing the system knows.

A **Dashboard** tab shows live metrics from the service `/metrics` endpoint,
trace summaries from `.aip_traces/`, and cache statistics.

Run with `make ui`, then open http://localhost:8501.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any

import pandas as pd
import requests
import streamlit as st

from aip.config import settings as aip_settings
from src.answer import answer as run_answer
from src.diagnose import diagnose, render
from src.retrieve import Index
from src.telemetry import (
    CostDelta,
    CostScope,
    Observation,
    clear_ui_telemetry,
    observe_ui,
    ui_snapshot,
)

EXAMPLES = [
    "What is the punishment for murder under the law currently in force?",
    "What does cheating by personation mean under BNS?",
    "How long does a warrant of arrest remain in force under BNSS?",
    "What did the Indian Penal Code say about adultery?",
    "What is the penalty for late filing of a patent application?",
    "Explain the doctrine of double jeopardy.",
]


def _index_summary() -> dict:
    try:
        index = Index.load()
    except Exception as error:  # noqa: BLE001 - the UI should say what is wrong
        return {"loaded": False, "error": f"{type(error).__name__}: {error}"}
    meta = index.meta
    return {
        "loaded": True,
        "n_chunks": meta.get("n_chunks"),
        "n_docs": meta.get("n_docs"),
        "n_repealed": meta.get("n_repealed"),
        "model": meta.get("embedding_model", ""),
        "chunks_sha256": str(meta.get("chunks_sha256", ""))[:12],
    }


def _sidebar(st) -> dict:
    st.sidebar.title("Settings")
    _default_mode = os.getenv("AIP_PROFILE", "local")
    _default_mode = "extractive" if _default_mode == "local" else "llm"
    mode = st.sidebar.selectbox(
        "answer mode",
        ["extractive", "llm", "agent"],
        index=["extractive", "llm", "agent"].index(
            _default_mode if _default_mode in ("extractive", "llm", "agent")
            else "extractive"
        ),
        help=(
            "extractive = deterministic, no chat model. "
            "llm = generate with the configured chat model (needs API/Ollama). "
            "agent = LLM drives retrieval with tools."
        ),
    )
    retriever = st.sidebar.selectbox(
        "retriever",
        ["hybrid+current", "hybrid", "dense", "bm25", "hybrid+rerank"],
        help="hybrid+current filters repealed provisions out. The others do not.",
    )
    k = st.sidebar.slider("provisions to retrieve", 1, 20, 8)
    current_only = st.sidebar.checkbox(
        "current law only",
        value=True,
        help=(
            "On: in-force law only, so a 2024 question cannot be answered from "
            "1860 text. Off: repealed provisions become reachable, which is "
            "what a question about the old law needs."
        ),
    )
    show_diagnosis = st.sidebar.checkbox("show the why panel", value=True)
    return {
        "retriever": retriever,
        "k": k,
        "current_only": current_only,
        "mode": mode,
        "show_diagnosis": show_diagnosis,
    }


def _render_hits(st, hits, cited: set[str]) -> None:
    st.subheader("Provisions retrieved")
    rows = []
    for position, hit in enumerate(hits, start=1):
        meta = hit.chunk.meta
        ref = meta.get("provision", "")
        rows.append({
            "marker": f"[{position}]" if ref in cited else "",
            "provision": ref,
            "heading": meta.get("heading", "")[:70],
            "status": "in force" if meta.get("in_force") != "False" else "REPEALED",
            "score": round(float(hit.score), 4),
            "counterpart": meta.get("superseded_by") or meta.get("replaces") or "",
        })
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    repealed = [r for r in rows if r["status"] == "REPEALED"]
    if repealed and not _is_historical(st.session_state.get("question", "")):
        st.warning(
            f"{len(repealed)} repealed provision(s) were retrieved. With "
            "'current law only' off they can be cited -- make sure the answer "
            "says they are repealed.",
            icon="⚠️",
        )


def _is_historical(question: str) -> bool:
    from src.answer import _HISTORICAL

    return bool(_HISTORICAL.search(question))


SERVICE_URL = "http://127.0.0.1:8000"
TRACES_DIR = aip_settings.trace_dir


def _fetch_service_metrics() -> dict[str, Any] | None:
    try:
        resp = requests.get(f"{SERVICE_URL}/metrics", timeout=3)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return None


def _service_is_healthy() -> bool:
    """Return whether the API can accept UI questions right now."""
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=1.5)
        return response.ok and response.json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


def _service_result(payload: dict[str, Any]) -> SimpleNamespace:
    """Adapt the API schema to the renderer's RagAnswer-like interface."""
    hits = []
    for item in payload.get("hits", []):
        meta = {
            "provision": item.get("provision", ""),
            "heading": item.get("heading", ""),
            "in_force": "True" if item.get("in_force", False) else "False",
            "superseded_by": item.get("superseded_by", ""),
            "replaces": item.get("replaces", ""),
        }
        hits.append(SimpleNamespace(
            chunk=SimpleNamespace(meta=meta),
            score=item.get("score", 0.0),
        ))
    return SimpleNamespace(
        question=payload.get("question", ""),
        answer=payload.get("answer", ""),
        refused=payload.get("refused", False),
        citations_valid=payload.get("citations_valid", True),
        invalid_citations=[],
        cited_provisions=payload.get("cited_provisions", []),
        hits=hits,
        stages=payload.get("stages", {}),
        n_searches=payload.get("n_searches", 0),
    )


def _service_cost(payload: dict[str, Any]) -> CostDelta:
    cost = payload.get("cost", {}) or {}
    return CostDelta(
        calls=int(cost.get("calls", 0)),
        cached_calls=int(cost.get("cached_calls", 0)),
        prompt_tokens=int(cost.get("prompt_tokens", 0)),
        completion_tokens=int(cost.get("completion_tokens", 0)),
        cost_usd=float(cost.get("cost_usd", 0.0)),
        unpriced_calls=int(cost.get("unpriced_calls", 0)),
        priced=bool(cost.get("priced", True)),
        models=list(cost.get("models", [])),
    )


class ServiceStreamError(RuntimeError):
    """The streaming API terminated without a usable final answer."""


def _read_sse_response(response, on_token=None) -> dict[str, Any]:
    event_name = ""
    data_lines: list[str] = []
    final_payload: dict[str, Any] | None = None

    def dispatch() -> None:
        nonlocal event_name, data_lines, final_payload
        if not event_name or not data_lines:
            event_name, data_lines = "", []
            return
        payload = json.loads("\n".join(data_lines))
        if event_name == "token" and on_token is not None:
            on_token(str(payload.get("text", "")),
                     bool(payload.get("unvalidated", True)))
        elif event_name == "final":
            final_payload = payload
        elif event_name == "error":
            raise ServiceStreamError(payload.get("detail", "stream failed"))
        event_name, data_lines = "", []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line:
            dispatch()
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    dispatch()
    if final_payload is None:
        raise ServiceStreamError("stream ended without a final answer event")
    return final_payload


def _run_question(
    question: str,
    settings: dict[str, Any],
    on_token=None,
):
    """Use the API when healthy; otherwise retain the standalone UI path."""
    started = time.perf_counter()
    if _service_is_healthy():
        request_body = {
            "question": question,
            "k": settings["k"],
            "retriever": settings["retriever"],
            "current_only": settings["current_only"],
            "mode": settings["mode"],
            "use_cache": True,
        }
        if settings["mode"] == "llm":
            with requests.post(
                f"{SERVICE_URL}/ask/stream",
                json=request_body,
                stream=True,
                timeout=(5, 180),
            ) as response:
                response.raise_for_status()
                payload = _read_sse_response(response, on_token=on_token)
        else:
            response = requests.post(
                f"{SERVICE_URL}/ask",
                json=request_body,
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return (
            _service_result(payload),
            _service_cost(payload),
            "service API",
            elapsed_ms,
            (payload.get("cache") or {}).get("layer", "none"),
        )

    with CostScope() as cost_scope:
        result = run_answer(
            question,
            k=settings["k"],
            retriever=settings["retriever"],
            current_only=settings["current_only"],
            mode=settings["mode"],
            on_token=(lambda text: on_token(text, True))
            if settings["mode"] == "llm" and on_token is not None
            else None,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return result, cost_scope.delta, "in-process UI", elapsed_ms, "none"


def _load_recent_traces(limit: int = 50) -> list[dict[str, Any]]:
    if not TRACES_DIR.exists():
        return []
    traces: list[dict[str, Any]] = []
    for path in sorted(TRACES_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        if path.name == "ui-telemetry.jsonl":
            continue
        if path.suffix == ".json":
            with contextlib.suppress(Exception):
                traces.append(json.loads(path.read_text()))
        elif path.suffix == ".jsonl":
            with contextlib.suppress(Exception):
                for line in path.read_text().splitlines():
                    if line.strip():
                        traces.append(json.loads(line))
        if len(traces) >= limit:
            break
    return traces[:limit]


def _render_dashboard(st) -> None:
    st.header("Dashboard")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("UI request history is persisted locally; service metrics are shown separately when available.")
    with col2:
        if st.button("Refresh"):
            st.rerun()

    metrics = _fetch_service_metrics()
    if metrics:
        _render_service_metrics(
            st, metrics, heading="Service API metrics (current process)"
        )
        st.divider()

    _render_inprocess_metrics(st)
    st.caption("Stored in `.aip_traces/ui-telemetry.jsonl` and restored after reload.")
    if st.button("Clear UI telemetry"):
        try:
            clear_ui_telemetry()
            st.success("Cleared stored UI observations.")
            st.rerun()
        except Exception as error:
            st.error(f"Failed to clear telemetry: {error}")

    st.divider()
    _render_traces(st)
    st.divider()
    _render_cache(st)


def _render_service_metrics(
    st, m: dict[str, Any], *, heading: str = "Metrics"
) -> None:
    st.subheader(heading)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requests", m.get("requests", 0))
    c2.metric("Uptime", f"{m.get('uptime_s', 0):.1f}s")
    c3.metric("Refusal rate", f"{m.get('refusal_rate', 0):.1%}")
    c4.metric("Tool calls/req", m.get("tool_calls_per_request", 0))

    lat = m.get("latency_ms", {})
    if lat:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latency p50", f"{lat.get('p50', 0):.1f} ms")
        c2.metric("Latency p95", f"{lat.get('p95', 0):.1f} ms")
        c3.metric("Latency p99", f"{lat.get('p99', 0):.1f} ms")
        c4.metric("Latency mean", f"{lat.get('mean', 0):.1f} ms")

    ttft = m.get("ttft_ms")
    if ttft:
        c1, c2 = st.columns(2)
        c1.metric("Time to first token p50", f"{ttft.get('p50', 0):.1f} ms")
        c2.metric("Time to first token p95", f"{ttft.get('p95', 0):.1f} ms")

    by_stage = m.get("by_stage_ms", {})
    if by_stage:
        st.markdown("**Latency by stage (ms)**")
        df = pd.DataFrame({
            stage: {"p50": v["p50"], "p95": v["p95"], "mean": v["mean"]}
            for stage, v in by_stage.items()
        }).T
        st.dataframe(df, use_container_width=True)

    cache_m = m.get("cache", {})
    if cache_m:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cache hits", cache_m.get("hits", 0))
        c2.metric("Hit rate", f"{cache_m.get('hit_rate', 0):.1%}")
        c3.metric("Semantic hits", cache_m.get("semantic", 0))
        sim = cache_m.get("similarity")
        if sim:
            st.caption(f"Similarity: min={sim.get('min'):.4f}, mean={sim.get('mean'):.4f}")

    cost = m.get("cost", {})
    if cost:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total cost", f"${cost.get('total_usd', 0):.6f}")
        c2.metric("Mean cost/req", f"${cost.get('mean_usd', 0):.8f}")
        c3.metric("Tokens", cost.get("tokens", 0))
        if not cost.get("priced", True):
            st.warning(f"Unpriced calls: {cost.get('unpriced_calls', 0)}")

    by_route = m.get("by_route", {})
    if by_route:
        st.markdown("**Requests by route**")
        st.bar_chart(by_route)


def _render_inprocess_metrics(st) -> None:
    st.subheader("UI requests (persisted on this machine)")
    m = ui_snapshot()
    if m.get("requests", 0) == 0:
        st.info("No requests recorded in this process yet.")
        return
    _render_service_metrics(st, m, heading="UI request metrics")


def _render_traces(st) -> None:
    st.subheader("Recent Traces")
    raw = _load_recent_traces(1000)
    if not raw:
        st.info("No traces found in .aip_traces/. Run an answer through the service/UI (and a chat model) to generate them.")
        return

    runs: dict[str, list[dict[str, Any]]] = {}
    for rec in raw:
        runs.setdefault(rec.get("run_id", "?"), []).append(rec)

    # Order runs by the timestamp of their first span, newest first.
    run_rows = []
    for run_id, spans in runs.items():
        spans_sorted = sorted(spans, key=lambda r: r.get("ts", 0))
        total_ms = sum(s.get("duration_ms", 0) or 0 for s in spans_sorted)
        started = spans_sorted[0].get("ts")
        names = ", ".join(sorted({s.get("name", "") for s in spans_sorted if s.get("name")}))
        # Pick a question from any span metadata we might have.
        question = next(
            (s.get("question") or s.get("hint") for s in spans_sorted
             if s.get("question") or s.get("hint")),
            "",
        )
        run_rows.append({
            "run_id": run_id,
            "started": str(started or ""),
            "spans": len(spans_sorted),
            "duration_ms": round(total_ms, 1),
            "names": names,
            "question": (str(question) if question else "")[:70],
        })
    run_rows.sort(key=lambda r: r["run_id"], reverse=True)

    df = pd.DataFrame(run_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    selected = st.selectbox("Inspect run", [""] + [r["run_id"] for r in run_rows])
    if selected:
        spans = sorted(runs[selected], key=lambda r: r.get("ts", 0))
        with st.expander("Run trace (spans)", expanded=True):
            for s in spans:
                st.json(s)


def _render_cache(st) -> None:
    service_metrics = _fetch_service_metrics()
    if service_metrics:
        st.subheader("Answer Cache (service on :8000)")
        sc = service_metrics.get("cache_stats", {}) or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Exact hits", sc.get("exact_hits", 0))
        c2.metric("Semantic hits", sc.get("semantic_hits", 0))
        c3.metric("Misses", sc.get("misses", 0))
        c4, c5, c6 = st.columns(3)
        c4.metric("Stores", sc.get("stores", 0))
        c5.metric("Lookups", sc.get("lookups", 0))
        c6.metric("Hit rate", f"{sc.get('hit_rate', 0):.1%}")
        if st.button("Clear cache"):
            try:
                import requests
                r = requests.post(f"{SERVICE_URL}/cache/clear", timeout=5)
                r.raise_for_status()
                data = r.json()
                st.success(f"Cleared {data.get('removed', 0)} entries.")
                st.rerun()
            except Exception as error:
                st.error(f"Failed to clear service cache: {error}")
        st.caption("UI submissions use this cache while the service is healthy; the standalone fallback bypasses it.")
    else:
        st.subheader("Service answer cache")
        st.info(
            "The UI answers in-process and bypasses the persistent answer cache. "
            "Start `make service` and send requests to its API to populate these metrics."
        )


def main() -> int:

    st.set_page_config(page_title="Legal-RAG", page_icon="§", layout="wide")

    tab_ask, tab_dash = st.tabs(["Ask", "Dashboard"])

    with tab_ask:
        st.title("Legal-RAG")
        st.caption(
            "Indian statutes, in force by default. Answer quality is not measured; "
            "retrieval is. See the README."
        )

        settings = _sidebar(st)

        summary = _index_summary()
        if not summary["loaded"]:
            st.error(f"index not available: {summary['error']}")
            st.stop()
        st.sidebar.caption(
            f"{summary['n_chunks']} chunks · {summary['n_docs']} Acts · "
            f"{summary['n_repealed']} repealed · {summary['chunks_sha256']}"
        )

        st.markdown("#### Example questions")
        for example in EXAMPLES:
            if st.button(example, key=f"ex-{example}"):
                st.session_state["question"] = example
                st.session_state.pop("last_result", None)
                st.session_state.pop("last_diagnosis", None)
        st.markdown("")

        with st.form("ask_form"):
            question = st.text_input(
                "Ask a question",
                key="question",
                placeholder="What is the punishment for murder under the law currently in force?",
            )
            submitted = st.form_submit_button("Get answer")

        if submitted:
            if not question.strip():
                st.warning("Type a question first.")
            else:
                draft = st.empty()
                draft_parts: list[str] = []
                request_started = time.perf_counter()
                first_token_ms: list[float | None] = [None]

                def show_token(text: str, unvalidated: bool = True) -> None:
                    if first_token_ms[0] is None:
                        first_token_ms[0] = round(
                            (time.perf_counter() - request_started) * 1000, 2
                        )
                    draft_parts.append(text)
                    status = (
                        "Draft — citations are being checked"
                        if unvalidated else "Cached answer"
                    )
                    draft.markdown(f"_{status}_\n\n" + "".join(draft_parts))

                try:
                    if settings["mode"] == "llm":
                        waiting_message = (
                            "Waiting for the chat model; the draft will stream "
                            "while its citations are validated…"
                        )
                    elif settings["mode"] == "agent":
                        waiting_message = "Agent is retrieving provisions and preparing an answer…"
                    else:
                        waiting_message = "Retrieving provisions and preparing an answer…"
                    with st.spinner(waiting_message):
                        result, cost_delta, backend, elapsed_ms, cache_layer = _run_question(
                            question,
                            settings,
                            on_token=show_token if settings["mode"] == "llm" else None,
                        )
                except (requests.RequestException, ServiceStreamError) as error:
                    st.error(f"Answer request failed: {error}")
                    st.session_state.pop("last_result", None)
                    st.session_state.pop("last_diagnosis", None)
                    result = None
                draft.empty()
                if result is None:
                    pass
                else:
                    st.session_state["last_result"] = result
                    st.session_state["last_question"] = question
                    st.session_state["last_mode"] = settings["mode"]
                    st.session_state["last_backend"] = backend
                    st.session_state.pop("last_diagnosis", None)
                    observe_ui(Observation(
                        trace_id=str(result.stages.get("trace_id", "") or ""),
                        route="/ui",
                        latency_ms=float(elapsed_ms),
                        latency_by_stage=dict(result.stages.get("latency_by_stage", {}) or {}),
                        cost=cost_delta,
                        refused=bool(result.refused),
                        ttft_ms=first_token_ms[0],
                        tool_calls=int(getattr(result, "n_searches", 0) or 0),
                        cache_layer=cache_layer,
                    ))

        result = st.session_state.get("last_result")
        if result is None:
            st.info("Pick an example or type a question, then select **Get answer**.")
        else:
            question = st.session_state.get("last_question", question)
            st.caption(
                f"Last submitted mode: {st.session_state.get('last_mode', 'extractive')} "
                f"· via {st.session_state.get('last_backend', 'in-process UI')}"
            )
            cited = set(result.cited_provisions)

            if not result.citations_valid and not result.refused:
                st.error("Answer withheld because its citations did not pass validation.")
            elif result.refused:
                st.warning("Refused.", icon="🚫")
                reason = result.stages.get("refusal_reason")
                if reason:
                    st.caption(str(reason))
                st.text(result.answer)
            else:
                st.markdown(result.answer)

            if not result.citations_valid:
                st.error(f"invalid citation markers: {result.invalid_citations}")

            if result.hits:
                _render_hits(st, result.hits, cited)

            if settings["show_diagnosis"]:
                diagnosis = st.session_state.get("last_diagnosis")
                if diagnosis is None:
                    diagnosis = diagnose(
                        question,
                        k=settings["k"],
                        retriever=settings["retriever"],
                        current_only=settings["current_only"],
                    )
                    st.session_state["last_diagnosis"] = diagnosis
                with st.expander("Why this answer", expanded=result.refused):
                    st.code(render(diagnosis), language="text")
                    st.caption(f"likely cause: {diagnosis.likely_cause}")

            with st.expander("raw stages"):
                st.json(result.stages)

    with tab_dash:
        _render_dashboard(st)

    return 0


if __name__ == "__main__":
    sys.exit(main())
