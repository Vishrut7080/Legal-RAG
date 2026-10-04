"""Browser UI over the same pipeline the gate measures.

It calls `src.answer` and `src.diagnose` in-process rather than going through
`src/service.py`, so the UI works without a server running. The two paths must
agree, and importing the same functions is what makes that true -- a UI that
reimplemented retrieval would drift silently from the CLI.

The one thing this adds on top of the answer is the **why** panel. When an
answer is wrong, the useful question is which stage got it wrong, and
`src/diagnose.py` answers that. A UI that only renders prose hides the most
interesting thing the system knows.

Run with `make ui`, then open http://localhost:8501.
"""
from __future__ import annotations

import sys

from src.answer import answer as run_answer
from src.diagnose import diagnose, render
from src.retrieve import Index

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
    llm = st.sidebar.checkbox(
        "generate with a chat model",
        value=False,
        help="Needs Ollama running. Off means deterministic extractive answers.",
    )
    show_diagnosis = st.sidebar.checkbox("show the why panel", value=True)
    return {
        "retriever": retriever,
        "k": k,
        "current_only": current_only,
        "llm": llm,
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


def main() -> int:
    import streamlit as st

    st.set_page_config(page_title="Legal-RAG", page_icon="§", layout="wide")
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
    st.markdown("")

    question = st.text_input(
        "Ask a question",
        key="question",
        placeholder="What is the punishment for murder under the law currently in force?",
    )
    if not question.strip():
        st.info("Pick an example or type a question.")
        return 0

    result = run_answer(
        question,
        k=settings["k"],
        retriever=settings["retriever"],
        current_only=settings["current_only"],
        llm=settings["llm"],
    )
    cited = set(result.cited_provisions)

    if result.refused:
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
        with st.expander("Why this answer", expanded=result.refused):
            diagnosis = diagnose(
                question,
                k=settings["k"],
                retriever=settings["retriever"],
                current_only=settings["current_only"],
            )
            st.code(render(diagnosis), language="text")
            st.caption(f"likely cause: {diagnosis.likely_cause}")

    with st.expander("raw stages"):
        st.json(result.stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())