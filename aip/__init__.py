"""
aip — the shared toolkit for Legal-RAG.

A small, readable library for retrieval-augmented generation over statutes and
court procedure. Everything here is meant to be opened: no module is over ~250
lines and there is no framework hiding the interesting parts.

    from aip import chat, structured, embed, Budget

Five rules the package follows, which the code in `src/` follows too:

1. Provider-agnostic.   Model names are LiteLLM strings. Swapping
                        `anthropic/claude-haiku-4-5-20251001` for
                        `ollama/llama3.1:8b` requires no other change.
2. Everything is cached. Identical requests never cost money twice.
3. Everything is counted. Tokens, cost, and latency are recorded on every call.
4. Everything is traceable. Every call appends a JSONL trace record.
5. Offline-capable.     With AIP_OFFLINE=1, cached responses replay and
                        uncached requests raise loudly rather than spending.

Rule 5 is what makes the regression gate in `src/gate.py` reproducible: the
committed snapshot of the corpus plus a warm response cache is enough to
re-measure every metric with no network and no API key.

The toolkit is the AI-in-Practice Module 1 reference library, extended with
the legal domain's needs (a `local` model profile, a `RERANK` tier, statutory
PII patterns, and `src/agent.py`'s tool-calling retrieval). The core modules
-- `llm`, `tracing`, `cost`, `chunking`, `guards`, `retrieval`, `rag`, `evals`
-- are shared with the labs and are read, not changed.
"""

__version__ = "1.0.0"

from aip.config import MODELS, resolve_model, settings
from aip.cost import Budget, BudgetExceeded, Usage, is_priced, price_of
from aip.embed import embed, embed_batch
from aip.evals import (
    JUDGE_RUBRIC_CORRECTNESS,
    JUDGE_RUBRIC_FAITHFULNESS,
    Case,
    CaseResult,
    EvalReport,
    compare,
    judge_agreement,
    llm_judge,
    retrieval_metrics,
    run_eval,
)
from aip.guards import (
    ToolDenied,
    ToolGuard,
    delimit_untrusted,
    detect_injection,
    enforce_citations,
    redact_pii,
)
from aip.llm import StructuredOutputError, chat, structured
from aip.tracing import read_traces, trace

__all__ = [
    "MODELS", "settings", "resolve_model",
    "Budget", "BudgetExceeded", "Usage", "is_priced", "price_of",
    "chat", "structured", "StructuredOutputError",
    "embed", "embed_batch",
    "trace", "read_traces",
    "ToolGuard", "ToolDenied", "detect_injection", "redact_pii",
    "delimit_untrusted", "enforce_citations",
    "Case", "CaseResult", "EvalReport", "run_eval", "compare",
    "retrieval_metrics", "llm_judge", "judge_agreement",
    "JUDGE_RUBRIC_FAITHFULNESS", "JUDGE_RUBRIC_CORRECTNESS",
]
