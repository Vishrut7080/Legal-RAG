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
"""

__version__ = "1.0.0"

from aip.config import MODELS, resolve_model, settings
from aip.cost import Budget, BudgetExceeded, Usage
from aip.embed import embed, embed_batch
from aip.llm import StructuredOutputError, chat, structured
from aip.tracing import read_traces, trace

__all__ = [
    "MODELS", "settings", "resolve_model",
    "Budget", "BudgetExceeded", "Usage",
    "chat", "structured", "StructuredOutputError",
    "embed", "embed_batch",
    "trace", "read_traces",
]
