from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import litellm

import aip.llm as llm


def test_chat_stream_emits_tokens_and_caches_the_assembled_response(monkeypatch):
    emitted: list[str] = []
    stored = []
    ledger = []
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return iter([
            SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="Fast "), finish_reason=None
                )],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="answer."), finish_reason="stop"
                )],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
            ),
        ])

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(llm.cache, "get", lambda key: None)
    monkeypatch.setattr(llm.cache, "put", lambda *args: stored.append(args))
    monkeypatch.setattr(llm.cost, "record", lambda usage: ledger.append(usage))
    monkeypatch.setattr(llm.tracing, "trace", lambda *a, **kw: nullcontext({}))

    result = llm.chat(
        "Explain the provision.",
        model="gemini/gemini-3.5-flash-lite",
        max_tokens=50,
        on_token=emitted.append,
    )

    assert result == "Fast answer."
    assert emitted == ["Fast ", "answer."]
    assert calls[0]["stream"] is True
    assert len(stored) == 1
    assert stored[0][3]["text"] == result
    assert ledger[0].prompt_tokens == 12
    assert ledger[0].completion_tokens == 4
