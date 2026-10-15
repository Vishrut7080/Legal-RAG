from __future__ import annotations

import json as jsonlib

import src.ui as ui


class _Response:
    ok = True

    def __init__(self, payload, lines=()):
        self._payload = payload
        self._lines = list(lines)

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_ui_uses_service_api_when_healthy(monkeypatch):
    calls = []
    payload = {
        "question": "question",
        "answer": "Answer [1]",
        "refused": False,
        "citations_valid": True,
        "cited_provisions": ["BNS s.126"],
        "hits": [{
            "provision": "BNS s.126",
            "heading": "Wrongful restraint",
            "in_force": True,
            "score": 0.8,
        }],
        "stages": {"trace_id": "trace-1", "latency_by_stage": {"retrieval": 2.0}},
        "n_searches": 1,
        "cost": {"calls": 1, "priced": True, "models": ["gemini/main"]},
        "cache": {"layer": "exact"},
    }
    monkeypatch.setattr(ui, "_service_is_healthy", lambda: True)

    def post(url, *, json, stream=False, timeout):
        calls.append((url, json, stream, timeout))
        lines = [
            'event: token',
            'data: {"text":"Answer [1]","unvalidated":true}',
            '',
            'event: final',
            f"data: {jsonlib.dumps(payload)}",
            '',
        ]
        return _Response(payload, lines)

    monkeypatch.setattr(ui.requests, "post", post)
    tokens = []
    settings = {
        "k": 6,
        "retriever": "hybrid+current",
        "current_only": True,
        "mode": "llm",
    }

    result, cost, backend, latency, cache_layer = ui._run_question(
        "question", settings, on_token=lambda text, pending: tokens.append((text, pending))
    )

    assert calls[0][0] == f"{ui.SERVICE_URL}/ask/stream"
    assert calls[0][1]["mode"] == "llm"
    assert calls[0][1]["use_cache"] is True
    assert calls[0][2] is True
    assert tokens == [("Answer [1]", True)]
    assert result.answer == "Answer [1]"
    assert result.hits[0].chunk.meta["provision"] == "BNS s.126"
    assert cost.calls == 1
    assert backend == "service API"
    assert latency >= 0
    assert cache_layer == "exact"


def test_ui_falls_back_to_in_process_when_service_is_down(monkeypatch):
    expected = type("Result", (), {
        "answer": "local answer",
        "stages": {"trace_id": "local-1"},
    })()
    monkeypatch.setattr(ui, "_service_is_healthy", lambda: False)
    monkeypatch.setattr(ui, "run_answer", lambda *args, **kwargs: expected)

    result, _, backend, latency, cache_layer = ui._run_question(
        "question",
        {"k": 4, "retriever": "hybrid", "current_only": True, "mode": "extractive"},
    )

    assert result is expected
    assert backend == "in-process UI"
    assert latency >= 0
    assert cache_layer == "none"
