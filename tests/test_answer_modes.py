from __future__ import annotations

import pytest

import src.answer as answer_module
from aip.chunking import Chunk
from aip.retrieval import Hit


def _hit() -> Hit:
    chunk = Chunk(
        "BNS s.319: cheating by personation.",
        "bns",
        "bns-319",
        {"provision": "BNS s.319", "in_force": "True"},
    )
    return Hit(chunk, 0.9, "test", 1)


def test_explicit_llm_mode_uses_generation_without_legacy_flag(monkeypatch):
    generated: list[str] = []
    monkeypatch.setattr(answer_module, "search", lambda *a, **kw: [_hit()])
    monkeypatch.setattr(answer_module, "_chunks_have_answer", lambda *a: (True, ""))

    def generate(question, hits, stages, **kwargs):
        generated.append(question)
        return "BNS s.319 covers cheating by personation [1]."

    monkeypatch.setattr(answer_module, "_generate", generate)
    result = answer_module.answer("What is BNS s.319?", mode="llm")

    assert generated == ["What is BNS s.319?"]
    assert result.stages["generator"] == "llm"
    assert result.citations_valid


def test_explicit_extractive_mode_overrides_legacy_llm_flag(monkeypatch):
    monkeypatch.setattr(answer_module, "search", lambda *a, **kw: [_hit()])
    monkeypatch.setattr(answer_module, "_chunks_have_answer", lambda *a: (True, ""))
    monkeypatch.setattr(
        answer_module,
        "_generate",
        lambda *a: pytest.fail("explicit extractive mode must not call the LLM"),
    )

    result = answer_module.answer("What is BNS s.319?", llm=True, mode="extractive")

    assert result.stages["generator"] == "extractive"


def test_unknown_answer_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        answer_module.answer("question", mode="unknown")


def test_gemini_small_tier_resolves_to_requested_flash_lite(monkeypatch):
    from aip.config import resolve_model, settings

    monkeypatch.setattr(settings, "profile", "gemini")

    assert answer_module.GENERATION_TIER == "SMALL"
    assert resolve_model(answer_module.GENERATION_TIER) == \
        "gemini/gemini-3.5-flash-lite"
