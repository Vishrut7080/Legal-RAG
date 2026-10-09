"""Tests for the service layer, the caches, telemetry, and the gate's ability to fail.

Grouped by what they defend rather than by module, because the invariants that
matter here are the ones that crossed a boundary: a cached answer replayed to the
wrong caller, a metric that reports a miss as a hit, a stream that puts
unvalidated text on the wire.

Two of these exist because the obvious version would have passed anyway. The
replayed-response test asserts on `cache.hit` in the body, because the bug it
guards was a *correct* cache lookup reported as a miss -- the answer was right
and the instrumentation was wrong, which no test of the answer would catch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.cache as cache_mod
from src.cache import AnswerCache, Config, scope_of, scopes_compatible
from src.retrieve import load_golden
from src.telemetry import (
    Observation,
    Registry,
    StageTimer,
    new_trace_id,
    percentile,
)

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "eval" / "golden.jsonl"
SWEEP = Path(__file__).resolve().parent.parent / "data" / "eval" / "cache_sweep.json"

# The pair the sweep found most confusable, at cosine ~0.82. It is the
# statute-shaped version of "Gold" versus "Silver", and it is the reason the
# shipped semantic threshold is measured rather than guessed.
TEMPORARY_INJUNCTION = "When may a temporary injunction be granted?"
PERPETUAL_INJUNCTION = "When is a perpetual injunction granted?"


@pytest.fixture()
def cache(tmp_path: Path) -> AnswerCache:
    return AnswerCache(tmp_path / "answers.sqlite3")


def _payload(provision: str, *, refused: bool = False,
             citations_valid: bool = True) -> dict:
    return {
        "answer": f"{provision} provides: ... [1]",
        "cited_provisions": [] if refused else [provision],
        "citations_valid": citations_valid,
        "refused": refused,
    }


# --------------------------------------------------------------------------
# Cache: the exact layer
# --------------------------------------------------------------------------
def test_exact_hit_ignores_case_and_trailing_punctuation(cache: AnswerCache) -> None:
    """Folding case and trailing punctuation must not create a false miss."""
    cache.store(TEMPORARY_INJUNCTION, Config(), _payload("CPC O.39 r.1"))
    lookup = cache.lookup(f"  {TEMPORARY_INJUNCTION.upper()}  ", Config())
    assert lookup.hit
    assert lookup.layer == "exact"
    assert lookup.similarity == 1.0


def test_exact_layer_refuses_to_fold_words(cache: AnswerCache) -> None:
    """Normalisation must never be semantic.

    Two questions differing by one word are different questions. A normaliser
    that stripped stopwords would answer one of them with the other.
    """
    cache.store("What is the punishment for murder?", Config(),
                _payload("BNS s.103"))
    lookup = cache.lookup("What is the punishment for culpable homicide?", Config())
    assert not lookup.hit


def test_config_is_part_of_the_key(cache: AnswerCache) -> None:
    """current_only=false must not be served a current-law answer."""
    cache.store(TEMPORARY_INJUNCTION, Config(), _payload("CPC O.39 r.1"))
    assert not cache.lookup(TEMPORARY_INJUNCTION, Config(current_only=False)).hit
    assert not cache.lookup(TEMPORARY_INJUNCTION, Config(k=3)).hit


def test_config_gates_the_semantic_layer_too(tmp_path: Path) -> None:
    """The config mask must apply to semantic hits, not only to the exact key.

    This was a real bug: the same question asked with current_only=false was
    served the answer recorded against the filtered request, which reintroduces
    exactly the defect the current-law filter exists to prevent.
    """
    cache = AnswerCache(tmp_path / "a.sqlite3")
    cache.store(TEMPORARY_INJUNCTION, Config(), _payload("CPC O.39 r.1"))

    reckless = AnswerCache(tmp_path / "a.sqlite3", semantic_threshold=0.0)
    lookup = reckless.lookup(TEMPORARY_INJUNCTION, Config(current_only=False))
    assert not lookup.hit, "a semantic hit ignored the retrieval configuration"
    assert "configuration" in lookup.reason


# --------------------------------------------------------------------------
# Cache: the semantic layer, and the trap it is
# --------------------------------------------------------------------------
def test_semantic_layer_refuses_temporary_for_perpetual(cache: AnswerCache) -> None:
    """At the shipped threshold the dangerous pair does not collide."""
    cache.store(TEMPORARY_INJUNCTION, Config(), _payload("CPC O.39 r.1"))
    lookup = cache.lookup(PERPETUAL_INJUNCTION, Config())
    assert not lookup.hit
    assert lookup.similarity is not None
    assert lookup.similarity < cache.semantic_threshold


def test_semantic_layer_would_be_wrong_at_a_low_threshold(tmp_path: Path) -> None:
    """The bug the measurement exists to prevent, reproduced on purpose.

    If this ever fails because the pair stopped being similar, the sweep result
    is stale and the shipped threshold should be revisited deliberately -- which
    is a more interesting failure than the one it guards against.
    """
    cache = AnswerCache(tmp_path / "a.sqlite3")
    cache.store(TEMPORARY_INJUNCTION, Config(), _payload("CPC O.39 r.1"))

    reckless = AnswerCache(tmp_path / "a.sqlite3", semantic_threshold=0.80)
    lookup = reckless.lookup(PERPETUAL_INJUNCTION, Config())
    assert lookup.hit, "the 0.82 cosine pair stopped colliding; re-run the sweep"
    assert lookup.payload["cited_provisions"] == ["CPC O.39 r.1"]
    # And that is exactly the wrong answer: perpetual injunctions live in the
    # Specific Relief Act, not the CPC.
    assert lookup.payload["cited_provisions"] != ["Specific Relief Act s.38"]


def test_scope_guard_blocks_different_acts() -> None:
    """The axis the guard can see: which Act is being asked about."""
    allowed, reason = scopes_compatible(
        scope_of("What is the punishment for murder under the BNS?"),
        scope_of("What is the punishment for murder under the IPC?"),
    )
    assert not allowed
    assert "different Acts" in reason


def test_scope_guard_treats_partial_act_overlap_as_a_conflict() -> None:
    """"BNS" and "BNS and IPC" are different questions, and the corpus holds both
    generations, so the difference is exactly the dangerous one."""
    allowed, _ = scopes_compatible(
        scope_of("Punishment for murder under the BNS?"),
        scope_of("Punishment for murder under the BNS and the IPC?"),
    )
    assert not allowed


def test_scope_guard_blocks_different_section_numbers() -> None:
    allowed, reason = scopes_compatible(
        scope_of("What does section 138 of the NI Act provide?"),
        scope_of("What does section 139 of the NI Act provide?"),
    )
    assert not allowed
    assert "section numbers" in reason


def test_scope_guard_allows_the_same_scope() -> None:
    allowed, _ = scopes_compatible(
        scope_of("Punishment for murder under the BNS?"),
        scope_of("What is the punishment for murder under the BNS?"),
    )
    assert allowed


def test_cached_refusal_is_never_served_to_a_similar_question(
        tmp_path: Path) -> None:
    """A refusal is a decision about one question.

    Serving it to a merely-similar question converts a correct refusal into a
    wrong answer, which is harder to notice than a missing one.
    """
    cache = AnswerCache(tmp_path / "a.sqlite3")
    cache.store("What is the corporate tax rate in India?", Config(),
                {"answer": "refused", "cited_provisions": [],
                 "citations_valid": True, "refused": True})

    reckless = AnswerCache(tmp_path / "a.sqlite3", semantic_threshold=0.80)
    lookup = reckless.lookup("What is the rate of corporate taxation in India?",
                             Config())
    assert not lookup.hit
    assert "refusal" in lookup.reason


def test_an_answer_failing_its_contract_is_never_cached(cache: AnswerCache) -> None:
    """Replaying a broken answer would make a transient bug permanent."""
    cache.store("something went wrong", Config(),
                _payload("BNS s.103", citations_valid=False))
    assert cache.size() == 0


def test_refusal_is_recorded_from_the_payload_not_the_caller(
        tmp_path: Path) -> None:
    """`store()` must not depend on the caller remembering to pass `refused`.

    A caller that forgets marks a refusal as an answer, and the semantic layer
    then hands it to the next similar question.
    """
    cache = AnswerCache(tmp_path / "a.sqlite3", semantic_threshold=0.80)
    cache.store("What is the corporate tax rate in India?", Config(),
                {"answer": "refused", "cited_provisions": [],
                 "citations_valid": True, "refused": True})
    lookup = cache.lookup("What is the rate of corporate taxation in India?",
                          Config())
    assert not lookup.hit


def test_semantic_layer_can_be_disabled(cache: AnswerCache) -> None:
    cache.store(TEMPORARY_INJUNCTION, Config(), _payload("CPC O.39 r.1"))
    off = AnswerCache(cache.path, enable_semantic=False)
    assert off.lookup(PERPETUAL_INJUNCTION, Config()).reason == \
        "semantic layer disabled"
    # The exact layer still works with the semantic layer off.
    assert off.lookup(TEMPORARY_INJUNCTION, Config()).hit


def test_clear_empties_the_cache(cache: AnswerCache) -> None:
    cache.store("a question", Config(), _payload("BNS s.103"))
    assert cache.size() == 1
    assert cache.clear() == 1
    assert cache.size() == 0
    assert cache.stats.exact_hits == 0


# --------------------------------------------------------------------------
# The sweep, which is where the threshold comes from
# --------------------------------------------------------------------------
def test_the_shipped_threshold_is_above_every_measured_similarity() -> None:
    """The shipped value must dominate the sweep's measured maximum.

    If this fails, either the golden set changed or the embedding model did, and
    the threshold has to be re-measured rather than inherited.
    """
    from src.cache import DEFAULT_SEMANTIC_THRESHOLD

    if not SWEEP.exists():
        pytest.skip("run `python -m src.cache_sweep` first")
    result = json.loads(SWEEP.read_text(encoding="utf-8"))
    worst = result["sim_summary"]["max_offdiagonal"]
    assert worst < DEFAULT_SEMANTIC_THRESHOLD, (
        f"shipped threshold {DEFAULT_SEMANTIC_THRESHOLD} is not above the "
        f"measured maximum off-diagonal similarity {worst}"
    )


def test_every_similarity_band_with_hits_is_purely_wrong_hits() -> None:
    """The headline finding is the absence of a safe band, so pin it.

    A band with any correct hit would be news, not a regression; this test says
    so explicitly rather than leaving it implied by a count.
    """
    if not SWEEP.exists():
        pytest.skip("run `python -m src.cache_sweep` first")
    result = json.loads(SWEEP.read_text(encoding="utf-8"))
    bands = [r for r in result["sweep"] if r["hits"] > 0]
    assert bands, "the sweep found no hits at all; it measured nothing"
    offenders = [r["threshold"] for r in bands if r["precision"] > 0.0]
    assert not offenders, (
        f"similarity bands {offenders} now produce some correct hits; a real "
        "operating range may exist and the threshold should be revisited"
    )
    assert result["knee"]["safe"] is False


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------
def test_percentile_returns_a_value_that_actually_occurred() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    for p in (0, 50, 95, 100):
        assert percentile(values, p) in values


def test_percentile_of_nothing_is_zero() -> None:
    assert percentile([], 95) == 0.0


def test_stage_timer_accumulates_repeated_stages() -> None:
    """A retried stage must not lose the first attempt's time."""
    timer = StageTimer()
    for _ in range(3):
        with timer.stage("retrieval"):
            sum(range(1000))
    assert set(timer.as_dict()) == {"retrieval"}
    assert timer.total_ms >= timer.get("retrieval")


def test_stage_timer_times_a_stage_that_raises() -> None:
    """The stage that throws is usually the slow one."""
    timer = StageTimer()
    with pytest.raises(ValueError), timer.stage("retrieval"):
        raise ValueError("boom")
    assert "retrieval" in timer.as_dict()


def test_registry_reports_percentiles_and_per_stage_breakdown() -> None:
    reg = Registry()
    for i in range(10):
        reg.record(Observation(
            trace_id=new_trace_id(), route="/ask", latency_ms=float(i * 10),
            latency_by_stage={"retrieval": float(i), "guards": 0.1},
        ))
    snap = reg.snapshot()
    assert snap["requests"] == 10
    # Nearest-rank, so the reported percentile is a latency that actually
    # occurred rather than an interpolation between two samples. For
    # [0, 10, ... 90] that is 40.0, not the arithmetic median of 45.
    assert snap["latency_ms"]["p50"] == 40.0
    assert snap["latency_ms"]["max"] == 90.0
    assert snap["latency_ms"]["p95"] >= snap["latency_ms"]["p50"]
    assert set(snap["by_stage_ms"]) == {"retrieval", "guards"}
    assert snap["by_route"] == {"/ask": 10}


def test_registry_counts_cache_hits_and_refusals() -> None:
    reg = Registry()
    reg.record(Observation(trace_id="a", route="/ask", latency_ms=1.0,
                           cache_layer="exact"))
    reg.record(Observation(trace_id="b", route="/ask", latency_ms=1.0,
                           cache_layer="semantic", cache_similarity=0.97))
    reg.record(Observation(trace_id="c", route="/ask", latency_ms=1.0,
                           refused=True))
    snap = reg.snapshot()
    assert snap["cache"]["hits"] == 2
    assert snap["cache"]["hit_rate"] == pytest.approx(2 / 3, abs=0.01)
    assert snap["refusals"] == 1


def test_registry_with_no_requests_says_so() -> None:
    snap = Registry().snapshot()
    assert snap["requests"] == 0
    assert "no requests" in snap["note"]


def test_a_zero_cost_is_reported_as_priced_not_as_unknown() -> None:
    """`local/` models are genuinely free; $0.00 must be labelled as known.

    A bare zero is indistinguishable from "we did not measure", which is the
    exact conflation `aip.cost.is_priced` exists to prevent.
    """
    from src.telemetry import CostDelta

    delta = CostDelta(calls=1, cost_usd=0.0, priced=True)
    assert delta.as_dict()["priced"] is True
    assert delta.as_dict()["cost_usd"] == 0.0

    unknown = CostDelta(calls=1, cost_usd=0.0, priced=False, unpriced_calls=1)
    assert unknown.as_dict()["priced"] is False
    assert unknown.as_dict()["unpriced_calls"] == 1


def test_trace_ids_are_distinct() -> None:
    assert len({new_trace_id() for _ in range(1000)}) == 1000


# --------------------------------------------------------------------------
# Service: observability in the body, and the cache it advertises
# --------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose cache is isolated from the real one."""
    monkeypatch.setattr(cache_mod, "_DEFAULT",
                        AnswerCache(tmp_path / "svc.sqlite3"))
    from src.service import app
    with TestClient(app) as test_client:
        yield test_client


def test_ask_returns_latency_cost_and_a_trace_id(client: TestClient) -> None:
    body = client.post("/ask", json={
        "question": "What is the punishment for murder?"}).json()
    assert body["trace_id"]
    assert body["latency_ms"] > 0
    assert "retrieval" in body["latency_by_stage"]
    assert body["cost"]["priced"] is True
    assert body["cost"]["cost_usd"] == 0.0


def test_a_refusal_still_carries_its_measurements(client: TestClient) -> None:
    """Timing must be stamped on the refusals too.

    A refusal is exactly when you most want to know why it was slow, and it is
    the path most likely to be forgotten.
    """
    body = client.post("/ask", json={
        "question": "What is the corporate tax rate in India?"}).json()
    assert body["refused"] is True
    assert body["trace_id"]
    assert body["latency_ms"] > 0
    assert body["latency_by_stage"]


def test_ask_reports_a_cache_hit_as_a_hit(client: TestClient) -> None:
    """The regression this pins: the answer was right and the report was wrong.

    The stored payload carries the metadata of the request that created it, so
    returning it verbatim made every hit announce `cache.hit: false`. No test of
    the answer alone would ever have caught that.
    """
    question = "What is the punishment for murder?"
    cold = client.post("/ask", json={"question": question}).json()
    warm = client.post("/ask", json={"question": question}).json()

    assert cold["cache"]["hit"] is False
    assert warm["cache"]["hit"] is True
    assert warm["cache"]["layer"] == "exact"
    assert warm["replayed"] is True
    assert warm["answer"] == cold["answer"]


def test_a_cache_hit_gets_its_own_trace_id_and_names_its_origin(
        client: TestClient) -> None:
    """A replay must be traceable back to the run that computed it."""
    question = "What is the punishment for murder?"
    cold = client.post("/ask", json={"question": question}).json()
    warm = client.post("/ask", json={"question": question}).json()
    assert warm["trace_id"] != cold["trace_id"]
    assert warm["cache"]["served_from_trace_id"] == cold["trace_id"]


def test_use_cache_false_bypasses_the_cache(client: TestClient) -> None:
    question = "What is the punishment for murder?"
    client.post("/ask", json={"question": question})
    bypass = client.post("/ask", json={"question": question,
                                       "use_cache": False}).json()
    assert bypass["cache"]["hit"] is False


def test_the_cache_does_not_make_the_current_law_default_weaker(
        client: TestClient) -> None:
    """A cache hit must not smuggle repealed text through a filtered request."""
    question = "What did the repealed IPC say about murder?"
    first = client.post("/ask", json={"question": question}).json()
    second = client.post("/ask", json={"question": question}).json()
    assert second["cache"]["hit"] is True
    assert second["answer"] == first["answer"]
    assert second["current_only"] is True


def test_health_reports_startup_and_cache_configuration(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["index_loaded"] is True
    assert body["startup_ms"] is not None
    assert body["cache"]["semantic_threshold"] > 0


def test_the_embedding_model_is_warmed_at_startup(client: TestClient) -> None:
    """The first request must not pay an 18-second model load.

    Left lazy, the first `/ask` after a deploy took ~18,370 ms -- against a
    1,500 ms time-to-first-token target -- while every subsequent request took
    ~20 ms. Nothing warms the model on the request path, so the p95 never showed
    it; only the max did.
    """
    body = client.get("/health").json()
    assert body["model_warm_error"] is None
    assert body["model_warm_ms"] is not None
    assert body["index_load_ms"] is not None
    assert body["startup_ms"] == body["index_load_ms"] + body["model_warm_ms"]


def test_the_first_request_is_not_the_slowest_one(client: TestClient) -> None:
    """Behavioural form of the same claim: no cold-start cliff after boot."""
    first = client.post("/ask", json={
        "question": "What is the punishment for murder?"}).json()
    later = client.post("/ask", json={
        "question": "How long may a warrant of arrest remain in force?"}).json()
    assert later["latency_ms"] < 2_000
    assert first["latency_ms"] < 2_000


def test_metrics_endpoint_reports_percentiles_and_cache_stats(
        client: TestClient) -> None:
    client.post("/ask", json={"question": "What is the punishment for murder?"})
    body = client.get("/metrics").json()
    assert body["requests"] >= 1
    assert body["latency_ms"]["p95"] >= body["latency_ms"]["p50"]
    assert "retrieval" in body["by_stage_ms"]
    assert body["cost"]["priced"] is True
    assert "cache_stats" in body


def test_cache_clear_endpoint_empties_the_cache(client: TestClient) -> None:
    client.post("/ask", json={"question": "What is the punishment for murder?"})
    body = client.post("/cache/clear").json()
    assert body["removed"] >= 1


# --------------------------------------------------------------------------
# Streaming, and the problem it creates
# --------------------------------------------------------------------------
def _read_sse(response) -> list[tuple[str, dict]]:
    events, name, data = [], None, []
    for raw in response.iter_lines():
        line = raw.rstrip("\n")
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            data.append(line[6:])
        elif line == "":
            if name:
                events.append((name, json.loads("".join(data))))
            name, data = None, []
    if name:
        events.append((name, json.loads("".join(data))))
    return events


def test_stream_sends_citations_before_any_token(client: TestClient) -> None:
    """The extractive generator's citations are valid before the first byte.

    The cited provision is chosen by retrieval and the generator is
    deterministic, so there is no window in which an unvalidated citation is on
    the wire. If this ordering ever changes, that guarantee is gone.
    """
    with client.stream("POST", "/ask/stream", json={
            "question": "What is the punishment for murder?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _read_sse(response)

    names = [name for name, _ in events]
    assert names[0] == "started"
    assert "citations" in names
    assert "token" in names
    assert names[-1] == "final"
    assert names.index("citations") < names.index("token")

    citations = dict(events)["citations"]
    assert citations["validated"] is True


def test_stream_final_event_has_a_stable_shape(client: TestClient) -> None:
    """A client must not have to branch on the cache to handle `final`."""
    question = "What is the punishment for murder?"
    with client.stream("POST", "/ask/stream", json={"question": question}) as r:
        cold = _read_sse(r)[-1][1]
    with client.stream("POST", "/ask/stream", json={"question": question}) as r:
        warm = _read_sse(r)[-1][1]

    for key in ("answer", "refused", "citations_valid", "retracted",
                "ttft_ms", "trace_id", "cost", "cache"):
        assert key in cold, f"missing {key} on a cold final event"
        assert key in warm, f"missing {key} on a replayed final event"
    assert warm["cache"]["hit"] is True
    assert warm["replayed"] is True


def test_stream_populates_the_cache_for_both_routes(client: TestClient) -> None:
    """/ask and /ask/stream answer the same question.

    A client that used one must warm the cache for the other, or the hit rate
    on /metrics would depend on which route a caller happened to choose.
    """
    question = "What is the punishment for murder under the BNS?"
    with client.stream("POST", "/ask/stream", json={"question": question}) as r:
        _read_sse(r)
    after = client.post("/ask", json={"question": question}).json()
    assert after["cache"]["hit"] is True
    assert after["cache"]["layer"] == "exact"


def test_stream_refusal_is_visible_in_the_stream(client: TestClient) -> None:
    with client.stream("POST", "/ask/stream", json={
            "question": "What is the corporate tax rate in India?"}) as r:
        final = _read_sse(r)[-1][1]
    assert final["refused"] is True
    assert final["refusal_reason"]


def test_stream_reports_a_real_time_to_first_token(client: TestClient) -> None:
    with client.stream("POST", "/ask/stream", json={
            "question": "What is the punishment for murder?"}) as r:
        final = _read_sse(r)[-1][1]
    assert final["ttft_ms"] is not None
    assert final["ttft_ms"] > 0
    # TTFT cannot exceed the whole computation by much; a number larger than
    # total latency means it was measured at the end of the loop.
    assert final["ttft_ms"] <= final["latency_ms"] * 1.5 + 50


# --------------------------------------------------------------------------
# The retriever menu must not offer something search() cannot resolve
# --------------------------------------------------------------------------
def test_every_retriever_the_service_offers_actually_resolves() -> None:
    """A 500 waiting to happen.

    `hybrid+rerank` was selectable in the UI, the service and `diagnose` while
    `build_retrievers` only defined four keys, so picking it raised `KeyError`.
    """
    from src.retrieve import _shared, search
    from src.service import RETRIEVERS

    _, retrievers = _shared()
    for name in RETRIEVERS:
        assert name in retrievers, f"{name} is offered but not registered"
        assert search("What is the punishment for murder?", k=2, which=name)


def test_reranker_degrades_instead_of_raising(monkeypatch: pytest.MonkeyPatch
                                              ) -> None:
    """An unreachable cross-encoder is a missing enhancement, not a crash."""
    import src.retrieve as retrieve_mod

    monkeypatch.setattr(retrieve_mod, "_reranker", lambda: None)
    hits = retrieve_mod.search("What is the punishment for murder?", k=3,
                               which="hybrid+rerank")
    assert len(hits) == 3


def test_reranker_is_not_invented_for_profiles_that_lack_one(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Under AIP_PROFILE=gemini there is no RERANK tier.

    `resolve_model` passes an unknown tier name through unchanged, so the old
    code handed `CrossEncoderReranker` the literal string "RERANK" and asked the
    HuggingFace Hub for a model repository by that name.
    """
    import src.retrieve as retrieve_mod

    monkeypatch.setattr(retrieve_mod, "resolve_model", lambda tier: "RERANK")
    retrieve_mod._reranker.cache_clear()
    try:
        assert retrieve_mod._reranker() is None
    finally:
        retrieve_mod._reranker.cache_clear()


# --------------------------------------------------------------------------
# The gate must be able to fail
# --------------------------------------------------------------------------
def test_breaking_the_retrieval_window_makes_the_gate_exit_non_zero() -> None:
    """A gate that has never been seen to fail is not known to be a gate."""
    from src.break_it import MODES, run_gate

    _, breaker = MODES["k"]
    target, original = breaker()
    try:
        code, out = run_gate()
    finally:
        target.write_text(original, encoding="utf-8")
    assert code != 0, f"the gate passed on a broken pipeline:\n{out}"
    assert "FAILED" in out


def test_editing_the_corpus_makes_the_gate_exit_non_zero() -> None:
    """The gap this found.

    Emptying a corpus act left every pre-existing check green, because the gate
    measured the index and nothing tied `data/corpus/*.md` back to `data/raw/`.
    """
    from src.break_it import MODES, run_gate

    _, breaker = MODES["corpus"]
    target, original = breaker()
    try:
        code, out = run_gate()
    finally:
        target.write_text(original, encoding="utf-8")
    assert code != 0, (
        "a corpus document can be edited with a green build; the gate no "
        f"longer detects it:\n{out}"
    )


def test_corrupting_the_index_checksum_makes_the_gate_exit_non_zero() -> None:
    from src.break_it import MODES, run_gate

    _, breaker = MODES["index"]
    target, original = breaker()
    try:
        code, out = run_gate()
    finally:
        target.write_text(original, encoding="utf-8")
    assert code != 0, f"the gate passed on a corrupt index:\n{out}"


def test_the_committed_corpus_reproduces_from_the_snapshot() -> None:
    """The positive counterpart: the gate's own integrity check must hold."""
    from src.gate import check_corpus_matches_snapshot

    check = check_corpus_matches_snapshot()
    assert check.ok, check.detail


def test_the_recorded_gate_failure_demo_shows_a_red_build() -> None:
    """The committed evidence, if it is ever re-generated green, is worthless."""
    demo = (Path(__file__).resolve().parent.parent
            / "reports" / "gate_failure_demo.txt")
    if not demo.exists():
        pytest.skip("run `make gate-record` first")
    text = demo.read_text(encoding="utf-8")
    assert "ALL BREAK MODES WERE CAUGHT" in text
    assert "FATAL" not in text


# --------------------------------------------------------------------------
# Results the report depends on
# --------------------------------------------------------------------------
def test_the_golden_set_is_the_size_the_report_claims() -> None:
    cases = load_golden(GOLDEN)
    assert len(cases) == 102
    assert sum(1 for c in cases if c.get("expect_refusal")) == 3
    assert sum(1 for c in cases if c.get("forbidden_provisions")) == 43


def test_the_sweep_probe_pairs_exist_in_the_golden_set() -> None:
    """The examples in the report are real golden cases, not invented ones."""
    questions = {c["question"] for c in load_golden(GOLDEN)}
    assert TEMPORARY_INJUNCTION in questions
    assert PERPETUAL_INJUNCTION in questions