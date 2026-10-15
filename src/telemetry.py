"""Latency, cost and trace identity for every request the service handles.

Why this module exists
----------------------
Six labs of components measure nothing about the thing they were assembled into.
A retriever reports hit rates; an evaluator reports precision. Neither tells you
that request X took nine seconds, because neither knows nine seconds happened.

So this module makes three things first-class, and every one of them is a number
rather than a claim:

  1. **Latency, broken down by stage.** A p95 with no stage attribution is not
     actionable -- it tells you the service is slow and nothing else. The stages
     here are the ones `src/answer.py` actually has: guards, retrieval,
     sufficiency, generation, citation check.
  2. **Cost, with its priced-ness attached.** `aip.cost.is_priced` draws a hard
     line between a model that is genuinely free (you own the hardware) and one
     whose price we do not know. Reporting $0.00 without that distinction is the
     bug `aip/cost.py:19-32` exists to prevent, so this module never prints a
     zero it cannot back.
  3. **A trace id.** Without one, a latency complaint is not actionable either:
     there is no way to get from "slow" to the record of the slow request.

Cost accounting, specifically
-----------------------------
Nothing here instruments a model call site. `aip.llm` and `aip.embed` already
call `aip.cost.record()` after every call, which lands in one process-global
`Budget`. So this module brackets a request with a before/after snapshot of that
budget and takes the difference. That has two useful consequences: the call sites
stay untouched, and a request that never calls a model reports zero tokens
because it genuinely made none -- not because nobody looked.

    with BudgetSnapshot() as snap: ...
    snap.delta.cost_usd        # attributable to this request

The default profile is `local`, where `is_prided` is true and the cost is a real
$0.00. That is a *correct* zero -- you own the hardware -- and it is reported as
such, never as an absence of measurement.
"""
from __future__ import annotations

import contextlib
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aip.config import settings as _aip_settings
from aip.cost import Budget, Usage, global_budget, is_priced

# Percentiles are computed over a bounded window rather than over all time. An
# unbounded latency list is a slow memory leak in a long-running service, and a
# percentile over "every request since the process started" stops describing
# current behaviour long before it stops being useful.
WINDOW = 5_000


def new_trace_id() -> str:
    """A short, unique-enough id for correlating a response with its trace."""
    return uuid.uuid4().hex[:12]


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. `p` is 0-100.

    Nearest-rank rather than interpolated, so the number returned is a latency
    that actually occurred. An interpolated p95 is a number no request had.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


# --------------------------------------------------------------------------
# Per-stage timing
# --------------------------------------------------------------------------
class StageTimer:
    """Accumulates wall-clock time per named stage.

        timer = StageTimer()
        with timer.stage("retrieval"):
            hits = search(q, k=8)
        timer.as_dict()   # {"retrieval": 41.2}
        timer.total_ms    # 41.2

    A stage entered twice accumulates rather than overwrites, so a loop that
    retries does not silently lose the first attempt's time -- which is exactly
    the kind of latency a dashboard should be showing.
    """

    def __init__(self) -> None:
        self._stages: dict[str, float] = {}

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            # `finally`, so a raised stage is still timed. A stage that throws is
            # usually the slowest one.
            elapsed = (time.perf_counter() - started) * 1000.0
            self._stages[name] = self._stages.get(name, 0.0) + elapsed

    def set(self, name: str, ms: float) -> None:
        self._stages[name] = ms

    def get(self, name: str) -> float:
        return self._stages.get(name, 0.0)

    @property
    def total_ms(self) -> float:
        return sum(self._stages.values())

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self._stages.items()}

    def slowest(self) -> str:
        return max(self._stages, key=lambda k: self._stages[k]) if self._stages else ""


# --------------------------------------------------------------------------
# Cost bracketing
# --------------------------------------------------------------------------
@dataclass
class BudgetSnapshot:
    """The process-global ledger at one instant."""

    calls: int
    cached_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    unpriced_calls: int

    @classmethod
    def take(cls, budget: Budget | None = None) -> BudgetSnapshot:
        b = budget or global_budget()
        return cls(
            calls=b.calls,
            cached_calls=b.cached_calls,
            prompt_tokens=b.prompt_tokens,
            completion_tokens=b.completion_tokens,
            cost_usd=b.spent_usd,
            unpriced_calls=b.unpriced_calls,
        )

    def difference(self, later: BudgetSnapshot) -> CostDelta:
        return CostDelta(
            calls=later.calls - self.calls,
            cached_calls=later.cached_calls - self.cached_calls,
            prompt_tokens=later.prompt_tokens - self.prompt_tokens,
            completion_tokens=later.completion_tokens - self.completion_tokens,
            cost_usd=round(later.cost_usd - self.cost_usd, 8),
            unpriced_calls=later.unpriced_calls - self.unpriced_calls,
        )


@dataclass
class CostDelta:
    """What one request cost, and whether that number means anything."""

    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    priced: bool = True
    models: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "priced": self.priced,
        }
        # Never let an unpriced run masquerade as a free one. Mirrors
        # `aip.cost.Budget.report`, for the same reason.
        if not self.priced or self.unpriced_calls:
            out["unpriced_calls"] = self.unpriced_calls
        if self.models:
            out["models"] = self.models
        return out

    @classmethod
    def from_usage(cls, usage: Usage, model: str) -> CostDelta:
        return cls(
            calls=usage.calls,
            cached_calls=int(bool(usage.cached)),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
            unpriced_calls=0 if is_priced(model) else max(usage.calls, 1),
            priced=is_priced(model),
            models=[model] if model else [],
        )


class CostScope:
    """Bracket a block of work and attribute its model spend to it.

        with CostScope() as scope:
            answer(question)
        scope.delta.as_dict()
    """

    def __init__(self) -> None:
        self._before = BudgetSnapshot.take()
        self.delta = CostDelta()
        self._entered = False

    def __enter__(self) -> CostScope:
        self._entered = True
        return self

    def __exit__(self, *exc: object) -> None:
        if not self._entered:
            return
        delta = self._before.difference(BudgetSnapshot.take())
        delta.priced = delta.unpriced_calls == 0
        self.delta = delta


# --------------------------------------------------------------------------
# Aggregate registry
# --------------------------------------------------------------------------
@dataclass
class Observation:
    """One completed request, as the registry sees it."""

    trace_id: str
    route: str
    latency_ms: float
    latency_by_stage: dict[str, float] = field(default_factory=dict)
    cost: CostDelta = field(default_factory=CostDelta)
    cache_layer: str = "none"
    cache_similarity: float | None = None
    refused: bool = False
    ttft_ms: float | None = None
    tool_calls: int = 0


class Registry:
    """Rolling aggregates for the /metrics endpoint.

    Keeps every recent observation (bounded) rather than only running sums,
    because a percentile cannot be updated incrementally and a mean hides the
    tail that a latency budget is written against.
    """

    def __init__(self, window: int = WINDOW) -> None:
        self._lock = threading.Lock()
        self._window = window
        self._obs: deque[Observation] = deque(maxlen=window)
        self._started = time.time()

    def record(self, observation: Observation) -> None:
        with self._lock:
            self._obs.append(observation)

    def reset(self) -> None:
        with self._lock:
            self._obs.clear()
            self._started = time.time()

    def _partition(self) -> list[Observation]:
        with self._lock:
            return list(self._obs)

    def snapshot(self) -> dict[str, Any]:
        obs = self._partition()
        if not obs:
            return {
                "requests": 0,
                "uptime_s": round(time.time() - self._started, 1),
                "latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0},
                "note": "no requests recorded yet",
            }

        latencies = [o.latency_ms for o in obs]
        by_stage: dict[str, list[float]] = {}
        for o in obs:
            for name, ms in o.latency_by_stage.items():
                by_stage.setdefault(name, []).append(ms)

        ttfts = [o.ttft_ms for o in obs if o.ttft_ms is not None]
        sims = [o.cache_similarity for o in obs
                if o.cache_similarity is not None]
        cached = [o for o in obs if o.cache_layer != "none"]
        semantic = [o for o in obs if o.cache_layer == "semantic"]
        refusals = sum(1 for o in obs if o.refused)
        tool_calls = sum(o.tool_calls for o in obs)

        total_cost = sum(o.cost.cost_usd for o in obs)
        total_tokens = sum(o.cost.total_tokens for o in obs)
        unpriced = sum(o.cost.unpriced_calls for o in obs)

        return {
            "requests": len(obs),
            "uptime_s": round(time.time() - self._started, 1),
            "latency_ms": {
                "p50": round(percentile(latencies, 50), 2),
                "p95": round(percentile(latencies, 95), 2),
                "p99": round(percentile(latencies, 99), 2),
                "mean": round(sum(latencies) / len(latencies), 2),
                "max": round(max(latencies), 2),
            },
            "by_stage_ms": {
                name: {
                    "p50": round(percentile(values, 50), 2),
                    "p95": round(percentile(values, 95), 2),
                    "mean": round(sum(values) / len(values), 2),
                }
                for name, values in sorted(by_stage.items())
            },
            "ttft_ms": ({"p50": round(percentile(ttfts, 50), 2),
                         "p95": round(percentile(ttfts, 95), 2)}
                        if ttfts else None),
            "cache": {
                "hits": len(cached),
                "hit_rate": round(len(cached) / len(obs), 4),
                "exact": sum(1 for o in obs if o.cache_layer == "exact"),
                "semantic": len(semantic),
                "similarity": ({"min": round(min(sims), 4),
                                "mean": round(sum(sims) / len(sims), 4)}
                               if sims else None),
            },
            "refusals": refusals,
            "refusal_rate": round(refusals / len(obs), 4),
            "tool_calls": tool_calls,
            "tool_calls_per_request": round(tool_calls / len(obs), 3),
            "cost": {
                "total_usd": round(total_cost, 6),
                "mean_usd": round(total_cost / len(obs), 8),
                "tokens": total_tokens,
                "unpriced_calls": unpriced,
                "priced": unpriced == 0,
            },
            "by_route": {
                route: sum(1 for o in obs if o.route == route)
                for route in sorted({o.route for o in obs})
            },
        }


_REGISTRY = Registry()
_UI_REGISTRY = Registry()

# UI observations are separate from service observations. Persist them so the
# UI dashboard survives a browser/process reload without contaminating the
# service's own /metrics counters.
_UI_TELEMETRY_FILE = Path(_aip_settings.trace_dir) / "ui-telemetry.jsonl"
_UI_TELEMETRY_LOCK = threading.Lock()
_UI_TELEMETRY_MAX_BYTES = 2 * 1024 * 1024


def _obs_to_dict(observation: Observation) -> dict[str, Any]:
    return {
        "trace_id": observation.trace_id,
        "route": observation.route,
        "latency_ms": observation.latency_ms,
        "latency_by_stage": observation.latency_by_stage or {},
        "cost": dict(observation.cost.as_dict()) if hasattr(observation.cost, "as_dict") else {},
        "cache_layer": observation.cache_layer,
        "cache_similarity": observation.cache_similarity,
        "refused": observation.refused,
        "ttft_ms": observation.ttft_ms,
        "tool_calls": observation.tool_calls,
    }


def _dict_to_obs(data: dict[str, Any]) -> Observation:
    cost_data = data.get("cost") or {}
    try:
        delta = CostDelta(**{k: cost_data.get(k, 0) for k in
                             ("calls", "cached_calls", "prompt_tokens",
                              "completion_tokens", "cost_usd", "unpriced_calls")})
        delta.priced = cost_data.get("priced", True)
        delta.models = list(cost_data.get("models") or [])
    except Exception:
        delta = CostDelta()
    return Observation(
        trace_id=str(data.get("trace_id") or ""),
        route=str(data.get("route") or ""),
        latency_ms=float(data.get("latency_ms") or 0.0),
        latency_by_stage=dict(data.get("latency_by_stage") or {}),
        cost=delta,
        cache_layer=str(data.get("cache_layer") or "none"),
        cache_similarity=(None if data.get("cache_similarity") is None
                          else float(data.get("cache_similarity"))),
        refused=bool(data.get("refused")),
        ttft_ms=(None if data.get("ttft_ms") is None
                 else float(data.get("ttft_ms"))),
        tool_calls=int(data.get("tool_calls") or 0),
    )


def clear_ui_telemetry() -> None:
    """Clear persisted and in-memory UI metrics without touching service data."""
    with _UI_TELEMETRY_LOCK:
        with contextlib.suppress(FileNotFoundError):
            _UI_TELEMETRY_FILE.unlink()
        _UI_REGISTRY.reset()


def _restore_ui_telemetry() -> None:
    """Load persisted UI observations into the bounded UI registry."""
    try:
        if _UI_TELEMETRY_FILE.exists():
            recent: deque[Observation] = deque(maxlen=WINDOW)
            with _UI_TELEMETRY_FILE.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        with contextlib.suppress(Exception):
                            recent.append(_dict_to_obs(json.loads(line)))
            for observation in recent:
                _UI_REGISTRY.record(observation)
    except Exception:
        pass


_restore_ui_telemetry()


def registry() -> Registry:
    return _REGISTRY


def observe(observation: Observation) -> None:
    """Record a service observation in this process only."""
    _REGISTRY.record(observation)


def observe_ui(observation: Observation) -> None:
    """Record and persist a UI request for the UI dashboard."""
    _UI_REGISTRY.record(observation)
    with contextlib.suppress(Exception), _UI_TELEMETRY_LOCK:
        _UI_TELEMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _UI_TELEMETRY_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_obs_to_dict(observation), default=str) + "\n")
        if _UI_TELEMETRY_FILE.stat().st_size > _UI_TELEMETRY_MAX_BYTES:
            recent: deque[str] = deque(maxlen=WINDOW)
            with _UI_TELEMETRY_FILE.open(encoding="utf-8") as fh:
                recent.extend(line for line in fh if line.strip())
            with _UI_TELEMETRY_FILE.open("w", encoding="utf-8") as fh:
                fh.writelines(recent)


def ui_snapshot() -> dict[str, Any]:
    """Metrics for UI-originated requests, including persisted history."""
    return _UI_REGISTRY.snapshot()


def snapshot() -> dict[str, Any]:
    return _REGISTRY.snapshot()
