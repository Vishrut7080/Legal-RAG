from __future__ import annotations

import json

from src import telemetry


def test_ui_observations_persist_separately_from_service_metrics(tmp_path, monkeypatch):
    path = tmp_path / "ui-telemetry.jsonl"
    monkeypatch.setattr(telemetry, "_UI_TELEMETRY_FILE", path)
    monkeypatch.setattr(telemetry, "_UI_REGISTRY", telemetry.Registry())
    monkeypatch.setattr(telemetry, "_REGISTRY", telemetry.Registry())

    telemetry.observe_ui(telemetry.Observation(
        trace_id="ui-1",
        route="/ui",
        latency_ms=125.0,
        latency_by_stage={"retrieval": 100.0, "generation": 25.0},
    ))
    telemetry.observe(telemetry.Observation(
        trace_id="api-1", route="/ask", latency_ms=40.0,
    ))

    assert telemetry.ui_snapshot()["requests"] == 1
    assert telemetry.snapshot()["requests"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))[
        "trace_id"
    ] == "ui-1"

    # Simulate a fresh UI process loading its persisted observations.
    monkeypatch.setattr(telemetry, "_UI_REGISTRY", telemetry.Registry())
    telemetry._restore_ui_telemetry()
    restored = telemetry.ui_snapshot()
    assert restored["requests"] == 1
    assert restored["latency_ms"]["mean"] == 125.0
    assert telemetry.snapshot()["requests"] == 1

    telemetry.clear_ui_telemetry()
    assert telemetry.ui_snapshot()["requests"] == 0
    assert telemetry.snapshot()["requests"] == 1
    assert not path.exists()
