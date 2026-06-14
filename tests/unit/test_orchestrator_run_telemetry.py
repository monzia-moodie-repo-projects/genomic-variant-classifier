"""test_orchestrator_run_telemetry.py -- Monzia Moodie
The orchestrator records per-agent run telemetry to 'agent_runs' (real runs only): ok/error records with
duration, capped at 50 per agent, never raises on a bad value, and dry-run pipelines record nothing.
"""
import pytest

from genomic_variant_classifier.agent_layer.shared_state import SharedState


def _orch(tmp_path, dry_run):
    try:
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
    except ImportError as e:
        pytest.skip(f"orchestrator import needs an unavailable dependency: {e}")
    return Orchestrator(SharedState(state_file=tmp_path / "s.json"), dry_run=dry_run)


def test_record_ok_and_error(tmp_path):
    orch = _orch(tmp_path, dry_run=False)
    orch._record_run_telemetry("A", {"action": "scan"}, 12.5, None)
    orch._record_run_telemetry("A", {"action": "error", "error": "boom"}, 5.0, "boom")
    runs = orch._state.get_section("agent_runs")["A"]
    assert len(runs) == 2
    assert runs[0]["status"] == "scan" and runs[0]["duration_ms"] == 12.5
    assert runs[1]["status"] == "error" and runs[1]["error"] == "boom"


def test_telemetry_capped_at_50(tmp_path):
    orch = _orch(tmp_path, dry_run=False)
    for i in range(60):
        orch._record_run_telemetry("A", {"action": "ok"}, float(i), None)
    assert len(orch._state.get_section("agent_runs")["A"]) == 50


def test_telemetry_never_raises_on_bad_duration(tmp_path):
    orch = _orch(tmp_path, dry_run=False)
    orch._record_run_telemetry("A", {"action": "ok"}, "not-a-number", None)   # must not raise
    assert orch._state.get_section("agent_runs").get("A", []) == []            # nothing recorded, no crash


def test_dry_run_pipeline_records_nothing(tmp_path):
    orch = _orch(tmp_path, dry_run=True)
    orch.run_pipeline("data_freshness")
    assert orch._state.get_section("agent_runs") == {}
