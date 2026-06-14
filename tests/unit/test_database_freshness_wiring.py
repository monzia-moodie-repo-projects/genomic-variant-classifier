"""test_database_freshness_wiring.py -- Monzia Moodie

DatabaseFreshnessMonitorAgent orchestrator wiring: a 'database_monitor' pipeline exists with the agent, the
agent is in 'full' and registered, and existing pipelines are untouched. Construction is guarded so it skips
where a heavy agent dependency is unavailable (mirrors the other wiring tests).
"""
import pytest

from genomic_variant_classifier.agent_layer.orchestrator import PIPELINE_DEFINITIONS

AGENT = "DatabaseFreshnessMonitorAgent"


def test_database_monitor_pipeline_defined():
    assert "database_monitor" in PIPELINE_DEFINITIONS, "no 'database_monitor' pipeline -- agent unreachable"
    assert PIPELINE_DEFINITIONS["database_monitor"] == [AGENT]
    assert AGENT in PIPELINE_DEFINITIONS["full"], "agent not in 'full' -> not active by default"
    # existing pipelines intact (catches accidental edits)
    assert PIPELINE_DEFINITIONS["data_freshness"] == ["DataFreshnessAgent"]
    assert len(PIPELINE_DEFINITIONS["drift"]) == len(set(PIPELINE_DEFINITIONS["drift"]))


def _orchestrator(tmp_path):
    try:
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        return Orchestrator(SharedState(state_file=tmp_path / "state.json"), dry_run=True)
    except ImportError as e:
        pytest.skip(f"orchestrator construction needs an unavailable dependency: {e}")


def test_agent_registered_and_constructible(tmp_path):
    orch = _orchestrator(tmp_path)
    reg = orch._agent_registry
    assert AGENT in reg, "agent not registered in _register_agents"
    # constructs as cls(shared_state) like every BaseAgent framework agent
    inst = reg[AGENT](orch._state)
    assert inst is not None
