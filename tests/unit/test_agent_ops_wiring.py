"""test_agent_ops_wiring.py -- Monzia Moodie
AgentOpsMonitorAgent orchestrator wiring: 'agent_ops' pipeline exists with the agent, agent in 'full' + the
auto-derived 'all', registered + constructs + dry-runs. SAFE membership checks only (no iterate-with-exclusion).
"""
import pytest

from genomic_variant_classifier.agent_layer.orchestrator import PIPELINE_DEFINITIONS

AGENT = "AgentOpsMonitorAgent"


def test_agent_ops_pipeline_defined():
    assert PIPELINE_DEFINITIONS["agent_ops"] == [AGENT]
    assert AGENT in PIPELINE_DEFINITIONS["full"] and AGENT in PIPELINE_DEFINITIONS["all"]
    assert PIPELINE_DEFINITIONS["data_readiness"] == ["DataReadinessAgent"]   # neighbour intact
    assert len(PIPELINE_DEFINITIONS["drift"]) == len(set(PIPELINE_DEFINITIONS["drift"]))


def _orchestrator(tmp_path):
    try:
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        return Orchestrator(SharedState(state_file=tmp_path / "state.json"), dry_run=True)
    except ImportError as e:
        pytest.skip(f"orchestrator needs an unavailable dependency: {e}")


def test_registered_and_constructs(tmp_path):
    orch = _orchestrator(tmp_path)
    assert AGENT in orch._agent_registry
    assert orch._agent_registry[AGENT](orch._state) is not None


def test_dry_run_returns_status(tmp_path):
    from genomic_variant_classifier.agent_layer.agents.agent_ops_monitor_agent import AgentOpsMonitorAgent
    from genomic_variant_classifier.agent_layer.shared_state import SharedState
    ss = SharedState(state_file=tmp_path / "state.json")
    res = AgentOpsMonitorAgent(ss, root=str(tmp_path)).run(dry_run=True)
    assert res["action"] == "agent_ops_scan" and res["ops_status"] in ("OK", "ATTENTION")
