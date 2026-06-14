"""test_finops_wiring.py -- Monzia Moodie
FinOpsAdvisorAgent orchestrator wiring: 'finops' pipeline exists with the agent, agent in 'full' + auto-derived
'all', registered + constructs + dry-runs (skipped without a snapshot). SAFE membership checks only.
"""
import pytest

from genomic_variant_classifier.agent_layer.orchestrator import PIPELINE_DEFINITIONS

AGENT = "FinOpsAdvisorAgent"


def test_finops_pipeline_defined():
    assert PIPELINE_DEFINITIONS["finops"] == [AGENT]
    assert AGENT in PIPELINE_DEFINITIONS["full"] and AGENT in PIPELINE_DEFINITIONS["all"]
    assert PIPELINE_DEFINITIONS["agent_ops"] == ["AgentOpsMonitorAgent"]      # neighbour intact
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


def test_dry_run_skips_without_snapshot(tmp_path):
    from genomic_variant_classifier.agent_layer.agents.finops_advisor_agent import FinOpsAdvisorAgent
    from genomic_variant_classifier.agent_layer.shared_state import SharedState
    ss = SharedState(state_file=tmp_path / "state.json")
    res = FinOpsAdvisorAgent(ss, snapshot_path=None, root=str(tmp_path)).run(dry_run=True)
    assert res["action"] == "skipped"                                        # no snapshot -> no live call, no spend
