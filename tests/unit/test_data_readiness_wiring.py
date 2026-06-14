"""test_data_readiness_wiring.py -- Monzia Moodie
DataReadinessAgent orchestrator wiring: a 'data_readiness' pipeline exists with the agent, the agent is in
'full' and the auto-derived 'all' superset and is registered, existing pipelines are intact, and the agent
constructs + dry-runs to a verdict. SAFE membership checks only -- NO iterate-PIPELINE_DEFINITIONS-with-exclusion.
"""
import pytest

from genomic_variant_classifier.agent_layer.orchestrator import PIPELINE_DEFINITIONS

AGENT = "DataReadinessAgent"


def test_data_readiness_pipeline_defined():
    assert "data_readiness" in PIPELINE_DEFINITIONS, "no 'data_readiness' pipeline -- agent unreachable"
    assert PIPELINE_DEFINITIONS["data_readiness"] == [AGENT]
    assert AGENT in PIPELINE_DEFINITIONS["full"], "agent not in 'full' -> not active by default"
    assert AGENT in PIPELINE_DEFINITIONS["all"], "auto-derived 'all' superset must include it"
    assert PIPELINE_DEFINITIONS["model_insights"] == ["ModelInsightsAgent"]   # neighbour intact
    assert len(PIPELINE_DEFINITIONS["drift"]) == len(set(PIPELINE_DEFINITIONS["drift"]))


def _orchestrator(tmp_path):
    try:
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        return Orchestrator(SharedState(state_file=tmp_path / "state.json"), dry_run=True)
    except ImportError as e:
        pytest.skip(f"orchestrator construction needs an unavailable dependency: {e}")


def test_agent_registered_and_constructs(tmp_path):
    orch = _orchestrator(tmp_path)
    assert AGENT in orch._agent_registry
    inst = orch._agent_registry[AGENT](orch._state)
    assert inst is not None


def test_dry_run_returns_a_verdict(tmp_path):
    # under a tmp root with no data/ -> assets MISSING -> NO_GO, but dry-run must not raise or gate
    from genomic_variant_classifier.agent_layer.agents.data_readiness_agent import DataReadinessAgent
    from genomic_variant_classifier.agent_layer.shared_state import SharedState
    ss = SharedState(state_file=tmp_path / "state.json")
    res = DataReadinessAgent(ss, root=str(tmp_path)).run(dry_run=True)
    assert res["action"] == "data_readiness_gate" and res["verdict"] in ("GO", "GO_WITH_WARNINGS", "NO_GO")
