"""test_model_insights_wiring.py -- Monzia Moodie
ModelInsightsAgent orchestrator wiring: a 'model_insights' pipeline exists with the agent, the agent is in
'full' and the auto-derived 'all' superset and is registered, existing pipelines are intact, and the agent
constructs + dry-runs to 'skipped' when no run artifacts exist. SAFE membership checks only -- NO
iterate-PIPELINE_DEFINITIONS-with-exclusion (that pattern is what broke CI in bbb9d5c).
"""
import pytest

from genomic_variant_classifier.agent_layer.orchestrator import PIPELINE_DEFINITIONS

AGENT = "ModelInsightsAgent"


def test_model_insights_pipeline_defined():
    assert "model_insights" in PIPELINE_DEFINITIONS, "no 'model_insights' pipeline -- agent unreachable"
    assert PIPELINE_DEFINITIONS["model_insights"] == [AGENT]
    assert AGENT in PIPELINE_DEFINITIONS["full"], "agent not in 'full' -> not active by default"
    assert AGENT in PIPELINE_DEFINITIONS["all"], "auto-derived 'all' superset must include it"
    # existing pipelines intact (catches accidental edits)
    assert PIPELINE_DEFINITIONS["interpretability"] == ["InterpretabilityAgent"]
    assert PIPELINE_DEFINITIONS["database_monitor"] == ["DatabaseFreshnessMonitorAgent"]
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
    reg = orch._agent_registry
    assert AGENT in reg, "agent not registered in _register_agents"
    inst = reg[AGENT](orch._state)        # constructs as cls(shared_state) like every framework agent
    assert inst is not None


def test_dry_run_skips_with_no_artifacts(tmp_path):
    from genomic_variant_classifier.agent_layer.agents.model_insights_agent import ModelInsightsAgent
    from genomic_variant_classifier.agent_layer.shared_state import SharedState
    ss = SharedState(state_file=tmp_path / "state.json")
    agent = ModelInsightsAgent(ss, outputs_root=str(tmp_path / "no_outputs"), root=str(tmp_path))
    res = agent.run(dry_run=True)
    assert res["action"] == "skipped" and res["reason"] == "no_run_artifacts"
