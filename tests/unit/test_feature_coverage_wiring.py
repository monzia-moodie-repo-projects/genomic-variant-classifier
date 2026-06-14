"""test_feature_coverage_wiring.py  --  Monzia Moodie

FeatureCoverageSentinel orchestrator wiring: the 'drift' pipeline exists with all nine drift
agents (incl. the sentinel), the sentinel is registered and routed via from_default_baseline,
and run_pipeline('drift') runs it end-to-end. The Orchestrator-construction tests skip cleanly
where a heavy agent dependency is unavailable (they run on the training box).
"""
import pytest

from genomic_variant_classifier.agent_layer.orchestrator import PIPELINE_DEFINITIONS

EXPECTED_DRIFT = {
    "SchemaDriftMonitorAgent", "ConceptDriftMonitorAgent", "LabelShiftMonitorAgent",
    "CalibrationDriftMonitorAgent", "InfrastructureDriftMonitorAgent", "FairnessSubgroupMonitorAgent",
    "AdversarialSubmissionMonitorAgent", "AnnotationPolicyMonitorAgent",
    "FeatureCoverageSentinelMonitorAgent",
}


def test_drift_pipeline_defined():
    assert "drift" in PIPELINE_DEFINITIONS, "no 'drift' pipeline -- drift agents unreachable via run_agents"
    assert set(PIPELINE_DEFINITIONS["drift"]) == EXPECTED_DRIFT
    assert len(PIPELINE_DEFINITIONS["drift"]) == 9
    # existing pipelines untouched
    assert PIPELINE_DEFINITIONS["data_freshness"] == ["DataFreshnessAgent"]


def _orchestrator(tmp_path):
    try:
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        return Orchestrator(SharedState(state_file=tmp_path / "state.json"), dry_run=True)
    except ImportError as e:  # a heavy agent dep (e.g. torch) is not installed in this env
        pytest.skip(f"orchestrator construction needs an unavailable dependency: {e}")


def test_sentinel_registered_and_routable(tmp_path):
    orch = _orchestrator(tmp_path)
    assert "FeatureCoverageSentinelMonitorAgent" in orch._agent_registry
    cls = orch._agent_registry["FeatureCoverageSentinelMonitorAgent"]
    assert hasattr(cls, "from_default_baseline"), "sentinel must expose from_default_baseline for the orchestrator hook"


def test_drift_pipeline_runs(tmp_path):
    orch = _orchestrator(tmp_path)
    results = orch.run_pipeline("drift")
    assert "FeatureCoverageSentinelMonitorAgent" in results
    status = results["FeatureCoverageSentinelMonitorAgent"]["status"]
    # no GVC_FEATURE_MATRIX / no reference in a fresh checkout -> awaiting_baseline; active -> ok
    assert status in ("awaiting_baseline", "ok"), status
    assert orch._state.get_section("feature_coverage"), "feature_coverage section not persisted"
