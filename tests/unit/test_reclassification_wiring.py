#!/usr/bin/env python3
"""test_reclassification_wiring.py -- ReclassificationSentinel orchestrator wiring (Monzia Moodie).

Robust membership checks (the known drift agents must all be present -> catches drops; new agents may be
appended -> tolerant of additions). The full-pipeline-run test skips where pandera is absent (CI), matching
the schema-drift tests: SchemaDriftMonitorAgent.from_default_baseline lazily imports pandera inside
run_pipeline('drift'), and pandera is an optional dep.
"""
import pytest

from genomic_variant_classifier.agent_layer.orchestrator import PIPELINE_DEFINITIONS

EXPECTED_DRIFT = {
    "SchemaDriftMonitorAgent", "ConceptDriftMonitorAgent", "LabelShiftMonitorAgent",
    "CalibrationDriftMonitorAgent", "InfrastructureDriftMonitorAgent", "FairnessSubgroupMonitorAgent",
    "AdversarialSubmissionMonitorAgent", "AnnotationPolicyMonitorAgent",
    "FeatureCoverageSentinelMonitorAgent", "ReclassificationSentinelMonitorAgent",
}


def test_drift_pipeline_includes_reclassification():
    drift = PIPELINE_DEFINITIONS["drift"]
    assert "ReclassificationSentinelMonitorAgent" in drift, "reclass sentinel not wired into the drift pipeline"
    assert EXPECTED_DRIFT <= set(drift), f"missing drift agents: {EXPECTED_DRIFT - set(drift)}"
    assert len(drift) == len(set(drift)), "duplicate drift agents in the pipeline"
    # reclass sentinel must not leak into any other pipeline -- except the deliberate auto-derived
    # 'all' superset (every agent appearing in any pipeline), which is not a hand-curated cadence pipeline.
    for name, agents in PIPELINE_DEFINITIONS.items():
        if name not in ("drift", "all"):
            assert "ReclassificationSentinelMonitorAgent" not in agents


def _orchestrator(tmp_path, dry_run=True):
    try:
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        return Orchestrator(SharedState(state_file=str(tmp_path / "st.json")), dry_run=dry_run)
    except ImportError as e:  # a heavy agent dep (e.g. torch) is unavailable in this env
        pytest.skip(f"orchestrator construction needs an unavailable dependency: {e}")


def test_reclassification_registered_and_hookable(tmp_path):
    orch = _orchestrator(tmp_path)
    assert "ReclassificationSentinelMonitorAgent" in orch._agent_registry
    cls = orch._agent_registry["ReclassificationSentinelMonitorAgent"]
    assert hasattr(cls, "from_default_baseline"), "agent must expose from_default_baseline for the orchestrator hook"


def test_run_pipeline_drift_runs_reclassification(tmp_path):
    pytest.importorskip("pandera")  # SchemaDriftMonitorAgent's lazy pandera import fires inside run_pipeline (optional dep)
    sf = tmp_path / "st.json"
    orch = _orchestrator(tmp_path, dry_run=False)
    orch.run_pipeline("drift")  # must not raise; reclassification reports awaiting_baseline (no reference)
    if sf.exists():
        assert "reclassification" in sf.read_text()
