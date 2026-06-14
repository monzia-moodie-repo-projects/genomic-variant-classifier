#!/usr/bin/env python3
"""test_reclassification_wiring.py -- ReclassificationSentinel orchestrator wiring (Monzia Moodie)."""
import json
from pathlib import Path
from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator, PIPELINE_DEFINITIONS
from genomic_variant_classifier.agent_layer.shared_state import SharedState

EXPECTED_DRIFT = [
    "SchemaDriftMonitorAgent", "ConceptDriftMonitorAgent", "LabelShiftMonitorAgent",
    "CalibrationDriftMonitorAgent", "InfrastructureDriftMonitorAgent", "FairnessSubgroupMonitorAgent",
    "AdversarialSubmissionMonitorAgent", "AnnotationPolicyMonitorAgent",
    "FeatureCoverageSentinelMonitorAgent", "ReclassificationSentinelMonitorAgent",
]


def test_drift_pipeline_lists_ten_agents():
    assert PIPELINE_DEFINITIONS["drift"] == EXPECTED_DRIFT
    for name, agents in PIPELINE_DEFINITIONS.items():
        if name != "drift":
            assert "ReclassificationSentinelMonitorAgent" not in agents


def test_reclassification_registered_and_hookable(tmp_path):
    orch = Orchestrator(SharedState(state_file=str(tmp_path / "st.json")), dry_run=True)
    assert "ReclassificationSentinelMonitorAgent" in orch._agent_registry
    cls = orch._agent_registry["ReclassificationSentinelMonitorAgent"]
    assert hasattr(cls, "from_default_baseline")


def test_run_pipeline_drift_runs_reclassification(tmp_path):
    sf = tmp_path / "st.json"
    orch = Orchestrator(SharedState(state_file=str(sf)), dry_run=False)
    orch.run_pipeline("drift")  # must not raise; reclassification reports awaiting_baseline (no reference)
    if sf.exists():
        assert "reclassification" in sf.read_text()
