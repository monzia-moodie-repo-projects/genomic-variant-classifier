"""Smoke tests: every drift-monitor wrapper instantiates with shared_state alone,
runs, and reports awaiting_baseline (wired but inactive) without crashing."""
from __future__ import annotations
import importlib
import pytest
from genomic_variant_classifier.agent_layer.shared_state import SharedState

CASES = [
    ("schema_drift_monitor_agent", "SchemaDriftMonitorAgent", "schema_drift"),
    ("concept_drift_monitor_agent", "ConceptDriftMonitorAgent", "concept_drift"),
    ("label_shift_monitor_agent", "LabelShiftMonitorAgent", "label_shift"),
    ("calibration_drift_monitor_agent", "CalibrationDriftMonitorAgent", "calibration_drift"),
    ("infrastructure_drift_monitor_agent", "InfrastructureDriftMonitorAgent", "infrastructure_drift"),
    ("fairness_subgroup_monitor_agent", "FairnessSubgroupMonitorAgent", "fairness_subgroup"),
    ("adversarial_submission_monitor_agent", "AdversarialSubmissionMonitorAgent", "adversarial_submission"),
    ("annotation_policy_monitor_agent", "AnnotationPolicyMonitorAgent", "annotation_policy"),
]


@pytest.mark.parametrize("module,cls,section", CASES)
def test_awaiting_baseline(module, cls, section):
    mod = importlib.import_module(f"genomic_variant_classifier.agent_layer.agents.{module}")
    agent = getattr(mod, cls)(SharedState())
    result = agent.run(dry_run=True)
    assert result["status"] == "awaiting_baseline"
    assert agent.section == section
    assert agent._get_section(section)["status"] == "awaiting_baseline"
