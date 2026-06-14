"""test_adversarial_submission_baseline.py  --  Monzia Moodie

AdversarialSubmissionMonitorAgent.from_default_baseline: detector always constructs from
literature-derived thresholds (model-free, no data baseline); optional thresholds JSON
overrides them; the four run-time DataFrame inputs move awaiting_baseline -> active detection.
"""
import json

import pandas as pd
import pytest

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.adversarial_submission_monitor_agent import (
    AdversarialSubmissionMonitorAgent,
)


def _state(tmp_path):
    return SharedState(state_file=tmp_path / "state.json")


def _clean():
    weekly = pd.DataFrame({"submitter_id": ["A", "A", "A"], "variant_id": ["v1", "v2", "v3"],
                           "classification": ["Pathogenic", "Benign", "Pathogenic"]})
    base = pd.DataFrame({"submitter_id": ["A"], "median_24h": [3.0]})
    agg = pd.DataFrame({"variant_id": ["v1", "v2", "v3"],
                        "classification_agg": ["Pathogenic", "Benign", "Pathogenic"]})
    meta = pd.DataFrame({"submitter_id": ["A"], "age_days": [1000]})
    return weekly, base, agg, meta


def test_active_green(tmp_path):
    w, b, a, m = _clean()
    agent = AdversarialSubmissionMonitorAgent.from_default_baseline(
        _state(tmp_path), weekly_submissions=w, submitter_baseline=b,
        aggregate_classifications=a, submitter_metadata=m,
        thresholds_path=tmp_path / "none.json", output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "green" and out["n_findings"] == 0


def test_red_on_bulk_and_flip(tmp_path):
    # B: 600 pathogenic submissions (>500 floor, base 0 -> R1 bulk), agg says benign (R4 flip)
    w = pd.DataFrame({"submitter_id": ["B"] * 600, "variant_id": [f"v{i}" for i in range(600)],
                      "classification": ["Pathogenic"] * 600})
    b = pd.DataFrame({"submitter_id": ["A"], "median_24h": [3.0]})
    a = pd.DataFrame({"variant_id": [f"v{i}" for i in range(600)], "classification_agg": ["Benign"] * 600})
    m = pd.DataFrame({"submitter_id": ["B"], "age_days": [1000]})
    agent = AdversarialSubmissionMonitorAgent.from_default_baseline(
        _state(tmp_path), weekly_submissions=w, submitter_baseline=b,
        aggregate_classifications=a, submitter_metadata=m,
        thresholds_path=tmp_path / "none.json", output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "red"
    assert "B" in out["quarantine_submitter_ids"] and out["n_findings"] >= 1


def test_thresholds_override_relaxes_bulk(tmp_path):
    # Raise the bulk floor above the spike so R1 no longer fires; clean agg -> green.
    tp = tmp_path / "thresholds.json"
    tp.write_text(json.dumps({"bulk_absolute_floor": 100000}), encoding="utf-8")
    w = pd.DataFrame({"submitter_id": ["B"] * 600, "variant_id": [f"v{i}" for i in range(600)],
                      "classification": ["Pathogenic"] * 600})
    b = pd.DataFrame({"submitter_id": ["B"], "median_24h": [10.0]})
    a = pd.DataFrame({"variant_id": [f"v{i}" for i in range(600)], "classification_agg": ["Pathogenic"] * 600})
    m = pd.DataFrame({"submitter_id": ["B"], "age_days": [1000]})
    agent = AdversarialSubmissionMonitorAgent.from_default_baseline(
        _state(tmp_path), weekly_submissions=w, submitter_baseline=b,
        aggregate_classifications=a, submitter_metadata=m,
        thresholds_path=tp, output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["severity"] == "green"  # bulk suppressed by the override


def test_awaiting_when_input_missing(tmp_path):
    w, b, a, _ = _clean()
    agent = AdversarialSubmissionMonitorAgent.from_default_baseline(
        _state(tmp_path), weekly_submissions=w, submitter_baseline=b,
        aggregate_classifications=a,  # submitter_metadata omitted -> None
        thresholds_path=tmp_path / "none.json", output_dir=tmp_path / "rep",
    )
    assert agent.run(dry_run=True)["status"] == "awaiting_baseline"
