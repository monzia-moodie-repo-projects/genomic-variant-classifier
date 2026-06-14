"""test_annotation_policy_baseline.py  --  Monzia Moodie

AnnotationPolicyMonitorAgent.from_default_baseline: detector always constructs from
literature-derived thresholds (model-free, no data baseline); optional thresholds JSON
overrides them; the four run-time inputs move awaiting_baseline -> active detection.
"""
import json

import pandas as pd
import pytest

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.annotation_policy_monitor_agent import (
    AnnotationPolicyMonitorAgent,
)

EMPTY_HIST = pd.DataFrame(columns=["submitter_id", "date", "outlier_rate"])


def _state(tmp_path):
    return SharedState(state_file=tmp_path / "state.json")


def test_active_green(tmp_path):
    agent = AnnotationPolicyMonitorAgent.from_default_baseline(
        _state(tmp_path), new_svi_pubs=[], clinvar_status_changes=pd.DataFrame({"variant_id": []}),
        submitter_history=EMPTY_HIST, n_inference_variants=1000,
        thresholds_path=tmp_path / "none.json", output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "green"


def test_red_on_new_svi_publication(tmp_path):
    agent = AnnotationPolicyMonitorAgent.from_default_baseline(
        _state(tmp_path), new_svi_pubs=["PMID:123"], clinvar_status_changes=pd.DataFrame({"variant_id": []}),
        submitter_history=EMPTY_HIST, n_inference_variants=1000,
        thresholds_path=tmp_path / "none.json", output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "red"
    assert out["requires_variant_scientist_review"] is True
    assert "PMID:123" in out["new_svi_publications"]


def test_red_on_status_change_rate(tmp_path):
    chg = pd.DataFrame({"variant_id": range(15)})  # 15/1000 = 1.5% >= red 1%
    agent = AnnotationPolicyMonitorAgent.from_default_baseline(
        _state(tmp_path), new_svi_pubs=[], clinvar_status_changes=chg,
        submitter_history=EMPTY_HIST, n_inference_variants=1000,
        thresholds_path=tmp_path / "none.json", output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "red"


def test_thresholds_override(tmp_path):
    # Lower the red band so 0.6% (6/1000) trips red.
    tp = tmp_path / "thresholds.json"
    tp.write_text(json.dumps({"review_status_red": 0.005, "review_status_amber": 0.001}), encoding="utf-8")
    chg = pd.DataFrame({"variant_id": range(6)})  # 0.6%
    agent = AnnotationPolicyMonitorAgent.from_default_baseline(
        _state(tmp_path), new_svi_pubs=[], clinvar_status_changes=chg,
        submitter_history=EMPTY_HIST, n_inference_variants=1000,
        thresholds_path=tp, output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["severity"] == "red"  # would be green under default 1% red band


def test_awaiting_when_input_missing(tmp_path):
    agent = AnnotationPolicyMonitorAgent.from_default_baseline(
        _state(tmp_path), new_svi_pubs=[], clinvar_status_changes=pd.DataFrame({"variant_id": []}),
        submitter_history=EMPTY_HIST,  # n_inference_variants omitted -> None
        thresholds_path=tmp_path / "none.json", output_dir=tmp_path / "rep",
    )
    assert agent.run(dry_run=True)["status"] == "awaiting_baseline"


def test_submitter_scan_runs_with_river(tmp_path):
    pytest.importorskip("river")
    hist = pd.DataFrame({"submitter_id": ["S1"] * 40,
                         "date": pd.date_range("2026-01-01", periods=40, freq="D"),
                         "outlier_rate": [0.0] * 5 + [1.0] * 35})
    agent = AnnotationPolicyMonitorAgent.from_default_baseline(
        _state(tmp_path), new_svi_pubs=[], clinvar_status_changes=pd.DataFrame({"variant_id": []}),
        submitter_history=hist, n_inference_variants=1000,
        thresholds_path=tmp_path / "none.json", output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] in ("green", "amber", "red")
