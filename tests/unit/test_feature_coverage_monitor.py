"""test_feature_coverage_monitor.py  --  Monzia Moodie

FeatureCoverageSentinelMonitorAgent.from_default_baseline: load the reference, resolve the
current matrix (arg / GVC_FEATURE_MATRIX env), and report green/red/awaiting through
DriftMonitorBase.run with a JSON-serialisable summary.
"""
import json

import numpy as np
import pandas as pd

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.feature_coverage_sentinel_monitor_agent import (
    FeatureCoverageSentinelMonitorAgent,
)

REF = {"reference": {"feat_a": "healthy", "feat_b": "healthy", "dead_x": "ALL_ZERO"},
       "near_constant_frac": 0.999}


def _ref_json(tmp_path):
    p = tmp_path / "ref.json"; p.write_text(json.dumps(REF), encoding="utf-8"); return p


def _matrix(**over):
    base = {"feat_a": np.linspace(0, 1, 40), "feat_b": np.r_[np.zeros(20), np.ones(20)],
            "dead_x": np.zeros(40)}
    base.update(over); return pd.DataFrame(base)


def test_green(tmp_path):
    a = FeatureCoverageSentinelMonitorAgent.from_default_baseline(
        SharedState(state_file=tmp_path / "s.json"), current_matrix=_matrix(),
        reference_path=_ref_json(tmp_path), output_dir=tmp_path)
    out = a.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "green"
    json.dumps(out)  # serialisable


def test_red_regression(tmp_path):
    a = FeatureCoverageSentinelMonitorAgent.from_default_baseline(
        SharedState(state_file=tmp_path / "s.json"), current_matrix=_matrix(feat_a=np.zeros(40)),
        reference_path=_ref_json(tmp_path), output_dir=tmp_path)
    out = a.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "red"
    assert out["n_regressed"] == 1 and out["regressed"][0][0] == "feat_a"


def test_awaiting_no_matrix(tmp_path):
    a = FeatureCoverageSentinelMonitorAgent.from_default_baseline(
        SharedState(state_file=tmp_path / "s.json"),
        reference_path=_ref_json(tmp_path), output_dir=tmp_path)
    assert a.run(dry_run=True)["status"] == "awaiting_baseline"


def test_awaiting_no_reference(tmp_path):
    a = FeatureCoverageSentinelMonitorAgent.from_default_baseline(
        SharedState(state_file=tmp_path / "s.json"), current_matrix=_matrix(),
        reference_path=tmp_path / "absent.json", output_dir=tmp_path)
    assert a.run(dry_run=True)["status"] == "awaiting_baseline"


def test_env_matrix(tmp_path, monkeypatch):
    mp = tmp_path / "m.parquet"; _matrix(feat_a=np.zeros(40)).to_parquet(mp)
    monkeypatch.setenv("GVC_FEATURE_MATRIX", str(mp))
    a = FeatureCoverageSentinelMonitorAgent.from_default_baseline(
        SharedState(state_file=tmp_path / "s.json"),
        reference_path=_ref_json(tmp_path), output_dir=tmp_path)
    out = a.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "red"
