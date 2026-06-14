"""test_feature_coverage_sentinel.py  --  Monzia Moodie

FeatureCoverageSentinelAgent: scores a current feature matrix against a reference health
verdict (from audit_split_feature_health.py) to catch silent feature regressions. Detector
only (the reference builder + monitor adapter + wiring land in the next step); references
are constructed directly here.
"""
import json

import numpy as np
import pandas as pd

from genomic_variant_classifier.agent_layer.agents.feature_coverage_sentinel_agent import (
    FeatureCoverageSentinelAgent,
)

REF = {"variant_id": "healthy", "feat_a": "healthy", "feat_b": "healthy",
       "dead_x": "ALL_ZERO", "const_y": "CONSTANT"}


def _matrix(**over):
    base = {"variant_id": [f"v{i}" for i in range(50)],
            "feat_a": np.linspace(0, 1, 50),
            "feat_b": np.r_[np.zeros(25), np.ones(25)],
            "dead_x": np.zeros(50),
            "const_y": np.full(50, 3.0)}
    base.update(over)
    return pd.DataFrame(base)


def _agent(reference=REF, **kw):
    return FeatureCoverageSentinelAgent(reference=dict(reference), output_dir=kw.pop("output_dir", "/tmp"), **kw)


def test_green_on_baseline_match():
    r = _agent().detect(_matrix())
    assert r.severity == "green"
    assert not r.regressed and not r.dropped
    assert set(r.still_degenerate) == {"dead_x", "const_y"}


def test_red_on_regression():
    r = _agent().detect(_matrix(feat_a=np.zeros(50)))
    assert r.severity == "red"
    assert r.regressed and r.regressed[0][0] == "feat_a" and "ALL_ZERO" in r.regressed[0][1]


def test_red_on_dropped_column():
    r = _agent().detect(_matrix().drop(columns=["const_y"]))
    assert r.severity == "red" and "const_y" in r.dropped


def test_recovered_is_green():
    r = _agent().detect(_matrix(dead_x=np.linspace(1, 2, 50)))
    assert r.severity == "green" and "dead_x" in r.recovered


def test_amber_on_new_column():
    r = _agent().detect(_matrix(brand_new=np.linspace(0, 1, 50)))
    assert r.severity == "amber" and "brand_new" in r.new_columns


def test_regression_and_drop_both_red():
    m = _matrix(feat_a=np.zeros(50)).drop(columns=["feat_b"])
    r = _agent().detect(m)
    assert r.severity == "red" and r.regressed and "feat_b" in r.dropped


def test_near_constant_frac_honored():
    nc = pd.DataFrame({"nc": np.r_[np.ones(96), 2 * np.ones(4)]})  # top_frac = 0.96
    red = _agent(reference={"nc": "healthy"}, near_constant_frac=0.95).detect(nc)
    green = _agent(reference={"nc": "healthy"}).detect(nc)  # default 0.999
    assert red.severity == "red" and red.regressed[0][0] == "nc"
    assert green.severity == "green"


def test_from_reference_canonical(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"reference": {"nc": "healthy"}, "near_constant_frac": 0.95}), encoding="utf-8")
    agent = FeatureCoverageSentinelAgent.from_reference(p, output_dir=tmp_path)
    assert agent.near_constant_frac == 0.95
    nc = pd.DataFrame({"nc": np.r_[np.ones(96), 2 * np.ones(4)]})
    assert agent.detect(nc).severity == "red"


def test_from_reference_flat(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"feat_a": "healthy", "dead_x": "ALL_ZERO"}), encoding="utf-8")
    agent = FeatureCoverageSentinelAgent.from_reference(p, output_dir=tmp_path)
    assert agent.near_constant_frac == 0.999
    assert agent.reference["dead_x"] == "ALL_ZERO"


def test_empty_reference_only_new_columns():
    # no reference at all -> every current column is 'new' -> amber (coverage gap)
    r = FeatureCoverageSentinelAgent(reference={}, output_dir="/tmp").detect(_matrix())
    assert r.severity == "amber" and "feat_a" in r.new_columns and not r.regressed
