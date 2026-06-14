"""test_feature_coverage_baseline.py  --  Monzia Moodie

build_feature_coverage_baseline.build_reference: replicate the audit's cross-file degeneracy
aggregation, guard the empty-degenerate -> NaN CSV pitfall, and round-trip through
FeatureCoverageSentinelAgent.from_reference.
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

from genomic_variant_classifier.agent_layer.agents.feature_coverage_sentinel_agent import (
    FeatureCoverageSentinelAgent,
)

_BUILDER = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_feature_coverage_baseline.py")


def _builder():
    spec = importlib.util.spec_from_file_location("build_feature_coverage_baseline", _BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _health():
    # feat_a healthy in both files; feat_b healthy in train, dead in val (cross-file degenerate);
    # dead_x degenerate; ok_c healthy.
    return pd.DataFrame({
        "column": ["feat_a", "feat_b", "dead_x", "ok_c", "feat_a", "feat_b", "dead_x", "ok_c"],
        "degenerate": ["", "", "ALL_ZERO", "", "", "CONSTANT;ALL_ZERO", "ALL_ZERO", ""],
        "file": ["train"] * 4 + ["val"] * 4,
    })


def test_cross_file_aggregation():
    ref = _builder().build_reference(_health(), 0.999)["reference"]
    assert ref["feat_a"] == "healthy" and ref["ok_c"] == "healthy"
    assert ref["dead_x"] == "ALL_ZERO"
    assert ref["feat_b"] != "healthy"  # dead in val -> degenerate cross-file
    # verdict is the first sorted reason across files
    assert ref["feat_b"] == "CONSTANT;ALL_ZERO"


def test_counts():
    p = _builder().build_reference(_health(), 0.999)
    assert p["n_healthy"] == 2 and p["n_degenerate"] == 2 and p["n_total"] == 4
    assert p["near_constant_frac"] == 0.999


def test_nan_on_empty_guard(tmp_path):
    # writing + re-reading via CSV turns empty 'degenerate' into NaN; build_reference must
    # treat NaN as healthy, not degenerate.
    csv = tmp_path / "h.csv"
    _health().to_csv(csv, index=False)
    reread = pd.read_csv(csv)
    assert int(reread["degenerate"].isna().sum()) > 0  # the empties became NaN
    p = _builder().build_reference(reread, 0.999)
    assert p["n_healthy"] == 2 and p["n_degenerate"] == 2  # guard worked


def test_round_trip_through_from_reference(tmp_path):
    payload = _builder().build_reference(_health(), 0.999)
    ref_json = tmp_path / "ref.json"
    ref_json.write_text(json.dumps(payload), encoding="utf-8")
    agent = FeatureCoverageSentinelAgent.from_reference(ref_json, output_dir=tmp_path)
    assert agent.near_constant_frac == 0.999
    # a matrix matching the reference (feat_a/ok_c healthy, feat_b/dead_x dead) -> green
    m = pd.DataFrame({"feat_a": np.linspace(0, 1, 30), "ok_c": np.linspace(1, 2, 30),
                      "feat_b": np.zeros(30), "dead_x": np.zeros(30)})
    r = agent.detect(m)
    assert r.severity == "green" and set(r.still_degenerate) == {"feat_b", "dead_x"}
    # feat_a regresses -> red
    m["feat_a"] = 0.0
    assert agent.detect(m).severity == "red"


def test_missing_columns_raises():
    import pytest
    with pytest.raises(ValueError):
        _builder().build_reference(pd.DataFrame({"column": ["x"]}), 0.999)  # no 'degenerate'
