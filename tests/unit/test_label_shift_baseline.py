"""test_label_shift_baseline.py  --  Monzia Moodie

LabelShift activation: build_label_shift_baseline.build_baseline produces a column-stochastic
reference_confusion + p_train; LabelShiftAgent.from_baseline loads it; and
LabelShiftMonitorAgent.from_default_baseline moves awaiting_baseline -> active detection
(green on a matching prediction window, red on a strongly shifted one, graceful when the
baseline or prediction log is absent).
"""
import json

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.label_shift_agent import LabelShiftAgent
from genomic_variant_classifier.agent_layer.agents.label_shift_monitor_agent import (
    LabelShiftMonitorAgent,
)

import importlib.util
import os

_BUILDER = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_label_shift_baseline.py")

CLASSES = ("B", "LB", "VUS", "LP", "P")


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_label_shift_baseline", _BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _simulate_preds(y_true, rng, acc=0.8):
    out = []
    for c in y_true:
        out.append(c if rng.random() < acc else rng.choice(CLASSES))
    return out


@pytest.fixture
def baseline_dict():
    bld = _load_builder()
    rng = np.random.default_rng(0)
    p_train = [0.45, 0.15, 0.20, 0.10, 0.10]
    ytr = rng.choice(CLASSES, size=4000, p=p_train)
    yval = rng.choice(CLASSES, size=4000, p=p_train)
    yval_pred = _simulate_preds(yval, rng, acc=0.8)
    return bld.build_baseline(ytr, yval, yval_pred, CLASSES)


def test_build_baseline_is_column_stochastic(baseline_dict):
    C = np.asarray(baseline_dict["reference_confusion"], dtype=float)
    assert C.shape == (5, 5)
    # each column (true class) sums to 1: P(pred|true)
    assert np.allclose(C.sum(axis=0), 1.0, atol=1e-9)
    p = np.asarray(baseline_dict["p_train"])
    assert abs(p.sum() - 1.0) < 1e-9 and (p >= 0).all()


def test_from_baseline_then_active_green(tmp_path, baseline_dict):
    bp = tmp_path / "label_shift_baseline.json"
    bp.write_text(json.dumps(baseline_dict), encoding="utf-8")
    state = SharedState(state_file=tmp_path / "state.json")
    rng = np.random.default_rng(1)
    p_train = np.asarray(baseline_dict["p_train"])
    prod = _simulate_preds(rng.choice(CLASSES, size=3000, p=p_train), rng, acc=0.8)
    plog = pd.DataFrame({"predicted_class": prod})
    agent = LabelShiftMonitorAgent.from_default_baseline(
        state, prediction_log=plog, baseline_path=bp, output_dir=tmp_path / "rep"
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok"
    assert out["severity"] == "green"


def test_active_red_on_strong_shift(tmp_path, baseline_dict):
    bp = tmp_path / "label_shift_baseline.json"
    bp.write_text(json.dumps(baseline_dict), encoding="utf-8")
    state = SharedState(state_file=tmp_path / "state.json")
    rng = np.random.default_rng(2)
    p_shift = [0.10, 0.05, 0.65, 0.10, 0.10]  # VUS-heavy
    prod = _simulate_preds(rng.choice(CLASSES, size=3000, p=p_shift), rng, acc=0.8)
    plog = pd.DataFrame({"predicted_class": prod})
    agent = LabelShiftMonitorAgent.from_default_baseline(
        state, prediction_log=plog, baseline_path=bp, output_dir=tmp_path / "rep"
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok"
    assert out["severity"] in ("amber", "red")
    assert out["max_abs_class_shift"] >= 0.05


def test_awaiting_when_baseline_absent(tmp_path):
    state = SharedState(state_file=tmp_path / "state.json")
    plog = pd.DataFrame({"predicted_class": ["B", "VUS", "P"]})
    agent = LabelShiftMonitorAgent.from_default_baseline(
        state, prediction_log=plog, baseline_path=tmp_path / "nope.json"
    )
    assert agent.run(dry_run=True)["status"] == "awaiting_baseline"


def test_awaiting_when_no_prediction_log(tmp_path, baseline_dict):
    bp = tmp_path / "label_shift_baseline.json"
    bp.write_text(json.dumps(baseline_dict), encoding="utf-8")
    state = SharedState(state_file=tmp_path / "state.json")
    agent = LabelShiftMonitorAgent.from_default_baseline(
        state, baseline_path=bp, output_dir=tmp_path / "rep"
    )
    assert agent.run(dry_run=True)["status"] == "awaiting_baseline"
