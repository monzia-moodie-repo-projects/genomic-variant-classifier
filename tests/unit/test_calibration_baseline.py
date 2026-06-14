#!/usr/bin/env python3
"""test_calibration_baseline.py -- CalibrationDrift activation machinery (Monzia Moodie)."""
import importlib.util
import os
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

_BUILDER = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_calibration_baseline.py")
_spec = importlib.util.spec_from_file_location("build_calibration_baseline", _BUILDER)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_baseline = _mod.build_baseline
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.calibration_drift_agent import (
    CalibrationDriftAgent,
)
from genomic_variant_classifier.agent_layer.agents.calibration_drift_monitor_agent import (
    CalibrationDriftMonitorAgent,
)

CLASSES = ("B", "LB", "VUS", "LP", "P")
K = len(CLASSES)


def _cal_df(n, miscal, seed):
    """Calibrated multiclass predictions; errors distributed UNIFORMLY across non-predicted classes
    so every class is calibrated at miscal=0 (overconfidence knob `miscal` raises top-label ECE)."""
    rng = np.random.default_rng(seed)
    others = {c: [x for x in CLASSES if x != c] for c in CLASSES}
    rows = []
    for _ in range(n):
        conf = rng.uniform(0.55, 0.97)
        accp = min(max(conf - miscal, 0.0), 1.0)
        pred = CLASSES[rng.integers(0, K)]
        true = pred if rng.random() < accp else rng.choice(others[pred])
        p = {f"p_{c}": (1 - conf) / (K - 1) for c in CLASSES}
        p[f"p_{pred}"] = conf
        rows.append({"predicted_class": pred, "true_class": true, **p})
    return pd.DataFrame(rows)


def _baseline(tmp_path):
    payload = build_baseline(_cal_df(4000, 0.0, 1), CLASSES, n_bins=15)
    p = tmp_path / "calibration_baseline.json"
    p.write_text(json.dumps(payload))
    return p, payload["baseline_ece"]


def _state(tmp_path):
    return SharedState(state_file=str(tmp_path / "st.json"))


def test_builder_matches_direct_detect():
    ref = _cal_df(3000, 0.0, 1)
    payload = build_baseline(ref, CLASSES, n_bins=15)
    direct = float(CalibrationDriftAgent(classes=CLASSES, baseline_ece=0.0,
                                         output_dir=Path("."), n_bins=15).detect(ref).ece_top_label)
    assert abs(payload["baseline_ece"] - direct) < 1e-12
    assert payload["classes"] == list(CLASSES) and payload["n_bins"] == 15


def test_from_baseline_roundtrip(tmp_path):
    bp, ece = _baseline(tmp_path)
    d = CalibrationDriftAgent.from_baseline(bp, output_dir=tmp_path)
    assert d.classes == CLASSES and abs(d.baseline_ece - ece) < 1e-12 and d.n_bins == 15


def test_active_green(tmp_path):
    bp, _ = _baseline(tmp_path)
    a = CalibrationDriftMonitorAgent.from_default_baseline(
        _state(tmp_path), labeled_predictions=_cal_df(4000, 0.0, 7), baseline_path=bp, output_dir=tmp_path)
    r = a.run(dry_run=False)
    assert r["status"] == "ok" and r["severity"] == "green"


def test_active_amber(tmp_path):
    bp, _ = _baseline(tmp_path)
    a = CalibrationDriftMonitorAgent.from_default_baseline(
        _state(tmp_path), labeled_predictions=_cal_df(4000, 0.04, 7), baseline_path=bp, output_dir=tmp_path)
    r = a.run(dry_run=False)
    assert r["severity"] == "amber" and r["delta_ece_vs_baseline"] >= 0.02 and r["ece_top_label"] < 0.05


def test_active_red(tmp_path):
    bp, _ = _baseline(tmp_path)
    a = CalibrationDriftMonitorAgent.from_default_baseline(
        _state(tmp_path), labeled_predictions=_cal_df(4000, 0.12, 7), baseline_path=bp, output_dir=tmp_path)
    r = a.run(dry_run=False)
    assert r["severity"] == "red" and r["ece_top_label"] >= 0.05


def test_awaiting_and_env(tmp_path, monkeypatch):
    bp, _ = _baseline(tmp_path)
    a = CalibrationDriftMonitorAgent.from_default_baseline(_state(tmp_path), baseline_path=bp, output_dir=tmp_path)
    assert a.run(dry_run=False)["status"] == "awaiting_baseline"
    pq = tmp_path / "preds.parquet"
    _cal_df(4000, 0.12, 7).to_parquet(pq)
    monkeypatch.setenv("GVC_CALIBRATION_LABELED_PREDICTIONS", str(pq))
    a2 = CalibrationDriftMonitorAgent.from_default_baseline(_state(tmp_path), baseline_path=bp, output_dir=tmp_path)
    r = a2.run(dry_run=False)
    assert r["status"] == "ok" and r["severity"] == "red"
