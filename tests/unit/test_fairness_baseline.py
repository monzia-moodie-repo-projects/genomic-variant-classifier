#!/usr/bin/env python3
"""test_fairness_baseline.py -- FairnessSubgroup activation machinery (Monzia Moodie).

Also documents the detector's max_dpd_change=0.0 stub (PHASE_2_FEATURES): see test_dpd_stub_is_zero.
"""
import importlib.util
import os
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

_BUILDER = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_fairness_baseline.py")
_spec = importlib.util.spec_from_file_location("build_fairness_baseline", _BUILDER)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_baseline = _mod.build_baseline
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.fairness_subgroup_agent import (
    FairnessSubgroupAgent,
)
from genomic_variant_classifier.agent_layer.agents.fairness_subgroup_monitor_agent import (
    FairnessSubgroupMonitorAgent,
)

CLASSES = ("B", "LB", "VUS", "LP", "P")
K = len(CLASSES)
AXES = {"ancestry": "gnomad_pop"}


def _fair_df(n, seed, *, afr_bad=False, nfe_skew=False):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        pop = rng.choice(["afr", "nfe", "eas"])
        conf = rng.uniform(0.6, 0.92)
        miscal = 0.35 if (afr_bad and pop == "afr") else 0.0
        accp = min(max(conf - miscal, 0.0), 1.0)
        p_pos = 0.85 if (nfe_skew and pop == "nfe") else 0.5
        pred = "P" if rng.random() < p_pos else "B"
        true = pred if rng.random() < accp else ("B" if pred == "P" else "P")
        p = {f"p_{c}": (1 - conf) / (K - 1) for c in CLASSES}
        p[f"p_{pred}"] = conf
        rows.append({"gnomad_pop": pop, "predicted_class": pred, "true_class": true, **p})
    return pd.DataFrame(rows)


def _baseline(tmp_path):
    payload = build_baseline(_fair_df(4000, 1), AXES, CLASSES, high_priority_strata=["afr", "sas", "amr"])
    p = tmp_path / "fairness_baseline.json"
    p.write_text(json.dumps(payload))
    return p


def _state(tmp_path):
    return SharedState(state_file=str(tmp_path / "st.json"))


def test_builder_p_train_counts(tmp_path):
    ref = _fair_df(3000, 1)
    payload = build_baseline(ref, AXES, CLASSES)
    afr = [r for r in payload["p_train_per_stratum"] if r["stratum"] == "afr"][0]
    sub = ref[ref["gnomad_pop"] == "afr"]
    assert afr["p_train"] == [int((sub["predicted_class"] == c).sum()) for c in CLASSES]


def test_from_baseline_roundtrip(tmp_path):
    d = FairnessSubgroupAgent.from_baseline(_baseline(tmp_path), output_dir=tmp_path)
    assert d.classes == CLASSES
    assert ("ancestry", "afr") in d.p_train_per_stratum
    assert isinstance(d.p_train_per_stratum[("ancestry", "afr")], np.ndarray)
    assert d.high_priority_strata == frozenset({"afr", "sas", "amr"})


def test_active_green(tmp_path):
    a = FairnessSubgroupMonitorAgent.from_default_baseline(
        _state(tmp_path), predictions=_fair_df(4000, 9), axes=AXES, baseline_path=_baseline(tmp_path), output_dir=tmp_path)
    r = a.run(dry_run=False)
    assert r["status"] == "ok" and r["severity"] == "green"


def test_active_amber(tmp_path):
    # nfe over-predicts P -> EOD gap (amber); calibrated so no ECE/priority red, nfe not high-priority
    a = FairnessSubgroupMonitorAgent.from_default_baseline(
        _state(tmp_path), predictions=_fair_df(4000, 9, nfe_skew=True), axes=AXES,
        baseline_path=_baseline(tmp_path), output_dir=tmp_path)
    r = a.run(dry_run=False)
    assert r["severity"] == "amber" and r["max_eod"] >= 0.05 and not r["high_priority_strata_flags"]


def test_active_red(tmp_path):
    a = FairnessSubgroupMonitorAgent.from_default_baseline(
        _state(tmp_path), predictions=_fair_df(4000, 9, afr_bad=True), axes=AXES,
        baseline_path=_baseline(tmp_path), output_dir=tmp_path)
    r = a.run(dry_run=False)
    assert r["severity"] == "red" and "afr" in r["high_priority_strata_flags"]


def test_dpd_stub_is_zero(tmp_path):
    """DOCUMENTS the PHASE_2_FEATURES stub: max_dpd_change is hardcoded 0.0 in the detector
    ('wire from training-time DPD baseline'). This is NOT a bug -- it records the known gap so a
    future DPD-baseline wiring is caught by a failing assertion here."""
    a = FairnessSubgroupMonitorAgent.from_default_baseline(
        _state(tmp_path), predictions=_fair_df(4000, 9, afr_bad=True), axes=AXES,
        baseline_path=_baseline(tmp_path), output_dir=tmp_path)
    r = a.run(dry_run=False)
    assert r["max_dpd_change"] == 0.0


def test_awaiting_and_env(tmp_path, monkeypatch):
    bp = _baseline(tmp_path)
    assert FairnessSubgroupMonitorAgent.from_default_baseline(
        _state(tmp_path), axes=AXES, baseline_path=bp, output_dir=tmp_path).run(dry_run=False)["status"] == "awaiting_baseline"
    assert FairnessSubgroupMonitorAgent.from_default_baseline(
        _state(tmp_path), predictions=_fair_df(100, 9), baseline_path=bp, output_dir=tmp_path).run(dry_run=False)["status"] == "awaiting_baseline"
    assert FairnessSubgroupMonitorAgent.from_default_baseline(
        _state(tmp_path), predictions=_fair_df(100, 9), axes=AXES, baseline_path=tmp_path / "nope.json",
        output_dir=tmp_path).run(dry_run=False)["status"] == "awaiting_baseline"
    pq = tmp_path / "preds.parquet"
    _fair_df(4000, 9, afr_bad=True).to_parquet(pq)
    monkeypatch.setenv("GVC_FAIRNESS_PREDICTIONS", str(pq))
    monkeypatch.setenv("GVC_FAIRNESS_AXES", json.dumps(AXES))
    r = FairnessSubgroupMonitorAgent.from_default_baseline(_state(tmp_path), baseline_path=bp, output_dir=tmp_path).run(dry_run=False)
    assert r["status"] == "ok" and r["severity"] == "red"
