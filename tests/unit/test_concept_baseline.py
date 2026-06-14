#!/usr/bin/env python3
"""test_concept_baseline.py -- ConceptDrift activation machinery (Monzia Moodie)."""
import importlib.util
import os
import json
import pytest
from pathlib import Path

_BUILDER = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_concept_baseline.py")
_spec = importlib.util.spec_from_file_location("build_concept_baseline", _BUILDER)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_baseline = _mod.build_baseline
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.concept_drift_agent import ConceptDriftAgent
from genomic_variant_classifier.agent_layer.agents.concept_drift_monitor_agent import (
    ConceptDriftMonitorAgent,
)

CLASSES_OK = (0.90, 0.01)  # baseline auroc, sigma


def _baseline(tmp_path) -> Path:
    p = tmp_path / "concept_baseline.json"
    p.write_text(json.dumps({"cbpe_baseline_auroc": 0.90, "cbpe_baseline_sigma": 0.01,
                             "sigma_drop_amber": 2.0, "auroc_drop_red": 0.03, "bbse_alpha": 0.05}))
    return p


def _state(tmp_path):
    return SharedState(state_file=str(tmp_path / "st.json"))


def test_build_baseline_shape():
    out = build_baseline(0.91, 0.013, auroc_drop_red=0.04)
    assert out["cbpe_baseline_auroc"] == 0.91 and out["cbpe_baseline_sigma"] == 0.013
    assert out["auroc_drop_red"] == 0.04 and "bbse_alpha" not in out


def test_from_baseline_roundtrip(tmp_path):
    d = ConceptDriftAgent.from_baseline(_baseline(tmp_path), output_dir=tmp_path)
    assert d.cbpe_baseline_auroc == 0.90 and d.cbpe_baseline_sigma == 0.01
    assert d.auroc_drop_red == 0.03 and d.bbse_alpha == 0.05


def test_active_green(tmp_path):
    a = ConceptDriftMonitorAgent.from_default_baseline(
        _state(tmp_path), cbpe_estimated_auroc=0.895, bbse_pvalue=0.5, n_samples=1000,
        baseline_path=_baseline(tmp_path), output_dir=tmp_path)
    r = a.run(dry_run=False)
    assert r["status"] == "ok" and r["severity"] == "green"


def test_active_amber(tmp_path):
    # drop 0.025 = 2.5 sigma (>= amber) but BBSE significant (0.01 < 0.05) -> not pure concept -> amber
    a = ConceptDriftMonitorAgent.from_default_baseline(
        _state(tmp_path), cbpe_estimated_auroc=0.875, bbse_pvalue=0.01, n_samples=1000,
        baseline_path=_baseline(tmp_path), output_dir=tmp_path)
    r = a.run(dry_run=False)
    assert r["severity"] == "amber" and r["likely_pure_concept"] is False


def test_active_red(tmp_path):
    # drop 0.05 (>= 0.03) AND BBSE not significant (0.5 >= 0.05) -> pure concept -> red
    a = ConceptDriftMonitorAgent.from_default_baseline(
        _state(tmp_path), cbpe_estimated_auroc=0.85, bbse_pvalue=0.5, n_samples=1000,
        baseline_path=_baseline(tmp_path), output_dir=tmp_path)
    r = a.run(dry_run=False)
    assert r["severity"] == "red" and r["likely_pure_concept"] is True


def test_awaiting_missing_scalar_and_baseline(tmp_path):
    bp = _baseline(tmp_path)
    a = ConceptDriftMonitorAgent.from_default_baseline(
        _state(tmp_path), cbpe_estimated_auroc=0.85, baseline_path=bp, output_dir=tmp_path)
    assert a.run(dry_run=False)["status"] == "awaiting_baseline"  # missing bbse/n
    a2 = ConceptDriftMonitorAgent.from_default_baseline(
        _state(tmp_path), cbpe_estimated_auroc=0.85, bbse_pvalue=0.5, n_samples=1000,
        baseline_path=tmp_path / "nope.json", output_dir=tmp_path)
    assert a2.run(dry_run=False)["status"] == "awaiting_baseline"  # missing baseline


def test_env_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("GVC_CONCEPT_CBPE_AUROC", "0.85")
    monkeypatch.setenv("GVC_CONCEPT_BBSE_PVALUE", "0.5")
    monkeypatch.setenv("GVC_CONCEPT_N_SAMPLES", "1000")
    a = ConceptDriftMonitorAgent.from_default_baseline(
        _state(tmp_path), baseline_path=_baseline(tmp_path), output_dir=tmp_path)
    r = a.run(dry_run=False)
    assert r["status"] == "ok" and r["severity"] == "red"
