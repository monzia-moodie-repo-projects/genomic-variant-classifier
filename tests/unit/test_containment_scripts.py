"""Containment in the scripts that fit or calibrate models on CACHED split parquets (2026-09-23).

A split written before the 2026-09-22 quarantine still carries the four structural columns. These scripts
feed every split column to raw LightGBM / XGBoost / CatBoost (or, for calibration, bypassed the pipeline's
admission), so each must refuse such a split before any model sees it.

Author: Monzia Moodie
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.containment import ContainmentError
from genomic_variant_classifier.quarantine_policy import QUARANTINED_FEATURES

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
Q0 = QUARANTINED_FEATURES[0]


def _splits(d: Path, quarantined: bool) -> Path:
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "n_pathogenic_in_gene": rng.integers(0, 5, n)})
    if quarantined:
        X[Q0] = 50.0                                   # the historical fabricated sentinel
    y = pd.DataFrame({"label": [0, 1] * (n // 2)})
    meta = pd.DataFrame({"gene_symbol": [f"G{i % 10}" for i in range(n)]})
    d.mkdir(parents=True, exist_ok=True)
    for part in ("train", "val", "test"):
        X.to_parquet(d / f"X_{part}.parquet"); y.to_parquet(d / f"y_{part}.parquet"); meta.to_parquet(d / f"meta_{part}.parquet")
    return d


def test_tune_hyperparams_refuses_a_quarantine_era_split(tmp_path):
    import tune_hyperparams as T
    d = _splits(tmp_path / "q", quarantined=True)
    with pytest.raises(ContainmentError, match="suspended"):
        T.load_splits(str(d / "X_train.parquet"), str(d / "y_train.parquet"), str(d / "X_val.parquet"), str(d / "y_val.parquet"))
    clean = _splits(tmp_path / "c", quarantined=False)
    Xtr, ytr, Xv, yv = T.load_splits(*(str(clean / f) for f in ("X_train.parquet", "y_train.parquet", "X_val.parquet", "y_val.parquet")))
    assert Xtr.shape == (60, 3) and Xv.shape == (60, 3)


def test_run11_hpo_refuses_before_any_trial(tmp_path, monkeypatch):
    fake = types.ModuleType("optuna"); fake.logging = types.SimpleNamespace(set_verbosity=lambda *a: None, WARNING=30)
    fake.create_study = lambda *a, **k: pytest.fail("reached optimisation: the split was not refused")
    pruners = types.ModuleType("optuna.pruners"); pruners.HyperbandPruner = lambda *a, **k: None; fake.pruners = pruners
    monkeypatch.setitem(sys.modules, "optuna", fake); monkeypatch.setitem(sys.modules, "optuna.pruners", pruners)
    monkeypatch.delitem(sys.modules, "run11_hpo", raising=False)
    import run11_hpo as H
    with pytest.raises(ContainmentError, match="suspended"):
        H.run_hpo(str(_splits(tmp_path / "q", quarantined=True)), str(tmp_path / "out"), models=["lightgbm"], n_trials=1)


def test_ablation_npig_refuses_before_fitting(tmp_path, monkeypatch):
    import ablation_npig_permutation as A
    d = _splits(tmp_path / "q", quarantined=True)
    monkeypatch.setattr(sys, "argv", ["x", "--splits-dir", str(d), "--n-permutations", "1", "--output", str(tmp_path / "o")])
    fitted = []
    monkeypatch.setattr(A, "_fit_eval_lgbm", lambda *a, **k: fitted.append(1) or 0.5)
    with pytest.raises(ContainmentError, match="suspended"):
        A.main()
    assert fitted == []


def _pipeline(columns):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from genomic_variant_classifier.api import pipeline as P
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(40, len(columns))), columns=columns); y = np.array([0, 1] * 20)
    sc = StandardScaler().fit(X); lr = LogisticRegression().fit(sc.transform(X), y)
    meta = LogisticRegression().fit(lr.predict_proba(sc.transform(X))[:, 1:], y)
    return P, sc, lr, meta


def test_calibrate_thresholds_loads_through_admission(tmp_path):
    import calibrate_thresholds as C
    from genomic_variant_classifier.models.variant_ensemble import engineer_features
    raw = pd.DataFrame({"chrom": ["1"] * 40, "pos": range(40), "ref": ["A"] * 40, "alt": ["G"] * 40,
                        "consequence": ["missense_variant"] * 40})
    feats = engineer_features(raw)
    usable = [c for c in feats.columns if not feats[c].isna().any()][:4]
    P, sc, lr, meta = _pipeline(usable + [Q0])
    legacy = P.InferencePipeline.__new__(P.InferencePipeline)
    legacy.__dict__.update(trained_models={"logistic_regression": lr}, meta_learner=meta, scaler=sc,
                           metadata=P.PipelineMetadata(feature_names=usable + [Q0]), gnn_scorer=None, preprocessor_=None)
    joblib.dump(legacy, tmp_path / "legacy.joblib")
    with pytest.raises(ContainmentError, match="suspended"):
        C.load_pipeline(str(tmp_path / "legacy.joblib"))
    P, sc, lr, meta = _pipeline(usable)
    pipe = P.InferencePipeline({"logistic_regression": lr}, meta, scaler=sc)
    pipe.save(tmp_path / "ok.joblib")
    loaded = C.load_pipeline(str(tmp_path / "ok.joblib"))
    np.testing.assert_array_equal(C.get_raw_scores(loaded, feats), pipe.predict_proba(raw))


# --- continual training (2026-09-23): the drift reference and the density ratio refuse quarantine-era cohorts ---
def test_lsif_alignment_refuses_a_quarantine_era_reference_cohort():
    from genomic_variant_classifier.training.continual_trainer import (
        ReferenceTrainingFeatures, _aligned_lsif_matrices)
    cols = ["a", "b", Q0]
    ref = ReferenceTrainingFeatures(frame=pd.DataFrame(np.ones((4, 3)), columns=cols),
                                    model_record_id="run15", feature_names=cols)
    with pytest.raises(ContainmentError, match="suspended"):
        _aligned_lsif_matrices(reference=ref, new_features=pd.DataFrame(np.ones((4, 3)), columns=cols))


def test_continual_learner_refuses_a_quarantine_era_reference_split_before_any_drift_work(tmp_path, monkeypatch):
    from genomic_variant_classifier.monitoring import drift_detector
    from genomic_variant_classifier.training.continual_trainer import ContinualLearner, ContinualLearningConfig
    d = _splits(tmp_path / "ref", quarantined=True)
    pd.DataFrame({"variant_id": ["v1"]}).to_parquet(d / "meta_test.parquet")
    for name in ("new.parquet", "old.parquet"):             # real inputs, so nothing ELSE stops the run first
        pd.DataFrame({"variant_id": ["v1"], "chrom": ["1"], "pos": [1], "ref": ["A"], "alt": ["G"]}).to_parquet(tmp_path / name)
    monkeypatch.setattr(drift_detector.DriftDetector, "from_reference", classmethod(
        lambda *a, **k: pytest.fail("a drift reference was built from a quarantine-era split")))
    learner = ContinualLearner(ContinualLearningConfig(output_dir=str(tmp_path / "out")))
    with pytest.raises(ContainmentError, match="suspended"):
        learner.run(reference_splits_dir=d, new_clinvar_path=tmp_path / "new.parquet",
                    old_clinvar_path=tmp_path / "old.parquet", current_model_path=tmp_path / "m.joblib")
