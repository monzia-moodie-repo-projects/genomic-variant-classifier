"""Regression test for INCIDENT_2026-05-23_run10a-no-checkpoints.

Runs 9 and 10a lost 27+ hours of training because the ensemble persisted only at pipeline end.
The fix (variant_ensemble.py:1255-1281) writes a per-model checkpoint quartet INSIDE the base-model
loop. This test fits a small ensemble (fast tabular base models only) and asserts every trained base
model leaves recoverable artifacts on disk -- so a crash mid-training is survivable.

Tabular-only (cnn_1d/tabular_nn/kan/svm/catboost/mc_dropout excluded) to stay fast; the checkpoint
code path is identical for every base model. Also asserts OOF and OOF-index arrays are length-matched
(the mechanism that prevents the oof-export length mismatch, INCIDENT_2026-05-16).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from genomic_variant_classifier.models.variant_ensemble import EnsembleConfig, VariantEnsemble

FAST = {"random_forest", "xgboost", "lightgbm", "logistic_regression"}


def test_per_model_checkpoints_written(tmp_path):
    rng = np.random.default_rng(0)
    n, d = 300, 20
    X_tab = pd.DataFrame(rng.standard_normal((n, d)), columns=[f"f{i}" for i in range(d)])
    # mild signal so models fit sensibly (not required for the persistence assertion itself)
    y = ((X_tab["f0"] + 0.5 * X_tab["f1"] + 0.3 * rng.standard_normal(n)) > 0).astype(int)

    cfg = EnsembleConfig(
        n_folds=3,
        random_state=0,
        model_dir=tmp_path / "models",
        skip_catboost=True,
        skip_svm=True,
        skip_kan=True,
        skip_mc_dropout=True,
    )
    ens = VariantEnsemble(cfg)
    for name in list(ens.base_estimators):
        if name not in FAST:
            ens.base_estimators.pop(name, None)
    assert ens.base_estimators and set(ens.base_estimators) <= FAST, "fixture: only fast models"

    trained = list(ens.base_estimators)  # snapshot: fit() clears base_estimators
    # X_seq=None: the FAST filter above drops every model but the fast tabular
    # ones, so nothing takes the sequence branch. The old placeholder carried
    # the comment "inert" -- since Part 3 (ff97c34) the code can say it instead.
    ens.fit(X_tab, None, y)

    md = cfg.model_dir
    for name in trained:
        for suffix in (".joblib", "_oof.npy", "_oof_indices.npy", "_meta.json"):
            p = md / f"{name}{suffix}"
            assert p.exists() and p.stat().st_size > 0, f"missing/empty checkpoint: {p.name}"
        meta = json.loads((md / f"{name}_meta.json").read_text())
        assert meta["name"] == name
        assert 0.0 <= meta["oof_auroc"] <= 1.0
        assert meta["n_samples"] > 0
        oof = np.load(md / f"{name}_oof.npy")
        idx = np.load(md / f"{name}_oof_indices.npy")
        assert len(oof) == len(idx), f"{name}: OOF/index length mismatch ({len(oof)} != {len(idx)})"
