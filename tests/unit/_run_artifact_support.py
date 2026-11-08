"""Test-only support for package 1, set 1b: REAL fitted ensembles and REAL completed runs, small and fast.

Production code never imports this. Every artefact here is produced by the production writers from a genuinely fitted
VariantEnsemble (two fast members: logistic regression and a shallow XGBoost), so a test that reads a completed run
reads what a real run writes, not a hand-made frame that happens to have the right column names.

Author: Monzia Moodie
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from genomic_variant_classifier.evaluation.oof_identity import compose_oof_identity
from genomic_variant_classifier.evaluation.prediction_artifacts import RunArtifactWriter
from genomic_variant_classifier.evaluation.split_identity import (
    SPLIT_UNIT_CONTEXT_SHA256,
    split_identity,
    split_observations,
)
from genomic_variant_classifier.models.variant_ensemble import EnsembleConfig, VariantEnsemble

FAST_MEMBERS = ("logistic_regression", "xgboost")


def fast_roster() -> dict:
    return {"logistic_regression": LogisticRegression(max_iter=300),
            "xgboost": xgb.XGBClassifier(n_estimators=25, max_depth=2, learning_rate=0.3, verbosity=0, n_jobs=1)}


def install_fast_roster(monkeypatch) -> None:
    """Every VariantEnsemble built while the patch is active gets the fast two-member roster (scripts under test
    construct their own ensembles)."""
    def _build(self) -> None:
        self.base_estimators = fast_roster()
        self.meta_learner = LogisticRegression(max_iter=300)
        self.blend_weights_ = None
        self.dropped_models_ = {}
        self.trained_models_ = {}
    monkeypatch.setattr(VariantEnsemble, "_build_estimators", _build)


def synthetic_split(n: int, *, seed: int, role: str, leaky: bool = False, n_genes: int = 40,
                    n_features: int = 5) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """(X, y, meta) with a real signal in f0; `leaky` makes f0 nearly determine the label."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    noise = 0.02 if leaky else 0.6
    y = pd.Series((X["f0"] + noise * rng.standard_normal(n) > 0.3).astype(np.int64), name="label")
    offset = {"train": 0, "val": 100_000, "test": 200_000, "holdout": 300_000}[role]
    meta = pd.DataFrame({
        "variant_id": [f"{role}:{i}" for i in range(n)],
        "chrom": ["1"] * n, "pos": np.arange(n, dtype=np.int64) + offset + 1,
        "ref": ["A"] * n, "alt": ["G"] * n,
        "gene_symbol": [f"{role}_G{i % n_genes}" for i in range(n)],
        "consequence": rng.choice(["missense_variant", "stop_gained"], n),
    })
    return X, y, meta


def fit_fast(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, model_dir: Optional[Path], *,
             gene_groups: bool = True, n_folds: int = 3) -> VariantEnsemble:
    ens = VariantEnsemble(EnsembleConfig(model_dir=model_dir, n_folds=n_folds))
    ens.base_estimators = fast_roster()
    ens.fit(X, None, y, gene_symbol=meta["gene_symbol"] if gene_groups else None)
    return ens


def completed_run(run_dir: Path, *, seed: int = 0, leaky: bool = False, gene_groups: bool = True,
                  n_train: int = 300, n_test: int = 160) -> Path:
    """A COMPLETE run written by the production writer: fitted ensemble, bound test capture, OOF identity, manifest."""
    run_dir = Path(run_dir)
    X, y, meta = synthetic_split(n_train, seed=seed, role="train", leaky=leaky)
    Xt, yt, mt = synthetic_split(n_test, seed=seed + 1, role="test", leaky=leaky)
    writer = RunArtifactWriter(run_id=run_dir.parent.name or "run", ablation=run_dir.name, output_dir=run_dir)
    ens = fit_fast(X, y, meta, run_dir / "models", gene_groups=gene_groups)
    ens.save(run_dir / "models" / "ensemble.joblib")
    writer.register_external("models/ensemble.joblib")
    train_id, train_vids = split_identity(X, y, meta, "train")
    test_id, test_vids = split_identity(Xt, yt, mt, "test")
    rows = split_observations(test_id, test_vids, unit_context_sha256=SPLIT_UNIT_CONTEXT_SHA256)
    scored = ens.score_split(Xt, None, rows, split_sha256=test_id.prediction_split_sha256)
    writer.save_split_predictions(scored, yt, mt, role="test")
    writer.save_oof_predictions(compose_oof_identity(ens, train=train_id, vids=train_vids, labels=y,
                                                     meta_train=meta))
    writer.complete(["test_predictions.parquet", "test_prediction_blocks.json", "oof_predictions.parquet",
                     "oof_identity.json", "models/ensemble.joblib"])
    return run_dir
