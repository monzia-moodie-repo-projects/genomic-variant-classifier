#!/usr/bin/env python3
"""
Run 11 Hyperparameter Optimization
====================================
Optuna-based HPO for LightGBM, XGBoost, and CatBoost.

Uses ASHA (HyperbandPruner) for early stopping of bad trials.
Results stored in SQLite for resumability across Vast.ai sessions.

Usage:
    python scripts/run11_hpo.py \
        --splits-dir outputs/run10b_final/full/splits \
        --output-dir outputs/run11/hpo \
        --n-trials 30 \
        --timeout 3600

Data collection:
    - Full Optuna study dashboard (HTML export)
    - Best trial params per model
    - Default vs optimized AUROC comparison
    - HPO wall-clock time per model
    - Number of pruned trials
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.pruners import HyperbandPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    logger.error("Optuna not installed. pip install optuna")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def lightgbm_objective(trial, X, y, cv):
    """LightGBM search space."""
    import lightgbm as lgb

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("lr", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "class_weight": "balanced",
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            params["device"] = "gpu"
    except ImportError:
        pass

    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        y_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        scores.append(score)

        # Prune if bad
        trial.report(score, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def xgboost_objective(trial, X, y, cv):
    """XGBoost search space."""
    import xgboost as xgb

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("lr", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 20),
        "eval_metric": "auc",
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 0,
    }

    try:
        import torch
        if torch.cuda.is_available():
            params["device"] = "cuda"
    except ImportError:
        pass

    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        scores.append(score)

        trial.report(score, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def catboost_objective(trial, X, y, cv):
    """CatBoost search space."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise optuna.TrialPruned()

    params = {
        "iterations": trial.suggest_int("iterations", 200, 1500),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("lr", 0.005, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10, log=True),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
        "auto_class_weights": "Balanced",
        "verbose": 0,
        "random_state": 42,
    }

    try:
        import torch
        if torch.cuda.is_available():
            params["task_type"] = "GPU"
    except ImportError:
        pass

    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
        )
        y_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        scores.append(score)

        trial.report(score, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_OBJECTIVES = {
    "lightgbm": lightgbm_objective,
    "xgboost": xgboost_objective,
    "catboost": catboost_objective,
}


def run_hpo(
    splits_dir: str,
    output_dir: str,
    models: Optional[list[str]] = None,
    n_trials: int = 30,
    timeout: int = 3600,
    n_folds: int = 3,
) -> dict:
    """
    Run HPO for specified models.

    Returns dict of {model_name: best_params}.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    splits = Path(splits_dir)
    X_train = pd.read_parquet(splits / "X_train.parquet").values
    y_train = pd.read_parquet(splits / "y_train.parquet").values.ravel()

    logger.info("HPO: %d training rows, %d features", X_train.shape[0], X_train.shape[1])

    # Subsample for HPO (full dataset is too slow for many trials)
    max_hpo_samples = 200_000
    if len(X_train) > max_hpo_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), max_hpo_samples, replace=False)
        X_hpo = X_train[idx]
        y_hpo = y_train[idx]
        logger.info("HPO: subsampled %d -> %d rows", len(X_train), max_hpo_samples)
    else:
        X_hpo = X_train
        y_hpo = y_train

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    if models is None:
        models = list(MODEL_OBJECTIVES.keys())

    results = {}
    db_path = output_path / "optuna_hpo.db"

    for model_name in models:
        if model_name not in MODEL_OBJECTIVES:
            logger.warning("Unknown model: %s", model_name)
            continue

        logger.info("=" * 60)
        logger.info("HPO: %s (%d trials, timeout=%ds)", model_name, n_trials, timeout)
        logger.info("=" * 60)

        t0 = time.perf_counter()

        study = optuna.create_study(
            direction="maximize",
            pruner=HyperbandPruner(
                min_resource=1,
                max_resource=n_folds,
                reduction_factor=3,
            ),
            storage=f"sqlite:///{db_path}",
            study_name=f"run11_{model_name}",
            load_if_exists=True,
        )

        objective_fn = MODEL_OBJECTIVES[model_name]
        study.optimize(
            lambda trial: objective_fn(trial, X_hpo, y_hpo, cv),
            n_trials=n_trials,
            timeout=timeout,
        )

        elapsed = time.perf_counter() - t0

        best = study.best_trial
        logger.info("  Best AUROC: %.4f", best.value)
        logger.info("  Best params: %s", best.params)
        logger.info("  Wall-clock: %.1f sec", elapsed)
        logger.info("  Trials: %d total, %d pruned",
                     len(study.trials),
                     len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))

        results[model_name] = {
            "best_auroc": best.value,
            "best_params": best.params,
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "wall_clock_sec": elapsed,
        }

        # Save per-model results
        model_result_path = output_path / f"{model_name}_best_params.json"
        with open(model_result_path, "w") as f:
            json.dump(results[model_name], f, indent=2, default=str)
        logger.info("  Saved: %s", model_result_path)

    # Save combined results
    combined_path = output_path / "best_params.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Combined HPO results: %s", combined_path)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 11 Hyperparameter Optimization")
    parser.add_argument("--splits-dir", required=True, help="Path to train/val/test splits")
    parser.add_argument("--output-dir", default="outputs/run11/hpo", help="Output directory")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to optimize (default: all)")
    parser.add_argument("--n-trials", type=int, default=30, help="Trials per model")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per model (sec)")
    parser.add_argument("--n-folds", type=int, default=3, help="CV folds for HPO")
    args = parser.parse_args()

    print("=" * 70)
    print("RUN 11: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    results = run_hpo(
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        models=args.models,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_folds=args.n_folds,
    )

    print("\n" + "=" * 70)
    print("HPO SUMMARY")
    print("=" * 70)
    for model, info in results.items():
        print(f"  {model:15s}  AUROC={info['best_auroc']:.4f}  "
              f"trials={info['n_trials']}  "
              f"pruned={info['n_pruned']}  "
              f"time={info['wall_clock_sec']:.0f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
