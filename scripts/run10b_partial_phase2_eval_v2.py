"""
Run 10b-partial salvage — Phase 2 v2.

Improvements over v1:
  - Auto-discovers split parquet locations (handles scp -r layout quirks)
  - Uses OOF arrays for PROPER CV-stacked meta-learner (not just val-trained)
  - Sanity-checks OOF alignment by reconstructing per-model OOF AUROCs
  - Falls back to simple-average ensemble if OOF alignment is broken
  - Frees memory after each model (random_forest is 302 MB)
  - Saves partial results even if some steps fail

Run from: C:\\Projects\\genomic-variant-classifier\\
Expected wall time: 10-30 minutes (random_forest inference is dominant cost).
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(r"C:\Projects\genomic-variant-classifier\outputs\run10b_final")
MODELS_DIR = ROOT / "full" / "models"
OUT_DIR = ROOT / "full"

EXPECTED_SPLITS = [
    "X_train.parquet", "X_val.parquet", "X_test.parquet",
    "y_train.parquet", "y_val.parquet", "y_test.parquet",
]
BROKEN_MODELS = {"cnn_1d"}  # OOF AUROC 0.5; exclude from ensembles


def find_file(root: Path, name: str) -> Path | None:
    """Search common locations + recursive fallback."""
    for c in [
        root / "full" / name,
        root / name,
        root / "full" / "full" / name,
        root / "full" / "splits" / name,
        root / "splits" / name,
    ]:
        if c.exists():
            return c.resolve()
    matches = list(root.rglob(name))
    return matches[0].resolve() if matches else None


def to_binary(y_in) -> np.ndarray:
    y = np.asarray(y_in).ravel()
    if y.dtype.kind in "fc":
        return (y >= 0.5).astype(int)
    if y.max() > 1:
        return (y >= 4).astype(int)
    return y.astype(int)


def load_y(path: Path) -> np.ndarray:
    df = pd.read_parquet(path)
    if df.shape[1] == 1:
        return to_binary(df.iloc[:, 0].values)
    for col in ["label", "y", "pathogenic", "is_pathogenic", "acmg_class"]:
        if col in df.columns:
            return to_binary(df[col].values)
    return to_binary(df.iloc[:, 0].values)


def main() -> int:
    print("=" * 72)
    print("Run 10b-partial salvage — Phase 2 v2")
    print(f"Search root: {ROOT}")
    print("=" * 72)
    t0_total = time.time()

    # [1/6] Discover splits
    print("\n[1/6] Discovering split files...")
    splits: dict[str, Path] = {}
    missing = []
    for name in EXPECTED_SPLITS:
        p = find_file(ROOT, name)
        if p is None:
            missing.append(name)
        else:
            splits[name] = p
            print(f"  OK  {name:20s} -> {p}")
    if missing:
        print(f"\n  MISSING: {missing}")
        print(f"\n  Recursive scan of {ROOT}:")
        for p in ROOT.rglob("*"):
            if p.is_file():
                rel = p.relative_to(ROOT)
                size = p.stat().st_size
                print(f"    {str(rel):60s} {size:>12d}")
        return 2

    # [2/6] Load val + test
    print("\n[2/6] Loading val + test splits...")
    X_val = pd.read_parquet(splits["X_val.parquet"])
    y_val = load_y(splits["y_val.parquet"])
    X_test = pd.read_parquet(splits["X_test.parquet"])
    y_test = load_y(splits["y_test.parquet"])
    print(f"  X_val  shape={X_val.shape}  prevalence={y_val.mean():.4f}")
    print(f"  X_test shape={X_test.shape}  prevalence={y_test.mean():.4f}")

    # [3/6] Per-model inference on val + test
    print("\n[3/6] Per-model inference:")
    print(f"  {'model':25s} {'load_s':>7s} {'pred_s':>7s} {'val_auroc':>10s} {'test_auroc':>11s}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*10} {'-'*11}")

    model_names = sorted(p.stem for p in MODELS_DIR.glob("*.joblib"))
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    per_model_rows = []

    for name in model_names:
        try:
            tt = time.time()
            model = joblib.load(MODELS_DIR / f"{name}.joblib")
            t_load = time.time() - tt
            tt = time.time()
            val_p = model.predict_proba(X_val)[:, 1]
            test_p = model.predict_proba(X_test)[:, 1]
            t_pred = time.time() - tt
            del model  # free RAM (random_forest is ~302 MB)
            v_au = roc_auc_score(y_val, val_p)
            t_au = roc_auc_score(y_test, test_p)
            print(f"  {name:25s} {t_load:>7.1f} {t_pred:>7.1f} {v_au:>10.4f} {t_au:>11.4f}")
            val_preds[name] = val_p
            test_preds[name] = test_p
            oof_au = json.loads((MODELS_DIR / f"{name}_meta.json").read_text())["oof_auroc"]
            per_model_rows.append({
                "model": name, "oof_auroc": oof_au,
                "val_auroc": float(v_au), "test_auroc": float(t_au),
                "val_ap": float(average_precision_score(y_val, val_p)),
                "test_ap": float(average_precision_score(y_test, test_p)),
                "test_brier": float(brier_score_loss(y_test, test_p)),
                "load_seconds": round(t_load, 2), "predict_seconds": round(t_pred, 2),
            })
        except Exception as exc:
            print(f"  {name:25s} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
            per_model_rows.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})

    # [4/6] Simple-average ensembles
    print("\n[4/6] Simple-average ensembles:")
    ensemble_results = {}
    useful = [n for n in val_preds if n not in BROKEN_MODELS]
    for label, members in [("all", list(val_preds)), ("useful_no_cnn1d", useful)]:
        if not members:
            continue
        val_avg = np.mean([val_preds[n] for n in members], axis=0)
        test_avg = np.mean([test_preds[n] for n in members], axis=0)
        v_au = float(roc_auc_score(y_val, val_avg))
        t_au = float(roc_auc_score(y_test, test_avg))
        print(f"  {label:20s} ({len(members)} models)  VAL={v_au:.4f}  TEST={t_au:.4f}")
        ensemble_results[label] = {
            "members": members, "n_members": len(members),
            "val_auroc": v_au, "test_auroc": t_au,
            "test_ap": float(average_precision_score(y_test, test_avg)),
            "test_brier": float(brier_score_loss(y_test, test_avg)),
        }

    # [5/6] OOF-based meta-learner with alignment sanity check
    print("\n[5/6] OOF-based meta-learner (proper CV-stacked):")
    stacking_results: dict = {}
    try:
        y_train = load_y(splits["y_train.parquet"])
        oof_arrays = {n: np.load(MODELS_DIR / f"{n}_oof.npy") for n in val_preds}
        oof_lens = sorted({len(v) for v in oof_arrays.values()})
        print(f"  OOF length(s): {oof_lens}")
        print(f"  y_train length: {len(y_train)}")

        if len(oof_lens) != 1:
            raise ValueError(f"OOF lengths not uniform: {oof_lens}")

        n_oof = oof_lens[0]
        if n_oof == len(y_train):
            print(f"  OOF length matches y_train exactly")
            y_for_meta = y_train
        elif n_oof < len(y_train):
            print(f"  OOF is subset ({n_oof}/{len(y_train)} = {n_oof/len(y_train):.1%})")
            print(f"  Assuming first {n_oof} rows; will verify by AUROC reconstruction")
            y_for_meta = y_train[:n_oof]
        else:
            raise ValueError(f"OOF length {n_oof} > y_train {len(y_train)}")

        # Alignment sanity: reconstruct per-model OOF AUROC vs meta.json
        print(f"\n  Alignment sanity check (reconstructed OOF AUROC vs meta.json):")
        print(f"  {'model':25s} {'expected':>10s} {'computed':>10s} {'delta':>8s} {'status':>12s}")
        max_delta = 0.0
        misaligned = []
        for n in val_preds:
            expected = json.loads((MODELS_DIR / f"{n}_meta.json").read_text())["oof_auroc"]
            try:
                computed = float(roc_auc_score(y_for_meta, oof_arrays[n]))
            except Exception:
                computed = float("nan")
            delta = abs(computed - expected) if not np.isnan(computed) else float("nan")
            max_delta = max(max_delta, delta if not np.isnan(delta) else 0)
            status = "OK" if delta < 0.005 else "MISALIGNED"
            if status == "MISALIGNED":
                misaligned.append(n)
            print(f"  {n:25s} {expected:>10.4f} {computed:>10.4f} {delta:>8.4f}   {status:>12s}")

        if max_delta > 0.005:
            print(f"\n  WARNING: max delta {max_delta:.4f} > 0.005 — OOF likely shuffled vs y_train order")
            print(f"  Skipping meta-learner; simple-average ensemble is the reliable salvage")
            stacking_results = {
                "error": "OOF rows not aligned with y_train row order",
                "max_delta": float(max_delta),
                "misaligned_models": misaligned,
                "suggestion": "Use simple-average ensemble TEST AUROC as the headline metric",
            }
        else:
            print(f"\n  Alignment OK (max delta {max_delta:.4f}); training meta-learner...")
            useful_oof = [n for n in val_preds if n not in BROKEN_MODELS]
            oof_stack = np.column_stack([oof_arrays[n].astype(np.float64) for n in useful_oof])
            meta = LogisticRegression(max_iter=5000, C=1.0)
            meta.fit(oof_stack, y_for_meta)

            val_stack = np.column_stack([val_preds[n] for n in useful_oof])
            test_stack = np.column_stack([test_preds[n] for n in useful_oof])
            meta_val_pred = meta.predict_proba(val_stack)[:, 1]
            meta_test_pred = meta.predict_proba(test_stack)[:, 1]

            stacked_val_au = float(roc_auc_score(y_val, meta_val_pred))
            stacked_test_au = float(roc_auc_score(y_test, meta_test_pred))
            print(f"  Meta-learner VAL AUROC : {stacked_val_au:.4f}")
            print(f"  Meta-learner TEST AUROC: {stacked_test_au:.4f}")
            stacking_results = {
                "members": useful_oof,
                "coefficients": dict(zip(useful_oof, meta.coef_[0].tolist())),
                "intercept": float(meta.intercept_[0]),
                "val_auroc": stacked_val_au,
                "test_auroc": stacked_test_au,
                "test_ap": float(average_precision_score(y_test, meta_test_pred)),
                "test_brier": float(brier_score_loss(y_test, meta_test_pred)),
                "oof_alignment_max_delta": float(max_delta),
                "n_train_rows_used": int(n_oof),
            }
    except Exception as exc:
        print(f"  Meta-learner step FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        stacking_results = {"error": f"{type(exc).__name__}: {exc}"}

    # [6/6] Write outputs
    print("\n[6/6] Writing outputs...")
    metrics = {
        "status": "partial — Run 10b instance destroyed mid-pipeline at ~06:00 UTC",
        "lost": ["deep_ensemble.joblib", "GNN", "cloud-computed test AUROC"],
        "recovered_via": "local CPU inference on 9 base models from --skip-kan checkpoints",
        "wall_time_seconds": round(time.time() - t0_total, 1),
        "discovered_splits": {k: str(v) for k, v in splits.items()},
        "per_model": per_model_rows,
        "ensembles": ensemble_results,
        "stacking_oof": stacking_results,
        "test_size": int(len(y_test)),
        "val_size": int(len(y_val)),
        "test_prevalence": float(y_test.mean()),
        "val_prevalence": float(y_val.mean()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics_partial.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(per_model_rows).to_csv(OUT_DIR / "per_model_metrics_partial.csv", index=False)
    print(f"  Wrote: {OUT_DIR / 'metrics_partial.json'}")
    print(f"  Wrote: {OUT_DIR / 'per_model_metrics_partial.csv'}")
    print(f"  Total wall time: {(time.time() - t0_total)/60:.1f} min")
    print("=" * 72)
    print("HEADLINE NUMBERS:")
    if "useful_no_cnn1d" in ensemble_results:
        print(f"  Simple-avg ensemble (8 useful)  TEST AUROC: {ensemble_results['useful_no_cnn1d']['test_auroc']:.4f}")
    if isinstance(stacking_results, dict) and "test_auroc" in stacking_results:
        print(f"  OOF-stacked meta-learner        TEST AUROC: {stacking_results['test_auroc']:.4f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
