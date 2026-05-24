"""
Run 10b-partial salvage — Phase 1: inventory.

Verifies the local artifact set is complete enough to attempt Phase 2 (test eval).
No model loading, no inference; just file listing + JSON parsing + OOF shape checks.

Run from: C:\\Projects\\genomic-variant-classifier\\
Expected wall time: <30 seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\Projects\genomic-variant-classifier\outputs\run10b_final\full")
MODELS_DIR = BASE_DIR / "models"

EXPECTED_MODELS = {
    "random_forest", "xgboost", "lightgbm", "logistic_regression",
    "gradient_boosting", "catboost", "tabular_nn", "cnn_1d", "mc_dropout",
}


def main() -> int:
    print("=" * 70)
    print("Run 10b-partial salvage — Phase 1 inventory")
    print("=" * 70)

    # --- 1. Directory layout
    if not BASE_DIR.exists():
        print(f"FAIL: {BASE_DIR} does not exist")
        return 2
    if not MODELS_DIR.exists():
        print(f"FAIL: {MODELS_DIR} does not exist")
        return 2

    # --- 2. Splits
    expected_splits = [
        "X_train.parquet", "X_val.parquet", "X_test.parquet",
        "y_train.parquet", "y_val.parquet", "y_test.parquet",
    ]
    print("\n[1/4] Splits:")
    split_ok = True
    for fn in expected_splits:
        p = BASE_DIR / fn
        if not p.exists():
            print(f"  MISSING {fn}")
            split_ok = False
            continue
        try:
            df = pd.read_parquet(p)
            print(f"  OK {fn:20s} shape={df.shape}")
        except Exception as exc:
            print(f"  CORRUPT {fn}: {exc}")
            split_ok = False

    # --- 3. Models
    print("\n[2/4] Base model triplets (.joblib + _oof.npy + _meta.json):")
    have_models = {p.stem for p in MODELS_DIR.glob("*.joblib")}
    missing = EXPECTED_MODELS - have_models
    extra = have_models - EXPECTED_MODELS
    if missing:
        print(f"  MISSING models: {sorted(missing)}")
    if extra:
        print(f"  EXTRA models:   {sorted(extra)}")
    if not (missing or extra):
        print(f"  OK — all 9 expected base models present")

    # --- 4. OOF + meta self-consistency
    print("\n[3/4] OOF arrays and meta JSON:")
    print(f"  {'model':25s} {'oof_auroc':>10s} {'n_samples':>12s} {'oof_shape':>14s}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*14}")
    rows = []
    for name in sorted(have_models):
        meta_p = MODELS_DIR / f"{name}_meta.json"
        oof_p = MODELS_DIR / f"{name}_oof.npy"
        if not meta_p.exists():
            print(f"  {name:25s} MISSING meta.json")
            continue
        if not oof_p.exists():
            print(f"  {name:25s} MISSING oof.npy")
            continue
        meta = json.loads(meta_p.read_text())
        oof = np.load(oof_p)
        print(f"  {name:25s} {meta['oof_auroc']:>10.4f} {meta['n_samples']:>12d} {str(oof.shape):>14s}")
        rows.append({
            "model": name,
            "oof_auroc": meta["oof_auroc"],
            "n_samples": meta["n_samples"],
            "oof_length": int(oof.shape[0]),
            "oof_dtype": str(oof.dtype),
        })

    # --- 5. Summary
    print("\n[4/4] Summary:")
    inventory = {
        "base_dir": str(BASE_DIR),
        "splits_present": split_ok,
        "models_expected": sorted(EXPECTED_MODELS),
        "models_present": sorted(have_models),
        "models_missing": sorted(missing),
        "per_model": rows,
    }
    out = BASE_DIR / "phase1_inventory.json"
    out.write_text(json.dumps(inventory, indent=2))
    print(f"  Wrote: {out}")
    print(f"  Models present: {len(have_models)} / {len(EXPECTED_MODELS)} expected")
    print(f"  Splits ok:      {split_ok}")
    print("=" * 70)
    if missing or not split_ok:
        print("STATUS: incomplete — Phase 2 may fail on missing pieces")
        return 1
    print("STATUS: ready for Phase 2 (load models + compute test AUROC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
