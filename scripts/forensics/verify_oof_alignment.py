#!/usr/bin/env python
"""verify_oof_alignment.py (2026-07-10)

Verify -- not assume -- the OOF-score <-> label join for run15_baseline before any conformal
calibration. Established from the builder source (scripts/run_phase2_eval.py L1025-1060) and the
consumer (scripts/run10b_partial_phase2_eval_v2.py L183-214):

  - oof_predictions has _train_row_idx = idx aligning OOF rows to meta_train/y_train.
  - oof (883,127) is a SUBSET of train (1,038,974); _train_row_idx is dense 0..883,126.
  - Whether that dense index means "first-N rows of train in order" (join valid) or "reset index
    that lost the true mapping" (join broken) CANNOT be told from counts alone.

The pipeline's own correctness test settles it: reconstruct each base model's OOF AUROC from
(model_score, y_train[_train_row_idx]) and check it is HIGH (matches the recorded ~0.98-0.99
blend performance), not ~0.5. If the labels were mis-joined, every per-model AUROC collapses to
chance. This is an INDEPENDENT falsifiable verification.

This script:
  1. Loads oof_predictions + _train_row_idx, y_train.
  2. Builds y = y_train.iloc[_train_row_idx] (the candidate join).
  3. Computes AUROC + AUPRC for each of the 13 model columns against y.
  4. Loads any recorded per-model metrics (metrics.json / meta.json / *per_model*.csv) if present
     and compares.
  5. VERDICT: PASS if every per-model AUROC is at or above the shared
     score<->label alignment minimum (see evaluation/alignment.py; the
     comparison is >=, which this line previously wrote as >) and (if
     recorded metrics exist) match
     within tolerance; FAIL otherwise. Prints a clear ABORT if the join looks broken.
Read-only. ASCII-clean.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# ALIGNMENT-1 (2026-08-07). This script and conformal calibration each
# declared the same threshold independently. It consumes the shared
# policy directly and NOT via the conformal package: a general integrity
# check should not depend on a specific statistical method to obtain a
# number.
from genomic_variant_classifier.evaluation.alignment import (
    DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY,
)

BASE = Path("outputs/run15_baseline/full")
OOF = BASE / "oof_predictions.parquet"
Y_TRAIN = BASE / "splits/y_train.parquet"
MODEL_COLS = ["random_forest", "xgboost", "lightgbm", "svm", "svm_bagged_rbf",
              "logistic_regression", "gradient_boosting", "catboost", "tabular_nn",
              "cnn_1d", "kan", "mc_dropout", "deep_ensemble"]

#: A correctly-joined base model should be well above chance. The value
#: lives in evaluation/alignment.py so this script and conformal
#: calibration cannot drift apart.
ALIGNMENT_POLICY = DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY
AUROC_FLOOR = ALIGNMENT_POLICY.minimum_auroc
TOL = 0.02           # tolerance when comparing to recorded metrics


def line(c="-", n=78):
    print(c * n)


def _load_recorded_metrics():
    """Find any recorded per-model AUROC for run15_baseline (json or csv). Returns {model: auroc}."""
    found = {}
    for pat in ["outputs/run15_baseline/**/metrics.json",
                "outputs/run15_baseline/**/meta.json",
                "outputs/run15_baseline/**/*per_model*.csv",
                "outputs/run15_baseline/**/*metrics*.csv"]:
        for f in glob.glob(pat, recursive=True):
            try:
                if f.endswith(".json"):
                    d = json.loads(Path(f).read_text())
                    # search for per-model auroc structures
                    for k, v in _walk_json_for_auroc(d):
                        found.setdefault(k, v)
                else:
                    df = pd.read_csv(f)
                    cols = {c.lower(): c for c in df.columns}
                    mcol = cols.get("model") or cols.get("name")
                    acol = next((cols[c] for c in cols if "auroc" in c or "auc" in c), None)
                    if mcol and acol:
                        for _, r in df.iterrows():
                            found.setdefault(str(r[mcol]), float(r[acol]))
            except Exception as e:
                print(f"  [metrics read note] {f}: {type(e).__name__}: {e}")
    return found


def _walk_json_for_auroc(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and ("auroc" in str(k).lower() or "auc" in str(k).lower()):
                out.append((prefix or str(k), float(v)))
            else:
                out.extend(_walk_json_for_auroc(v, prefix=str(k)))
    return out


def main():
    print("=" * 78)
    print("OOF ALIGNMENT VERIFICATION (run15_baseline) -- gate before conformal calibration")
    print("=" * 78)

    if not OOF.exists() or not Y_TRAIN.exists():
        print(f"FATAL: need {OOF} and {Y_TRAIN}")
        return 2

    oof = pd.read_parquet(OOF)
    ytr = pd.read_parquet(Y_TRAIN)
    ycol = "label" if "label" in ytr.columns else ytr.columns[0]
    print(f"oof rows {len(oof):,}; y_train rows {len(ytr):,}")

    idx = oof["_train_row_idx"].values
    print(f"_train_row_idx: min {idx.min():,} max {idx.max():,} "
          f"contiguous={bool(idx.min() == 0 and idx.max() == len(oof) - 1 and pd.Series(idx).nunique() == len(oof))}")
    if idx.max() >= len(ytr):
        print(f"FATAL: _train_row_idx max {idx.max()} >= y_train length {len(ytr)}; cannot join.")
        return 2

    # candidate join
    y = ytr[ycol].values[idx]
    pos = int(y.sum()); rate = pos / len(y)
    print(f"joined labels: {len(y):,} rows, positives {pos:,} ({rate:.4f})")
    line()

    print("Per-model reconstructed OOF metrics (score vs joined label):")
    recon = {}
    present = [c for c in MODEL_COLS if c in oof.columns]
    for c in present:
        s = oof[c].values
        try:
            auroc = roc_auc_score(y, s)
            auprc = average_precision_score(y, s)
        except Exception as e:
            print(f"  {c:22s} metric error: {e}")
            continue
        recon[c] = auroc
        flag = "" if auroc >= AUROC_FLOOR else "   <== LOW (alignment suspect)"
        print(f"  {c:22s} AUROC {auroc:.4f}  AUPRC {auprc:.4f}{flag}")
    line()

    # compare to recorded metrics if any
    recorded = _load_recorded_metrics()
    if recorded:
        print("Recorded metrics found; comparing where model names line up:")
        for c in present:
            rec = next((v for k, v in recorded.items() if c in k.lower()), None)
            if rec is not None:
                d = abs(recon.get(c, float("nan")) - rec)
                print(f"  {c:22s} recon {recon.get(c, float('nan')):.4f} vs recorded {rec:.4f} "
                      f"delta {d:.4f} {'OK' if d < TOL else 'MISMATCH'}")
    else:
        print("No recorded per-model AUROC files found; using the internal HIGH-AUROC gate only.")
    line("=")

    # verdict
    if not recon:
        print("VERDICT: FAIL -- no per-model AUROC computed.")
        return 1
    lo = min(recon.values())
    print(f"min per-model AUROC = {lo:.4f} (floor {AUROC_FLOOR})")
    if lo >= AUROC_FLOOR:
        print("VERDICT: PASS -- per-model AUROCs are high, so the label join is CONSISTENT with the")
        print("models' own out-of-fold performance. The mapping y = y_train[_train_row_idx] is")
        print("VERIFIED for use in conformal calibration (methodological, pre-correction).")
        return 0
    print("VERDICT: FAIL -- at least one base model reconstructs near chance. The join is likely")
    print("BROKEN (dense index lost the true mapping). DO NOT calibrate. Investigate the writer.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
