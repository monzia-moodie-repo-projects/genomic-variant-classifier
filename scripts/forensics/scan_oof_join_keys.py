#!/usr/bin/env python
"""scan_oof_join_keys.py (2026-07-10)

run15_baseline's oof_predictions has a RESET dense _train_row_idx that lost the true label
mapping (verify_oof_alignment.py proved all 13 per-model AUROCs = chance). So run15 cannot be the
conformal methods-demo substrate.

But the OOF format evolved: model_insights_agent.py REQUIRES oof to have a 'label' column, which
proves some runs write oof WITH labels (or variant_id) -- a directly recoverable, verifiable join.

This scans EVERY oof_predictions / test_predictions parquet in the repo and classifies each by how
recoverable its score<->label join is:
  - HAS_LABEL      : an explicit label/y column in the oof itself (best; direct join).
  - HAS_VARIANT_ID : a variant_id column (join to meta by variant_id; robust).
  - IDX_ONLY       : only _train_row_idx / positional (run15 case; likely unrecoverable).
  - SCORES_ONLY    : neither; unusable without external alignment.

For HAS_LABEL / HAS_VARIANT_ID candidates it computes a quick per-model AUROC (using the in-file
label, or a variant_id join to the run's meta) so we can SEE which run is a valid substrate with
real signal (AUROC high) -- i.e. run the alignment gate inline for every candidate.

Read-only. ASCII-clean. Prints a ranked shortlist of viable substrates.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

MODEL_TOKENS = ("random_forest", "xgboost", "lightgbm", "svm", "logistic", "gradient",
                "catboost", "tabular_nn", "cnn", "kan", "mc_dropout", "deep_ensemble",
                "gnn", "ensemble", "blend", "_prob")
LABEL_NAMES = ("label", "y", "y_true", "target")
EXCLUDE = (".venv312", ".venv", ".git", "site-packages", "__pycache__")


def line(c="-", n=78):
    print(c * n)


def _model_cols(cols):
    return [c for c in cols if any(t in str(c).lower() for t in MODEL_TOKENS)
            and str(c).lower() not in LABEL_NAMES and c != "_train_row_idx"]


def _classify(cols):
    low = [str(c).lower() for c in cols]
    if any(l in LABEL_NAMES for l in low):
        return "HAS_LABEL"
    if "variant_id" in low:
        return "HAS_VARIANT_ID"
    if "_train_row_idx" in low:
        return "IDX_ONLY"
    return "SCORES_ONLY"


def main():
    print("=" * 78)
    print("OOF JOIN-KEY SCAN (find a verifiable conformal substrate; run15 is broken)")
    print("=" * 78)

    files = []
    for pat in ["outputs/**/oof_predictions.parquet", "outputs/**/test_predictions.parquet",
                "outputs/**/predictions.parquet"]:
        files += glob.glob(pat, recursive=True)
    files = sorted(set(f for f in files if not any(e in f for e in EXCLUDE)))
    if not files:
        print("No oof/test prediction parquets found under outputs/.")
        return 0

    rows = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"  [read error] {f}: {type(e).__name__}: {e}")
            continue
        cls = _classify(df.columns)
        mcols = _model_cols(df.columns)
        rows.append((f, len(df), cls, mcols, df))

    # print classification table
    print(f"{'rows':>10}  {'class':14}  {'#models':>7}  path")
    for f, n, cls, mcols, _ in sorted(rows, key=lambda r: (r[2], -r[1])):
        print(f"{n:>10,}  {cls:14}  {len(mcols):>7}  {f}")
    line("=")

    # for HAS_LABEL candidates, run the inline AUROC gate
    print("Inline alignment check for HAS_LABEL candidates (per-model AUROC via in-file label):")
    viable = []
    for f, n, cls, mcols, df in rows:
        if cls != "HAS_LABEL" or roc_auc_score is None or not mcols:
            continue
        lc = next(c for c in df.columns if str(c).lower() in LABEL_NAMES)
        y = pd.to_numeric(df[lc], errors="coerce").values
        if len(np.unique(y[~np.isnan(y)])) < 2:
            print(f"  {Path(f).parent.name}: label not binary/degenerate; skip")
            continue
        aurocs = []
        for c in mcols:
            s = pd.to_numeric(df[c], errors="coerce").values
            m = ~(np.isnan(s) | np.isnan(y))
            if m.sum() > 100 and len(np.unique(y[m])) == 2:
                try:
                    aurocs.append(roc_auc_score(y[m], s[m]))
                except Exception:
                    pass
        if aurocs:
            lo, hi = min(aurocs), max(aurocs)
            ok = lo >= 0.90
            print(f"  {f}")
            print(f"      rows {n:,}  models {len(aurocs)}  per-model AUROC min {lo:.4f} max {hi:.4f}"
                  f"  {'<== VIABLE (aligned)' if ok else '<== labels present but low AUROC'}")
            if ok:
                viable.append((f, n, lo, hi))
    line("=")

    if viable:
        print("VIABLE SUBSTRATES (label present + per-model AUROC high == verifiable join):")
        for f, n, lo, hi in sorted(viable, key=lambda r: -r[1]):
            print(f"  rows {n:,}  AUROC {lo:.4f}-{hi:.4f}  {f}")
        print("\n=> Pick the largest VIABLE substrate for the conformal methods demo.")
    else:
        print("NO viable HAS_LABEL substrate found with high AUROC. Next option: HAS_VARIANT_ID")
        print("runs (join to meta by variant_id) or go straight to the re-baseline retrain.")
    line("=")
    print("SCAN COMPLETE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
