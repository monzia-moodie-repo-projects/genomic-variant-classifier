#!/usr/bin/env python
"""cosmic_leakage_audit.py -- READ-ONLY leakage audit for the COSMIC CMC features.

COSMIC is somatic; the label is germline ClinVar pathogenicity -- so recurrence should
be an INDEPENDENT feature. But CMC ships a CLINVAR_CLNSIG column, and recurrence could
still proxy the label on overlapping variants. This audit answers, on the real annotated
matrix, whether cosmic_recurrence / cosmic_sig_tier are honest features or a laundered
label. A non-zero coverage count is NOT sufficient; the lone-feature AUROC is the test.

Reads a parquet that carries cosmic_recurrence + cosmic_sig_tier + the binary label
(e.g. a splits X_*.parquet joined to its meta_*.parquet, or any matrix with both). No
writes, no model training beyond a rank-based lone-feature AUROC.

    python scripts/cosmic_leakage_audit.py --matrix outputs/run/splits/train_matrix.parquet
    python scripts/cosmic_leakage_audit.py --x outputs/run/splits/X_train.parquet --meta outputs/run/meta_train.parquet
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

_LABEL_CANDIDATES = ["label", "y", "y_true", "binary_label", "target"]
_LEAK_PATTERNS = ["clnsig", "clinvar_cln", "clinvar_clnsig"]


def _rank_auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney rank AUROC of a single score vs a binary label; NaN if one class."""
    m = ~np.isnan(score)
    s, y = score[m], label[m]
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {i: (csum[i] - counts[i] + 1 + csum[i]) / 2.0 for i in range(len(counts))}
    ranks = np.array([avg[i] for i in inv])
    r_pos = ranks[y == 1].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=None, help="one parquet with cosmic_* + label")
    ap.add_argument("--x", default=None, help="features parquet (cosmic_*)")
    ap.add_argument("--meta", default=None, help="meta parquet (label); row-aligned to --x")
    ap.add_argument("--flag-auroc", type=float, default=0.70,
                    help="lone-feature AUROC above this => flag as possible label proxy")
    args = ap.parse_args()

    if args.matrix:
        df = pd.read_parquet(args.matrix)
    elif args.x and args.meta:
        x = pd.read_parquet(args.x).reset_index(drop=True)
        m = pd.read_parquet(args.meta).reset_index(drop=True)
        if len(x) != len(m):
            print(f"FAIL: --x ({len(x)}) and --meta ({len(m)}) row counts differ."); return 2
        df = pd.concat([x, m[[c for c in m.columns if c not in x.columns]]], axis=1)
    else:
        print("Provide --matrix, or both --x and --meta."); return 2

    print("=" * 70)
    print("COSMIC CMC leakage audit")
    print("=" * 70)
    print(f"rows: {len(df)}  cols: {df.shape[1]}")

    # -- hard leak guard: CLINVAR_CLNSIG (the label) must NOT be in the matrix ----
    leaks = [c for c in df.columns if any(p in c.lower() for p in _LEAK_PATTERNS)]
    if leaks:
        print(f"*** LEAK: label-bearing column(s) present in the matrix: {leaks}")
        print("*** connector or wiring is reading CLINVAR_CLNSIG -- STOP and fix.")
        return 3
    print("[ok] no CLINVAR_CLNSIG-type column in the matrix")

    for c in ("cosmic_recurrence", "cosmic_sig_tier"):
        if c not in df.columns:
            print(f"FAIL: {c} absent -- was COSMIC wiring installed + activated?"); return 2

    label_col = next((c for c in _LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        print(f"FAIL: no label column found (looked for {_LABEL_CANDIDATES})."); return 2
    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy()
    if set(np.unique(y[~np.isnan(y)])) - {0.0, 1.0}:
        print(f"note: label '{label_col}' is not strictly binary; treating >0 as positive.")
        y = (y > 0).astype(float)
    print(f"label column: '{label_col}'  (pos={int(np.nansum(y))}, neg={int((y==0).sum())})")

    rec = pd.to_numeric(df["cosmic_recurrence"], errors="coerce").to_numpy()
    tier = pd.to_numeric(df["cosmic_sig_tier"], errors="coerce").to_numpy()

    # -- coverage overall + by class -------------------------------------------
    hit = rec > 0
    print("\n-- coverage --")
    print(f"cosmic_recurrence > 0 : {hit.sum()} / {len(df)}  ({100*hit.mean():.2f}%)")
    for cls, name in ((1, "pathogenic"), (0, "benign")):
        msk = (y == cls)
        if msk.sum():
            print(f"  within {name:10s}: {int((hit & msk).sum())}/{int(msk.sum())} "
                  f"({100*(hit & msk).sum()/msk.sum():.2f}%) have recurrence>0")
    print(f"cosmic_sig_tier  > 0  : {int((tier>0).sum())} / {len(df)}  ({100*(tier>0).mean():.2f}%)")

    # -- association + lone-feature AUROC (the leakage test) --------------------
    print("\n-- association with label (lone-feature) --")
    auc_rec = _rank_auroc(rec, y)
    auc_tier = _rank_auroc(tier, y)
    # point-biserial (Pearson of feature vs 0/1 label) on overlap
    def _corr(a):
        m = ~np.isnan(a) & ~np.isnan(y)
        if m.sum() < 3 or np.std(a[m]) == 0:
            return float("nan")
        return float(np.corrcoef(a[m], y[m])[0, 1])
    print(f"cosmic_recurrence : lone-feature AUROC = {auc_rec:.4f}   point-biserial r = {_corr(rec):+.4f}")
    print(f"cosmic_sig_tier   : lone-feature AUROC = {auc_tier:.4f}   point-biserial r = {_corr(tier):+.4f}")
    # mean feature value by class (interpretability)
    for name, a in (("recurrence", rec), ("sig_tier", tier)):
        mp = np.nanmean(a[y == 1]) if (y == 1).any() else float("nan")
        mn = np.nanmean(a[y == 0]) if (y == 0).any() else float("nan")
        print(f"  mean {name:10s}: pathogenic={mp:.4f}  benign={mn:.4f}")

    # -- verdict ---------------------------------------------------------------
    print("\n-- verdict --")
    flagged = [n for n, a in (("cosmic_recurrence", auc_rec), ("cosmic_sig_tier", auc_tier))
               if not np.isnan(a) and a >= args.flag_auroc]
    if flagged:
        print(f"FLAG: {flagged} have lone-feature AUROC >= {args.flag_auroc:.2f} -- recurrence may")
        print("be proxying the label on overlapping variants. Investigate before trusting the")
        print("feature: check whether COSMIC coverage concentrates on known-pathogenic genes.")
        return 1
    print(f"CLEAN: both lone-feature AUROCs < {args.flag_auroc:.2f} -- cosmic_* behave as independent")
    print("recurrence signal, not a laundered label. (Coverage above is the honest picture.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
