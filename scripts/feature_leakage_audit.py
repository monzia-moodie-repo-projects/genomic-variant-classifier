#!/usr/bin/env python
"""feature_leakage_audit.py -- READ-ONLY lone-feature leakage audit for the Phase-2 connectors.

Generalises cosmic_leakage_audit.py to audit ANY set of engineered features against the
binary label on a real split matrix. Default targets are this session's new connectors:
    genomiclm_delta_norm, genomiclm_llr   (Nucleotide Transformer)
    cosmic_recurrence, cosmic_sig_tier     (COSMIC CMC)
    kegg_pathway_count, kegg_disease_pathway_flag  (KEGG)

For each feature it reports coverage (overall + by class), point-biserial correlation with
the label, and the LONE-FEATURE rank AUROC (Mann-Whitney) -- the actual leakage test: a
single feature that ranks the label too well may be a label proxy, not an honest signal.
A non-zero coverage count is NOT sufficient evidence a feature is safe.

Also hard-guards that no label-bearing column (CLINVAR_CLNSIG-type) is in the matrix.

    python scripts/feature_leakage_audit.py --matrix <X_with_label.parquet>
    python scripts/feature_leakage_audit.py --x <X_train.parquet> --meta <meta_train.parquet>
    python scripts/feature_leakage_audit.py --x ... --meta ... --features genomiclm_delta_norm cosmic_recurrence
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

_LABEL_CANDIDATES = ["label", "y", "y_true", "binary_label", "target"]
_LEAK_PATTERNS = ["clnsig", "clinvar_cln"]
_DEFAULT_FEATURES = [
    "genomiclm_delta_norm", "genomiclm_llr",
    "cosmic_recurrence", "cosmic_sig_tier",
    "kegg_pathway_count", "kegg_disease_pathway_flag",
]


def _rank_auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney rank AUROC of a single score vs a binary label; NaN if one class/empty."""
    m = ~np.isnan(score)
    s, y = score[m], label[m]
    pos_n = int((y == 1).sum()); neg_n = int((y == 0).sum())
    if pos_n == 0 or neg_n == 0:
        return float("nan")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {i: (csum[i] - counts[i] + 1 + csum[i]) / 2.0 for i in range(len(counts))}
    ranks = np.array([avg[i] for i in inv])
    r_pos = ranks[y == 1].sum()
    return float((r_pos - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n))


def _corr(a: np.ndarray, y: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(y)
    if m.sum() < 3 or np.std(a[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], y[m])[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=None, help="one parquet with features + label")
    ap.add_argument("--x", default=None, help="features parquet")
    ap.add_argument("--meta", default=None, help="meta parquet (label); row-aligned to --x")
    ap.add_argument("--features", nargs="*", default=None,
                    help=f"features to audit (default: {_DEFAULT_FEATURES})")
    ap.add_argument("--flag-auroc", type=float, default=0.70,
                    help="lone-feature AUROC >= this => flag as possible label proxy")
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

    print("=" * 72)
    print("Feature leakage audit (lone-feature rank AUROC)")
    print("=" * 72)
    print(f"rows: {len(df)}  cols: {df.shape[1]}")

    leaks = [c for c in df.columns if any(p in c.lower() for p in _LEAK_PATTERNS)]
    if leaks:
        print(f"*** LEAK: label-bearing column(s) in the matrix: {leaks} -- STOP and fix wiring.")
        return 3
    print("[ok] no CLINVAR_CLNSIG-type column in the matrix")

    label_col = next((c for c in _LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        print(f"FAIL: no label column found (looked for {_LABEL_CANDIDATES})."); return 2
    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy()
    if set(np.unique(y[~np.isnan(y)])) - {0.0, 1.0}:
        print(f"note: label '{label_col}' not strictly binary; treating >0 as positive.")
        y = (y > 0).astype(float)
    print(f"label: '{label_col}'  pos={int(np.nansum(y))} neg={int((y==0).sum())}\n")

    feats = args.features if args.features else _DEFAULT_FEATURES
    present = [f for f in feats if f in df.columns]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"note: not in matrix (skipped): {missing}\n")
    if not present:
        print("FAIL: none of the requested features are in the matrix."); return 2

    flagged, dead = [], []
    hdr = f"{'feature':30s} {'cover%':>7s} {'path%':>6s} {'ben%':>6s} {'r':>7s} {'AUROC':>7s}"
    print(hdr); print("-" * len(hdr))
    for f in present:
        v = pd.to_numeric(df[f], errors="coerce").to_numpy()
        hit = v != 0
        cover = 100.0 * np.nanmean(hit) if len(v) else 0.0
        pth = 100.0 * (hit & (y == 1)).sum() / max((y == 1).sum(), 1)
        ben = 100.0 * (hit & (y == 0)).sum() / max((y == 0).sum(), 1)
        auroc = _rank_auroc(v, y)
        r = _corr(v, y)
        astr = f"{auroc:.3f}" if not np.isnan(auroc) else "  n/a"
        print(f"{f:30s} {cover:7.2f} {pth:6.1f} {ben:6.1f} {r:+7.3f} {astr:>7s}")
        if np.isnan(auroc) or cover == 0.0:
            dead.append(f)
        elif auroc >= args.flag_auroc or auroc <= (1.0 - args.flag_auroc):
            flagged.append((f, auroc))

    print("\n-- verdict --")
    if dead:
        print(f"DEAD/undefined (0 coverage or one-class): {dead}")
    if flagged:
        print(f"FLAG (lone-feature AUROC extreme, possible label proxy): "
              f"{[(f, round(a,3)) for f,a in flagged]}")
        print("Investigate: does coverage concentrate on known-pathogenic genes? Is the feature")
        print("derived (directly or via its source) from anything label-adjacent?")
        return 1
    if not dead:
        print(f"CLEAN: all audited features have lone-feature AUROC within "
              f"[{1-args.flag_auroc:.2f}, {args.flag_auroc:.2f}] -- honest signal, not label proxies.")
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
