"""Uncertainty for the repair experiment primary contrasts, using the
project EXISTING reviewed gene-cluster bootstrap. No second bootstrap
implementation is introduced.

The paired Brier delta on a set of rows is the MEAN of a per-row quantity:
    d_i = (p_corrected_i - y_i)**2 - (p_legacy_i - y_i)**2
    delta = mean(d_i)
so passing d as the score argument with fn = mean makes cluster_bootstrap_ci
resample WHOLE GENES and recompute the paired delta on each resample.

Negative delta means the corrected-cohort model has LOWER Brier loss.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("logistic_regression", "lightgbm")
CELLS = ("common", "added")


def paired_delta(df, model):
    y = df["label"].to_numpy().astype(float)
    pl = df[f"{model}__legacy"].to_numpy().astype(float)
    pc = df[f"{model}__corrected"].to_numpy().astype(float)
    for name, p in (("legacy", pl), ("corrected", pc)):
        if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
            raise ValueError(f"{model}/{name}: probabilities outside [0,1] or non-finite")
    return (pc - y) ** 2 - (pl - y) ** 2


def run(predictions_path, src_root, output_path, n_boot, seed):
    sys.path.insert(0, str(Path(src_root)))
    from genomic_variant_classifier.evaluation.metrics import cluster_bootstrap_ci

    df = pd.read_parquet(predictions_path)
    required = {"variant_id", "gene_symbol", "evaluation_cell", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    if df["variant_id"].duplicated().any():
        raise ValueError("Duplicate variant_id in predictions")

    print(f"rows: {len(df):,} | genes: {df['gene_symbol'].nunique():,}")
    for c in CELLS:
        sub = df[df["evaluation_cell"] == c]
        print(f"  {c}: {len(sub):,} rows, {sub['gene_symbol'].nunique():,} genes, "
              f"prevalence {sub['label'].mean():.6f}")

    mean_fn = lambda yy, ss: float(np.mean(ss))
    results = {}
    print()
    print(f"=== paired Brier deltas, gene-cluster bootstrap (n_boot={n_boot}) ===")
    for model in MODELS:
        results[model] = {}
        d_all = paired_delta(df, model)
        for c in CELLS:
            sel = (df["evaluation_cell"] == c).to_numpy()
            d = d_all[sel]
            y = df.loc[sel, "label"].to_numpy()
            genes = df.loc[sel, "gene_symbol"].to_numpy()
            point = float(np.mean(d))
            lo, hi, de = cluster_bootstrap_ci(mean_fn, y, d, genes, n_boot=n_boot,
                                               seed=seed, return_design_effect=True)
            excludes_zero = bool((lo > 0) or (hi < 0))
            results[model][c] = {
                "n_rows": int(sel.sum()), "n_genes": int(pd.unique(genes).size),
                "delta_brier": point, "ci_low": float(lo), "ci_high": float(hi),
                "design_effect": float(de), "ci_excludes_zero": excludes_zero,
            }
            verdict = "excludes zero" if excludes_zero else "INCLUDES zero"
            print(f"  {model} / {c}: delta={point:+.6f} "
                  f"CI=[{lo:+.6f}, {hi:+.6f}] design_effect={de:.3f} -> {verdict}")

    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": "cluster_bootstrap_ci from the project reviewed evaluation.metrics; "
                     "paired per-row Brier delta supplied as score with fn=mean",
        "clusters": "gene_symbol",
        "n_boot": n_boot, "seed": seed,
        "sign_convention": "negative delta = corrected-cohort model has lower Brier loss",
        "predictions_path": str(Path(predictions_path).resolve()),
        "results": results,
        "interaction_note": (
            "A confidence interval for the interaction (delta_added - delta_common) "
            "requires resampling BOTH cells jointly within one gene draw. The reviewed "
            "single-score interface cannot express that without repurposing its label "
            "argument, which would change its validated semantics. An interval for it "
            "needs a deliberate, reviewed extension to evaluation.metrics."
        ),
    }
    Path(output_path).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Wrote {output_path}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--src-root", required=True, help="Path to the repo src directory")
    p.add_argument("--output", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.predictions, args.src_root, args.output, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
