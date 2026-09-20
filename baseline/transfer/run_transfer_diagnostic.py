"""Does gene-level constraint harm the stronger model specifically on genes
whose constraint annotation is PRESENT?

On the repaired cohort, evaluated on the gene-disjoint validation partition,
adding loeuf and mis_z made LightGBM Brier loss WORSE by +0.006159 with the
interval excluding zero in 5/5 seeds, while logistic regression showed no
detectable effect. loeuf and mis_z are GENE-level quantities and every
validation gene is unseen in training. A candidate mechanism is that gene-level
constraint supports calibration within genes the model has seen and misleads on
genes it has not.

That mechanism makes a falsifiable prediction: the harm should concentrate in
rows whose gene HAS constraint data, and be absent where it does not.

Strata come from EXECUTION-MATCHED availability persisted by the run, not from
a recomputed join. Estimators are the reviewed cluster bootstrap (per stratum)
and the paired-cell contrast built on its resampling design (between strata).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from paired_contrast import cluster_bootstrap_paired_contrast_ci

MODELS = ("logistic_regression", "lightgbm")
TIERS = ("core", "core_plus_constraint")


def run(predictions_path, src_root, output_path, seeds, n_boot):
    sys.path.insert(0, str(Path(src_root)))
    from genomic_variant_classifier.evaluation.metrics import cluster_bootstrap_ci

    df = pd.read_parquet(predictions_path)

    required = {"variant_id", "gene_symbol", "label", "loeuf_is_missing", "mis_z_is_missing"}
    absent = required - set(df.columns)
    if absent:
        raise ValueError(
            f"Predictions lack execution-matched availability columns: {sorted(absent)}. "
            f"This file was written before the persistence fix; re-run the experiment "
            f"rather than recomputing the constraint join.")
    for m in MODELS:
        for t in TIERS:
            if f"{m}__{t}" not in df.columns:
                raise ValueError(f"missing prediction column {m}__{t}")
    if df["variant_id"].duplicated().any():
        raise ValueError("duplicate variant_id")

    df["constraint_available"] = np.where(df["loeuf_is_missing"] == 0, "present", "absent")
    print(f"rows {len(df):,} | genes {df['gene_symbol'].nunique():,}")
    for s in ("present", "absent"):
        sub = df[df["constraint_available"] == s]
        if sub.empty:
            print(f"  {s}: 0 rows")
            continue
        print(f"  constraint {s}: {len(sub):,} rows, {sub['gene_symbol'].nunique():,} genes, "
              f"prevalence {sub['label'].mean():.6f}")

    y = df["label"].to_numpy().astype(float)
    genes = df["gene_symbol"].to_numpy()
    strata = df["constraint_available"].to_numpy()
    mean_fn = lambda yy, ss: float(np.mean(ss))

    results = {}
    for m in MODELS:
        d = (df[f"{m}__core_plus_constraint"].to_numpy() - y) ** 2 \
            - (df[f"{m}__core"].to_numpy() - y) ** 2
        entry = {"overall_delta_brier": float(np.mean(d)), "by_stratum": {}}
        print()
        print(f"=== {m}: paired tier delta (positive = constraint WORSE) ===")
        print(f"  overall: {np.mean(d):+.6f}")
        for s in ("present", "absent"):
            sel = strata == s
            if sel.sum() == 0:
                continue
            point = float(np.mean(d[sel]))
            per_seed = []
            for seed in seeds:
                lo, hi, de = cluster_bootstrap_ci(mean_fn, y[sel], d[sel], genes[sel],
                                                   n_boot=n_boot, seed=seed,
                                                   return_design_effect=True)
                per_seed.append({"seed": int(seed), "ci_low": float(lo), "ci_high": float(hi),
                                 "design_effect": float(de),
                                 "excludes_zero": bool((lo > 0) or (hi < 0))})
            verdicts = {r["excludes_zero"] for r in per_seed}
            entry["by_stratum"][s] = {
                "n_rows": int(sel.sum()),
                "n_genes": int(pd.unique(genes[sel]).size),
                "delta_brier": point,
                "per_seed": per_seed,
                "verdict_stable": len(verdicts) == 1,
            }
            flag = "STABLE" if len(verdicts) == 1 else "*** UNSTABLE ***"
            print(f"  constraint {s}: delta={point:+.6f} n={int(sel.sum()):,} -> {flag}")
            for r in per_seed:
                print(f"      seed {r['seed']}: [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] "
                      f"design_effect={r['design_effect']:.3f} excludes_zero={r['excludes_zero']}")

        if {"present", "absent"} <= set(np.unique(strata)):
            inter = []
            point_i = None
            for seed in seeds:
                pt, lo, hi, k = cluster_bootstrap_paired_contrast_ci(
                    d, strata, genes, cell_a="absent", cell_b="present",
                    n_boot=n_boot, seed=seed)
                point_i = pt
                inter.append({"seed": int(seed), "ci_low": lo, "ci_high": hi,
                              "effective_replicates": k,
                              "excludes_zero": bool((lo > 0) or (hi < 0))})
            verdicts = {r["excludes_zero"] for r in inter}
            entry["stratum_contrast_present_minus_absent"] = {
                "point": point_i, "per_seed": inter,
                "verdict_stable": len(verdicts) == 1,
            }
            flag = "STABLE" if len(verdicts) == 1 else "*** UNSTABLE ***"
            print(f"  contrast (present minus absent): {point_i:+.6f} -> {flag}")
            for r in inter:
                print(f"      seed {r['seed']}: [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] "
                      f"excludes_zero={r['excludes_zero']}")
        results[m] = entry

    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "question": "does gene-level constraint harm specifically where its annotation "
                    "is present, on unseen genes?",
        "strata_source": "execution-matched loeuf_is_missing persisted by the experiment",
        "sign_convention": "positive delta = constraint tier has HIGHER Brier loss",
        "estimators": ["cluster_bootstrap_ci (per stratum)",
                       "cluster_bootstrap_paired_contrast_ci (between strata, one gene draw)"],
        "predictions_path": str(Path(predictions_path).resolve()),
        "seeds": list(map(int, seeds)), "n_boot": n_boot,
        "results": results,
        "caveat": "A positive contrast is consistent with the gene-level-transfer "
                  "mechanism but does not establish it: rows with constraint present "
                  "differ from rows without it in ways beyond annotation availability.",
    }
    Path(output_path).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Wrote {output_path}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--src-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    run(args.predictions, args.src_root, args.output, tuple(args.seeds), args.n_boot)


if __name__ == "__main__":
    main()
