"""Interval for the repair experiment interaction, across seeds."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from paired_contrast import cluster_bootstrap_paired_contrast_ci

MODELS = ("logistic_regression", "lightgbm")
REPORTED = {"logistic_regression": -0.126662, "lightgbm": -0.003073}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    df = pd.read_parquet(args.predictions)
    y = df["label"].to_numpy().astype(float)
    cells = df["evaluation_cell"].to_numpy()
    genes = df["gene_symbol"].to_numpy()

    results = {}
    for model in MODELS:
        pl = df[f"{model}__legacy"].to_numpy().astype(float)
        pc = df[f"{model}__corrected"].to_numpy().astype(float)
        d = (pc - y) ** 2 - (pl - y) ** 2
        per_seed = []
        point = None
        for s in args.seeds:
            pt, lo, hi, k = cluster_bootstrap_paired_contrast_ci(
                d, cells, genes, cell_a="common", cell_b="added",
                n_boot=args.n_boot, seed=s)
            point = pt
            per_seed.append({"seed": int(s), "ci_low": lo, "ci_high": hi,
                             "effective_replicates": k,
                             "excludes_zero": bool((lo > 0) or (hi < 0))})
        verdicts = {r["excludes_zero"] for r in per_seed}
        agrees = abs(point - REPORTED[model]) < 5e-6
        results[model] = {"interaction_point": point,
                          "matches_reported_point": bool(agrees),
                          "reported_point": REPORTED[model],
                          "per_seed": per_seed,
                          "verdict_stable": len(verdicts) == 1}
        flag = "STABLE" if len(verdicts) == 1 else "*** UNSTABLE ***"
        print(f"{model}: interaction={point:+.6f} "
              f"(matches reported point: {agrees}) -> {flag}")
        for r in per_seed:
            print(f"    seed {r['seed']}: [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] "
                  f"excludes_zero={r['excludes_zero']} reps={r['effective_replicates']}")

    manifest = {"run_utc": datetime.now(timezone.utc).isoformat(),
                "estimator": "cluster_bootstrap_paired_contrast_ci (proposed addition "
                             "to evaluation.metrics; same resampling design as "
                             "cluster_bootstrap_ci, both cells in one gene draw)",
                "contrast": "mean(d|added) - mean(d|common)",
                "sign_convention": "negative = corrected improves added cell more than common",
                "seeds": list(map(int, args.seeds)), "n_boot": args.n_boot,
                "results": results}
    Path(args.output).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
