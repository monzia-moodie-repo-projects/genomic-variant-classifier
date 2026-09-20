"""Direct paired contrast between the strong arms.

The four-arm run contrasted everything against lr_current, so the differences
BETWEEN lr_representation, lr_splines and lightgbm have no interval. Whether a
well-specified linear model MATCHES gradient boosting is a different question
from whether it beats a badly-specified one, and it needs its own contrast.
"""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

PAIRS = [("lr_representation", "lightgbm"),
         ("lr_splines", "lightgbm"),
         ("lr_representation", "lr_splines")]

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--predictions", required=True)
p.add_argument("--src-root", required=True)
p.add_argument("--output", required=True)
p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
p.add_argument("--n-boot", type=int, default=2000)
a = p.parse_args()

sys.path.insert(0, str(Path(a.src_root)))
from genomic_variant_classifier.evaluation.metrics import cluster_bootstrap_ci

df = pd.read_parquet(a.predictions)
y = df["label"].to_numpy().astype(float)
genes = df["gene_symbol"].to_numpy()
mean_fn = lambda yy, ss: float(np.mean(ss))

results = {}
print("negative delta = FIRST arm has lower Brier loss")
for first, second in PAIRS:
    for c in (first, second):
        if c not in df.columns:
            raise ValueError(f"missing prediction column {c}")
    d = (df[first].to_numpy() - y) ** 2 - (df[second].to_numpy() - y) ** 2
    point = float(np.mean(d))
    per_seed = []
    for s in a.seeds:
        lo, hi, de = cluster_bootstrap_ci(mean_fn, y, d, genes, n_boot=a.n_boot,
                                          seed=s, return_design_effect=True)
        per_seed.append({"seed": int(s), "ci_low": float(lo), "ci_high": float(hi),
                         "design_effect": float(de),
                         "excludes_zero": bool((lo > 0) or (hi < 0))})
    verdicts = {r["excludes_zero"] for r in per_seed}
    key = f"{first}__minus__{second}"
    results[key] = {"delta_brier": point, "per_seed": per_seed,
                    "verdict_stable": len(verdicts) == 1}
    flag = "STABLE" if len(verdicts) == 1 else "*** UNSTABLE ***"
    print(f"  {key}: delta={point:+.6f} -> {flag}")
    for r in per_seed:
        print(f"      seed {r['seed']}: [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] "
              f"design_effect={r['design_effect']:.3f} excludes_zero={r['excludes_zero']}")

Path(a.output).write_text(json.dumps({
    "run_utc": datetime.now(timezone.utc).isoformat(),
    "question": "does a well-specified linear model MATCH gradient boosting?",
    "sign_convention": "negative = first arm lower Brier loss",
    "predictions_path": str(Path(a.predictions).resolve()),
    "seeds": list(map(int, a.seeds)), "n_boot": a.n_boot,
    "results": results,
}, indent=2, default=str), encoding="utf-8")
print()
print(f"Wrote {a.output}")
