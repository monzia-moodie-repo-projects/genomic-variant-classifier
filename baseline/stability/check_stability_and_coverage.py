"""Two things required before the repair result is reportable.

(1) REPEAT-RUN STABILITY. The LightGBM common-population interval sits very
close to zero. A conclusion that hinges on one bootstrap seed is not a
conclusion. Re-runs the SAME reviewed estimator across several seeds.

(2) ANNOTATION AVAILABILITY BY POPULATION. If added rows have systematically
different constraint coverage from common rows, that composition difference
must be reported alongside any metric difference.

Read-only. Introduces no new estimator.
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
    return (pc - y) ** 2 - (pl - y) ** 2


def load_constraint_genes(path, prefer="ENST"):
    cols = ["gene", "transcript", "mane_select", "lof.oe_ci.upper", "mis.z_score"]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in cols, low_memory=False)
    c = df[df["mane_select"] == True].copy()
    t = c["transcript"].astype(str)
    c["namespace"] = "other"
    c.loc[t.str.startswith("ENST"), "namespace"] = "ENST"
    c.loc[t.str.startswith(("NM_", "NR_", "XM_", "XR_")), "namespace"] = "NCBI"
    c = c[c["namespace"].isin(["ENST", "NCBI"]) & c["gene"].notna()].copy()
    fallback = "NCBI" if prefer == "ENST" else "ENST"
    c["_rank"] = c["namespace"].map({prefer: 0, fallback: 1})
    c = c.sort_values(["gene", "_rank"]).drop_duplicates(subset=["gene"], keep="first")
    return c.rename(columns={"lof.oe_ci.upper": "loeuf", "mis.z_score": "mis_z"})[
        ["gene", "loeuf", "mis_z"]]


def run(predictions_path, src_root, gnomad_path, output_path, seeds, n_boot):
    sys.path.insert(0, str(Path(src_root)))
    from genomic_variant_classifier.evaluation.metrics import cluster_bootstrap_ci

    df = pd.read_parquet(predictions_path)
    mean_fn = lambda yy, ss: float(np.mean(ss))

    print(f"=== (1) repeat-run stability across seeds {list(seeds)} (n_boot={n_boot}) ===")
    stability = {}
    for model in MODELS:
        d_all = paired_delta(df, model)
        stability[model] = {}
        for c in CELLS:
            sel = (df["evaluation_cell"] == c).to_numpy()
            d, y = d_all[sel], df.loc[sel, "label"].to_numpy()
            genes = df.loc[sel, "gene_symbol"].to_numpy()
            point = float(np.mean(d))
            per_seed = []
            for s in seeds:
                lo, hi = cluster_bootstrap_ci(mean_fn, y, d, genes, n_boot=n_boot, seed=s)
                per_seed.append({"seed": int(s), "ci_low": float(lo), "ci_high": float(hi),
                                 "excludes_zero": bool((lo > 0) or (hi < 0))})
            verdicts = {r["excludes_zero"] for r in per_seed}
            stability[model][c] = {"delta_brier": point, "per_seed": per_seed,
                                    "verdict_stable": len(verdicts) == 1,
                                    "verdict": (per_seed[0]["excludes_zero"]
                                                if len(verdicts) == 1 else "UNSTABLE")}
            flag = "STABLE" if len(verdicts) == 1 else "*** UNSTABLE ACROSS SEEDS ***"
            print(f"  {model}/{c}: delta={point:+.6f} -> {flag}")
            for r in per_seed:
                print(f"      seed {r['seed']}: [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] "
                      f"excludes_zero={r['excludes_zero']}")

    print()
    print("=== (2) annotation availability by evaluation population ===")
    constraint = load_constraint_genes(gnomad_path)
    merged = df.merge(constraint, left_on="gene_symbol", right_on="gene", how="left")
    coverage = {}
    for c in CELLS:
        sub = merged[merged["evaluation_cell"] == c]
        cov = {
            "n_rows": int(len(sub)),
            "n_genes": int(sub["gene_symbol"].nunique()),
            "prevalence": float(sub["label"].mean()),
            "loeuf_present_rows": int(sub["loeuf"].notna().sum()),
            "loeuf_present_fraction": float(sub["loeuf"].notna().mean()),
            "mis_z_present_fraction": float(sub["mis_z"].notna().mean()),
            "mean_rows_per_gene": float(len(sub) / max(sub["gene_symbol"].nunique(), 1)),
        }
        coverage[c] = cov
        print(f"  {c}: rows={cov['n_rows']:,} genes={cov['n_genes']:,} "
              f"prevalence={cov['prevalence']:.4f} "
              f"loeuf_present={cov['loeuf_present_fraction']:.4f} "
              f"mis_z_present={cov['mis_z_present_fraction']:.4f} "
              f"rows/gene={cov['mean_rows_per_gene']:.1f}")

    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(map(int, seeds)), "n_boot": n_boot,
        "stability": stability,
        "annotation_availability": coverage,
        "note": "Composition differences between evaluation cells (prevalence, gene "
                "concentration, annotation coverage) must be reported alongside any metric "
                "difference. A metric change across cells is not attributable to the model.",
    }
    Path(output_path).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Wrote {output_path}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--src-root", required=True)
    p.add_argument("--gnomad-constraint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    run(args.predictions, args.src_root, args.gnomad_constraint, args.output,
        args.seeds, args.n_boot)


if __name__ == "__main__":
    main()
