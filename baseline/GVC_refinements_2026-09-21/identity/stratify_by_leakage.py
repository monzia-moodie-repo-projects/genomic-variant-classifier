"""Re-evaluate EXISTING validation predictions split by gene-component leakage.

A validation row is 'leaked' if its gene component also contains training rows in the FULL
registry membership (spans are never computed from the restricted rows), otherwise 'unseen'.
No model is refitted. Strata differ in composition as well as leakage, so between-stratum
differences are DESCRIPTIVE. Paired contrasts are computed WITHIN a stratum, with percentile
intervals from resampling whole components. Prevalence and AUROC are reported beside Brier.
Read-only; writes one JSON.
"""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def auroc(y, p):
    order = np.argsort(p, kind="mergesort"); y = y[order]; p = p[order]
    ranks = np.empty(len(p)); i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[j + 1] == p[i]: j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1; i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return None
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_ci(d, comp, n_boot, seed):
    uniq, inv = np.unique(comp, return_inverse=True)
    sums = np.bincount(inv, weights=d); cnts = np.bincount(inv)
    rng = np.random.default_rng(seed); stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        stats[b] = sums[pick].sum() / cnts[pick].sum()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)), int(uniq.size)


def influence(g, delta, top):
    """Exact per-component attribution: stratum mean delta = sum_c (n_c/N) * mean_c(delta).
    Leave-out values are POINT ESTIMATES for sensitivity only: the components removed are chosen
    after seeing the data, so no interval is attached to them."""
    import pandas as pd
    f = pd.DataFrame({"component": g["component"].to_numpy(), "delta": delta,
                      "gene": g["gene_symbol"].astype(str).to_numpy() if "gene_symbol" in g else "n/a"})
    N = len(f); total = float(delta.sum())
    # Share of the SIGNED total is exact but unbounded: when the total is near zero it
    # amplifies noise (e.g. -320%). Share of total ABSOLUTE contribution stays in [0, 1].
    per = f.groupby("component").agg(rows=("delta", "size"), mean_delta=("delta", "mean"),
                                      sum_delta=("delta", "sum"), genes=("gene", lambda x: sorted(set(x))[:6]))
    per["contribution"] = per["sum_delta"] / N
    abs_total = float(per["contribution"].abs().sum())
    if not np.isclose(per["contribution"].sum(), delta.mean(), atol=1e-12):
        raise AssertionError("component contributions do not sum to the stratum delta")
    per = per.reindex(per["contribution"].abs().sort_values(ascending=False, kind="mergesort").index)
    top_rows = [{"component": str(c)[:30], "rows": int(r.rows), "mean_delta": float(r.mean_delta),
                 "contribution": float(r.contribution),
                 "share_of_stratum_delta": float(r.sum_delta / total) if total else None,
                 "share_of_absolute_contribution": float(abs(r.contribution) / abs_total) if abs_total else None,
                 "genes": list(r.genes)} for c, r in per.head(top).iterrows()]
    leave = {}
    for k in (1, 3, 5):
        drop = per.head(k); rest = N - int(drop["rows"].sum())
        leave[f"without_top_{k}"] = float((total - drop["sum_delta"].sum()) / rest) if rest else None
    return {"top": top_rows, "leave_out_point_estimates": leave}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("predictions", "components", "cohort", "membership", "output"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--component-column", default="component_genes_only")
    ap.add_argument("--models", nargs="+", default=None,
                    help="model prediction columns to evaluate. If omitted, columns are AUTO-DETECTED by value "
                         "range, which cannot tell a model from a feature; the output then flags that choice.")
    ap.add_argument("--contrast", nargs=2, action="append", default=[], metavar=("TREATED", "REFERENCE"))
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-components", type=int, default=30,
                    help="declared policy: below this many components a percentile cluster interval is "
                         "reported as UNRELIABLE and no zero-exclusion claim is made (not a statistical guarantee)")
    ap.add_argument("--top-components", type=int, default=10)
    a = ap.parse_args()
    if Path(a.output).exists(): sys.exit(f"ABORT: {a.output} exists; refusing to overwrite")
    import pandas as pd
    pred = pd.read_parquet(a.predictions)
    for c in ("variant_id", "label"):
        if c not in pred: sys.exit(f"ABORT: predictions lack {c!r}; columns {list(pred.columns)}")
    if pred["variant_id"].duplicated().any(): sys.exit("ABORT: duplicate variant_id in predictions")
    if not set(pred["label"].unique()) <= {0, 1}: sys.exit("ABORT: labels are not binary 0/1")
    if a.models:
        missing = [m for m in a.models if m not in pred.columns]
        if missing: sys.exit(f"ABORT: --models not in predictions: {missing}; columns {list(pred.columns)}")
        models, selection = list(a.models), "explicit"
    else:
        models = [c for c in pred.columns if c not in ("variant_id", "label") and pd.api.types.is_float_dtype(pred[c])
                  and pred[c].between(0, 1).all()]
        selection = "AUTO-DETECTED by value range; may include feature columns"
    constant = [m for m in models if pred[m].nunique() <= 1]
    for t, r in a.contrast:
        if t not in models or r not in models: sys.exit(f"ABORT: contrast columns not found among {models}")

    comps = pd.read_parquet(a.components, columns=["variation_id", a.component_column])
    cohort = pd.read_parquet(a.cohort, columns=["variant_id", "source_id"]); cohort["source_id"] = cohort["source_id"].astype(str)
    mem = pd.read_parquet(a.membership, columns=["variant_id", "partition"])
    full = cohort.merge(mem, on="variant_id", validate="many_to_one").merge(
        comps.rename(columns={"variation_id": "source_id", a.component_column: "component"}), on="source_id", how="left")
    span = full.dropna(subset=["component"]).groupby("component")["partition"].agg(lambda s: frozenset(s))

    d = pred.merge(full[["variant_id", "component"]].drop_duplicates("variant_id"), on="variant_id", how="left", validate="one_to_one")
    n_nocomp = int(d["component"].isna().sum())
    d["component"] = d["component"].fillna("no-component:" + d["variant_id"].astype(str))
    d["stratum"] = np.where(d["component"].map(lambda c: "train" in span.get(c, frozenset())), "leaked", "unseen")

    out = {"component_column": a.component_column, "rows": len(d), "rows_without_component": n_nocomp,
           "models": models, "model_selection": selection, "constant_columns": constant, "strata": {}}
    for s in ("unseen", "leaked", "all"):
        g = d if s == "all" else d[d["stratum"] == s]
        y = g["label"].to_numpy(float); comp = g["component"].to_numpy()
        entry = {"rows": int(len(g)), "components": int(g["component"].nunique()),
                 "prevalence": float(y.mean()) if len(g) else None, "per_model": {}, "contrasts": {}}
        for m in models:
            p = g[m].to_numpy(float)
            entry["per_model"][m] = {"brier": float(np.mean((p - y) ** 2)), "auroc": auroc(y, p)}
        for t, r in a.contrast:
            delta = (g[t].to_numpy(float) - y) ** 2 - (g[r].to_numpy(float) - y) ** 2
            lo, hi, k = cluster_ci(delta, comp, a.n_boot, a.seed)
            reliable = k >= a.min_components
            entry["contrasts"][f"{t} - {r}"] = {"brier_delta": float(delta.mean()), "ci95_component": [lo, hi],
                                                "interval_reliable": reliable,
                                                "excludes_zero": bool(lo > 0 or hi < 0) if reliable else None,
                                                "components": k,
                                                "influence": influence(g, delta, a.top_components)}
        out["strata"][s] = entry
    out["run_utc"] = datetime.now(timezone.utc).isoformat()
    Path(a.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"rows {len(d):,} | without component {n_nocomp:,} | component column {a.component_column}")
    print(f"model selection: {selection}")
    if constant:
        print(f"WARNING constant columns (a single value, so not a model's predictions): {constant}")
    for s, e in out["strata"].items():
        print(f"\n[{s}] rows {e['rows']:,} ({e['rows']/len(d):.1%}) | components {e['components']:,} | prevalence {e['prevalence']:.4f}")
        for m, v in e["per_model"].items():
            au = f"{v['auroc']:.4f}" if v["auroc"] is not None else "undefined"
            print(f"    {m:40} Brier {v['brier']:.5f}  AUROC {au}")
        for k, v in e["contrasts"].items():
            print(f"    CONTRAST {k}: {v['brier_delta']:+.6f}  95% CI [{v['ci95_component'][0]:+.6f}, "
                  f"{v['ci95_component'][1]:+.6f}]  excludes zero: "
                  f"{v['excludes_zero'] if v['interval_reliable'] else 'NOT ASSESSED - too few components'}  "
                  f"({v['components']:,} components)")
            inf = v["influence"]
            print("      leave-out point estimates: " + ", ".join(
                f"{k2} {x:+.6f}" for k2, x in inf["leave_out_point_estimates"].items() if x is not None))
            for t_ in inf["top"][:5]:
                share = (f"{t_['share_of_absolute_contribution']:.1%} of |total|"
                         if t_["share_of_absolute_contribution"] is not None else "n/a")
                print(f"      {t_['rows']:>7,} rows  mean {t_['mean_delta']:+.5f}  contribution "
                      f"{t_['contribution']:+.6f}  share {share}  {t_['genes']}")
    print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
