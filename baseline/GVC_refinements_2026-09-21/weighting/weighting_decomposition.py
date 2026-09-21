"""Exact decomposition of the variant-weighted vs group-weighted disagreement.

    Delta_variant - Delta_group = G * Cov_g(n_g / N, delta_g)

where delta_g is group g's mean per-row difference, n_g its row count, N total
rows, G the number of groups, and Cov is the POPULATION covariance over groups.
The identity is exact, not asymptotic. The two weightings disagree if and only
if group size covaries with the per-group effect.

The size-stratified table localises WHERE the disagreement arises. It is
descriptive: group size is not randomised, and large ClinVar genes differ from
small ones in curation depth, disease area and variant mix.

The unit MUST be a biological independence unit (a resolved gene component).
Computed on raw gene_symbol strings, TTN and 'TTN-AS1;TTN' are separate groups
and the decomposition describes registry strings, not genes.
"""
import numpy as np
import pandas as pd


class WeightingError(ValueError):
    pass


def decompose(values, groups, *, n_size_bins=5):
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    if v.ndim != 1 or v.shape != g.shape:
        raise WeightingError(f"shape mismatch: {v.shape} vs {g.shape}")
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise WeightingError("values must be non-empty and finite")
    frame = pd.DataFrame({"v": v, "g": g})
    per = frame.groupby("g", sort=True)["v"].agg(delta="mean", n="size")
    G, N = len(per), int(per["n"].sum())
    if G < 2:
        raise WeightingError("need at least two groups")

    variant = float(v.mean())
    group = float(per["delta"].mean())
    w = per["n"].to_numpy() / N
    d = per["delta"].to_numpy()
    cov = float(np.mean((w - w.mean()) * (d - d.mean())))
    implied = G * cov
    gap = variant - group
    if not np.isclose(gap, implied, rtol=1e-9, atol=1e-12):
        raise WeightingError(f"identity failed: gap {gap} vs G*cov {implied}")

    # Bins hold roughly EQUAL ROWS, not equal groups. Equal-group quantiles hide
    # the mechanism when sizes are heavily skewed (one ClinVar gene can hold
    # 22,537 rows): a handful of large genes share a bin with hundreds of small
    # ones and the reversal disappears from the table.
    per = per.sort_values("n", kind="mergesort")
    cum = per["n"].cumsum() / N
    k = min(n_size_bins, len(per))
    per = per.assign(size_bin=[f"R{min(int(c * k - 1e-12), k - 1) + 1}" for c in cum])
    bins = []
    for label, sub in per.groupby("size_bin", observed=True):
        bins.append({"size_bin": str(label), "n_groups": int(len(sub)),
                     "rows": int(sub["n"].sum()),
                     "min_group_size": int(sub["n"].min()),
                     "max_group_size": int(sub["n"].max()),
                     "group_weighted_delta": float(sub["delta"].mean()),
                     "row_weighted_delta": float(np.average(sub["delta"], weights=sub["n"])),
                     "share_of_rows": float(sub["n"].sum() / N)})
    return {
        "variant_weighted": variant,
        "group_weighted": group,
        "gap": gap,
        "G_times_cov": implied,
        "identity_holds": True,
        # The identity proves the two means DIFFER whenever the covariance is
        # nonzero. A sign REVERSAL additionally requires opposite signs.
        "means_differ": bool(not np.isclose(gap, 0.0, atol=1e-15)),
        "sign_reversal": bool(variant * group < 0),
        "n_groups": G, "n_rows": N,
        "groups_improving": int((d < 0).sum()),
        "groups_worsening": int((d > 0).sum()),
        "largest_group_share": float(per["n"].max() / N),
        # Undefined when every group has the same size (or the same delta):
        # rank correlation divides by a zero spread. Report None with the reason
        # rather than a NaN that reads as a number.
        "spearman_size_vs_delta": (
            None if per["n"].nunique() < 2 or np.unique(d).size < 2
            else float(pd.Series(per["n"].to_numpy()).rank().corr(pd.Series(d).rank()))),
        "spearman_undefined_reason": (
            "all groups have equal size" if per["n"].nunique() < 2
            else "all groups have equal delta" if np.unique(d).size < 2 else None),
        "by_size_bin": bins,
    }


def per_group_contributions(values, groups):
    """Exact per-group terms (n_g/N - 1/G) * delta_g, summing to the gap.

    This is the primary decomposition. Size bins are only a visualisation:
    they depend on boundary and tie policies, and whole groups make exact
    equal-row-mass bins impossible.
    """
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    if v.ndim != 1 or v.shape != g.shape or v.size == 0 or not np.all(np.isfinite(v)):
        raise WeightingError("values must be finite, 1-D and aligned with groups")
    per = pd.DataFrame({"v": v, "g": g}).groupby("g", sort=True)["v"].agg(delta="mean", n="size")
    if len(per) < 2:
        raise WeightingError("need at least two groups")
    N, G = int(per["n"].sum()), len(per)
    per["contribution"] = (per["n"] / N - 1 / G) * per["delta"]
    gap = float(v.mean() - per["delta"].mean())
    if not np.isclose(per["contribution"].sum(), gap, rtol=1e-9, atol=1e-12):
        raise WeightingError("contributions do not sum to the gap")
    return per.sort_values("contribution", key=np.abs, ascending=False)
