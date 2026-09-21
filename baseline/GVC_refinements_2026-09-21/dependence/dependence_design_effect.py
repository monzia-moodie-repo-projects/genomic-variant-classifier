"""A dependence design effect that isolates the resampling UNIT.

The project's cluster_bootstrap_ci(return_design_effect=True) divides a
whole-gene interval (n_boot replicates, class balance free to vary) by a
row interval from bootstrap_ci (min(n_boot, 500) replicates, class-STRATIFIED).
Three things therefore differ between numerator and denominator: the unit,
the class-balance treatment, and the Monte Carlo precision. Only the first
is within-gene dependence.

Demonstrated consequence: on data with ZERO within-gene dependence, where
the paired statistic differs by class as Brier deltas do, that ratio reports
up to 2.55. This module changes ONE thing between arms -- the unit -- and
reports the variance ratio (Kish's design effect), tracking the closed form
1 + (m-1)*rho within about 1%.

Deliberately NOT a replacement for cluster_bootstrap_ci's interval, which is
unaffected. This concerns only the COMPARISON between interval widths.
"""
import numpy as np


class DependenceRatioError(ValueError):
    pass


def dependence_design_effect(statistic, values, clusters, *, n_boot=4000, seed=0):
    """Return a dict with the variance design effect and its width analogue.

    statistic: callable taking a 1-D array, e.g. np.mean.
    values:    per-row values, e.g. paired Brier differences.
    clusters:  per-row independence unit (a gene component, not a raw string).

    Both arms: identical n_boot, identical seed, neither class-stratified.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    if v.ndim != 1 or v.shape != c.shape:
        raise DependenceRatioError(f"shape mismatch: values {v.shape}, clusters {c.shape}")
    if v.size < 2:
        raise DependenceRatioError("need at least two rows")
    if not np.all(np.isfinite(v)):
        raise DependenceRatioError("values must be finite")
    if type(n_boot) is not int or n_boot < 200:
        raise DependenceRatioError("n_boot must be an int of at least 200")
    uniq = np.unique(c)
    if uniq.size < 2:
        raise DependenceRatioError("need at least two clusters")

    index_of = {u: np.flatnonzero(c == u) for u in uniq}
    rng = np.random.default_rng(seed)
    cluster_stats = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(uniq, uniq.size, replace=True)
        cluster_stats[b] = statistic(v[np.concatenate([index_of[u] for u in drawn])])
    rng = np.random.default_rng(seed)
    row_stats = np.empty(n_boot)
    for b in range(n_boot):
        row_stats[b] = statistic(v[rng.integers(0, v.size, v.size)])

    sd_cluster = float(np.std(cluster_stats, ddof=1))
    sd_row = float(np.std(row_stats, ddof=1))
    if not (np.isfinite(sd_row) and sd_row > 0):
        raise DependenceRatioError("row-bootstrap spread is zero; ratio undefined")
    sd_ratio = sd_cluster / sd_row
    return {
        "variance_design_effect": sd_ratio ** 2,
        "sd_ratio": sd_ratio,
        "n_boot": n_boot,
        "n_rows": int(v.size),
        "n_clusters": int(uniq.size),
        "definition": "SD ratio of whole-cluster to row bootstrap distributions; "
                      "identical n_boot and seed; neither arm class-stratified. "
                      "variance_design_effect = sd_ratio**2 (Kish).",
    }
