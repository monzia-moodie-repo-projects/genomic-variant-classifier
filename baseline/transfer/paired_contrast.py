"""PROPOSED ADDITION to evaluation/metrics.py -- not yet installed.

Extends the reviewed gene-cluster bootstrap to a PAIRED-CELL contrast, which
the existing single-score interface cannot express. It reuses that function's
exact resampling design (whole clusters, percentile interval, finite-value
guard); it does not introduce a different estimator.

The repair experiment interaction

    I = mean(d | cell_b) - mean(d | cell_a)

requires BOTH cells resampled inside ONE gene draw, because a gene can
contribute rows to both cells and that dependence must be preserved. Computing
two separate intervals and comparing them is not the same quantity.
"""
import numpy as np


def cluster_bootstrap_paired_contrast_ci(values, cells, clusters, *,
                                          cell_a, cell_b,
                                          n_boot=2000, alpha=0.05, seed=0,
                                          two_stage=False):
    """Percentile bootstrap of mean(values|cell_b) - mean(values|cell_a).

    Whole clusters are drawn ONCE per replicate and both cells are recomputed
    from that same draw, preserving within-cluster dependence across cells.
    A replicate contributing no rows to either cell is skipped rather than
    counted as zero.

    Returns (point, lo, hi, n_effective_replicates).
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(cells)
    g = np.asarray(clusters)
    if not (len(v) == len(c) == len(g)):
        raise ValueError(f"length mismatch: values={len(v)}, cells={len(c)}, clusters={len(g)}")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite")
    if cell_a == cell_b:
        raise ValueError("cell_a and cell_b must differ")

    in_a = c == cell_a
    in_b = c == cell_b
    if not in_a.any() or not in_b.any():
        raise ValueError(f"both cells must be present; got {int(in_a.sum())} and {int(in_b.sum())}")

    point = float(v[in_b].mean() - v[in_a].mean())

    uniq = np.unique(g)
    index_of = {u: np.flatnonzero(g == u) for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=uniq.size, replace=True)
        parts = []
        for u in drawn:
            idx = index_of[u]
            if two_stage and idx.size > 1:
                idx = rng.choice(idx, idx.size, replace=True)
            parts.append(idx)
        i = np.concatenate(parts) if parts else np.empty(0, dtype=int)
        if i.size == 0:
            continue
        sel_a = in_a[i]
        sel_b = in_b[i]
        if not sel_a.any() or not sel_b.any():
            continue
        stat = v[i][sel_b].mean() - v[i][sel_a].mean()
        if np.isfinite(stat):
            vals.append(stat)
    if not vals:
        return point, float("nan"), float("nan"), 0
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), len(vals)
