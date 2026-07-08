"""
genomic_variant_classifier.evaluation.metrics  (2026-07-08)
==========================================================================
Core of the Run-17 metric stack. Pure functions over (y, score) arrays: no I/O, no
model objects, no sklearn. Every function is independently unit-testable, and the
test suite feeds each one the input that SHOULD trip it.

WHY THIS EXISTS

    The project's declared primary metric is AUPRC. AUPRC's no-skill floor equals the
    positive rate, so an AUPRC quoted without its `pos_rate` is uninterpretable, and
    two AUPRCs measured at different positive rates are not comparable. Runs 9-14 ran
    at pos_rate 20.34%; Runs 15-17 at 14.15%. Nothing in any run artifact recorded
    which. See docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md sec 6.

    Every metric here therefore returns `pos_rate` alongside itself, and every panel
    is STRATIFIABLE -- because a headline number computed over a cohort that is 86%
    SNVs and 8.5% deletions (the deletions 3.5x pathogenic-enriched and carrying
    sixteen constant features) tells you almost nothing about either class.

DESIGN RULES
    * A metric that cannot be computed returns NaN and says so. It never returns 0.5
      or 0.0 as if that were a measurement.
    * Confidence intervals are STRATIFIED bootstrap by default: resampling positives
      and negatives separately preserves `pos_rate`, so the CI reflects sampling
      noise in the score, not noise in the class balance.
    * Ties in the score are handled explicitly: average ranks for AUROC, and tied
      scores collapsed into a single threshold for AUPRC. This matters -- every padded
      deletion in the Run-14 splits carries sixteen literally constant features. Both
      are cross-validated against sklearn to 1e-16 (see tests).
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "auroc", "auprc", "no_skill_auprc", "brier_score",
    "expected_calibration_error", "calibration_slope_intercept",
    "bootstrap_ci", "evaluate", "stratified_evaluate",
]

_EPS = 1e-12


def _clean(y: Sequence, s: Sequence) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(float).ravel()
    s = np.asarray(s).astype(float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y and score length mismatch: {y.shape} vs {s.shape}")
    ok = np.isfinite(y) & np.isfinite(s)
    return y[ok].astype(int), s[ok]


def _degenerate(y: np.ndarray) -> bool:
    """A single class present -> ranking metrics are undefined. Say so; do not guess."""
    return y.size == 0 or y.sum() == 0 or y.sum() == y.size


def auroc(y: Sequence, score: Sequence) -> float:
    """Rank-based AUROC (Mann-Whitney U). Ties get average ranks. NaN if one class."""
    y, s = _clean(y, score)
    if _degenerate(y):
        return float("nan")
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y: Sequence, score: Sequence) -> float:
    """Average precision, aggregating TIED scores into a single threshold.

    Ties are not a corner case here: every padded deletion in the Run-14 splits carries
    sixteen literally constant features, and any binary indicator ties massively. A
    row-by-row walk after an arbitrary tie-break inflates AP (it credits an ordering the
    score never expressed). Cross-checked against sklearn.average_precision_score to
    1e-16 on balanced, 10%-imbalanced, 1%-imbalanced, near-random and heavily-tied data.
    """
    y, s = _clean(y, score)
    if _degenerate(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yy, ss = y[order], s[order]
    # collapse runs of equal score: keep only the last index of each run
    last_of_run = np.r_[np.flatnonzero(np.diff(ss)), yy.size - 1]
    tp = np.cumsum(yy)[last_of_run]
    fp = np.cumsum(1 - yy)[last_of_run]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / tp[-1]
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def no_skill_auprc(y: Sequence) -> float:
    """A random classifier's AUPRC == the positive rate. Quote AUPRC against this."""
    y = np.asarray(y).astype(int).ravel()
    return float(y.mean()) if y.size else float("nan")


def brier_score(y: Sequence, prob: Sequence) -> float:
    y, p = _clean(y, prob)
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def expected_calibration_error(y: Sequence, prob: Sequence, n_bins: int = 10) -> float:
    """Equal-width binning. |accuracy - confidence| weighted by bin occupancy."""
    y, p = _clean(y, prob)
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / y.size) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_slope_intercept(y: Sequence, prob: Sequence,
                                max_iter: int = 100, tol: float = 1e-10) -> tuple[float, float]:
    """Fit y ~ sigmoid(intercept + slope * logit(p)) by IRLS.

    Perfect calibration -> slope 1.0, intercept 0.0.
    slope < 1 means over-confident; slope > 1 means under-confident.
    """
    y, p = _clean(y, prob)
    if _degenerate(y):
        return float("nan"), float("nan")
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), _EPS)
        z = eta + (y - mu) / w
        XtW = X.T * w
        try:
            beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[1]), float(beta[0])  # slope, intercept


def bootstrap_ci(fn: Callable[[np.ndarray, np.ndarray], float],
                 y: Sequence, score: Sequence, *,
                 n_boot: int = 200, alpha: float = 0.05, seed: int = 0,
                 stratified: bool = True) -> tuple[float, float]:
    """Percentile bootstrap CI. Stratified by class so pos_rate is preserved.

    n_boot defaults to 200: each replicate is O(n log n), so 1000 replicates over a
    1.2M-row split is minutes, not seconds. Raise it for a final publication panel.
    """
    y, s = _clean(y, score)
    if _degenerate(y):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals: list[float] = []
    for _ in range(n_boot):
        if stratified:
            i = np.concatenate([rng.choice(pos, pos.size, replace=True),
                                rng.choice(neg, neg.size, replace=True)])
        else:
            i = rng.integers(0, y.size, y.size)
        v = fn(y[i], s[i])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def evaluate(y: Sequence, score: Sequence, *,
             prob: Sequence | None = None,
             n_boot: int = 0, seed: int = 0, n_bins: int = 10) -> dict:
    """Full single-population panel.

    `score` ranks; `prob` (default: score) is used for calibration. Pass both when the
    ranking score is not a probability.
    """
    y_c, s_c = _clean(y, score)
    p = s_c if prob is None else _clean(y, prob)[1]
    base = no_skill_auprc(y_c)
    ap = auprc(y_c, s_c)
    slope, intercept = calibration_slope_intercept(y_c, p)
    out = {
        "n": int(y_c.size),
        "n_pos": int(y_c.sum()),
        "pos_rate": base,
        "auroc": auroc(y_c, s_c),
        "auprc": ap,
        "auprc_no_skill": base,
        "auprc_lift": (ap / base) if (base and np.isfinite(ap)) else float("nan"),
        "brier": brier_score(y_c, p),
        "ece": expected_calibration_error(y_c, p, n_bins=n_bins),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }
    if n_boot:
        out["auroc_ci95"] = bootstrap_ci(auroc, y_c, s_c, n_boot=n_boot, seed=seed)
        out["auprc_ci95"] = bootstrap_ci(auprc, y_c, s_c, n_boot=n_boot, seed=seed)
    return out


def stratified_evaluate(y: Sequence, score: Sequence, groups: Iterable, *,
                        prob: Sequence | None = None,
                        n_boot: int = 0, seed: int = 0,
                        min_n: int = 30) -> pd.DataFrame:
    """One panel per group, plus an ALL row.

    A headline metric over a cohort that is 86% SNVs and 8.5% deletions -- the latter
    3.5x pathogenic-enriched and carrying sixteen constant features -- describes
    neither class. Strata below `min_n` are reported with NaN metrics, never dropped
    silently.
    """
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(score).astype(float).ravel()
    g = pd.Series(list(groups), dtype="object")
    p = s if prob is None else np.asarray(prob).astype(float).ravel()
    if not (len(y) == len(s) == len(g) == len(p)):
        raise ValueError(f"length mismatch y={len(y)} s={len(s)} g={len(g)} p={len(p)}")

    rows: dict[str, dict] = {"ALL": evaluate(y, s, prob=p, n_boot=n_boot, seed=seed)}
    for name in sorted(g.dropna().unique(), key=str):
        m = (g == name).to_numpy()
        if m.sum() < min_n:
            rows[str(name)] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                               "pos_rate": float(y[m].mean()) if m.sum() else float("nan"),
                               "auroc": float("nan"), "auprc": float("nan"),
                               "auprc_no_skill": float("nan"), "auprc_lift": float("nan"),
                               "brier": float("nan"), "ece": float("nan"),
                               "cal_slope": float("nan"), "cal_intercept": float("nan")}
            continue
        rows[str(name)] = evaluate(y[m], s[m], prob=p[m], n_boot=n_boot, seed=seed)
    df = pd.DataFrame(rows).T
    for c in df.columns:
        if c not in ("auroc_ci95", "auprc_ci95"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
