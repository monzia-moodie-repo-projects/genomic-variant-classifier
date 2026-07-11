"""Coverage diagnostics for conformal prediction sets (from scratch, numpy/pandas).

Given prediction SETS (boolean membership per class) and true labels, computes the diagnostics
that certify whether a conformal procedure actually delivers its promised guarantee, sliced the
ways that matter for GenAssoc:

  - marginal coverage: overall P(true label in set).
  - per-class coverage: coverage within each true class (exposes rare-class under-coverage that
    marginal coverage hides -- the exact failure Mondrian fixes).
  - per-stratum coverage: coverage within groups of a stratifier column (ReviewStatus, consequence,
    is_missense). Handles a NaN slice explicitly via an 'unknown' bin -- never silently dropped.
  - gene-disjoint-holdout coverage: coverage measured with genes as the unit, matching the split.
  - set-size distribution: mean/median/quantiles of |set|; efficiency of the procedure.
  - abstention / uncertainty rates: empty-set rate (abstains) and full-set rate (maximally unsure).

All functions are pure and deterministic. A set is an (n, K) boolean array; y is (n,) int in [0,K).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_UNKNOWN = "__unknown__"


def _validate(sets: np.ndarray, y: np.ndarray):
    sets = np.asarray(sets, dtype=bool)
    y = np.asarray(y, dtype=int)
    if sets.ndim != 2:
        raise ValueError("sets must be (n, K) boolean")
    if len(sets) != len(y):
        raise ValueError(f"length mismatch: sets {len(sets)} vs y {len(y)}")
    if y.min() < 0 or y.max() >= sets.shape[1]:
        raise ValueError("y contains a class index outside [0, K)")
    return sets, y


def covered(sets: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean per-row: was the TRUE label in the set?"""
    sets, y = _validate(sets, y)
    return sets[np.arange(len(y)), y]


def marginal_coverage(sets: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(covered(sets, y)))


def per_class_coverage(sets: np.ndarray, y: np.ndarray, K: int | None = None) -> pd.Series:
    sets, y = _validate(sets, y)
    K = K or sets.shape[1]
    cov = covered(sets, y)
    out = {}
    for k in range(K):
        m = y == k
        out[k] = float(np.mean(cov[m])) if m.any() else float("nan")
    return pd.Series(out, name="per_class_coverage")


def per_stratum_coverage(sets: np.ndarray, y: np.ndarray, strata: np.ndarray) -> pd.DataFrame:
    """Coverage within each stratum value. NaN stratum values go into an explicit 'unknown' bin
    so no rows are silently dropped."""
    sets, y = _validate(sets, y)
    cov = covered(sets, y)
    s = pd.Series(strata).astype("object")
    s = s.where(~s.isna(), _UNKNOWN)
    df = pd.DataFrame({"stratum": s.values, "covered": cov})
    g = df.groupby("stratum")["covered"]
    out = pd.DataFrame({"n": g.size(), "coverage": g.mean()})
    return out.sort_values("n", ascending=False)


def group_coverage_disjoint(sets: np.ndarray, y: np.ndarray, groups: np.ndarray,
                            agg: str = "all") -> float:
    """Gene-level coverage: for each group, is the group 'covered'? agg='all' requires every
    member covered (strict); agg='any' requires at least one. Reports the fraction of groups."""
    sets, y = _validate(sets, y)
    cov = covered(sets, y)
    df = pd.DataFrame({"g": np.asarray(groups), "c": cov})
    if agg == "all":
        gc = df.groupby("g")["c"].all()
    elif agg == "any":
        gc = df.groupby("g")["c"].any()
    else:
        raise ValueError("agg must be 'all' or 'any'")
    return float(np.mean(gc.values))


def set_size_summary(sets: np.ndarray) -> dict:
    sets = np.asarray(sets, dtype=bool)
    sizes = sets.sum(axis=1)
    return {
        "mean": float(np.mean(sizes)),
        "median": float(np.median(sizes)),
        "q10": float(np.quantile(sizes, 0.10)),
        "q90": float(np.quantile(sizes, 0.90)),
        "min": int(sizes.min()),
        "max": int(sizes.max()),
    }


def abstention_rates(sets: np.ndarray) -> dict:
    """empty-set rate (procedure abstains) and full-set rate (maximally uncertain)."""
    sets = np.asarray(sets, dtype=bool)
    sizes = sets.sum(axis=1)
    K = sets.shape[1]
    return {
        "empty_rate": float(np.mean(sizes == 0)),
        "full_rate": float(np.mean(sizes == K)),
        "singleton_rate": float(np.mean(sizes == 1)),
    }


def coverage_report(sets: np.ndarray, y: np.ndarray, alpha: float,
                    strata: np.ndarray | None = None,
                    groups: np.ndarray | None = None) -> dict:
    """One-call diagnostic bundle. Includes a boolean 'marginal_ok' = coverage >= 1 - alpha - slack,
    where slack is a small finite-sample allowance (2 standard errors of the coverage estimate)."""
    sets, y = _validate(sets, y)
    n = len(y)
    marg = marginal_coverage(sets, y)
    se = float(np.sqrt(max(marg * (1 - marg), 1e-12) / n))
    rep = {
        "n": n,
        "alpha": alpha,
        "target": 1 - alpha,
        "marginal_coverage": marg,
        "marginal_se": se,
        "marginal_ok": bool(marg >= (1 - alpha) - 2 * se),
        "per_class_coverage": per_class_coverage(sets, y).to_dict(),
        "set_size": set_size_summary(sets),
        "abstention": abstention_rates(sets),
    }
    if strata is not None:
        rep["per_stratum_coverage"] = per_stratum_coverage(sets, y, strata).to_dict("index")
    if groups is not None:
        rep["group_coverage_all"] = group_coverage_disjoint(sets, y, groups, "all")
        rep["group_coverage_any"] = group_coverage_disjoint(sets, y, groups, "any")
    return rep
