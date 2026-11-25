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

  - family-weighted class coverage (2026-11-24): the mean, over the families that contain the class,
    of each family's within-family coverage, beside the variant-weighted rate, with an explicit
    support state. See family_class_coverage.

All functions are pure and deterministic. A set is an (n, K) boolean array; y is (n,) int in [0,K).

INPUT DISCIPLINE (2026-11-24, COVERAGE-SILENT-COERCION-1). _validate used to cast with
np.asarray(..., dtype=int) and np.asarray(..., dtype=bool). Measured on main 2b55f06f: a label of
0.7 was truncated to class 0 and reported as covered (marginal coverage 1.0), and a matrix of
probabilities (0.3, 0.2) was read as set membership. Both now refuse. A set is boolean, or integer
0/1; a label is an integer, or a float holding an integral value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_UNKNOWN = "__unknown__"


def _membership(sets) -> np.ndarray:
    arr = np.asarray(sets)
    if arr.dtype == np.bool_:
        return arr
    if arr.dtype.kind in "iu" and arr.size and bool(np.isin(arr, (0, 1)).all()):
        return arr.astype(bool)
    raise ValueError(
        f"sets must be boolean set membership (or integers 0/1); got dtype {arr.dtype}. "
        "A matrix of probabilities is not a prediction set.")


def _class_index(y) -> np.ndarray:
    arr = np.asarray(y)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    if arr.dtype.kind == "f" and arr.size and bool(np.isfinite(arr).all()) and bool(
            (arr == np.floor(arr)).all()):
        return arr.astype(np.int64)
    raise ValueError(
        f"y must hold integer class indexes; got dtype {arr.dtype}"
        + (" with a non-integral or non-finite value" if arr.dtype.kind == "f" else ""))


def _validate(sets: np.ndarray, y: np.ndarray):
    sets = _membership(sets)
    y = _class_index(y)
    if sets.ndim != 2:
        raise ValueError("sets must be (n, K) boolean")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if len(sets) != len(y):
        raise ValueError(f"length mismatch: sets {len(sets)} vs y {len(y)}")
    if len(y) == 0:
        raise ValueError("no rows to evaluate")
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
    member covered (strict); agg='any' requires at least one. Reports the fraction of groups.

    THESE ARE NOT FAMILY-WEIGHTED COVERAGE. For a group of ten members each covered
    independently with probability 0.9, P(all) = 0.9**10 = 0.349 and P(any) = 1 - 0.1**10,
    while the mean within-group coverage is 0.9. Their meanings are kept exactly as they were;
    the family-weighted estimand is family_class_coverage."""
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
    where slack is a small finite-sample allowance (2 standard errors of the coverage estimate).

    'marginal_ok' is a VARIANT-LEVEL heuristic: its standard error treats rows as independent.
    Rows of one gene or protein family are not, so it is not evidence of validity for a new gene or
    family, and it is not a finite-sample guarantee. Family-level behaviour is reported by
    family_class_coverage, which this bundle does not call."""
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


# --------------------------------------------------------------------------- #
# Family-weighted class coverage (2026-11-24)
# --------------------------------------------------------------------------- #
#: The support states. A point estimate is always reported when the class is present; the
#: interval is withheld below the declared family support, because a percentile bootstrap over
#: two families has three attainable means and its "95% interval" is [0, 1] (measured).
FAMILY_SUPPORT_ESTIMATED = "estimated"
FAMILY_SUPPORT_INSUFFICIENT = "insufficient_family_support"
FAMILY_SUPPORT_CLASS_ABSENT = "class_absent"


@dataclass(frozen=True)
class FamilyClassCoverage:
    """Empirical class-conditional coverage at two weightings, with its support.

    family_weighted = (1 / |F_c|) * sum over f in F_c of (covered rows of class c in f) / n_{f,c},
    F_c being the families with at least one row of class c. variant_weighted is the ordinary
    per-class rate. Neither substitutes for the other, and neither is a finite-sample guarantee:
    the interval is a family-cluster percentile bootstrap, a DESCRIPTIVE interval whose
    sampling interpretation needs independent families.
    """
    class_label: int
    status: str
    n_rows: int
    n_families: int
    family_size_min: int | None
    family_size_median: float | None
    family_size_max: int | None
    variant_weighted: float | None
    family_weighted: float | None
    mean_set_size: float | None
    empty_set_rate: float | None
    singleton_rate: float | None
    full_set_rate: float | None
    singleton_error_rate: float | None
    interval: tuple[float, float] | None
    interval_level: float
    min_families: int
    n_boot: int
    seed: int
    interval_method: str = "family_cluster_percentile_bootstrap"
    interpretation: str = "empirical_diagnostic_not_finite_sample_guarantee"


def _family_identities(families, n: int) -> np.ndarray:
    arr = np.asarray(families, dtype=object)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"families must be one-dimensional and aligned with the {n} rows")
    for v in arr:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(
                f"a family identity must be a non-blank string; got {v!r}. An unknown family "
                "is not a cluster, and pooling unknowns would invent one.")
    return arr


def family_class_coverage(sets, y, families, class_label: int, *, min_families: int = 20,
                          n_boot: int = 2000, seed: int = 0,
                          interval_level: float = 0.95) -> FamilyClassCoverage:
    """Family-weighted and variant-weighted coverage for one true class, with explicit support.

    `families` are canonical identities supplied by the caller (a protein-homology family, a
    canonical gene); this function never derives them, and refuses a missing or blank one.
    """
    sets, y = _validate(sets, y)
    fam = _family_identities(families, len(y))
    K = sets.shape[1]
    if isinstance(class_label, bool) or not isinstance(class_label, (int, np.integer)) or not (
            0 <= int(class_label) < K):
        raise ValueError(f"class_label must be an integer in [0, {K})")
    if isinstance(min_families, bool) or not isinstance(min_families, int) or min_families < 2:
        raise ValueError("min_families must be an integer >= 2")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 100:
        raise ValueError("n_boot must be an integer >= 100")
    if not (0.0 < interval_level < 1.0):
        raise ValueError("interval_level must lie in (0, 1)")
    c = int(class_label)
    common = dict(class_label=c, interval_level=float(interval_level), min_families=min_families,
                  n_boot=n_boot, seed=seed)
    mask = y == c
    if not mask.any():
        return FamilyClassCoverage(status=FAMILY_SUPPORT_CLASS_ABSENT, n_rows=0, n_families=0,
                                   family_size_min=None, family_size_median=None,
                                   family_size_max=None, variant_weighted=None,
                                   family_weighted=None, mean_set_size=None, empty_set_rate=None,
                                   singleton_rate=None, full_set_rate=None,
                                   singleton_error_rate=None, interval=None, **common)
    hit = sets[mask, c]
    sizes = sets[mask].sum(axis=1)
    names, inverse = np.unique(fam[mask].astype(str), return_inverse=True)
    counts = np.bincount(inverse)
    per_family = np.bincount(inverse, weights=hit.astype(float)) / counts
    single = sizes == 1
    status = (FAMILY_SUPPORT_ESTIMATED if len(names) >= min_families
              else FAMILY_SUPPORT_INSUFFICIENT)
    interval = None
    if status == FAMILY_SUPPORT_ESTIMATED:
        rng = np.random.default_rng(seed)
        F = len(per_family)
        means = np.empty(n_boot, dtype=float)
        step = max(1, min(n_boot, 4_000_000 // F))
        for start in range(0, n_boot, step):
            stop = min(n_boot, start + step)
            idx = rng.integers(0, F, size=(stop - start, F))
            means[start:stop] = per_family[idx].mean(axis=1)
        tail = (1.0 - interval_level) / 2.0
        lo, hi = np.quantile(means, [tail, 1.0 - tail])
        interval = (float(lo), float(hi))
    return FamilyClassCoverage(
        status=status, n_rows=int(mask.sum()), n_families=int(len(names)),
        family_size_min=int(counts.min()), family_size_median=float(np.median(counts)),
        family_size_max=int(counts.max()), variant_weighted=float(hit.mean()),
        family_weighted=float(per_family.mean()), mean_set_size=float(sizes.mean()),
        empty_set_rate=float(np.mean(sizes == 0)), singleton_rate=float(np.mean(single)),
        full_set_rate=float(np.mean(sizes == K)),
        singleton_error_rate=(float(np.mean(~hit[single])) if single.any() else None),
        interval=interval, **common)
