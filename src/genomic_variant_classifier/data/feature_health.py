"""feature_health.py  --  Monzia Moodie

Single source of truth for per-column feature-health / degeneracy verdicts.

Extracted verbatim from scripts/audit_split_feature_health.py so that both the
audit (which produces a reference) and FeatureCoverageSentinelAgent (which scores
a current matrix against that reference) compute health with identical semantics --
a divergence here would let a silent feature regression slip past the sentinel.

A column is DEGENERATE if (among non-null values) it is all-null (ALL_NULL),
constant (CONSTANT), near-constant (NEAR_CONSTANT, one value covers
>= near_constant of non-null rows), or numeric all-zero (ALL_ZERO).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_NEAR_CONSTANT_FRAC = 0.999


def unique_and_top(nn: pd.Series):
    """n_unique and top-value frequency, safe against unhashable cells
    (dict/list/ndarray) by stringifying ONLY for the count -- numeric stats
    are computed separately on the real values."""
    if len(nn) == 0:
        return 0, 0.0, False
    try:
        nu = int(nn.nunique())
        top = float(nn.value_counts(normalize=True).iloc[0])
        return nu, top, False
    except TypeError:
        ss = nn.map(lambda v: repr(v))  # deterministic, hashable
        nu = int(ss.nunique())
        top = float(ss.value_counts(normalize=True).iloc[0])
        return nu, top, True


def col_health(s: pd.Series, near_constant: float = DEFAULT_NEAR_CONSTANT_FRAC) -> dict:
    n = int(len(s))
    n_null = int(s.isna().sum())
    nn = s.dropna()
    n_unique, top_frac, unhashable = unique_and_top(nn)
    out = {"n": n, "null_pct": round(100.0 * n_null / n, 3) if n else 0.0,
           "n_unique": n_unique, "dtype": str(s.dtype),
           "unhashable": unhashable}
    is_num = pd.api.types.is_numeric_dtype(s)
    if is_num and len(nn):
        arr = nn.to_numpy(dtype=float)
        out.update({"zero_pct": round(100.0 * float((arr == 0).sum()) / len(arr), 3),
                    "min": float(np.min(arr)), "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)), "std": float(np.std(arr))})
    else:
        out.update({"zero_pct": np.nan, "min": np.nan, "max": np.nan,
                    "mean": np.nan, "std": np.nan})
    # degeneracy
    reasons = []
    if len(nn) == 0:
        reasons.append("ALL_NULL")
    else:
        if n_unique <= 1:
            reasons.append("CONSTANT")
        elif top_frac >= near_constant:
            reasons.append(f"NEAR_CONSTANT({top_frac:.4f})")
        if is_num and out.get("zero_pct", 0) == 100.0:
            reasons.append("ALL_ZERO")
    out["degenerate"] = ";".join(reasons) if reasons else ""
    return out


def is_degenerate(health: dict) -> bool:
    """True iff the col_health dict carries any degeneracy reason."""
    return bool(health.get("degenerate"))


def verdict(health: dict) -> str:
    """Per-column verdict string: the degeneracy reasons, or 'healthy'."""
    return health.get("degenerate") or "healthy"
