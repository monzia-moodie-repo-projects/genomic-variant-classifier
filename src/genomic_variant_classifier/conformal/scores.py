"""Nonconformity score functions for conformal prediction (from scratch, numpy).

A nonconformity score s(x, y) is LOW when (x, y) conforms (model confident in y) and HIGH when it
does not. The conformal machinery thresholds these scores. All scores are multiclass-general so
the 5-class milestone (M2) reuses them unchanged.

Implemented:
  - LAC  (Least Ambiguous set-valued Classifier): s(x,y) = 1 - P[y].
  - APS  (Adaptive Prediction Sets, Romano-Sesia-Candes 2020): for the class at sorted rank r,
         s = (probability mass STRICTLY above r) + u * P[that class], with a single U(0,1) draw
         u PER SAMPLE (shared across candidate classes at prediction time). This exclusive-mass +
         per-row-u construction is what yields exact coverage; an inclusive cumsum or per-cell
         randomization subtly breaks exchangeability (verified empirically during development).
  - RAPS (Regularized APS, Angelopoulos et al. 2021): APS + lam * max(0, rank - k_reg).

CRITICAL CONVENTION: calibration (true-label) scores and prediction (all-class) scores MUST use
the identical rule and the identical per-row u, or class-conditional coverage drifts. Both paths
below share _aps_components() to guarantee that.
"""
from __future__ import annotations

import numpy as np


def _validate(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D (n, K); got shape {P.shape}")
    if np.any(P < -1e-9) or np.any(P > 1 + 1e-9):
        raise ValueError("P entries must lie in [0, 1]")
    return P


# ---------------------------------------------------------------- LAC
def lac_scores_true(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    P = _validate(P)
    y = np.asarray(y, dtype=int)
    return 1.0 - P[np.arange(len(y)), y]


def lac_scores_all(P: np.ndarray) -> np.ndarray:
    P = _validate(P)
    return 1.0 - P


# ---------------------------------------------------------------- APS core
def _aps_exclusive_mass(P: np.ndarray) -> np.ndarray:
    """For each (row, class), the probability mass of classes STRICTLY more probable (rho_k).
    Ties broken by the argsort order (stable)."""
    n, K = P.shape
    order = np.argsort(-P, axis=1)
    sorted_P = np.take_along_axis(P, order, axis=1)
    cum_excl_sorted = np.cumsum(sorted_P, axis=1) - sorted_P     # mass strictly above each rank
    excl = np.empty_like(P)
    rows = np.arange(n)[:, None]
    excl[rows, order] = cum_excl_sorted
    return excl


def _aps_ranks(P: np.ndarray) -> np.ndarray:
    """1-based rank of each class within its row's descending order."""
    n, K = P.shape
    order = np.argsort(-P, axis=1)
    ranks = np.empty((n, K), dtype=int)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(1, K + 1)[None, :]
    return ranks


def aps_scores_true(P: np.ndarray, y: np.ndarray, u: np.ndarray | None = None,
                    rng: np.random.Generator | None = None, randomize: bool = True) -> np.ndarray:
    """APS score of the TRUE label: rho_y + u * P[y]. u is one draw per row."""
    P = _validate(P)
    y = np.asarray(y, dtype=int)
    n = len(y)
    excl = _aps_exclusive_mass(P)
    rho_true = excl[np.arange(n), y]
    p_true = P[np.arange(n), y]
    if not randomize:
        return rho_true + p_true            # deterministic (upper) variant
    if u is None:
        if rng is None:
            rng = np.random.default_rng(0)
        u = rng.uniform(size=n)
    return rho_true + u * p_true


def aps_scores_all(P: np.ndarray, u: np.ndarray | None = None,
                   rng: np.random.Generator | None = None, randomize: bool = True) -> np.ndarray:
    """APS score for EVERY class: rho_k + u * P[k], with the SAME per-row u across classes."""
    P = _validate(P)
    n, K = P.shape
    excl = _aps_exclusive_mass(P)
    if not randomize:
        return excl + P
    if u is None:
        if rng is None:
            rng = np.random.default_rng(0)
        u = rng.uniform(size=n)
    return excl + u[:, None] * P


# ---------------------------------------------------------------- RAPS
def raps_scores_true(P: np.ndarray, y: np.ndarray, u: np.ndarray | None = None,
                     rng: np.random.Generator | None = None, randomize: bool = True,
                     k_reg: int = 1, lam: float = 0.0) -> np.ndarray:
    P = _validate(P)
    y = np.asarray(y, dtype=int)
    base = aps_scores_true(P, y, u=u, rng=rng, randomize=randomize)
    ranks = _aps_ranks(P)[np.arange(len(y)), y]
    return base + lam * np.maximum(0, ranks - k_reg)


def raps_scores_all(P: np.ndarray, u: np.ndarray | None = None,
                    rng: np.random.Generator | None = None, randomize: bool = True,
                    k_reg: int = 1, lam: float = 0.0) -> np.ndarray:
    P = _validate(P)
    base = aps_scores_all(P, u=u, rng=rng, randomize=randomize)
    ranks = _aps_ranks(P)
    return base + lam * np.maximum(0, ranks - k_reg)
