"""src/genomic_variant_classifier/models/disagreement.py

Author: Monzia Moodie

Schema-2 uncertainty for a finite, equally weighted set of Bernoulli draws (MC-dropout passes, fitted-ensemble members,
or an algorithm roster viewed as a research diagnostic). Gate G5 of the frozen package-1 acceptance matrix.

Two exact identities, kept in their own units and NEVER added together (a variance plus an entropy has no unit):

  entropy, in nats:              H(q)      = mean_t H(p_t)        + mean_t KL(p_t || q)
  variance, in probability^2:    q (1 - q) = mean_t p_t (1 - p_t) + mean_t (p_t - q)^2

with q = mean_t p_t. The legacy names "aleatoric" (mean entropy) and "epistemic" (variance) are retired: the first
term of each identity is the expected within-draw uncertainty and the second is the between-draw spread of THIS finite
draw set; neither is a population quantity, and the variance is not an "epistemic uncertainty" of anything.

The disagreement term uses the near/far Kullback-Leibler kernel adapted (owner review bundle revision 4,
disagreement_reference.py) from the audited refinements.py, with row-local numerical status:
  exact_agreement  every draw holds the same represented value; the disagreement is exactly 0
  resolved         computed in the qualified normal-number domain
  unresolved       null with a reason: the rounded mixture mean is an endpoint, or the value is below the smallest
                   normal binary64 number (subnormal results are not qualified)
Rows whose draws are all >= 1/2 are complemented first, where 1 - p is exact (Sterbenz), which rescues the pair
(1, 1 - 2^-53) at 3.8477e-17 nats instead of an endpoint. Inputs outside [0, 1] are refused; nothing is clipped.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from genomic_variant_classifier.prediction_contract import ContractError, binary_entropy

logger = logging.getLogger(__name__)

__all__ = ["DISAGREEMENT_METHOD", "bernoulli_kl", "disagreement_with_status", "draw_decomposition"]

DISAGREEMENT_METHOD = "bernoulli_kl_near_far_exact_complement_v2"

_NEAR = 0.5          # |t| <= 1/2 selects the cancellation-free branch; then q/2 <= p <= 2q and p - q is exact
_SERIES_TERMS = 24   # |u| <= 1/3 on the near branch; (1/9)**24 < 1e-22, far below double precision


def _log1p_minus_identity(t: np.ndarray) -> np.ndarray:
    """g(t) = log1p(t) - t for |t| <= 1/2 without cancellation.

    With u = t / (2 + t): log1p(t) = 2 atanh(u) and 2u - t = -t u, so g(t) = -t u + 2u sum_{k>=1} u^(2k) / (2k + 1).
    """
    u = t / (2.0 + t)
    u2 = u * u
    s = np.zeros_like(u)
    for k in range(_SERIES_TERMS, 0, -1):     # Horner from the smallest term
        s = u2 * (1.0 / (2 * k + 1) + s)
    return -t * u + 2.0 * u * s


def bernoulli_kl(p: Any, q: Any) -> np.ndarray:
    """KL(Bernoulli(p) || Bernoulli(q)) in nats, elementwise; q strictly inside (0, 1) wherever p differs from it.

    Near p == q (|t1|, |t2| <= 1/2, t1 = (p - q)/q, t2 = (q - p)/(1 - q)) the two log terms cancel to order
    (p - q)^2, so that branch is rewritten exactly as KL = -t1 t2 + p g(t1) + (1 - p) g(t2). p - q is formed ONCE from
    the inputs; elsewhere each term uses the form without cancellation (log of a normal ratio, else a log difference,
    and log1p for the complement).
    """
    p, q = np.asarray(p), np.asarray(q)
    if (p.dtype.kind != "f" or q.dtype.kind != "f" or p.dtype.itemsize not in (4, 8)
            or q.dtype.itemsize not in (4, 8)):
        raise ContractError("float32/64 probabilities required; no string/bool coercion")
    p, q = np.broadcast_arrays(p.astype(np.float64), q.astype(np.float64))
    if (not np.isfinite(p).all() or not np.isfinite(q).all()
            or ((p < 0) | (p > 1) | (q < 0) | (q > 1)).any()):
        raise ContractError("finite probabilities in [0, 1] required")
    tiny = np.finfo(np.float64).tiny
    d = p - q
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        t1 = d / q
        t2 = -d / (1.0 - q)
        near = (np.abs(t1) <= _NEAR) & (np.abs(t2) <= _NEAR) & (q > 0) & (q < 1)
        t1n, t2n = np.where(near, t1, 0.0), np.where(near, t2, 0.0)
        # (p - q)^2 / (q (1 - q)) == -t1 t2: no square of a small difference, so no spurious underflow.
        near_val = -t1n * t2n + p * _log1p_minus_identity(t1n) + (1.0 - p) * _log1p_minus_identity(t2n)
        r1 = p / q
        a_far = np.where((r1 >= tiny) & np.isfinite(r1), np.log(r1), np.log(p) - np.log(q))
        both_high = (p >= 0.5) & (q >= 0.5)           # then 1 - p and 1 - q are exact (Sterbenz)
        r2 = (1.0 - p) / (1.0 - q)
        b_far = np.where(both_high & (r2 >= tiny) & np.isfinite(r2), np.log(r2), np.log1p(-p) - np.log1p(-q))
        a = np.where(np.abs(t1) <= _NEAR, np.log1p(t1), a_far)
        b = np.where(np.abs(t2) <= _NEAR, np.log1p(t2), b_far)
        far_val = np.where(p > 0, p * a, 0.0) + np.where(p < 1, (1.0 - p) * b, 0.0)
    out = np.where(p == q, 0.0, np.where(near, near_val, far_val))
    if not np.isfinite(out).all():
        raise ArithmeticError("KL is nonfinite: check support and the rounded reference mean")
    if (out < 0).any():
        raise ArithmeticError("negative computed KL; no silent clipping")
    return out


def _validated_draws(stack: Any) -> tuple[np.ndarray, str]:
    raw = np.asarray(stack)
    if raw.dtype.kind != "f" or raw.dtype.itemsize not in (4, 8):
        raise ContractError("float32/64 draws required")
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] == 0:
        raise ContractError("nonempty (draws >= 2, rows) matrix required")
    p = raw.astype(np.float64)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ContractError("finite draws in [0, 1] required; nothing is clipped")
    return p, raw.dtype.name


def disagreement_with_status(stack: Any) -> dict[str, Any]:
    """Finite equal-weight empirical-mixture disagreement (Jensen-Shannon form), in nats, with row-local status."""
    p, precision = _validated_draws(stack)
    agreement = (p == p[0]).all(axis=0)
    flip = (p >= 0.5).all(axis=0) & ~agreement
    q = p.copy()
    q[:, flip] = 1.0 - q[:, flip]
    mean = q.mean(axis=0)
    endpoint = ~agreement & ((mean == 0) | (mean == 1))
    eligible = ~agreement & ~endpoint
    computed = np.zeros(p.shape[1])
    with np.errstate(under="ignore"):
        if eligible.any():
            computed[eligible] = bernoulli_kl(q[:, eligible], mean[None, eligible]).mean(axis=0)
        mixture_entropy = binary_entropy(mean)
        draw_entropy = binary_entropy(q.ravel()).reshape(q.shape).mean(axis=0)
    tiny = np.finfo(np.float64).tiny
    values: list[float | None] = []
    statuses: list[str] = []
    reasons: list[str | None] = []
    for i in range(p.shape[1]):
        if agreement[i]:
            values.append(0.0); statuses.append("exact_agreement"); reasons.append(None)
        elif endpoint[i]:
            values.append(None); statuses.append("unresolved"); reasons.append("rounded_mixture_endpoint")
        elif computed[i] < tiny:
            values.append(None); statuses.append("unresolved"); reasons.append("below_qualified_normal_domain")
        else:
            values.append(float(computed[i])); statuses.append("resolved"); reasons.append(None)
    return {"values": values, "status": statuses, "reason": reasons,
            "complemented_rows": flip.tolist(),
            "mixture_predictive_entropy_nats": [None if endpoint[i] else float(v)
                                               for i, v in enumerate(mixture_entropy)],
            "mean_draw_entropy_nats": draw_entropy.tolist(),
            "method": DISAGREEMENT_METHOD,
            "producer_precision": precision,
            "interpretation": "finite_draw_mixture_disagreement_nats"}


def draw_decomposition(stack: Any) -> dict[str, Any]:
    """Both identities for an equally weighted (draws, rows) stack, each in its own unit (ddof = 0).

    Returns plain vectors plus the per-row disagreement status. The mixture mean is computed in float64 from the
    validated draws; a producer that reports its own mean (for example a float32 wrapper) keeps that value as its
    prediction, and this mean is a diagnostic.
    """
    p, precision = _validated_draws(stack)
    # Rows whose draws all hold the same value: the mean IS that value and the spread is EXACTLY zero. Measured
    # 2026-11-07: ten draws of 0.3 average to 0.30000000000000004 in floating point and np.var then reports a
    # nonzero spread for draws that agree exactly.
    agreement = (p == p[0]).all(axis=0)
    mean = np.where(agreement, p[0], p.mean(axis=0))
    within_variance = np.where(agreement, p[0] * (1.0 - p[0]), (p * (1.0 - p)).mean(axis=0))
    spread_variance = np.where(agreement, 0.0, ((p - mean) ** 2).mean(axis=0))
    kl = disagreement_with_status(p)
    return {"schema": "gvc.uncertainty.schema-2",
            "n_draws": int(p.shape[0]),
            "mixture_mean": mean,
            "expected_draw_entropy_nats": np.asarray(kl["mean_draw_entropy_nats"], dtype=np.float64),
            "mixture_predictive_entropy_nats": kl["mixture_predictive_entropy_nats"],
            "disagreement_nats": kl["values"],
            "disagreement_status": kl["status"],
            "disagreement_reason": kl["reason"],
            "expected_within_draw_variance_probability_squared": within_variance,
            "between_draw_spread_variance_probability_squared": spread_variance,
            "ddof": 0,
            "weighting": "uniform_over_declared_draws",
            "disagreement_method": DISAGREEMENT_METHOD,
            "producer_precision": precision}
