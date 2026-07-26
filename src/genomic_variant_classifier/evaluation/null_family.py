"""Matched-spectrum null operators for the R3 recovery test.

WHY THIS EXISTS
===============
A recovery statistic must be compared against a null that matches the observed
transform's STRENGTH while destroying only the property under test. Panel R's R3
measures whether TRAIN-fitted whitening recovers held-out structure. The wrong
null is a Haar rotation: angular concentration is exactly rotation-invariant
(see test_norm_angle_probe.py), so a rotation cannot separate whitening from
chance -- it is an invariance control, not an inferential reference.

The right null preserves the whitening GAIN SPECTRUM -- the singular values,
determinant, Frobenius norm, operator norm and condition number, i.e. the entire
numerical character of the transform -- while randomising the one thing whose
scientific value is under test: the ALIGNMENT of the rescaling gains with the
covariance eigen-directions. Two such nulls, per Monzia Moodie's design:

  PRIMARY -- eigenvalue-assignment permutation.
    W_perm = U @ diag(gains[P]) @ U.T, P a random permutation.
    Keeps the empirical eigenvectors U and the exact gain spectrum; breaks only
    the gain-to-direction correspondence (a low-variance covariance direction no
    longer receives the large whitening gain that generated it). This is the
    cleanest conditional-randomisation null: preserve nuisance structure,
    randomise the hypothesised informative relationship. It is cheap -- an O(d)
    permutation, no QR per draw.

  SECONDARY -- matched-spectrum random orientation.
    W_orient = Q @ diag(gains) @ Q.T, Q Haar-orthogonal.
    Same gains, a completely unrelated orientation. A broader test of alignment
    specificity: it destroys alignment with BOTH the covariance eigenvectors and
    the representation axes, not just the gain-to-direction assignment.

GAIN-SPREAD PRECONDITION (a robustness guard)
---------------------------------------------
The permutation null is only informative if the gains actually differ. Under
heavy shrinkage or a near-isotropic covariance the gains flatten, and permuting
near-equal gains barely moves the operator -- a diffuse, uninformative null that
an ensemble-diversity check on operator differences would still wave through.
require_informative_gain_spread refuses a permutation null whose gain coefficient
of variation is below a floor, so a flat spectrum fails loud rather than yielding
a null that cannot reject anything. The orientation null does not need this (it
reorients rather than reassigns), so it is the fallback when gains are flat.

STATISTICAL WORDING
-------------------
Downstream, the comparison of an observed statistic against these nulls yields a
Monte Carlo matched-null tail probability, NOT an exact permutation p-value:
exactness would require an established invariance/exchangeability theorem for
this transformation family, which is not claimed here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .norm_angle_probe import WhiteningTransform

logger = logging.getLogger(__name__)

__all__ = [
    "NullKind",
    "NullOperator",
    "haar_orthogonal",
    "eigenvalue_assignment_null",
    "matched_spectrum_orientation_null",
    "require_informative_gain_spread",
    "assert_spectrum_matched",
    "FlatGainSpectrumError",
    "MissingEigendecompositionError",
]

_EPS = 1e-12
_DEFAULT_GAIN_CV_FLOOR = 0.05   # below this, permuting gains is uninformative


class NullKind(str, Enum):
    EIGENVALUE_PERMUTATION = "eigenvalue_permutation"
    MATCHED_SPECTRUM_ORIENTATION = "matched_spectrum_orientation"


class FlatGainSpectrumError(ValueError):
    """Raised when a permutation null is requested on gains too flat to inform:
    permuting near-equal gains barely changes the operator, so the null cannot
    reject anything and would be silently anti-conservative."""


class MissingEigendecompositionError(ValueError):
    """Raised when a null is requested from a WhiteningTransform that does not
    carry its eigenvectors/gains (e.g. constructed directly rather than via
    fit_whitening). The null family cannot re-derive matched operators without
    the factors."""


@dataclass(frozen=True)
class NullOperator:
    """A matched-spectrum null transform and the provenance of how it was made."""

    kind: NullKind
    matrix: np.ndarray            # (dim, dim) symmetric, same gains as observed
    gains: np.ndarray             # the (sorted) gain spectrum it was built from
    condition_number: float
    preserves_eigenvectors: bool

    def transform(self, x: np.ndarray, mean: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - mean) @ self.matrix


def _require_factors(fit: WhiteningTransform) -> tuple[np.ndarray, np.ndarray]:
    if fit.eigenvectors is None or fit.gains is None:
        raise MissingEigendecompositionError(
            "this WhiteningTransform carries no eigenvectors/gains; build it via "
            "fit_whitening so the matched-null family can re-derive operators")
    return np.asarray(fit.eigenvectors, dtype=np.float64), \
        np.asarray(fit.gains, dtype=np.float64)


def gain_coefficient_of_variation(gains: np.ndarray) -> float:
    """Spread of the gains: std / mean. Zero means all gains equal (permutation
    is a no-op); large means a strongly anisotropic whitening."""
    g = np.asarray(gains, dtype=np.float64)
    m = float(g.mean())
    if m <= _EPS:
        return 0.0
    return float(g.std() / m)


def require_informative_gain_spread(
    fit: WhiteningTransform, *, floor: float = _DEFAULT_GAIN_CV_FLOOR,
) -> None:
    """Refuse a permutation null when the gains are too flat to inform. This is
    the robustness guard: a near-isotropic covariance flattens the gains, and a
    permutation of near-equal gains cannot reject anything."""
    _, gains = _require_factors(fit)
    cv = gain_coefficient_of_variation(gains)
    if cv < floor:
        raise FlatGainSpectrumError(
            f"gain coefficient of variation {cv:.4f} is below the floor {floor}; "
            "permuting near-equal gains yields an uninformative null. Use the "
            "matched-spectrum orientation null, which reorients rather than "
            "reassigns, or raise the anisotropy of the fit.")


def haar_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-uniform random orthogonal matrix via QR of a Gaussian, with the
    sign correction that makes the distribution genuinely uniform."""
    if dim < 1:
        raise ValueError("dimension must be positive")
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


def eigenvalue_assignment_null(
    fit: WhiteningTransform, rng: np.random.Generator,
    *, gain_cv_floor: float = _DEFAULT_GAIN_CV_FLOOR,
) -> NullOperator:
    """PRIMARY null: keep U and the gain spectrum, permute the gain-to-direction
    assignment. Refuses a flat gain spectrum (see require_informative_gain_spread)."""
    U, gains = _require_factors(fit)
    require_informative_gain_spread(fit, floor=gain_cv_floor)
    perm = rng.permutation(gains.size)
    permuted = gains[perm]
    matrix = (U * permuted) @ U.T
    return NullOperator(
        kind=NullKind.EIGENVALUE_PERMUTATION,
        matrix=matrix,
        gains=np.sort(permuted),
        condition_number=float(permuted.max() / permuted.min()),
        preserves_eigenvectors=True,
    )


def matched_spectrum_orientation_null(
    fit: WhiteningTransform, rng: np.random.Generator,
) -> NullOperator:
    """SECONDARY null: keep the gain spectrum, apply a completely random
    orientation. Does not need the gain-spread guard -- it reorients rather than
    reassigns, so it is informative even when the gains are flat."""
    _, gains = _require_factors(fit)
    q = haar_orthogonal(gains.size, rng)
    matrix = (q * gains) @ q.T
    return NullOperator(
        kind=NullKind.MATCHED_SPECTRUM_ORIENTATION,
        matrix=matrix,
        gains=np.sort(gains),
        condition_number=float(gains.max() / gains.min()),
        preserves_eigenvectors=False,
    )


def assert_spectrum_matched(
    observed: np.ndarray, candidate: np.ndarray,
    *, atol: float = 1e-9, rtol: float = 1e-7,
) -> None:
    """Assert two symmetric operators share the same spectrum (eigenvalues). This
    is the contract a matched null MUST satisfy: same singular values, hence same
    determinant, Frobenius/operator norm and condition number. A null that fails
    this is not matched and any comparison against it confounds strength with
    alignment."""
    ov = np.sort(np.linalg.eigvalsh(np.asarray(observed, dtype=np.float64)))
    cv = np.sort(np.linalg.eigvalsh(np.asarray(candidate, dtype=np.float64)))
    if not np.allclose(ov, cv, atol=atol, rtol=rtol):
        raise AssertionError(
            "candidate operator spectrum does not match the observed spectrum; "
            "the null is not matched and would confound transform strength with "
            "alignment")
