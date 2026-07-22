"""The matched-spectrum null family preserves the whitening spectrum, breaks
alignment, and refuses a flat gain spectrum for the permutation null.

WHAT IS PINNED
==============
  1. spectrum match  -- both nulls share the observed whitening's eigenvalues
                        (hence condition number, determinant, norms) exactly.
  2. alignment break -- the observed whitening whitens the data (covariance ->
                        identity); the nulls do not.
  3. gain-spread guard -- the permutation null is refused when gains are too flat
                        to inform; the orientation null still works (fallback).
  4. factor requirement -- a null cannot be built from a transform lacking its
                        eigendecomposition.
  5. haar orthogonal -- the orientation null's random matrix is truly orthogonal.

No torch. Artifacts built via the real extraction boundary.

Author: written for Monzia Moodie, 2026-07-22.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.representation_artifact import (
    extract_focal_embeddings)
from genomic_variant_classifier.evaluation.norm_angle_probe import (
    fit_whitening, WhiteningTransform)
from genomic_variant_classifier.evaluation.null_family import (
    NullKind, NullOperator, haar_orthogonal, eigenvalue_assignment_null,
    matched_spectrum_orientation_null, require_informative_gain_spread,
    assert_spectrum_matched, gain_coefficient_of_variation,
    FlatGainSpectrumError, MissingEigendecompositionError)


class _MockOut:
    def __init__(self, e):
        self.focal_embeddings = e

    @property
    def has_embeddings(self):
        return self.focal_embeddings is not None


def _artifact(emb, role="TRAIN"):
    return extract_focal_embeddings(
        _MockOut(emb), [f"v{i}" for i in range(len(emb))],
        representation_name="r", partition_role=role, model_class="M", git_sha="t")


def _anisotropic(n=400, d=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x[:, :3] *= 6.0
    return x


def _fit(seed=0, ridge=1e-6):
    return fit_whitening(_artifact(_anisotropic(seed=seed)), ridge=ridge)


# --------------------------------------------------------------------------- #
# 1. spectrum match
# --------------------------------------------------------------------------- #
def test_permutation_null_matches_spectrum():
    t = _fit()
    n = eigenvalue_assignment_null(t, np.random.default_rng(0))
    assert_spectrum_matched(t.W, n.matrix)  # raises if not matched


def test_orientation_null_matches_spectrum():
    t = _fit()
    n = matched_spectrum_orientation_null(t, np.random.default_rng(0))
    assert_spectrum_matched(t.W, n.matrix)


def test_both_nulls_share_the_observed_condition_number():
    t = _fit()
    obs_cond = float(np.linalg.cond(t.W))
    for factory in (eigenvalue_assignment_null, matched_spectrum_orientation_null):
        n = factory(t, np.random.default_rng(1))
        assert abs(n.condition_number - obs_cond) / obs_cond < 1e-6


def test_assert_spectrum_matched_rejects_unmatched():
    t = _fit()
    # a scaled operator has a different spectrum
    with pytest.raises(AssertionError):
        assert_spectrum_matched(t.W, 2.0 * t.W)


# --------------------------------------------------------------------------- #
# 2. alignment break
# --------------------------------------------------------------------------- #
def test_observed_whitening_beats_nulls_at_whitening():
    x = _anisotropic()
    t = fit_whitening(_artifact(x))
    xc = x - t.mean

    def whiten_err(M):
        return np.linalg.norm(np.cov(xc @ M, rowvar=False) - np.eye(x.shape[1]), "fro")

    obs = whiten_err(t.W)
    perm = whiten_err(eigenvalue_assignment_null(t, np.random.default_rng(0)).matrix)
    orient = whiten_err(matched_spectrum_orientation_null(t, np.random.default_rng(0)).matrix)
    assert obs < perm, "observed whitening must align with covariance; permutation must not"
    assert obs < orient, "observed whitening must align with covariance; orientation must not"


def test_orientation_null_is_genuinely_reoriented():
    """The orientation null must apply a RANDOM rotation, not the identity. With
    q=identity it would collapse to diag(gains): axis-aligned (no off-diagonal
    mass) and identical across draws. A real Haar rotation mixes axes, so the
    operator has substantial off-diagonal mass and two draws differ. This catches
    substituting the identity for the Haar orthogonal."""
    t = _fit()
    a = matched_spectrum_orientation_null(t, np.random.default_rng(0)).matrix
    b = matched_spectrum_orientation_null(t, np.random.default_rng(1)).matrix
    # two independent draws must differ
    assert not np.allclose(a, b), "two orientation draws must differ (random)"
    # the operator must NOT be diagonal -- a real rotation mixes axes
    off_diag_mass = np.linalg.norm(a - np.diag(np.diag(a)), "fro")
    diag_mass = np.linalg.norm(np.diag(a))
    assert off_diag_mass > 0.1 * diag_mass, (
        "orientation null must be reoriented (off-diagonal), not axis-aligned; "
        "a diagonal operator means the identity was used instead of a rotation")


def test_permutation_preserves_eigenvectors_flag():
    t = _fit()
    assert eigenvalue_assignment_null(t, np.random.default_rng(0)).preserves_eigenvectors
    assert not matched_spectrum_orientation_null(t, np.random.default_rng(0)).preserves_eigenvectors


# --------------------------------------------------------------------------- #
# 3. gain-spread guard
# --------------------------------------------------------------------------- #
def test_permutation_null_refuses_flat_gains():
    # heavy ridge on near-isotropic data flattens the gains
    flat = fit_whitening(_artifact(np.random.default_rng(0).normal(size=(500, 16))),
                         ridge=1.0)
    with pytest.raises(FlatGainSpectrumError):
        eigenvalue_assignment_null(flat, np.random.default_rng(0))


def test_orientation_null_works_on_flat_gains():
    flat = fit_whitening(_artifact(np.random.default_rng(0).normal(size=(500, 16))),
                         ridge=1.0)
    n = matched_spectrum_orientation_null(flat, np.random.default_rng(0))
    assert n.kind is NullKind.MATCHED_SPECTRUM_ORIENTATION


def test_permutation_null_allowed_on_anisotropic_gains():
    t = _fit()  # anisotropic
    n = eigenvalue_assignment_null(t, np.random.default_rng(0))
    assert n.kind is NullKind.EIGENVALUE_PERMUTATION


def test_gain_cv_is_higher_for_anisotropic_than_flat():
    aniso = _fit(ridge=1e-6)
    flat = fit_whitening(_artifact(np.random.default_rng(0).normal(size=(500, 16))),
                         ridge=1.0)
    assert gain_coefficient_of_variation(aniso.gains) > \
        gain_coefficient_of_variation(flat.gains)


def test_require_informative_gain_spread_raises_below_floor():
    flat = fit_whitening(_artifact(np.random.default_rng(0).normal(size=(500, 16))),
                         ridge=1.0)
    with pytest.raises(FlatGainSpectrumError):
        require_informative_gain_spread(flat)


# --------------------------------------------------------------------------- #
# 4. factor requirement
# --------------------------------------------------------------------------- #
def test_null_requires_eigendecomposition():
    # a transform built WITHOUT eigenvectors/gains cannot seed a null
    bare = WhiteningTransform(
        mean=np.zeros(4), W=np.eye(4), fit_partition_role="TRAIN",
        n_fit_rows=10, ridge=1e-6)  # eigenvectors/gains default None
    with pytest.raises(MissingEigendecompositionError):
        eigenvalue_assignment_null(bare, np.random.default_rng(0))
    with pytest.raises(MissingEigendecompositionError):
        matched_spectrum_orientation_null(bare, np.random.default_rng(0))


def test_fit_whitening_populates_the_factors():
    t = _fit()
    assert t.eigenvectors is not None and t.gains is not None
    # W reconstructs exactly from the factors
    np.testing.assert_allclose(
        (t.eigenvectors * t.gains) @ t.eigenvectors.T, t.W, atol=1e-10)


# --------------------------------------------------------------------------- #
# 5. haar orthogonal
# --------------------------------------------------------------------------- #
def test_haar_orthogonal_is_orthogonal():
    q = haar_orthogonal(12, np.random.default_rng(0))
    np.testing.assert_allclose(q @ q.T, np.eye(12), atol=1e-10)


def test_haar_orthogonal_rejects_bad_dim():
    with pytest.raises(ValueError):
        haar_orthogonal(0, np.random.default_rng(0))
