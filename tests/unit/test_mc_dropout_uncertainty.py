"""tests/unit/test_mc_dropout_uncertainty.py

Regression tests for `_decompose_uncertainty` in
`genomic_variant_classifier.models.mc_dropout`.

Background
----------
Run 14 anomaly catalogue (Run 15 backlog) flagged A1: potential `np.log(0)`
at mc_dropout.py:87:

    entropy_per_pass = -(clipped * np.log(clipped)
                         + (1 - clipped) * np.log(1 - clipped))

View-first inspection (2026-05-26) revealed the immediately preceding
line clips the input:

    L85: eps = 1e-8
    L86: clipped = np.clip(probs_stack, eps, 1.0 - eps)

So the log is mathematically safe: at the worst-case boundary,
log(1e-8) ~= -18.42, and the surrounding multiplication
1e-8 * log(1e-8) ~= -1.84e-7 is finite. A1 is a verified false anomaly
(see docs/runs/RUN_15_PLAN.md B.O2).

These tests lock that boundary-safe behaviour in:
  - Any future refactor that removes the line-86 clip will produce
    nan/-inf for probs_stack containing 0.0 or 1.0 and fail
    test_all_zero_probs_stack_finite / test_all_one_probs_stack_finite.
  - Interior correctness (p=0.5 -> aleatoric = log(2)) ensures the
    overall entropy formula stays intact.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.models.mc_dropout import _decompose_uncertainty


class TestDecomposeUncertaintyBoundary:
    """A1 regression guard -- boundary inputs must produce finite outputs."""

    def test_all_zero_probs_stack_finite(self):
        """probs_stack containing exactly 0.0 must yield finite outputs."""
        probs = np.zeros((10, 5))
        mean_p, epi, alea = _decompose_uncertainty(probs)
        assert np.all(np.isfinite(mean_p)), "mean_prob non-finite on zeros"
        assert np.all(np.isfinite(epi)),    "epistemic non-finite on zeros"
        assert np.all(np.isfinite(alea)),   "aleatoric non-finite on zeros"
        assert np.all(alea >= 0.0)
        assert np.all(alea < 1e-5), f"boundary aleatoric too large: max={alea.max()}"

    def test_all_one_probs_stack_finite(self):
        """probs_stack containing exactly 1.0 must yield finite outputs."""
        probs = np.ones((10, 5))
        mean_p, epi, alea = _decompose_uncertainty(probs)
        assert np.all(np.isfinite(mean_p))
        assert np.all(np.isfinite(epi))
        assert np.all(np.isfinite(alea))
        assert np.all(alea >= 0.0)
        assert np.all(alea < 1e-5)

    def test_mixed_boundary_and_interior_finite(self):
        """Each column hits a different boundary or interior value."""
        probs = np.array([
            [0.0, 1.0, 1e-15, 0.5, 1.0 - 1e-15],
            [0.0, 1.0, 1e-15, 0.5, 1.0 - 1e-15],
            [0.0, 1.0, 1e-15, 0.5, 1.0 - 1e-15],
        ])
        mean_p, epi, alea = _decompose_uncertainty(probs)
        assert np.all(np.isfinite(mean_p))
        assert np.all(np.isfinite(epi))
        assert np.all(np.isfinite(alea))


class TestDecomposeUncertaintyCorrectness:
    """Interior-case correctness of the entropy + variance formula."""

    def test_interior_p_half_gives_log2(self):
        """At p=0.5 (max entropy), aleatoric must equal log(2)."""
        probs = np.full((20, 3), 0.5)
        _, _, alea = _decompose_uncertainty(probs)
        expected = np.log(2.0)
        np.testing.assert_allclose(alea, expected, rtol=1e-9, atol=1e-12)

    def test_epistemic_zero_when_passes_agree(self):
        """If all T passes return identical probability, epistemic must be 0."""
        probs = np.full((10, 5), 0.3)
        _, epi, _ = _decompose_uncertainty(probs)
        np.testing.assert_allclose(epi, 0.0, atol=1e-12)

    def test_mean_prob_matches_arithmetic_mean(self):
        """mean_prob must equal the across-passes arithmetic mean."""
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.0, 1.0, size=(15, 7))
        mean_p, _, _ = _decompose_uncertainty(probs)
        np.testing.assert_allclose(mean_p, probs.mean(axis=0), rtol=1e-12)

    def test_output_shapes(self):
        """All three returned arrays must have shape (n_samples,)."""
        probs = np.full((10, 8), 0.4)
        mean_p, epi, alea = _decompose_uncertainty(probs)
        assert mean_p.shape == (8,)
        assert epi.shape    == (8,)
        assert alea.shape   == (8,)