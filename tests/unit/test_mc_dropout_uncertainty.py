"""tests/unit/test_mc_dropout_uncertainty.py

Schema-2 uncertainty of a finite, equally weighted draw set (models/disagreement.draw_decomposition), package 1 gate G5.
Author: Monzia Moodie

Rewritten 2026-11-07 for package 1, set 1a. The function these tests guarded, `_decompose_uncertainty`, returned a
draw variance labelled "epistemic" and a mean draw entropy labelled "aleatoric", and clipped its input. Both names are
retired and nothing is clipped. The intents are KEPT and now stated exactly:

  * boundary draws (0 and 1, the Run 14 anomaly A1) give finite values -- and certain draws give EXACT zeros, because
    the entropy uses 0 log 0 := 0 rather than a clip floor;
  * p = 1/2 gives log 2; identical draws give an exactly zero spread; the mean is the arithmetic mean; shapes are
    (rows,).
New: both identities close, each in its own unit; inputs outside [0, 1] and fewer than two draws are REFUSED.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.models.disagreement import draw_decomposition
from genomic_variant_classifier.prediction_contract import ContractError


class TestBoundaryDraws:
    """A1 regression guard, strengthened: boundary inputs give finite AND exact values."""

    def test_all_zero_draws_give_exact_zeros(self):
        r = draw_decomposition(np.zeros((10, 5)))
        assert np.array_equal(r["mixture_mean"], np.zeros(5))
        assert np.array_equal(r["expected_draw_entropy_nats"], np.zeros(5))
        assert np.array_equal(r["between_draw_spread_variance_probability_squared"], np.zeros(5))
        assert r["disagreement_status"] == ["exact_agreement"] * 5 and r["disagreement_nats"] == [0.0] * 5

    def test_all_one_draws_give_exact_zeros(self):
        r = draw_decomposition(np.ones((10, 5)))
        assert np.array_equal(r["expected_draw_entropy_nats"], np.zeros(5))
        assert r["mixture_predictive_entropy_nats"] == [0.0] * 5
        assert r["disagreement_status"] == ["exact_agreement"] * 5

    def test_mixed_boundary_and_interior_are_finite(self):
        probs = np.array([[0.0, 1.0, 1e-15, 0.5, 1.0 - 1e-15]] * 3)
        r = draw_decomposition(probs)
        for key in ("mixture_mean", "expected_draw_entropy_nats",
                    "between_draw_spread_variance_probability_squared",
                    "expected_within_draw_variance_probability_squared"):
            assert np.isfinite(r[key]).all(), key


class TestCorrectness:

    def test_interior_p_half_gives_log2(self):
        r = draw_decomposition(np.full((20, 3), 0.5))
        np.testing.assert_allclose(r["expected_draw_entropy_nats"], np.log(2.0), rtol=1e-15)

    def test_spread_is_exactly_zero_when_draws_agree(self):
        r = draw_decomposition(np.full((10, 5), 0.3))
        assert np.array_equal(r["between_draw_spread_variance_probability_squared"], np.zeros(5))
        assert r["disagreement_nats"] == [0.0] * 5

    def test_mean_matches_the_arithmetic_mean(self):
        probs = np.random.default_rng(42).uniform(0.0, 1.0, size=(15, 7))
        np.testing.assert_allclose(draw_decomposition(probs)["mixture_mean"], probs.mean(axis=0), rtol=1e-15)

    def test_output_shapes_and_units(self):
        r = draw_decomposition(np.full((10, 8), 0.4))
        assert r["mixture_mean"].shape == (8,)
        assert r["expected_draw_entropy_nats"].shape == (8,)
        assert r["between_draw_spread_variance_probability_squared"].shape == (8,)
        assert r["ddof"] == 0 and r["schema"] == "gvc.uncertainty.schema-2"
        assert "total" not in " ".join(r), "a variance and an entropy are never summed"

    def test_both_identities_close_in_their_own_units(self):
        rng = np.random.default_rng(7)
        probs = rng.uniform(0.02, 0.98, size=(9, 200))
        r = draw_decomposition(probs)
        q = r["mixture_mean"]
        np.testing.assert_allclose(
            q * (1 - q),
            r["expected_within_draw_variance_probability_squared"]
            + r["between_draw_spread_variance_probability_squared"], rtol=0, atol=1e-15)
        h = np.asarray(r["mixture_predictive_entropy_nats"])
        np.testing.assert_allclose(h, r["expected_draw_entropy_nats"] + np.asarray(r["disagreement_nats"]),
                                   rtol=0, atol=1e-14)


class TestRefusals:

    @pytest.mark.parametrize("bad", [np.array([[0.2, 1.5], [0.2, 0.3]]), np.array([[-1e-300, 0.2], [0.1, 0.2]]),
                                     np.array([[np.nan, 0.2], [0.1, 0.2]])])
    def test_values_outside_the_unit_interval_are_refused_never_clipped(self, bad):
        with pytest.raises(ContractError):
            draw_decomposition(bad)

    def test_one_draw_is_not_a_mixture(self):
        with pytest.raises(ContractError):
            draw_decomposition(np.full((1, 4), 0.3))

    def test_integer_input_is_refused(self):
        with pytest.raises(ContractError):
            draw_decomposition(np.zeros((3, 2), dtype=np.int64))
