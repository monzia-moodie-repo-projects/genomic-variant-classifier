"""Tests for conformal nonconformity scores (LAC / APS / RAPS)."""
import numpy as np
import pytest

from genomic_variant_classifier.conformal import scores as SC


def _probs(n, K, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, K))
    P = np.exp(logits)
    return P / P.sum(1, keepdims=True)


def test_lac_true_is_one_minus_prob():
    P = _probs(50, 3, 1)
    y = np.random.default_rng(2).integers(0, 3, 50)
    s = SC.lac_scores_true(P, y)
    assert np.allclose(s, 1 - P[np.arange(50), y])


def test_lac_all_shape_and_range():
    P = _probs(40, 4, 3)
    S = SC.lac_scores_all(P)
    assert S.shape == (40, 4)
    assert np.all(S >= 0) and np.all(S <= 1)


def test_aps_true_within_zero_one():
    P = _probs(60, 5, 4)
    y = np.random.default_rng(5).integers(0, 5, 60)
    s = SC.aps_scores_true(P, y, u=None, randomize=False)
    assert np.all(s >= 0) and np.all(s <= 1 + 1e-9)


def test_aps_all_monotone_in_prob_order():
    # For a single row, APS cumulative score should be ordered by descending prob inclusion.
    P = np.array([[0.5, 0.3, 0.2]])
    S = SC.aps_scores_all(P, u=None, randomize=False)
    # the most probable class has the smallest cumulative-inclusion score
    assert S[0, 0] <= S[0, 1] <= S[0, 2]


def test_randomized_aps_smaller_than_nonrandomized_on_average():
    P = _probs(2000, 4, 7)
    rng = np.random.default_rng(8)
    u = rng.uniform(size=2000)
    S_rand = SC.aps_scores_all(P, u=u, randomize=True)
    S_det = SC.aps_scores_all(P, u=None, randomize=False)
    # randomized scores are <= deterministic (they subtract a uniform fraction of the last class)
    assert S_rand.mean() <= S_det.mean()
