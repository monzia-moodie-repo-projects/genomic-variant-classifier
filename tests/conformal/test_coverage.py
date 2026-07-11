"""Tests for coverage diagnostics."""
import numpy as np

from genomic_variant_classifier.conformal import coverage as C


def test_marginal_perfect_and_zero():
    n, K = 500, 3
    y = np.random.default_rng(0).integers(0, K, n)
    perfect = np.zeros((n, K), bool); perfect[np.arange(n), y] = True
    assert C.marginal_coverage(perfect, y) == 1.0
    wrong = np.zeros((n, K), bool); wrong[np.arange(n), (y + 1) % K] = True
    assert C.marginal_coverage(wrong, y) == 0.0


def test_per_class_detects_rare_undercoverage():
    n = 3000
    y = np.concatenate([np.zeros(2850, int), np.ones(150, int)])
    sets = np.zeros((n, 2), bool)
    r = np.random.default_rng(1)
    sets[np.arange(2850), 0] = r.random(2850) < 0.95
    sets[np.arange(2850, 3000), 1] = r.random(150) < 0.60
    pcc = C.per_class_coverage(sets, y)
    assert pcc[0] > 0.9 and pcc[1] < 0.7


def test_per_stratum_keeps_nan_rows():
    strata = np.array((["A"] * 400 + ["B"] * 400 + [np.nan] * 200), dtype=object)
    y = np.random.default_rng(2).integers(0, 2, 1000)
    sets = np.zeros((1000, 2), bool); sets[np.arange(1000), y] = True
    psc = C.per_stratum_coverage(sets, y, strata)
    assert psc["n"].sum() == 1000
    assert "__unknown__" in psc.index


def test_set_size_and_abstention():
    sets = np.array([[False, False, False], [True, True, True], [True, False, False]], bool)
    ab = C.abstention_rates(sets)
    assert np.isclose(ab["empty_rate"], 1 / 3)
    assert np.isclose(ab["full_rate"], 1 / 3)
    assert np.isclose(ab["singleton_rate"], 1 / 3)
    ss = C.set_size_summary(sets)
    assert ss["min"] == 0 and ss["max"] == 3


def test_report_marginal_ok_flag():
    n, K = 2000, 3
    rng = np.random.default_rng(3)
    y = rng.integers(0, K, n)
    sets = np.zeros((n, K), bool)
    cover = rng.random(n) < 0.92
    sets[np.arange(n), y] = cover
    sets[np.arange(n), (y + 1) % K] = ~cover
    rep = C.coverage_report(sets, y, alpha=0.1)
    assert rep["marginal_ok"] is True
