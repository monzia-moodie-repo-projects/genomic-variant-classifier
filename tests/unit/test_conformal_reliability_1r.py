"""Reliability set 1R (2026-11-24): family-weighted coverage, exact predicted-class deferral,
truth-class routing, the Clopper-Pearson correction, and four silent coercions removed.

WHAT EACH GROUP ASSERTS
-----------------------
* family_class_coverage: the family-weighted and variant-weighted estimands against hand
  computations; the three support states; refusal of unknown families, probabilities as sets,
  non-integral labels; a bootstrap that is reproducible, bounded, and withheld below support.
* predicted_class_deferral: equality with an independent brute-force oracle on random inputs, the
  measured subtraction collapse that the two 2026-11-21 research packages fell into, exact quotas.
* truth_class_routing: every denominator against a hand count; an empty class is not zero risk.
* Clopper-Pearson: its exact coverage recomputed by an independent math.comb sum (at least 1 - delta,
  and strictly above it somewhere, so "conservative" is the true word); a graded risk refused.
* calibrate: missing gene identities refused; dropped rows counted instead of vanishing.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.conformal import calibrate as CAL
from genomic_variant_classifier.conformal import coverage as C
from genomic_variant_classifier.conformal import risk_control as RC


# --------------------------------------------------------------------------- #
# family_class_coverage
# --------------------------------------------------------------------------- #
def _sets_from_hits(hit: np.ndarray, y: np.ndarray, K: int = 2) -> np.ndarray:
    sets = np.zeros((len(y), K), dtype=bool)
    sets[np.arange(len(y)), y] = hit
    sets[np.arange(len(y)), (y + 1) % K] = ~hit
    return sets


def test_family_and_variant_weightings_differ_and_match_hand_computation():
    y = np.ones(12, dtype=int)
    fam = ["A"] * 10 + ["B", "C"]
    hit = np.array([True] * 8 + [False] * 2 + [True, False])
    r = C.family_class_coverage(_sets_from_hits(hit, y), y, fam, 1, min_families=2, n_boot=200)
    assert r.status == C.FAMILY_SUPPORT_ESTIMATED
    assert (r.n_rows, r.n_families) == (12, 3)
    assert r.variant_weighted == pytest.approx(9 / 12)
    assert r.family_weighted == pytest.approx((0.8 + 1.0 + 0.0) / 3)
    assert (r.family_size_min, r.family_size_median, r.family_size_max) == (1, 1.0, 10)
    assert r.interpretation == "empirical_diagnostic_not_finite_sample_guarantee"


def test_the_all_and_any_group_statistics_are_not_the_family_weighted_one():
    """The legacy statistics keep their meaning; the new estimand is a different number."""
    rng = np.random.default_rng(3)
    y = np.ones(400, dtype=int)
    fam = np.array([f"F{i // 10}" for i in range(400)], dtype=object)
    hit = rng.random(400) < 0.9
    sets = _sets_from_hits(hit, y)
    r = C.family_class_coverage(sets, y, fam, 1, min_families=20, n_boot=200)
    g_all = C.group_coverage_disjoint(sets, y, fam, "all")
    g_any = C.group_coverage_disjoint(sets, y, fam, "any")
    per_family = pd.Series(hit).groupby(fam).mean().mean()
    assert r.family_weighted == pytest.approx(per_family)
    assert g_all < r.family_weighted < g_any


def test_eligibility_is_class_specific():
    sets = np.array([[True, False], [True, False], [False, True]])
    r = C.family_class_coverage(sets, np.array([0, 0, 1]), ["A", "A", "B"], 1, min_families=2,
                                n_boot=100)
    assert (r.n_rows, r.n_families, r.family_weighted) == (1, 1, 1.0)
    assert r.status == C.FAMILY_SUPPORT_INSUFFICIENT and r.interval is None


def test_a_class_absent_is_a_state_not_a_zero():
    sets = np.array([[True, False], [True, False]])
    r = C.family_class_coverage(sets, np.array([0, 0]), ["A", "B"], 1, min_families=2, n_boot=100)
    assert r.status == C.FAMILY_SUPPORT_CLASS_ABSENT
    assert r.family_weighted is None and r.variant_weighted is None and r.interval is None


def test_two_families_get_a_point_estimate_but_no_interval():
    """Measured on the 2026-11-21 package: two families gave a '95% interval' of [0, 1]."""
    sets = np.array([[False, True], [True, False]])
    r = C.family_class_coverage(sets, np.array([1, 1]), ["A", "B"], 1, n_boot=1000)
    assert r.status == C.FAMILY_SUPPORT_INSUFFICIENT
    assert r.family_weighted == 0.5 and r.interval is None


def test_the_bootstrap_is_reproducible_bounded_and_seed_sensitive():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    fam = np.array([f"F{i % 97}" for i in range(2000)], dtype=object)
    sets = _sets_from_hits(rng.random(2000) < 0.85, y)
    a = C.family_class_coverage(sets, y, fam, 1, n_boot=500, seed=7)
    b = C.family_class_coverage(sets, y, fam, 1, n_boot=500, seed=7)
    c = C.family_class_coverage(sets, y, fam, 1, n_boot=500, seed=8)
    assert a.interval == b.interval and a.interval != c.interval
    lo, hi = a.interval
    assert 0.0 <= lo <= a.family_weighted <= hi <= 1.0


def test_the_chunked_bootstrap_equals_one_draw():
    """The chunking bounds memory; it must not change a single resampled mean."""
    rng = np.random.default_rng(1)
    y = np.ones(300, dtype=int)
    fam = np.array([f"F{i % 30}" for i in range(300)], dtype=object)
    hit = rng.random(300) < 0.8
    r = C.family_class_coverage(_sets_from_hits(hit, y), y, fam, 1, n_boot=300, seed=11)
    per_family = pd.Series(hit.astype(float)).groupby(fam).mean()
    per_family = per_family.sort_index().to_numpy()
    idx = np.random.default_rng(11).integers(0, 30, size=(300, 30))
    lo, hi = np.quantile(per_family[idx].mean(axis=1), [0.025, 0.975])
    assert r.interval == (pytest.approx(lo), pytest.approx(hi))


def test_set_informativeness_is_reported():
    sets = np.array([[True, True], [False, True], [True, False], [False, False]])
    y = np.array([1, 1, 1, 1])
    r = C.family_class_coverage(sets, y, ["A", "B", "C", "D"], 1, min_families=2, n_boot=100)
    assert r.mean_set_size == pytest.approx(1.0)
    assert r.full_set_rate == pytest.approx(0.25)
    assert r.empty_set_rate == pytest.approx(0.25)
    assert r.singleton_rate == pytest.approx(0.5)
    assert r.singleton_error_rate == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [None, "", "  ", float("nan"), 7])
def test_an_unknown_family_is_refused(bad):
    sets = np.array([[False, True], [False, True]])
    with pytest.raises(ValueError, match="family identity"):
        C.family_class_coverage(sets, np.array([1, 1]), ["A", bad], 1, min_families=2, n_boot=100)


@pytest.mark.parametrize("kwargs", [{"min_families": 1}, {"n_boot": 99}, {"interval_level": 1.0},
                                    {"min_families": True}])
def test_impossible_settings_are_refused(kwargs):
    sets = np.array([[False, True], [False, True]])
    with pytest.raises(ValueError):
        C.family_class_coverage(sets, np.array([1, 1]), ["A", "B"], 1, **kwargs)


@pytest.mark.parametrize("label", [2, -1, True, 1.0])
def test_an_impossible_class_label_is_refused(label):
    sets = np.array([[False, True], [False, True]])
    with pytest.raises(ValueError, match="class_label"):
        C.family_class_coverage(sets, np.array([1, 1]), ["A", "B"], label, min_families=2,
                                n_boot=100)


# --------------------------------------------------------------------------- #
# COVERAGE-SILENT-COERCION-1
# --------------------------------------------------------------------------- #
def test_a_non_integral_label_is_refused_where_it_used_to_be_truncated():
    """Measured on main 2b55f06f: label 0.7 became class 0 and was reported covered (1.0)."""
    sets = np.array([[True, False], [False, True]])
    with pytest.raises(ValueError, match="integer class"):
        C.marginal_coverage(sets, np.array([0.7, 1.0]))


def test_probabilities_are_not_prediction_sets():
    with pytest.raises(ValueError, match="not a prediction set"):
        C.marginal_coverage(np.array([[0.3, 0.0], [0.0, 0.2]]), np.array([0, 1]))


def test_integral_floats_and_zero_one_integers_are_still_accepted():
    sets = np.array([[1, 0], [0, 1]], dtype=np.int64)
    assert C.marginal_coverage(sets, np.array([0.0, 1.0])) == 1.0


@pytest.mark.parametrize("y", [np.array([0.0, np.nan]), np.array([0.0, np.inf]),
                               np.array(["0", "1"])])
def test_non_finite_and_text_labels_are_refused(y):
    with pytest.raises(ValueError):
        C.marginal_coverage(np.array([[True, False], [False, True]]), y)


def test_empty_input_is_refused():
    with pytest.raises(ValueError):
        C.marginal_coverage(np.zeros((0, 2), dtype=bool), np.zeros(0, dtype=int))


# --------------------------------------------------------------------------- #
# predicted_class_deferral
# --------------------------------------------------------------------------- #
def _oracle(scores, t, fn, fp):
    """Independent brute force: rank by EXACT distance with Fraction, ties by index."""
    rows = list(range(len(scores)))
    neg = [i for i in rows if scores[i] < t]
    pos = [i for i in rows if scores[i] >= t]
    dist = {i: abs(Fraction(scores[i]) - Fraction(t)) for i in rows}
    out = [False] * len(scores)
    for group, frac in ((neg, fn), (pos, fp)):
        k = math.floor(len(group) * Fraction(str(frac)))
        for i in sorted(group, key=lambda j: (dist[j], j))[:k]:
            out[i] = True
    return out


@pytest.mark.parametrize("seed", range(25))
def test_deferral_equals_an_exact_oracle(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    pool = np.concatenate([rng.random(n), rng.choice([0.0, 1.0, 0.5, 1e-20, 2e-20, 3e-20], n)])
    scores = rng.choice(pool, n)
    t = float(rng.choice([0.05, 0.25, 0.5, 0.71, 0.9]))
    fn, fp = float(rng.choice([0.0, 0.1, 0.29, 0.5, 1.0])), float(rng.choice([0.0, 0.3, 0.7, 1.0]))
    got = RC.predicted_class_deferral(scores, t, negative_fraction=fn, positive_fraction=fp)
    assert got.tolist() == _oracle(scores.tolist(), t, fn, fp)


def test_the_measured_subtraction_collapse_is_not_reproduced():
    """|p - 0.5| is 0.5 for all three; the row CLOSEST to the threshold is index 2."""
    p = np.array([1e-20, 2e-20, 3e-20, 0.9])
    assert abs(p[0] - 0.5) == abs(p[1] - 0.5) == abs(p[2] - 0.5)
    got = RC.predicted_class_deferral(p, 0.5, negative_fraction=Fraction(1, 3), positive_fraction=0)
    assert np.flatnonzero(got).tolist() == [2]


def test_quotas_are_exact_on_the_declared_decimal():
    """100 * 0.29 is 28.999999999999996 in floating point; the declared quota is 29."""
    assert 100 * 0.29 < 29
    got = RC.predicted_class_deferral(np.linspace(0.0, 0.4, 100), 0.5, negative_fraction=0.29,
                                      positive_fraction=0.0)
    assert int(got.sum()) == 29


def test_deferral_is_within_predicted_class_and_ties_go_to_the_lower_index():
    p = np.array([0.4, 0.4, 0.6, 0.6, 0.1, 0.95])
    got = RC.predicted_class_deferral(p, 0.5, negative_fraction=Fraction(1, 3),
                                      positive_fraction=Fraction(1, 3))
    assert np.flatnonzero(got).tolist() == [0, 2]


@pytest.mark.parametrize("kwargs", [
    {"scores": [0.2, float("nan")]}, {"scores": [1.2]}, {"scores": []}, {"scores": [True, False]},
    {"threshold": 0.0}, {"threshold": 1.0}, {"threshold": True},
    {"negative_fraction": 1.5}, {"negative_fraction": -0.1}, {"negative_fraction": True},
    {"positive_fraction": float("nan")},
])
def test_impossible_deferral_inputs_are_refused(kwargs):
    args = {"scores": [0.2, 0.8], "threshold": 0.5, "negative_fraction": 0.5,
            "positive_fraction": 0.5}
    args.update(kwargs)
    with pytest.raises(ValueError):
        RC.predicted_class_deferral(np.asarray(args.pop("scores")), args.pop("threshold"), **args)


# --------------------------------------------------------------------------- #
# truth_class_routing
# --------------------------------------------------------------------------- #
def test_routing_denominators_match_a_hand_count():
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.3, 0.7])
    labels = np.array([0, 0, 1, 1, 1, 0])
    deferred = np.array([False, False, False, True, False, True])
    r = RC.truth_class_routing(labels, scores, deferred, 0.5)
    benign, path = r["classes"][0], r["classes"][1]
    assert (benign["n_total"], benign["n_accepted"], benign["n_wrong_automatic"]) == (3, 2, 1)
    assert benign["wrong_automatic_per_class"] == pytest.approx(1 / 3)
    assert benign["wrong_rate_among_accepted"] == pytest.approx(1 / 2)
    assert (path["n_total"], path["n_accepted"], path["n_wrong_automatic"]) == (3, 2, 1)
    assert path["acceptance_rate"] == pytest.approx(2 / 3)
    assert path["wrong_automatic_per_class"] == pytest.approx(1 / 3)
    assert r["accepted_fraction"] == pytest.approx(4 / 6)
    assert r["interpretation"] == "descriptive_routing_not_risk_control"


def test_all_deferred_is_unavailable_not_zero():
    r = RC.truth_class_routing(np.array([0, 1]), np.array([0.1, 0.9]), np.array([True, True]), 0.5)
    assert r["classes"][1]["wrong_rate_among_accepted"] is None
    assert r["classes"][1]["wrong_automatic_per_class"] == 0.0


def test_a_missing_true_class_is_reported_as_unobserved():
    r = RC.truth_class_routing(np.array([0, 0]), np.array([0.1, 0.9]), np.array([False, False]), 0.5)
    assert r["classes"][1]["status"] == "no_observations"
    assert r["classes"][1]["acceptance_rate"] is None


@pytest.mark.parametrize("labels,deferred", [
    (np.array([0, 2]), np.array([False, False])),
    (np.array([0.0, 1.0]), np.array([False, False])),
    (np.array([0, 1]), np.array([0, 1])),
    (np.array([0]), np.array([False, False])),
])
def test_impossible_routing_inputs_are_refused(labels, deferred):
    with pytest.raises(ValueError):
        RC.truth_class_routing(labels, np.array([0.1, 0.9]), deferred, 0.5)


# --------------------------------------------------------------------------- #
# Clopper-Pearson
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [20, 50, 200])
def test_clopper_pearson_exact_coverage_is_conservative(n):
    """Exact coverage P_R(UB(K) >= R) by an independent math.comb sum: never below 1 - delta,
    and above it at every measured R -- the docstring's 'neither conservative' was false."""
    delta = 0.10
    ub = [RC.clopper_pearson_upper_bound(k, n, delta) for k in range(n + 1)]
    for R in (0.01, 0.05, 0.10, 0.20):
        cov = math.fsum(math.comb(n, k) * R ** k * (1 - R) ** (n - k)
                        for k in range(n + 1) if ub[k] >= R)
        assert cov >= 1 - delta - 1e-12
        assert cov > 1 - delta


def test_a_graded_risk_is_refused_by_clopper_pearson():
    """Measured on main 2b55f06f: r_hat 0.105 at n 10 (1.05 events) was bounded as 1 event."""
    with pytest.raises(RC.NonBinaryRiskError, match="not an integer"):
        RC.control_risk([0.0, 1.0], [0.105, 0.105], 10, 0.5, 0.1, bound="clopper_pearson")
    assert issubclass(RC.NonBinaryRiskError, ValueError)
    assert issubclass(RC.NonBinaryRiskError, RC.RiskControlError)


def test_a_count_in_floating_point_is_still_a_count():
    k = [int(v) for v in np.random.default_rng(4).integers(0, 400, 50)]
    risks = sorted((v / 400 for v in k), reverse=True)
    out = RC.control_risk(np.linspace(0, 1, 50), risks, 400, 0.9, 0.1, bound="clopper_pearson")
    assert out["upper_bounds"][0] == pytest.approx(
        RC.clopper_pearson_upper_bound(max(k), 400, 0.1))


def test_the_other_bounds_still_take_a_graded_risk():
    out = RC.control_risk([0.0, 1.0], [0.105, 0.105], 10, 0.9, 0.1, bound="hoeffding_bentkus")
    assert out["upper_bounds"][0] > 0.105


# --------------------------------------------------------------------------- #
# calibrate: CALIBRATE-NULL-GENE-POOL-1 and CALIBRATE-SILENT-ROW-DROP-1
# --------------------------------------------------------------------------- #
def _substrate(n=400, seed=5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(y * 0.8 + rng.normal(0, 0.1, n), 0, 1)
    genes = np.array([f"G{i % 40}" for i in range(n)], dtype=object)
    return pd.DataFrame({"selected_score": p, "label": y.astype(float), "gene_symbol": genes,
                         "consequence": "missense"})


@pytest.mark.parametrize("bad", [None, np.nan, "", "   "])
def test_a_missing_gene_is_refused_not_pooled(bad):
    df = _substrate()
    df["gene_symbol"] = df["gene_symbol"].astype(object)
    df.loc[:59, "gene_symbol"] = bad
    with pytest.raises(ValueError, match="no gene identity"):
        CAL._verify_frame(df, CAL.CalibrationConfig())


def test_dropped_rows_are_counted_in_the_result():
    df = _substrate()
    df.loc[60:69, "label"] = np.nan
    df.loc[70:74, "selected_score"] = np.nan
    cfg = CAL.CalibrationConfig()
    res = CAL._calibrate_frame(CAL._verify_frame(df, cfg), cfg, {"bound": False})
    assert res.rows == {"input": 400, "dropped_missing_label": 10,
                        "dropped_missing_score_only": 5, "usable": 385}
    assert res.n_total == 385


def test_a_frame_that_bypassed_the_gate_says_so():
    df = _substrate()
    cfg = CAL.CalibrationConfig()
    res = CAL._calibrate_frame(df, cfg, {"bound": False})
    assert res.rows is None
