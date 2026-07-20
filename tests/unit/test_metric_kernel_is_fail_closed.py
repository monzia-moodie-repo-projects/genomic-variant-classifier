"""The metric kernel refuses what it cannot measure -- 2026-07-20.

Every test here pins a defect confirmed by reading metrics.py on 2026-07-20, raised by an
independent audit of the metric stack. Each names the behaviour it replaces, because a
regression test that does not say what it is defending is a test nobody dares delete.
"""

from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.metrics import (
    CalibrationFit,
    CleanArrays,
    MISSING_STRATUM,
    auprc,
    auprc_gain,
    auroc,
    bootstrap_ci,
    brier_score,
    calibration_slope_intercept,
    clean_arrays,
    cluster_bootstrap_ci,
    evaluate,
    is_probability,
    log_loss,
    no_skill_auprc,
    stratified_evaluate,
)


# --------------------------------------------------------------- defect A --

def test_score_and_probability_are_cleaned_on_one_joint_mask():
    """The defect: separate masks returned equal-length arrays describing different rows."""
    y = [0, 1, 0, 1, 0, 1]
    score = [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]   # row 0 unusable
    prob = [0.1, 0.2, np.nan, 0.4, 0.5, 0.6]    # row 2 unusable
    c = clean_arrays(y, score, prob)
    assert c.n == 4
    assert c.score.tolist() == [1.0, 3.0, 4.0, 5.0]
    assert c.probability.tolist() == pytest.approx([0.2, 0.4, 0.5, 0.6])
    assert c.y.tolist() == [1, 1, 0, 1]


def test_the_old_two_mask_approach_would_have_misaligned_by_construction():
    """Same lengths, different rows -- which is why nothing downstream noticed."""
    y = np.array([0, 1, 0, 1, 0, 1])
    score = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0])
    prob = np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6])
    old_score = score[np.isfinite(y) & np.isfinite(score)]
    old_prob = prob[np.isfinite(y) & np.isfinite(prob)]
    assert len(old_score) == len(old_prob) == 5      # EQUAL LENGTH -- nothing complained
    # ...but they describe DIFFERENT ROWS. old_score dropped row 0 and kept row 2;
    # old_prob kept row 0 and dropped row 2. Position i in one is not position i in the other.
    assert old_score[0] == 1.0 and old_prob[0] == pytest.approx(0.1)
    assert 2.0 in old_score and np.nan not in old_prob.tolist()
    # The joint cleaner keeps only rows usable in BOTH, so position i means one row.
    c = clean_arrays(y, score, prob)
    assert c.n == 4 and c.mask.tolist() == [False, True, False, True, True, True]


def test_evaluate_reports_how_many_rows_it_dropped():
    y = [0, 1] * 50
    score = [0.2, 0.8] * 50
    score[0] = np.nan
    out = evaluate(y, score)
    assert out["n_input"] == 100
    assert out["n"] == 99
    assert out["n_dropped"] == 1
    assert out["dropped_fraction"] == pytest.approx(0.01)


def test_mismatched_lengths_raise_rather_than_broadcast():
    with pytest.raises(ValueError, match="different shapes"):
        clean_arrays([0, 1, 0], [0.1, 0.2], [0.1, 0.2, 0.3])


# --------------------------------------------------------------- defect B --

@pytest.mark.parametrize("bad", [0.9, 1.2, 2.0, -1.0, 7.0])
def test_non_binary_labels_raise_instead_of_being_truncated(bad):
    """astype(int) turned 0.9 into 0 and left 2.0 as 2. Silently."""
    with pytest.raises(ValueError, match="must be exactly 0 or 1"):
        clean_arrays([0, 1, bad], [0.1, 0.2, 0.3])


def test_the_error_names_the_offending_values():
    with pytest.raises(ValueError) as exc:
        clean_arrays([0, 1, 2.0, 3.0], [0.1, 0.2, 0.3, 0.4])
    assert "2.0" in str(exc.value) and "3.0" in str(exc.value)


def test_a_coerced_label_corrupts_auroc_in_two_distinct_ways():
    """The audit understated this, and the first witness I chose was the wrong one.

    Two different corruptions, and they need different witnesses:

      [0, 1, 2]  ->  y.sum() == 3 == y.size, so `_degenerate` fires SPURIOUSLY and the old
                     code returned NaN by accident. Wrong answer, right-looking reason.
      [0, 1, 3]  ->  y.sum() == 4 != y.size, so `_degenerate` stays quiet, and
                     (1 - y).sum() == -1 makes the denominator n_pos * n_neg NEGATIVE.
                     A SIGNED AUROC, silently.
    """
    spurious = np.array([0, 1, 2])
    assert spurious.sum() == spurious.size          # degeneracy fires by accident
    assert int((1 - spurious).sum()) == 0

    signed = np.array([0, 1, 3])
    assert signed.sum() != signed.size              # degeneracy does NOT fire
    assert int((1 - signed).sum()) == -1            # the denominator goes negative

    for bad in (spurious, signed):
        with pytest.raises(ValueError, match="must be exactly 0 or 1"):
            auroc(bad, [0.1, 0.2, 0.3])


def test_boolean_labels_are_accepted():
    out = evaluate(np.array([True, False] * 50), [0.9, 0.1] * 50)
    assert out["n"] == 100
    assert out["n_pos"] == 50


# --------------------------------------------------------------- defect D --

def test_an_empty_probability_vector_is_not_a_valid_probability():
    assert is_probability([]) is False
    assert is_probability([np.nan, np.nan]) is False


def test_calibration_valid_is_false_when_nothing_finite_remains():
    assert np.isnan(brier_score([0, 1], [np.nan, np.nan]))
    assert np.isnan(log_loss([0, 1], [np.nan, np.nan]))


def test_a_real_probability_vector_is_still_valid():
    assert is_probability([0.0, 0.5, 1.0]) is True


# --------------------------------------------------------------- defect E --

def test_calibration_reports_convergence():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 500)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.1, 500), 0.01, 0.99)
    fit = calibration_slope_intercept(y, p)
    assert isinstance(fit, CalibrationFit)
    assert fit.converged is True
    assert fit.iterations >= 1
    assert 0.0 <= fit.clipped_fraction <= 1.0


def test_calibration_fit_still_unpacks_as_slope_intercept():
    """Every existing caller does `slope, intercept = ...`. It must keep working."""
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.4, 400)
    p = np.clip(0.4 + 0.3 * y + rng.normal(0, 0.1, 400), 0.01, 0.99)
    slope, intercept = calibration_slope_intercept(y, p)
    assert np.isfinite(slope) and np.isfinite(intercept)


def test_a_non_probability_input_returns_an_unconverged_nan_fit():
    fit = calibration_slope_intercept([0, 1] * 10, [-3.0, 4.5] * 10)
    assert fit.converged is False
    assert np.isnan(fit.slope) and np.isnan(fit.intercept)


def test_a_single_class_returns_an_unconverged_nan_fit():
    fit = calibration_slope_intercept([1] * 20, [0.6] * 20)
    assert fit.converged is False


# --------------------------------------------------------------- defect F --

def test_rows_with_a_missing_group_get_their_own_stratum():
    y = ([0, 1] * 40)
    score = ([0.2, 0.8] * 40)
    groups = (["a"] * 40) + ([None] * 40)
    df = stratified_evaluate(y, score, groups, min_n=5, min_pos=1, min_neg=1)
    assert MISSING_STRATUM in df.index
    assert int(df.loc[MISSING_STRATUM, "n_input"]) == 40


def test_the_strata_partition_the_cohort():
    y = ([0, 1] * 40)
    score = ([0.2, 0.8] * 40)
    groups = (["a"] * 30) + ([None] * 20) + (["b"] * 30)
    df = stratified_evaluate(y, score, groups, min_n=5, min_pos=1, min_neg=1)
    total = int(df.drop(index="ALL")["n_input"].astype(int).sum())
    assert total == 80


# --------------------------------------------------------------- defect G --

def test_a_stratum_with_almost_no_positives_is_reported_insufficient():
    y = [0] * 999 + [1]
    score = list(np.linspace(0, 1, 1000))
    groups = ["g"] * 1000
    df = stratified_evaluate(y, score, groups, min_n=30, min_pos=10, min_neg=10)
    assert df.loc["g", "status"].startswith("insufficient")
    assert "pos<10" in df.loc["g", "status"]
    assert np.isnan(df.loc["g", "auprc"])


def test_an_insufficient_stratum_is_reported_never_dropped():
    y = [0, 1] * 50
    score = [0.2, 0.8] * 50
    groups = (["big"] * 90) + (["tiny"] * 10)
    df = stratified_evaluate(y, score, groups, min_n=30, min_pos=5, min_neg=5)
    assert "tiny" in df.index
    assert df.loc["tiny", "status"].startswith("insufficient")


def test_the_status_names_every_floor_that_failed():
    y = [0, 1]
    score = [0.2, 0.8]
    groups = ["t", "t"]
    df = stratified_evaluate(y, score, groups, min_n=30, min_pos=10, min_neg=10)
    status = df.loc["t", "status"]
    assert "n<30" in status and "pos<10" in status and "neg<10" in status


# ------------------------------------------------------- cluster bootstrap --

def _genes_carry_the_signal(seed: int = 0):
    """A cohort where DISCRIMINATION varies by gene, not merely the score's level.

    My first fixture added a gene-level shift to the SCORE while y stayed independent of
    gene. Resampling genes then swapped in noise carrying no label information, so the AUROC
    distribution barely moved and the test failed -- correctly. The clustering has to be in
    the RELATIONSHIP being measured, which is what happens biologically: some genes are
    well-characterised and separate cleanly, others are not.

    Here six of thirty genes have INVERTED discrimination. Which genes land in a resample
    moves AUROC a great deal; which ROWS land in one does not, because the proportion of
    inverted rows stays near 20% either way. Measured design effect on this fixture: ~3.2x.
    """
    rng = np.random.default_rng(seed)
    genes, y, s = [], [], []
    for g in range(30):
        yy = rng.binomial(1, 0.5, 20)
        ss = (-yy if g % 5 == 0 else yy) + rng.normal(0, 0.1, 20)
        genes += [f"G{g}"] * 20
        y += yy.tolist()
        s += ss.tolist()
    return genes, y, s


def test_cluster_bootstrap_returns_a_wider_interval_when_genes_carry_signal():
    """The whole point: independent resampling understates variance under clustering."""
    genes, y, s = _genes_carry_the_signal(0)
    n_lo, n_hi = bootstrap_ci(auroc, y, s, n_boot=400, seed=0)
    c_lo, c_hi, de = cluster_bootstrap_ci(auroc, y, s, genes, n_boot=400, seed=0,
                                          return_design_effect=True)
    assert (c_hi - c_lo) > (n_hi - n_lo), (
        f"cluster width {c_hi - c_lo:.4f} should exceed row width {n_hi - n_lo:.4f}")
    assert de > 1.5


def test_the_naive_interval_is_the_one_that_is_wrong():
    """Stated as a direction, not a magnitude: the old estimator was ANTI-conservative."""
    genes, y, s = _genes_carry_the_signal(1)
    n_lo, n_hi = bootstrap_ci(auroc, y, s, n_boot=400, seed=2)
    c_lo, c_hi = cluster_bootstrap_ci(auroc, y, s, genes, n_boot=400, seed=2)
    assert c_lo < n_lo and c_hi > n_hi


def test_the_design_effect_quantifies_the_understatement():
    """The number that lets a historical interval be re-read rather than merely replaced."""
    genes, y, s = _genes_carry_the_signal(3)
    lo, hi, de = cluster_bootstrap_ci(auroc, y, s, genes, n_boot=400, seed=3,
                                      return_design_effect=True)
    assert np.isfinite(de) and de > 1.0
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi


def test_design_effect_is_near_one_when_clusters_carry_nothing():
    """A control. If genes are uninformative, the correction should be ~neutral."""
    rng = np.random.default_rng(21)
    y = rng.binomial(1, 0.4, 600)
    s = y + rng.normal(0, 0.8, 600)
    genes = [f"G{i % 30}" for i in range(600)]     # gene assignment is arbitrary
    _, _, de = cluster_bootstrap_ci(auroc, y, s, genes, n_boot=400, seed=4,
                                    return_design_effect=True)
    assert 0.5 < de < 2.0


def test_cluster_bootstrap_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        cluster_bootstrap_ci(auroc, [0, 1, 0], [0.1, 0.2, 0.3], ["a", "b"])


def test_two_stage_bootstrap_runs_and_returns_finite_bounds():
    rng = np.random.default_rng(5)
    genes = sum(([f"G{g}"] * 20 for g in range(20)), [])
    y = rng.binomial(1, 0.35, 400).tolist()
    s = rng.normal(0, 1, 400).tolist()
    lo, hi = cluster_bootstrap_ci(auroc, y, s, genes, n_boot=100, seed=1, two_stage=True)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


# ------------------------------------------------------------ new metrics --

def test_auprc_gain_is_the_absolute_gain_not_the_ratio():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.1, 500)
    s = y * 0.6 + rng.normal(0, 0.4, 500)
    ap = auprc(y, s)
    base = no_skill_auprc(y)
    assert auprc_gain(y, s) == pytest.approx(ap - base)


def test_log_loss_punishes_confident_errors_harder_than_brier():
    y = [1, 1, 1, 1]
    timid = [0.4] * 4
    confident_and_wrong = [0.01] * 4
    assert log_loss(y, confident_and_wrong) > log_loss(y, timid)
    ratio_ll = log_loss(y, confident_and_wrong) / log_loss(y, timid)
    ratio_br = brier_score(y, confident_and_wrong) / brier_score(y, timid)
    assert ratio_ll > ratio_br


def test_log_loss_is_nan_for_a_non_probability():
    assert np.isnan(log_loss([0, 1] * 10, [-2.0, 3.0] * 10))


# --------------------------------------------------- preserved behaviours --

def test_auroc_and_auprc_still_match_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.25, 800)
    s = y * 0.7 + rng.normal(0, 0.5, 800)
    assert auroc(y, s) == pytest.approx(sklearn_metrics.roc_auc_score(y, s), abs=1e-12)
    assert auprc(y, s) == pytest.approx(
        sklearn_metrics.average_precision_score(y, s), abs=1e-12)


def test_a_single_class_still_returns_nan_not_a_number():
    assert np.isnan(auroc([1] * 20, list(range(20))))
    assert np.isnan(auprc([0] * 20, list(range(20))))


def test_evaluate_still_returns_every_original_key():
    out = evaluate([0, 1] * 50, [0.2, 0.8] * 50)
    for key in ("n", "n_pos", "pos_rate", "auroc", "auprc", "auprc_no_skill",
                "auprc_lift", "brier", "ece", "cal_slope", "cal_intercept",
                "calibration_valid"):
        assert key in out, key


def test_stratified_evaluate_still_has_an_all_row():
    y = [0, 1] * 50
    df = stratified_evaluate(y, [0.2, 0.8] * 50, ["a"] * 50 + ["b"] * 50,
                             min_n=5, min_pos=5, min_neg=5)
    assert "ALL" in df.index
    assert int(df.loc["ALL", "n"]) == 100


def test_clean_arrays_is_a_frozen_dataclass():
    c = clean_arrays([0, 1], [0.1, 0.9])
    assert isinstance(c, CleanArrays)
    with pytest.raises(Exception):
        c.y = np.array([1, 0])
