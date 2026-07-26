"""Tests for the gene-disjoint calibration carve in VariantEnsemble.

WHY THIS FILE EXISTS
====================
The post-hoc isotonic calibrator was fitted on genes the base models had trained
on, in every production run.

WHERE THE DEFECT LIVED. scripts/run_phase2_eval.py:590 -- the production entry
point, the one preflight_run17.emit_command() emits and every Run 14-17 launcher
uses -- calls fit() with no *_cal_ext argument. That takes the self-carve branch,
which was:

    idx_fit, idx_cal = _tts(idx, test_size=0.15, stratify=y_arr,
                            random_state=self.config.random_state)

`stratify=y_arr`, and no `groups=`. Measured 2026-07-21:

    cohort                     cal genes   also in fit   cal rows from trained genes
    500 genes                        319           319                      100.0 %
    ClinVar-like, 8,000 genes      7,864         7,856                      100.0 %

Twenty lines below, the inner cross-validation is carefully gene-disjoint via
GroupKFold, citing INCIDENT_2026-06-13 by name. So out-of-fold PREDICTIONS were
gene-disjoint while the fold the CALIBRATOR was fitted on was not -- an
inconsistency inside a single function. The calibrator learned the
score-to-probability map in the seen-gene regime; those probabilities were then
reported for unseen genes. It affects xgboost, lightgbm and random_forest (the
_RECALIBRATE set) and through them the stacking meta-learner.

WHAT WAS NOT THE DEFECT. scripts/train.py has a v2_conformal path that fits the
calibrator on the `tune` partition -- the selection set. That is also wrong, but
NO launcher passes --split-protocol, so it has never run. Chasing it would have
fixed a dormant path and left the live one intact. The entry point was checked
before the fix was designed, which is the only reason this file is about
variant_ensemble.py rather than train.py.

THE ROW-VERSUS-GROUP TRAP. GroupShuffleSplit interprets test_size as a
proportion of GROUPS, not rows, and ClinVar per-gene counts are heavy-tailed --
the same trap found in split_protocol_v2.group_shuffle on 2026-07-21, where a
row-based rescale crashed 3 of 12 seeds. The carve therefore VALIDATES the fold
it produced rather than assuming it is usable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split as _tts

from genomic_variant_classifier.models.variant_ensemble import (
    EnsembleConfig,
    VariantEnsemble,
)


def _cohort(n_genes: int, lo: int, hi: int, seed: int = 0):
    """Genes with variable variant counts, as a real cohort has."""
    rng = np.random.default_rng(seed)
    genes, labels = [], []
    for i in range(n_genes):
        k = int(rng.integers(lo, hi))
        genes += [f"G{i:05d}"] * k
        labels += list(rng.integers(0, 2, k))
    return np.array(genes), np.array(labels)


@pytest.fixture(scope="module")
def small():
    return _cohort(500, 3, 12)


@pytest.fixture(scope="module")
def heavy_tailed():
    """1,200 genes, counts spanning 1 to 300 -- the shape ClinVar actually has."""
    return _cohort(1200, 1, 300, seed=3)


def _carve(genes, y, mode="gene_disjoint", **cfg):
    ens = VariantEnsemble(EnsembleConfig(calibration_carve=mode, **cfg))
    idx = np.arange(len(y))
    g = None if genes is None else pd.Series(genes)
    return ens._carve_calibration_fold(idx, y, g)


# --------------------------------------------------------------------------- #
# 1. the defect, and its repair
# --------------------------------------------------------------------------- #
def test_the_legacy_carve_puts_every_calibration_gene_in_the_fit_fold(small):
    """CHARACTERISATION. Not an endorsement -- this documents exactly what the
    production runs did, so the repair cannot be quietly reverted."""
    genes, y = small
    i_fit, i_cal, how = _carve(genes, y, mode="legacy_stratified")
    shared = set(genes[i_fit]) & set(genes[i_cal])
    assert shared, "expected gene overlap under the legacy carve"
    assert np.isin(genes[i_cal], list(shared)).mean() == 1.0, (
        "every calibration row should come from a gene in the fit fold")
    assert how == "legacy_stratified:configured"


@pytest.mark.parametrize("fixture", ["small", "heavy_tailed"])
def test_the_gene_disjoint_carve_shares_no_gene(request, fixture):
    """THE REPAIR. Zero shared genes, so the isotonic calibrator never sees a
    gene the base models were fitted on."""
    genes, y = request.getfixturevalue(fixture)
    i_fit, i_cal, how = _carve(genes, y)
    assert not (set(genes[i_fit]) & set(genes[i_cal]))
    assert how.startswith("gene_disjoint:")


@pytest.mark.parametrize("fixture", ["small", "heavy_tailed"])
def test_no_calibration_row_comes_from_a_trained_gene(request, fixture):
    """The property that actually matters, stated in rows rather than genes:
    100.0 per cent before, 0.0 per cent after."""
    genes, y = request.getfixturevalue(fixture)
    _, i_cal, _ = _carve(genes, y)
    fit_genes = set(genes[np.setdiff1d(np.arange(len(y)), i_cal)])
    assert np.isin(genes[i_cal], list(fit_genes)).mean() == 0.0


@pytest.mark.parametrize("seed", [42, 7, 123, 2026])
def test_disjointness_holds_across_seeds(small, seed):
    genes, y = small
    ens = VariantEnsemble(EnsembleConfig(random_state=seed))
    i_fit, i_cal, _ = ens._carve_calibration_fold(np.arange(len(y)), y, pd.Series(genes))
    assert not (set(genes[i_fit]) & set(genes[i_cal]))


# --------------------------------------------------------------------------- #
# 2. the fold must be USABLE, not merely disjoint
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture", ["small", "heavy_tailed"])
def test_the_calibration_fold_holds_both_classes(request, fixture):
    """Isotonic regression on a single class is degenerate: it returns a
    constant and reports no error. Disjointness alone does not prevent that."""
    genes, y = request.getfixturevalue(fixture)
    _, i_cal, _ = _carve(genes, y)
    assert set(np.unique(y[i_cal]).tolist()) == {0, 1}


def test_the_realized_row_fraction_stays_near_the_target(heavy_tailed):
    """GroupShuffleSplit's test_size is a proportion of GROUPS. With
    heavy-tailed gene sizes the realized ROW fraction can wander, so the carve
    checks the fold it produced rather than trusting the request."""
    genes, y = heavy_tailed
    _, i_cal, _ = _carve(genes, y)
    frac = len(i_cal) / len(y)
    assert 0.05 < frac < 0.30, f"realized calibration row fraction {frac:.3f}"


def test_a_cohort_too_small_for_a_gene_disjoint_fold_fails_loud():
    """Six genes cannot yield a fold with both classes and enough rows. The
    carve must say so, not return a fold that cannot support the fit it is
    for."""
    genes, y = _cohort(6, 2, 4, seed=1)
    with pytest.raises(ValueError, match="no gene-disjoint carve"):
        _carve(genes, y)


def test_the_failure_message_carries_the_attempts():
    """A refusal without the evidence behind it sends the reader guessing."""
    genes, y = _cohort(6, 2, 4, seed=1)
    with pytest.raises(ValueError) as ei:
        _carve(genes, y)
    msg = str(ei.value)
    assert "Attempts" in msg and "seed" in msg
    assert "legacy_stratified" in msg, "the message should name the escape hatch"


def test_min_rows_is_honoured(small):
    """A deliberately huge floor must be refused rather than silently ignored."""
    genes, y = small
    with pytest.raises(ValueError, match="no gene-disjoint carve"):
        _carve(genes, y, calibration_min_rows=10 ** 7)


# --------------------------------------------------------------------------- #
# 3. no silent fallback, ever
# --------------------------------------------------------------------------- #
def test_absent_gene_labels_fall_back_but_say_so(small, caplog):
    """run_phase2_eval.py reads gene_symbol from splits/meta_train.parquet and
    passes None when it is absent. That must be visible, not inferred."""
    _, y = small
    with caplog.at_level("WARNING"):
        _, _, how = _carve(None, y)
    assert how == "legacy_stratified:no_gene_labels"
    assert "gene labels were NOT supplied" in caplog.text


def test_explicit_legacy_warns_about_what_it_costs(small, caplog):
    genes, y = small
    with caplog.at_level("WARNING"):
        _, _, how = _carve(genes, y, mode="legacy_stratified")
    assert how == "legacy_stratified:configured"
    assert "optimistic" in caplog.text


@pytest.mark.parametrize("mode,expected", [
    ("gene_disjoint", "gene_disjoint:"),
    ("legacy_stratified", "legacy_stratified:configured"),
])
def test_the_carve_reports_what_it_actually_did(small, mode, expected):
    genes, y = small
    _, _, how = _carve(genes, y, mode=mode)
    assert how.startswith(expected)


def test_an_unknown_mode_is_refused(small):
    genes, y = small
    with pytest.raises(ValueError, match="calibration_carve must be"):
        _carve(genes, y, mode="whatever_seems_fine")


# --------------------------------------------------------------------------- #
# 4. the legacy path is preserved EXACTLY
# --------------------------------------------------------------------------- #
def test_legacy_is_byte_for_byte_the_pre_repair_behaviour(small):
    """Historical runs must remain reproducible. This reconstructs the original
    call and demands identical indices, not merely a similar split."""
    genes, y = small
    idx = np.arange(len(y))
    got_fit, got_cal, _ = _carve(genes, y, mode="legacy_stratified")
    want_fit, want_cal = _tts(idx, test_size=0.15, stratify=y, random_state=42)
    assert np.array_equal(got_fit, want_fit)
    assert np.array_equal(got_cal, want_cal)


@pytest.mark.parametrize("seed", [42, 7, 2026])
def test_legacy_matches_the_original_at_every_seed(small, seed):
    genes, y = small
    idx = np.arange(len(y))
    ens = VariantEnsemble(EnsembleConfig(calibration_carve="legacy_stratified",
                                         random_state=seed))
    got_fit, got_cal, _ = ens._carve_calibration_fold(idx, y, pd.Series(genes))
    want_fit, want_cal = _tts(idx, test_size=0.15, stratify=y, random_state=seed)
    assert np.array_equal(got_fit, want_fit) and np.array_equal(got_cal, want_cal)


def test_the_default_is_gene_disjoint():
    """If this ever flips back to the legacy carve by default, the defect
    returns silently and every reported probability becomes optimistic again."""
    assert EnsembleConfig().calibration_carve == "gene_disjoint"


# --------------------------------------------------------------------------- #
# 5. determinism and coverage
# --------------------------------------------------------------------------- #
def test_the_carve_is_deterministic(small):
    genes, y = small
    outs = [_carve(genes, y)[2] for _ in range(3)]
    assert len(set(outs)) == 1


def test_the_two_folds_partition_the_rows_exactly(small):
    """Coverage: every row lands in exactly one fold. A row in neither is
    silently dropped from both fitting and calibration."""
    genes, y = small
    i_fit, i_cal, _ = _carve(genes, y)
    both = np.concatenate([i_fit, i_cal])
    assert len(both) == len(y)
    assert len(np.unique(both)) == len(y)


def test_folds_are_non_empty(small):
    genes, y = small
    i_fit, i_cal, _ = _carve(genes, y)
    assert len(i_fit) > 0 and len(i_cal) > 0


# --------------------------------------------------------------------------- #
# 6. the both-classes guard, pinned by a cohort that forces it
# --------------------------------------------------------------------------- #
# ADDED AFTER A SABOTAGE TEST FAILED TO FAIL. Removing `ok_classes` left all 28
# tests passing, because every cohort above happens to yield both classes in the
# first carve. A guard whose removal no test notices is not guarded -- the same
# finding made four times on 2026-07-21, here in my own tests.
#
# This cohort makes the label a property of the GENE: only 4 of 40 genes are
# pathogenic. A gene-level carve can then easily select 15 per cent of genes that
# are all benign, which is exactly the situation the guard exists for.
def _gene_determined_cohort():
    rng = np.random.default_rng(0)
    genes, labels = [], []
    for i in range(40):
        cls = 1 if i < 4 else 0
        k = int(rng.integers(8, 20))
        genes += [f"G{i:03d}"] * k
        labels += [cls] * k
    return np.array(genes), np.array(labels)


def test_a_single_class_carve_is_rejected_and_another_seed_tried():
    """MEASURED: on this cohort GroupShuffleSplit at random_state=42 returns an
    89-row calibration fold containing ONLY benign variants. Isotonic regression
    fitted on one class returns a constant and raises nothing, so the calibrator
    would silently become a constant map and every calibrated probability would
    be the class base rate."""
    genes, y = _gene_determined_cohort()
    idx = np.arange(len(y))
    from sklearn.model_selection import GroupShuffleSplit
    (_, c_), = GroupShuffleSplit(1, test_size=0.15, random_state=42).split(
        idx, y, groups=genes)
    assert set(np.unique(y[c_]).tolist()) == {0}, (
        "fixture no longer forces a single-class first carve; the guard would "
        "not be exercised and this test would stop testing anything")

    _, i_cal, how = _carve(genes, y, calibration_min_rows=20)
    assert set(np.unique(y[i_cal]).tolist()) == {0, 1}
    assert how != "gene_disjoint:seed=42", "the single-class carve was accepted"
    assert how == "gene_disjoint:seed=45"


def test_the_retry_still_returns_a_gene_disjoint_fold():
    """Retrying for both classes must not quietly abandon disjointness."""
    genes, y = _gene_determined_cohort()
    i_fit, i_cal, _ = _carve(genes, y, calibration_min_rows=20)
    assert not (set(genes[i_fit]) & set(genes[i_cal]))


# --------------------------------------------------------------------------- #
# 7. cohort SIZE -- the regression my own tests missed
# --------------------------------------------------------------------------- #
# ADDED 2026-07-21 AFTER SHIPPING A BREAK. The first version used a single
# absolute floor of 200 rows. It failed
# test_level2_leakfree_oof::test_fit_accepts_gene_symbol_and_runs on a 427-row,
# 30-gene cohort, where a 15 per cent carve of ANY kind yields about 64 rows.
# The floor refused gene-disjoint folds of 63-76 rows while the legacy
# stratified carve it replaced had been using a 62-row fold for the whole
# history of the project without complaint. Refusing what the predecessor
# accepted is a regression, not a stricter standard.
#
# Every cohort in sections 1-6 has 500 or more genes, so none of them could have
# caught it. The guard now asks "did the carve deliver roughly what was ASKED
# for?" rather than "is this fold small?", and these tests pin both ends.
def _small_cohort():
    """The shape of tests/unit/test_level2_leakfree_oof.py::_cohort."""
    rng = np.random.default_rng(0)
    genes, labels = [], []
    for gi in range(30):
        n = int(rng.integers(8, 20))
        prob = rng.uniform(0.2, 0.8)
        for _ in range(n):
            genes.append(f"G{gi}")
            labels.append(int(rng.random() < prob))
    return np.array(genes), np.array(labels)


def _dominant_gene_cohort():
    """One gene holding 9,000 of 11,000 rows -- heavy-tailed to the extreme."""
    genes = np.array(["BIG"] * 9000 + [f"S{i}" for i in range(400) for _ in range(5)])
    labels = np.random.default_rng(1).integers(0, 2, len(genes))
    return genes, labels


def test_a_small_cohort_is_not_refused(caplog):
    """THE REGRESSION. 427 rows across 30 genes must produce a usable fold."""
    genes, y = _small_cohort()
    i_fit, i_cal, how = _carve(genes, y)
    assert how.startswith("gene_disjoint:")
    assert not (set(genes[i_fit]) & set(genes[i_cal]))
    assert set(np.unique(y[i_cal]).tolist()) == {0, 1}


def test_a_small_fold_warns_rather_than_failing(caplog):
    """Below the advisory threshold the isotonic fit really is noisy. Saying so
    is useful; refusing is a regression against the legacy carve, which used a
    62-row fold on this very cohort without comment."""
    genes, y = _small_cohort()
    with caplog.at_level("WARNING"):
        _, i_cal, _ = _carve(genes, y)
    assert len(i_cal) < 200
    assert "advisory threshold" in caplog.text
    assert "Proceeding" in caplog.text


def test_a_large_fold_is_not_warned_about(caplog):
    """The advisory must not fire on a healthy fold, or it becomes noise that
    gets filtered out and stops being read."""
    genes, y = _cohort(1200, 1, 300, seed=3)
    with caplog.at_level("WARNING"):
        _carve(genes, y)
    assert "advisory threshold" not in caplog.text


def test_a_dominant_gene_cannot_swallow_the_calibration_fold():
    """THE TRAP RUNS BOTH WAYS, found by probing the lower bound. A gene holding
    9,000 of 11,000 rows can land wholly IN the calibration fold: measured
    2026-07-21, that gave a 9,300-row fold against a 1,650-row target -- 85 per
    cent of the data, leaving the models 1,700 rows to fit on."""
    genes, y = _dominant_gene_cohort()
    with pytest.raises(ValueError) as ei:
        _carve(genes, y)
    assert "too many rows" in str(ei.value)


def test_the_refusal_names_both_bounds():
    """A refusal that does not say WHICH bound was missed leaves the reader to
    guess whether to change the cohort or the configuration."""
    genes, y = _dominant_gene_cohort()
    with pytest.raises(ValueError) as ei:
        _carve(genes, y)
    msg = str(ei.value)
    assert "absolute floor" in msg
    assert "starve the fit" in msg


def test_the_relative_floor_catches_an_undersized_carve():
    """The lower half of the same trap: a fold far smaller than requested."""
    genes, y = _cohort(500, 3, 12)
    with pytest.raises(ValueError, match="no gene-disjoint carve"):
        _carve(genes, y, calibration_min_fraction_of_target=50.0)


@pytest.mark.parametrize("fixture,expected_pct", [
    ("small_cohort", (0.05, 0.40)),
    ("heavy_tailed", (0.05, 0.30)),
])
def test_the_realized_fraction_lands_in_a_sane_band(fixture, expected_pct):
    genes, y = (_small_cohort() if fixture == "small_cohort"
                else _cohort(1200, 1, 300, seed=3))
    _, i_cal, _ = _carve(genes, y)
    frac = len(i_cal) / len(y)
    assert expected_pct[0] < frac < expected_pct[1], f"{frac:.3f}"


def test_clinvar_scale_is_unaffected_by_the_new_bounds():
    """The bounds must not disturb the regime the project actually runs in."""
    genes, y = _cohort(8000, 1, 300, seed=3)
    _, i_cal, how = _carve(genes, y)
    frac = len(i_cal) / len(y)
    assert how == "gene_disjoint:seed=42"
    assert 0.12 < frac < 0.18, f"realized {frac:.3f} against a 0.15 target"
