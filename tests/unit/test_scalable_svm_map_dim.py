"""The Nystrom/RFF map dimension must be clamped to the rows the map ACTUALLY sees.

Added 2026-07-13.

THE DEFECT
----------
`ScalableSVM._build_headline` clamped the feature-map dimension D with

    d = int(min(self.n_components, max(n_samples - 1, 1)))

-- against the size of the FULL training set. But `calibrate=True` is the DEFAULT, and it
wraps the pipeline (StandardScaler -> Nystroem -> LinearSVC) in a CalibratedClassifierCV,
which REFITS THE ENTIRE PIPELINE ON EACH CROSS-VALIDATION TRAINING FOLD. Every fold is
strictly smaller than n_samples.

Nystroem selects `n_components` ROWS of the data it is fitted on as landmarks, so it
requires n_components <= (rows fitted on). When that is violated, scikit-learn does not
raise -- it SILENTLY REDUCES n_components and emits:

    UserWarning: n_components > n_samples. This is not possible.
    n_components was set to n_samples, which results in inefficient evaluation of the
    full kernel.

The clamp was therefore off by the cross-validation factor.

THE MEASUREMENT (2026-07-13), n=100, n_components=1024, old clamp giving d=99
-----------------------------------------------------------------------------
    calibrate=True,  calibration_cv=3   -> 3 warnings   (one per fold, ~66 rows each)
    calibrate=True,  calibration_cv=5   -> 5 warnings   (one per fold, ~80 rows each)
    calibrate=False (no CV refit)       -> 0 warnings

One warning per fold, exactly. That is the signature of a per-fold refit -- conclusive.

WHY IT SURVIVED
---------------
At production scale the clamp never binds: n ~ 1.7e6 and D = 1024, so min(1024, n-1) = 1024
and each fold (~1.1e6 rows) dwarfs it. The defect is INVISIBLE at the scale the model is
trained at, and appears only on the small fixtures the tests use -- where 18 of these
warnings per suite run were waved away as "test-scale noise" instead of being read. That is
the same reflex that let a NON-CONVERGED logistic regression train for weeks
(tests/unit/test_logistic_regression_is_scaled.py): a warning is not a failure, so nobody
looked. A finding in a log is a comment; a finding that fails a test is a gate.

WHAT THIS FILE GATES
--------------------
1. The arithmetic of the clamp, asserted directly against StratifiedKFold's actual smallest
   training fold -- not merely "does it warn".
2. That fitting emits ZERO n_components warnings across a range of n / cv combinations.
3. That the un-calibrated path still uses the full row count (no needless shrinkage).
4. That production-scale parameters leave D at its requested value (the clamp must not bind
   where it should not).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from genomic_variant_classifier.models.scalable_svm import ScalableSVM

_WARN = "n_components > n_samples"


def _xy(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _count_warnings(**kw) -> int:
    n = kw.pop("n", 100)
    X, y = _xy(n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ScalableSVM(**kw).fit(X, y)
    return sum(1 for w in caught if _WARN in str(w.message))


# ---------------------------------------------------------------------------
# 1. The arithmetic, checked against scikit-learn's ACTUAL fold sizes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [40, 60, 100, 250])
@pytest.mark.parametrize("cv", [2, 3, 5])
def test_clamp_matches_the_smallest_real_training_fold(n, cv):
    """Do not trust `n - ceil(n/k)`; verify it against StratifiedKFold itself.

    The formula is only correct if it equals the smallest training fold scikit-learn
    actually produces. Derive that empirically and compare -- a formula asserted against
    itself proves nothing.
    """
    X, y = _xy(n)
    svm = ScalableSVM(mode="nystrom", n_components=1024, calibrate=True, calibration_cv=cv)

    smallest_real_fold = min(
        len(tr) for tr, _ in StratifiedKFold(n_splits=cv).split(X, y)
    )
    claimed = svm._rows_the_map_is_fitted_on(n)

    assert claimed == n - math.ceil(n / cv), "the closed form drifted from its own definition"
    assert claimed <= smallest_real_fold, (
        f"the clamp ({claimed}) EXCEEDS the smallest training fold scikit-learn actually "
        f"builds ({smallest_real_fold}) for n={n}, cv={cv}. Nystroem would silently truncate."
    )
    # And the dimension handed to Nystroem must never exceed that fold.
    assert svm._map_dim(n) <= smallest_real_fold


# ---------------------------------------------------------------------------
# 2. The behavioural gate: fitting must be SILENT.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cv", [2, 3, 5])
def test_calibrated_nystrom_fit_emits_no_n_components_warning(cv):
    """The 18 warnings that ran in every suite for weeks. Zero, now, or this fails."""
    hits = _count_warnings(
        n=100, mode="nystrom", n_components=1024, calibrate=True, calibration_cv=cv
    )
    assert hits == 0, (
        f"ScalableSVM(nystrom, calibrate=True, calibration_cv={cv}) still emits {hits} "
        f"'{_WARN}' warning(s) -- one per fold. The map dimension is being clamped against "
        f"n_samples instead of the cross-validation TRAINING FOLD; see this module's "
        f"docstring."
    )


def test_rff_fit_is_also_silent():
    """RBFSampler has no n<=D requirement, but the clamp is applied to it deliberately."""
    assert _count_warnings(
        n=100, mode="rff", n_components=1024, calibrate=True, calibration_cv=3
    ) == 0


def test_uncalibrated_path_was_never_broken_and_stays_silent():
    """The control from the 2026-07-13 measurement: calibrate=False emitted 0 warnings."""
    assert _count_warnings(
        n=100, mode="nystrom", n_components=1024, calibrate=False
    ) == 0


# ---------------------------------------------------------------------------
# 3. The clamp must NOT bind where it should not.
# ---------------------------------------------------------------------------
def test_uncalibrated_clamp_uses_the_full_row_count():
    """With no cross-validation refit, the map sees every row -- do not shrink it."""
    svm = ScalableSVM(mode="nystrom", n_components=1024, calibrate=False)
    assert svm._rows_the_map_is_fitted_on(500) == 500
    assert svm._map_dim(500) == 500          # min(1024, 500)
    assert svm._map_dim(5000) == 1024        # min(1024, 5000) -- requested D honoured


def test_at_production_scale_the_clamp_does_not_bind():
    """n ~ 1.7e6, D = 1024: the requested dimension must survive untouched.

    This is the case the model is ACTUALLY trained at, and the case in which the original
    bug was invisible. If a future 'fix' to the clamp ever starts shrinking D here, the
    headline SVM result silently changes -- so pin it.
    """
    svm = ScalableSVM(mode="nystrom", n_components=1024, calibrate=True, calibration_cv=3)
    n = 1_700_000
    assert svm._rows_the_map_is_fitted_on(n) == n - math.ceil(n / 3)
    assert svm._map_dim(n) == 1024, (
        "at production scale the clamp must not bind; D must remain exactly as requested"
    )


def test_degenerate_tiny_n_never_produces_a_zero_width_map():
    """A map of width 0 would be a silent catastrophe; the floor must hold."""
    svm = ScalableSVM(mode="nystrom", n_components=1024, calibrate=True, calibration_cv=3)
    for n in (1, 2, 3, 4):
        assert svm._rows_the_map_is_fitted_on(n) >= 1
        assert svm._map_dim(n) >= 1
