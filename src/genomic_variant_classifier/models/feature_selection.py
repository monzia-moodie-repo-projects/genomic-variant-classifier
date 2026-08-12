"""Decide which columns a fitted estimator keeps, and RECORD why.

NEURALNAN-1
===========
`TabularNNClassifier.fit` selected columns with

    self.feature_mask_ = X.var(axis=0) > 0.0

as a zero-variance guard. The guard is legitimate -- a constant column carries
no signal and StandardScaler would divide by zero on it. What is wrong is that
`np.var` over a column containing ANY NaN returns NaN, and `NaN > 0.0` is False.
So the guard silently reclassifies "contains a missing value" as "is degenerate"
and DELETES the whole feature, including every observed value it still holds.

MEASURED 2026-08-10, with a responsiveness control at two sample sizes:

    tabular_nn / mc_dropout / deep_ensemble
        delta on a NaN-bearing column : EXACTLY 0.000000 at n=8 AND at n=400
        delta on columns never missing: 0.28 - 0.99

At n=400 that column retained 360 OBSERVED values, and perturbing them changed
nothing. The mask is then pickled with the estimator, so the deletion persists
into every later prediction.

Three of thirteen estimators therefore trained on a different feature space than
the other ten, invisibly. In a project whose stated purpose includes comparing
machine-learning algorithms empirically, that makes a per-model comparison
measure feature availability alongside algorithm.

A GUARD I DID NOT KNOW WAS THERE
=================================
Line 1921 adds

    if not self.feature_mask_.any():        # degenerate: keep all, never 0-width
        self.feature_mask_ = np.ones(X.shape[1], dtype=bool)

so the failure is PARTIAL, not all-or-nothing. A matrix in which every feature
had a missing value passes through untouched; one with a single missing feature
loses exactly that feature. That is harder to notice, not easier, and it is why
nothing ever raised.

WHY THIS COMMIT RECORDS RATHER THAN REFUSES
===========================================
Refusing a missing-value column is the correct end state, and it is NOT the
correct step today. The semantic matrix now legitimately carries NaN in two
constraint features; refusing would make three estimators fail, the ensemble's
fail-loud dropout machinery would raise, and Run 17 could not start -- while no
rendering layer yet exists to impute them. A guard that forces a correct
representation before that representation can be produced is not a repair.

So the two conditions are SEPARATED and both are RECORDED. Behaviour is
unchanged; the loss becomes visible. That mirrors the project's own precedent
at variant_ensemble.py:2923-2956, where a base-model failure is recorded with
its cause and logged at error level rather than absorbed.

`np.nanvar` alone would have been the wrong repair in the other direction: it
keeps the column alive and sends NaN into a network that PROPAGATES it -- kan
demonstrated exactly that, returning non-finite probabilities from a completed
fit. Curing the diagnostic while preserving the defect.

Author: Monzia Moodie
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSelectionRecord:
    """WHY each column was kept or dropped. Immutable once fitted.

    `dropped_constant` and `dropped_missing` are separate because they are
    different findings with different repairs. A constant column is a data
    question -- did a connector produce anything? A dropped-for-missingness
    column is a REPRESENTATION defect: the value exists and the estimator
    never saw it.
    """
    n_input: int
    kept: tuple = ()
    dropped_constant: tuple = ()
    dropped_missing: tuple = ()
    #: True when every column was degenerate and the mask was discarded to
    #: avoid a zero-width matrix. Recorded because in that state NOTHING was
    #: dropped, and a reader comparing counts would otherwise be puzzled.
    degenerate_fallback: bool = False

    def as_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_kept": len(self.kept),
            "dropped_constant": list(self.dropped_constant),
            "dropped_missing": list(self.dropped_missing),
            "degenerate_fallback": self.degenerate_fallback,
        }

    @property
    def lost_to_missingness(self) -> int:
        """The NEURALNAN-1 quantity. Non-zero means an estimator is training on
        fewer features than the contract declares, for a reason that is a
        defect rather than a property of the data."""
        return len(self.dropped_missing)


def select_model_features(X, *, estimator_name: str = "tabular_nn",
                          feature_names=None):
    """Return (mask, record) for a numeric design matrix.

    CONSTANCY IS MEASURED AMONG OBSERVED VALUES. `np.nanvar` ignores NaN, so a
    column that is constant where observed is genuinely degenerate whether or
    not it also has gaps -- and a column that VARIES where observed is not
    degenerate at all, however many gaps it has. Those two facts cannot be
    distinguished by `X.var(axis=0)`, which returns NaN for both.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            "feature selection requires a 2-D matrix, got {} dimension(s)".format(
                arr.ndim))
    n = arr.shape[1]
    names = list(feature_names) if feature_names is not None else list(range(n))
    if len(names) != n:
        raise ValueError(
            "{} feature name(s) for {} column(s)".format(len(names), n))

    has_missing = np.isnan(arr).any(axis=0)
    all_missing = np.isnan(arr).all(axis=0)

    # np.nanvar over an ALL-NaN column emits "Degrees of freedom <= 0" -- a
    # RuntimeWarning, which np.errstate does NOT suppress (that governs
    # floating-point errors, not NumPy's warning machinery). The verdict is
    # already correct, because all_missing catches those columns below; but the
    # warning would appear in every run log carrying an entirely-absent feature,
    # and this cohort has ten. Noise inside a defect-detection instrument
    # teaches a reader to ignore it.
    #
    # Computing variance only where there is something to compute is both
    # quieter and more honest: an undefined variance is left undefined.
    observed_var = np.full(arr.shape[1], np.nan, dtype=float)
    computable = ~all_missing
    if computable.any():
        observed_var[computable] = np.nanvar(arr[:, computable], axis=0)

    # A column with NO observed values has an undefined variance; it is
    # degenerate for the same reason a constant one is -- nothing to learn from.
    constant = (~np.isnan(observed_var) & (observed_var <= 0.0)) | all_missing
    # Dropped FOR MISSINGNESS ONLY: it varies where observed, and is discarded
    # solely because the estimator cannot consume NaN. This is the defect.
    missing_only = has_missing & ~constant

    mask = ~(constant | missing_only)
    degenerate_fallback = False
    if not mask.any():
        # Mirrors the existing line-1921 guard: never hand the network a
        # zero-width matrix. In this state nothing is dropped, so the record
        # says so rather than reporting phantom losses.
        mask = np.ones(n, dtype=bool)
        degenerate_fallback = True
        record = FeatureSelectionRecord(
            n_input=n, kept=tuple(names), degenerate_fallback=True)
    else:
        record = FeatureSelectionRecord(
            n_input=n,
            kept=tuple(np.asarray(names, dtype=object)[mask].tolist()),
            dropped_constant=tuple(
                np.asarray(names, dtype=object)[constant & ~missing_only].tolist()),
            dropped_missing=tuple(
                np.asarray(names, dtype=object)[missing_only].tolist()),
            degenerate_fallback=False)

    if record.lost_to_missingness:
        logger.warning(
            "%s: %d feature(s) DROPPED FOR MISSINGNESS ALONE -- they vary where "
            "observed and are discarded only because this estimator cannot "
            "consume NaN: %s. This estimator is training on a SMALLER FEATURE "
            "SPACE than the contract declares, which makes a per-model "
            "comparison against native-missing estimators measure feature "
            "availability alongside algorithm. See NEURALNAN-1.",
            estimator_name, record.lost_to_missingness,
            list(record.dropped_missing)[:10])
    if record.dropped_constant:
        logger.info(
            "%s: %d constant feature(s) dropped (no signal where observed): %s",
            estimator_name, len(record.dropped_constant),
            list(record.dropped_constant)[:10])
    return mask, record
