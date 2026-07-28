"""
Evaluation Module for Genomic Variant Classification
Author: Monzia Moodie

=============================================================================
REVISION 2026-07-08 -- METRIC STACK ADDED **ALONGSIDE** THE ORIGINAL API.

SUPERSEDED 2026-07-21: the original API below the header -- `compute_classification_metrics`
and `ModelEvaluator` are restored verbatim from commit 87e32ad^, after 87e32ad
overwrote them. Behaviour, signatures, and sklearn backing are identical.

WHY THE ADDITIONS EXIST

  The project's declared primary metric is AUPRC, whose no-skill floor equals the
  positive rate. An AUPRC quoted without `pos_rate` is uninterpretable, and two AUPRCs
  measured at different positive rates are not comparable. Runs 9-14 ran at pos_rate
  20.34%; Runs 15-17 at 14.15%. No run artifact recorded which. See
  docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md sec 6.

  The additions also make every panel STRATIFIABLE, because a headline over a cohort
  that is 86% SNVs and 8.5% padded deletions -- the latter 3.5x pathogenic-enriched and
  carrying sixteen literally constant features -- describes neither class.

DESIGN RULES (additions only)
  * A metric that cannot be computed returns NaN and says so. It never returns 0.5 or
    0.0 as if that were a measurement.
  * Confidence intervals are STRATIFIED bootstrap by default: positives and negatives
    resampled separately, so `pos_rate` is preserved and the CI reflects noise in the
    score, not in the class balance.
  * Ties are handled explicitly: average ranks for AUROC, tied scores collapsed to a
    single threshold for AUPRC. This matters -- every padded deletion in the Run-14
    splits carries sixteen constant features. Both cross-validated against sklearn to
    1e-16, including all-constant scores (tests/unit/test_evaluation_metrics.py).

KNOWN PRE-EXISTING WART, LEFT ALONE: `calibration_curve` is imported and never used.
Removing it is a behaviour-neutral cleanup for a separate commit, not this one.
=============================================================================
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

# =============================================================================
# METRIC STACK (2026-07-08) -- ADDITIVE. Pure numpy/pandas primitives, every one
# independently unit-tested and cross-validated against sklearn.
#
# HARDENED 2026-07-20 -- REMEDIATION OF AN INDEPENDENT AUDIT.
#
# Six confirmed defects repaired. Each is named at the function that fixes it, with the
# behaviour it replaces, so a reader can tell what changed and why without a diff.
#
#   A  score and prob were cleaned with SEPARATE masks -> silent row misalignment
#   B  labels were COERCED with .astype(int), not validated -> signed, wrong AUROC
#   D  an EMPTY probability vector was reported as a valid probability
#   E  the calibration solver could not report nonconvergence
#   F  rows with a missing subgroup label counted in ALL and appeared in no stratum
#   G  subgroup sufficiency tested total n only, ignoring class support
#   +  the bootstrap resampled VARIANTS, ignoring within-gene correlation
#
# 2026-07-21 -- THE SEPARATE COMMIT THE NOTE ABOVE ANTICIPATED.
#
# `compute_classification_metrics` and `ModelEvaluator` stood above this banner until
# today, described there as "unsafe in ways this stack explicitly rejects" with their
# unification deferred to a separate, separately-measured commit. This is it. They are
# REMOVED, not wrapped.
#
# Removal rather than delegation, because delegation preserves the contract that is the
# actual problem. The old API returned a dict of bare floats, which cannot express
# undefined, insufficient support, dependency unavailable, or computationally deferred --
# all first-class scientific states here. float("nan") inside such a dict is already a
# compromise; `specificity: 0` for an undefined quantity is worse. And on a single-class
# cohort it did neither: confusion_matrix(...).ravel() yields one cell, not four, so it
# raised "not enough values to unpack (expected 4, got 1)".
#
# Neither had a production caller. One test pinned them; one notebook imported them from
# `src.evaluation.metrics`, a package path that has not existed since the rename. Two
# evaluation contracts in one module invite divergence, and this project was bitten four
# times in a single day by a check reporting success where nothing was measured. Use
# `evaluate()`: one contract, one status model, one source of truth.
# =============================================================================

import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from genomic_variant_classifier.evaluation.capabilities import (  # noqa: E402
    BootstrapUnit,
    MetricStatus,
)
from typing import Callable, Iterable, Iterator, Sequence  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = [
    # `compute_classification_metrics` and `ModelEvaluator` were removed 2026-07-21;
    # the "do not remove" note that stood here was itself stale. See the banner above.
    # metric stack
    "auroc", "auprc", "auprc_gain", "no_skill_auprc", "brier_score", "log_loss",
    "expected_calibration_error", "calibration_slope_intercept",
    "bootstrap_ci", "cluster_bootstrap_ci", "evaluate", "stratified_evaluate",
    "BootstrapUnit", "BootstrapResult", "bootstrap_metric",
    "InsufficientSupportError", "DEFAULT_MIN_VALID_REPLICATES",
    "DEFAULT_MIN_VALID_FRACTION",
    "is_probability", "clean_arrays", "CleanArrays", "CalibrationFit",
]

_EPS = 1e-12

# Sufficiency floors for a subgroup panel. NOT universal constants -- project policy, chosen
# so a stratum whose metrics cannot be stable is reported as insufficient rather than given
# numbers that look like measurements. Overridable per call.
DEFAULT_MIN_N = 30
DEFAULT_MIN_POS = 10
DEFAULT_MIN_NEG = 10


@dataclass(frozen=True)
class CleanArrays:
    """Labels, ranking score and probability on ONE shared row mask, with accounting.

    DEFECT A. The previous code cleaned `score` and `prob` independently:

        y_c, s_c = _clean(y, score)
        p = s_c if prob is None else _clean(y, prob)[1]

    Those are two different masks. A non-finite score on row 1 and a non-finite probability
    on row 2 produced two arrays of the SAME LENGTH describing DIFFERENT ROWS, after which
    every calibration metric paired a probability with the wrong label -- silently, because
    the length check downstream passed.

    `n_dropped` is not decoration. A panel computed after silently discarding a large
    fraction of the cohort is a different measurement from the one the caller asked for, and
    the count is the only way to notice.
    """

    y: "np.ndarray"
    score: "np.ndarray"
    probability: "np.ndarray"
    mask: "np.ndarray"          # the joint keep-mask, over the ORIGINAL rows
    n_input: int
    n_dropped_nonfinite_y: int
    n_dropped_nonfinite_score: int
    n_dropped_nonfinite_probability: int

    @property
    def n(self) -> int:
        return int(self.y.size)

    @property
    def n_dropped(self) -> int:
        return int(self.n_input - self.n)

    @property
    def dropped_fraction(self) -> float:
        return float(self.n_dropped / self.n_input) if self.n_input else float("nan")


# --------------------------------------------------------------------------- #
# The prediction-input contract. FAIL CLOSED.
#
# Ruled 2026-07-27: no numerical kernel may select, filter, normalise or
# redefine its evaluation population. Population construction is an explicit
# upstream operation, and every result must describe exactly that population.
#
# A non-finite predicted score or probability is a MODEL-OUTPUT FAILURE, not an
# ordinary missing observation. The kernels below therefore ASSERT this contract
# rather than repairing the input. Validation happens in the registry, which
# refuses before invoking anything; reaching one of these raises means an
# unvalidated caller bypassed that gate, which is a defect and must be loud.
#
# VALIDATION IS METRIC-SPECIFIC, NOT UNIVERSAL. Each kernel checks only the
# prediction array IT consumes. A universal assertion over every supplied array
# would fail a probability-only metric because an irrelevant score array carried
# a non-finite value, contradicting the descriptor-specific accounting the
# registry already establishes.
#
# FINITENESS RAISES; RANGE DOES NOT. These are different categories and the
# distinction is load-bearing. A vector outside [0, 1] was never a probability
# vector -- `is_probability` returns False and the calibration kernels return
# NaN, a contract pinned by test_calibration_metrics_are_nan_on_non_probability_scores
# -- and THE SAME ARRAY remains a perfectly valid score for a ranking metric on
# the same rows, which that test also asserts. A non-finite value is different:
# the model did not produce an output at all. Raising on range here would break a
# landed, correct contract and conflate "not a probability" with "no prediction".
# --------------------------------------------------------------------------- #
def _require_finite_scores(scores: Sequence, *, metric_name: str) -> "np.ndarray":
    arr = np.asarray(scores, dtype=float).ravel()
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(
            f"{metric_name}: scores contain non-finite model outputs "
            f"({n_bad} of {arr.size}); prediction arrays must be validated "
            "before kernel invocation. The kernel does not filter them: a "
            "value computed over the survivors would describe a population "
            "nobody named.")
    return arr


def _require_finite_probabilities(probabilities: Sequence, *,
                                  metric_name: str) -> "np.ndarray":
    arr = np.asarray(probabilities, dtype=float).ravel()
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(
            f"{metric_name}: probabilities contain non-finite model outputs "
            f"({n_bad} of {arr.size}); prediction arrays must be validated "
            "before kernel invocation.")
    return arr


def clean_arrays(y: Sequence, score: Sequence,
                 probability: Sequence | None = None) -> CleanArrays:
    """Validate labels strictly; drop non-finite rows on ONE joint mask.

    DEFECT B. The previous `_clean` ended `y[ok].astype(int)`. numpy truncates toward zero,
    so 0.9 became 0, 1.2 became 1 and 2.0 stayed 2 -- silently. A stray 2.0 then made
    `(1 - y).sum()` NEGATIVE, so AUROC's `n_pos * n_neg` denominator was negative and the
    metric came back signed and wrong; `_degenerate`'s `y.sum() == y.size` test could also be
    satisfied by accident. Labels are now REJECTED, never coerced.

    Booleans are accepted -- they are unambiguously binary. Everything else must be exactly
    0 or 1 after the non-finite rows are removed.
    """
    y_arr = np.asarray(y)
    if y_arr.dtype == bool:
        y_arr = y_arr.astype(float)
    y_arr = y_arr.astype(float).ravel()
    s_arr = np.asarray(score, dtype=float).ravel()
    p_arr = s_arr.copy() if probability is None else np.asarray(
        probability, dtype=float).ravel()

    if not (y_arr.shape == s_arr.shape == p_arr.shape):
        raise ValueError(
            "evaluation arrays have different shapes: "
            f"y={y_arr.shape}, score={s_arr.shape}, probability={p_arr.shape}"
        )

    # LEGACY PATH ONLY. Label selection is a POPULATION decision and now belongs
    # to `EvaluationPopulation`, which records what it removed and why. The mask
    # below survives solely for `metrics.evaluate`, the survivor-filtering
    # compatibility composite, which constructs its own population and discloses
    # the narrowing as n_input / n / n_dropped. The registry never reaches this
    # code: it refuses non-finite predictions before dispatch and receives arrays
    # already projected through a population.
    fy = np.isfinite(y_arr)
    fs = np.isfinite(s_arr)
    fp = np.isfinite(p_arr)
    keep = fy & fs & fp

    y_keep = y_arr[keep]
    bad = y_keep[~np.isin(y_keep, (0.0, 1.0))]
    if bad.size:
        offenders = np.unique(bad)[:10]
        raise ValueError(
            f"binary labels must be exactly 0 or 1; found {offenders.tolist()} "
            f"in {bad.size} of {y_keep.size} rows. Labels are validated, never coerced: "
            "astype(int) would truncate 0.9 to 0 and leave 2.0 as 2, which makes "
            "(1 - y).sum() negative and AUROC signed."
        )

    return CleanArrays(
        y=y_keep.astype(np.int8),
        score=s_arr[keep],
        probability=p_arr[keep],
        mask=keep,
        n_input=int(y_arr.size),
        n_dropped_nonfinite_y=int((~fy).sum()),
        n_dropped_nonfinite_score=int((~fs).sum()),
        n_dropped_nonfinite_probability=int((~fp).sum()),
    )


def _clean(y: Sequence, s: Sequence) -> "tuple[np.ndarray, np.ndarray]":
    """Two-array compatibility wrapper over `clean_arrays`.

    Kept so every existing caller -- auroc, auprc, brier_score, the calibration functions,
    bootstrap_ci -- gains strict label validation without a signature change.
    """
    c = clean_arrays(y, s)
    return c.y.astype(int), c.score


def _degenerate(y: "np.ndarray") -> bool:
    """A single class present -> ranking metrics are undefined. Say so; do not guess."""
    return y.size == 0 or y.sum() == 0 or y.sum() == y.size


def auroc(y: Sequence, score: Sequence) -> float:
    """Rank-based AUROC (Mann-Whitney U). Ties get average ranks. NaN if one class.

    RAISES on non-finite scores. See the prediction-input contract above.
    """
    _require_finite_scores(score, metric_name="auroc")
    y, s = _clean(y, score)
    if _degenerate(y):
        return float("nan")
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y: Sequence, score: Sequence) -> float:
    """Average precision, aggregating TIED scores into a single threshold.

    Ties are not a corner case here: every padded deletion in the Run-14 splits carries
    sixteen literally constant features, and any binary indicator ties massively. A
    row-by-row walk after an arbitrary tie-break inflates AP (it credits an ordering the
    score never expressed). Cross-checked against sklearn.average_precision_score to
    1e-16 on balanced, 10%- and 1%-imbalanced, near-random, heavily-tied, binary, and
    all-constant scores.
    """
    _require_finite_scores(score, metric_name="auprc")
    y, s = _clean(y, score)
    if _degenerate(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yy, ss = y[order], s[order]
    last_of_run = np.r_[np.flatnonzero(np.diff(ss)), yy.size - 1]
    tp = np.cumsum(yy)[last_of_run]
    fp = np.cumsum(1 - yy)[last_of_run]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / tp[-1]
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def no_skill_auprc(y: Sequence) -> float:
    """A random classifier's AUPRC == the positive rate. Quote AUPRC against this."""
    y = np.asarray(y).astype(float).ravel()
    y = y[np.isfinite(y)]
    return float(y.mean()) if y.size else float("nan")


def auprc_gain(y: Sequence, score: Sequence) -> float:
    """AUPRC minus its no-skill floor -- the ABSOLUTE gain.

    `auprc_lift` (the ratio) explodes when prevalence is tiny: at pos_rate 0.001 an AUPRC of
    0.01 is a lift of 10 and a gain of 0.009. Reporting only the ratio makes a negligible
    improvement look transformative. Both are now reported.
    """
    base = no_skill_auprc(y)
    ap = auprc(y, score)
    if not (np.isfinite(base) and np.isfinite(ap)):
        return float("nan")
    return float(ap - base)


def _sigmoid(z: "np.ndarray") -> "np.ndarray":
    """Overflow-safe logistic. np.exp(-z) overflows for z << 0."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def is_probability(p: Sequence, tol: float = 1e-9) -> bool:
    """True iff there is at least one finite value and every one lies in [0, 1].

    DEFECT D. This previously returned True for an EMPTY vector (`p.size == 0 or ...`), so
    `calibration_valid=True` could be recorded for a probability vector containing nothing.
    A missing probability vector is not a valid probability vector.

    Calibration metrics -- Brier, ECE, calibration slope/intercept -- are defined only for
    probabilities. Handed a raw standardized feature (range e.g. -0.44 .. 4.89), `evaluate()`
    used to clip, take a logit, and report numbers. Those numbers meant nothing and nothing
    said so. A metric that cannot be computed must say so.
    """
    p = np.asarray(p, dtype=float).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return False
    return bool((p >= -tol).all() and (p <= 1.0 + tol).all())


def brier_score(y: Sequence, prob: Sequence) -> float:
    """NaN if `prob` is not a probability -- Brier is undefined outside [0, 1]."""
    if not is_probability(prob):
        return float("nan")
    _require_finite_probabilities(prob, metric_name="brier_score")
    y, p = _clean(y, prob)
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def log_loss(y: Sequence, prob: Sequence, eps: float = 1e-15) -> float:
    """Binary cross-entropy. NaN if `prob` is not a probability.

    Added 2026-07-20. Brier and log loss disagree in a way that matters clinically: log loss
    is unbounded and punishes CONFIDENT errors far harder, which is the failure mode that
    costs most when a pathogenic variant is called benign at p = 0.99.
    """
    if not is_probability(prob):
        return float("nan")
    _require_finite_probabilities(prob, metric_name="log_loss")
    y, p = _clean(y, prob)
    if y.size == 0:
        return float("nan")
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


# --------------------------------------------------------------------------- #
# THE CALIBRATION BINNING TABLE
#
# One binning, derived once, from which BOTH the expected and the maximum
# calibration error are read. Until 2026-07-27 the two statistics were computed
# by two separate implementations -- this module binned one way and
# `ClinicalEvaluator._calibration_error` another -- and they disagreed about
# every probability sitting exactly on an interior edge.
#
# THE DEFECT, MEASURED. The kernel used `np.digitize(..., right=True)`, which
# makes EVERY bin `(lo, hi]` -- left-open. The first line of the docstring below
# says "TOP BIN CLOSED", describing `[lo, hi)` with only the final bin closed,
# which is what the evaluator implemented. Every edge-exact probability therefore
# landed one bin LOWER in the kernel than documented, for seventeen days.
#
# On a cohort where an edge-exact value shares a bin with non-edge values of the
# OPPOSITE calibration sign, the two returned 0.3242857 and 0.0642857 -- a
# relative difference of 404%.
#
# WHY IT SURVIVED SEVENTEEN DAYS. The expected calibration error is
#
#     (1/N) * sum_b | sum_{i in b} (y_i - p_i) |
#
# which is INVARIANT to regrouping whenever every merged group shares the sign of
# (accuracy - confidence): combining same-sign groups cannot change the total.
# Ordinary fixtures land in that regime by default, and the fixtures in
# tests/unit/test_calibration_implementations_agree.py contain no interior-edge
# value at all -- so that module separated the TOP-bin definitions and was
# structurally unable to separate these.
#
# REACHABILITY. Continuous scores and the thirteen-model ensemble mean put NO
# rows on an interior decade edge. Probabilities averaged over ten folds put 55%
# there, and probabilities rounded to one decimal 60%.
#
# No published figure moves: every published calibration number came from the
# evaluator, which was already correct, and the evaluator now reads this table
# rather than carrying a second implementation of it.
# --------------------------------------------------------------------------- #
CALIBRATION_BINNING = "equal_width"
CALIBRATION_INTERVAL_CONVENTION = "[lo, hi) with the top bin closed at 1.0"
CALIBRATION_DEFINITION_VERSION = 2


def equal_width_bin_indices(values: Sequence, n_bins: int = 10) -> "np.ndarray":
    """Bin index for each probability under the documented convention.

    `[lo, hi)` for every bin except the last, which is closed at 1.0 so that
    predictions of exactly 1.0 -- pure decision-tree and ensemble leaves -- are
    counted rather than silently dropped.

    `searchsorted(edges, v, side="right") - 1` places an edge-exact value in the
    bin it OPENS, and the clip closes the top. Expressed as a named function
    rather than an inline expression because the convention is a scientific
    decision, it has been got wrong once already, and a named function is
    somewhere a validation and a test can attach.

    Fails closed on non-finite and out-of-range input rather than silently
    placing such values in the first or last bin, which is what an unguarded
    clip would do.
    """
    v = np.asarray(values, dtype=float).ravel()
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool):
        raise TypeError(f"n_bins must be an integer, got {type(n_bins).__name__}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if not np.isfinite(v).all():
        raise ValueError(
            "equal_width_bin_indices: values contain non-finite entries; an "
            "unguarded clip would place them in the first or last bin and the "
            "calibration figure would silently describe a different population")
    if v.size and (v.min() < 0.0 or v.max() > 1.0):
        raise ValueError(
            f"equal_width_bin_indices: values must lie in [0, 1]; observed "
            f"[{v.min()}, {v.max()}]. Clipping them would move mass into the "
            "edge bins and misstate calibration there.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.searchsorted(edges, v, side="right") - 1
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


@dataclass(frozen=True)
class CalibrationBins:
    """Occupied calibration bins, and the statistics read from them.

    The expected and maximum calibration errors are two summaries of ONE table.
    Computing them separately is how they come to disagree, so they are derived
    here from a single binning rather than by two functions that each bin again.

    Only OCCUPIED bins are retained. An empty bin has no accuracy and no
    confidence, and inventing zero for either would drag the maximum toward the
    bin's own midpoint and pull the weighted mean toward nothing.
    """

    bin_index: "np.ndarray"
    accuracy: "np.ndarray"
    confidence: "np.ndarray"
    weight: "np.ndarray"
    n_bins: int
    n_observations: int

    @classmethod
    def from_predictions(cls, y: Sequence, prob: Sequence,
                         n_bins: int = 10) -> "CalibrationBins":
        y_arr, p_arr = _clean(y, prob)
        idx = equal_width_bin_indices(p_arr, n_bins)
        occupied = np.unique(idx)
        n = y_arr.size
        acc = np.array([y_arr[idx == b].mean() for b in occupied], dtype=float)
        conf = np.array([p_arr[idx == b].mean() for b in occupied], dtype=float)
        wgt = np.array([(idx == b).sum() / n for b in occupied], dtype=float)
        return cls(bin_index=occupied.astype(np.int64), accuracy=acc,
                   confidence=conf, weight=wgt, n_bins=int(n_bins),
                   n_observations=int(n))

    @property
    def gap(self) -> "np.ndarray":
        return np.abs(self.accuracy - self.confidence)

    @property
    def expected(self) -> float:
        """Occupancy-weighted mean gap."""
        return float(np.sum(self.weight * self.gap)) if self.bin_index.size else float("nan")

    @property
    def maximum(self) -> float:
        """Largest gap over OCCUPIED bins."""
        return float(self.gap.max()) if self.bin_index.size else float("nan")

    @property
    def n_occupied(self) -> int:
        return int(self.bin_index.size)

    def definition(self) -> dict:
        """How these numbers were produced, for the artifact.

        A calibration figure without its binning convention is not reproducible:
        the same predictions under `(lo, hi]` and under `[lo, hi)` gave 0.3242857
        and 0.0642857 on a measured cohort.
        """
        return {"binning": CALIBRATION_BINNING,
                "interval_convention": CALIBRATION_INTERVAL_CONVENTION,
                "n_bins": self.n_bins,
                "metric_definition_version": CALIBRATION_DEFINITION_VERSION}


def expected_calibration_error(y: Sequence, prob: Sequence, n_bins: int = 10) -> float:
    """Equal-width binning, TOP BIN CLOSED. |accuracy - confidence| weighted by occupancy.

    Returns NaN if `prob` is not a probability.

    THE CLOSED TOP BIN, AND WHY IT IS NOT OPTIONAL. A half-open final bin `(p >= lo) &
    (p < hi)` with `hi == 1.0` drops every prediction of exactly 1.0 -- a pure decision-tree
    or ensemble leaf -- so the rows the model is most confident about contribute nothing.
    Measured under-report on a 20%-pure-leaf split: 86.5% when first audited on 2026-07-08,
    86.7% when re-measured independently on 2026-07-20.

    HISTORY, kept because the dates matter. The defect was diagnosed in `evaluator.py`'s
    `_calibration_error` on 2026-07-08 (docs/audits/EVALUATION_STACK_AUDIT_2026-07-08.md)
    and REPAIRED THERE ON 2026-07-10; see the dated comment at evaluator.py:321-323. An
    earlier version of this docstring described that defect in the present tense and was
    never updated, so it misstated the state of `evaluator.py` for ten days. Corrected
    2026-07-20.

    A survey on 2026-07-20 evaluated nine independent implementations of this metric on
    identical fixtures and found the same open-top defect live in three further places --
    scripts/calibrate_thresholds.py, scripts/validate_external.py and
    scripts/calibration_analysis.py -- plus a second, distinct defect in three more, where
    `calibration_curve`'s non-empty bins were zipped against `np.histogram`'s full bin
    counts. Both were repaired on 2026-07-20 and are pinned by
    tests/unit/test_calibration_implementations_agree.py.
    """
    if not is_probability(prob):
        return float("nan")
    _require_finite_probabilities(prob, metric_name="expected_calibration_error")
    y, p = _clean(y, prob)
    if y.size == 0:
        return float("nan")
    # Reads the shared table rather than summing again. Binning once but summing
    # twice still leaves two implementations that can drift: before this change
    # the two agreed on every bin and still differed by 3.5e-18 through
    # floating-point summation order alone. One table, one summation, no residual.
    return CalibrationBins.from_predictions(y, prob, n_bins).expected


def maximum_calibration_error(y: Sequence, prob: Sequence, n_bins: int = 10) -> float:
    """Largest |accuracy - confidence| over OCCUPIED bins.

    Reads the same `CalibrationBins` table the expected calibration error reads,
    so the two cannot come to disagree about which rows fell in which bin. That
    is not hypothetical: two independent binnings in this codebase disagreed
    about every interior edge for seventeen days.

    Empty bins are excluded rather than counted as a perfect gap of zero or an
    infinite one. A bin nobody landed in carries no evidence either way.

    Returns NaN if `prob` is not a probability, matching
    `expected_calibration_error`, so a raw feature yields NaN from both rather
    than NaN from one and an exception from the other.
    """
    if not is_probability(prob):
        return float("nan")
    _require_finite_probabilities(prob, metric_name="maximum_calibration_error")
    y_arr, _p = _clean(y, prob)
    if y_arr.size == 0:
        return float("nan")
    return CalibrationBins.from_predictions(y, prob, n_bins).maximum


@dataclass(frozen=True)
class CalibrationFit:
    """Slope and intercept WITH the solver's own account of whether it converged.

    DEFECT E. The previous function fell out of `max_iter` and returned coefficients
    regardless, so separation, quasi-separation and numerical instability were
    indistinguishable from a clean fit.

    Iterable, yielding (slope, intercept), so `slope, intercept = calibration_slope_intercept(...)`
    keeps working at every existing call site while `.converged` becomes available.
    """

    slope: float
    intercept: float
    converged: bool
    iterations: int
    clipped_fraction: float

    def __iter__(self) -> Iterator[float]:
        yield self.slope
        yield self.intercept


# --------------------------------------------------------------------------- #
# THRESHOLD-DEPENDENT AND POPULATION KERNELS
#
# Added 2026-07-27 with the registry vocabulary completion. Each takes its
# threshold and comparison operator EXPLICITLY rather than closing over a
# constant, because a threshold without its operator is incomplete provenance:
# the two differ at probabilities exactly equal to the threshold, which is the
# commonest single value a calibrated model emits.
#
# ZERO DENOMINATORS ARE NOT ZERO CORRELATION. Both kernels return NaN when their
# denominator vanishes, so that a constant classifier cannot be reported as
# "evaluated, and uncorrelated". Whether that NaN becomes a refusal or a failure
# is the registry's decision, not the kernel's -- see the applicability
# predicates, which catch the degenerate margin BEFORE dispatch so the result is
# UNDEFINED (a property of the cohort) rather than FAILED (an implementation
# defect).
# --------------------------------------------------------------------------- #
def apply_decision_threshold(prob: Sequence, *, threshold: float,
                             operator: str) -> "np.ndarray":
    """Hard labels from probabilities, under an EXPLICIT comparison operator.

    `>=` and `>` differ exactly at `prob == threshold`. With the conventional
    0.5 that is not a rare corner: it is the value a maximally uncertain model
    emits, and the value a two-model average produces whenever the pair
    disagrees. An implicit operator is a silent scientific choice.
    """
    arr = np.asarray(prob, dtype=float).ravel()
    if operator == ">=":
        return arr >= threshold
    if operator == ">":
        return arr > threshold
    raise ValueError(
        f"unsupported decision operator {operator!r}; must be '>=' or '>'")


def _confusion_counts(y: Sequence, predicted: "np.ndarray") -> tuple:
    y_arr = np.asarray(y, dtype=float).ravel()
    pos = y_arr == 1
    tp = int(np.sum(predicted & pos))
    fp = int(np.sum(predicted & ~pos))
    fn = int(np.sum(~predicted & pos))
    tn = int(np.sum(~predicted & ~pos))
    return tp, fp, fn, tn


def matthews_correlation_coefficient(y: Sequence, prob: Sequence, *,
                                     threshold: float, operator: str = ">=") -> float:
    """Matthews correlation coefficient at an explicit decision threshold.

    Returns NaN when any confusion-matrix margin is zero. scikit-learn returns
    0.0 there, which is indistinguishable from a genuinely evaluated classifier
    whose predictions carry no correlation with the outcome. A constant
    classifier has an undefined coefficient, not a measured one of zero.
    """
    if not is_probability(prob):
        return float("nan")
    _require_finite_probabilities(prob, metric_name="matthews_correlation_coefficient")
    y_arr, p_arr = _clean(y, prob)
    if y_arr.size == 0:
        return float("nan")
    predicted = apply_decision_threshold(p_arr, threshold=threshold, operator=operator)
    tp, fp, fn, tn = _confusion_counts(y_arr, predicted)
    denominator = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    if denominator <= 0.0:
        return float("nan")
    return float((tp * tn - fp * fn) / np.sqrt(denominator))


def f1_at_threshold(y: Sequence, prob: Sequence, *, threshold: float,
                    operator: str = ">=") -> float:
    """Positive-class F1 at an explicit decision threshold.

    Returns NaN when `2*tp + fp + fn` is zero -- no positive reference labels
    and no positive predictions. scikit-learn returns 0.0 under its
    `zero_division` policy; reporting that as observed performance would state
    that the classifier was measured and scored nothing, when in fact there was
    nothing to measure.
    """
    if not is_probability(prob):
        return float("nan")
    _require_finite_probabilities(prob, metric_name="f1_at_threshold")
    y_arr, p_arr = _clean(y, prob)
    if y_arr.size == 0:
        return float("nan")
    predicted = apply_decision_threshold(p_arr, threshold=threshold, operator=operator)
    tp, fp, fn, _tn = _confusion_counts(y_arr, predicted)
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return float("nan")
    return float(2 * tp / denominator)


def prevalence(y: Sequence) -> float:
    """Proportion of positive reference labels in the evaluation population.

    A POPULATION STATISTIC, not a metric of predictions: it takes no
    probabilities and no scores, and it is well defined on a single-class cohort
    where the ranking metrics are not. It does NOT filter labels -- the
    population was already constructed upstream by `EvaluationPopulation`, and
    filtering here would silently describe a different denominator than the one
    the result names.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.size == 0:
        return float("nan")
    if not np.isfinite(y_arr).all():
        raise ValueError(
            "prevalence: reference labels contain non-finite entries. Label "
            "eligibility is an upstream population decision; this kernel must "
            "not narrow the denominator it was handed.")
    return float(np.mean(y_arr == 1))


def calibration_slope_intercept(y: Sequence, prob: Sequence,
                                max_iter: int = 100,
                                tol: float = 1e-10) -> CalibrationFit:
    """Fit y ~ sigmoid(intercept + slope * logit(p)) by iteratively-reweighted least squares.

    Perfect calibration -> slope 1.0, intercept 0.0.
    slope < 1 means over-confident; slope > 1 means under-confident.

    Returns a CalibrationFit. A run that did NOT converge returns NaN coefficients and
    `converged=False` -- it does not return the last iterate as though it were an answer.
    `clipped_fraction` reports how much of the input sat at the [1e-6, 1-1e-6] clip, because
    a fit dominated by clipped values is describing the clip, not the model.
    """
    nan_fit = CalibrationFit(float("nan"), float("nan"), False, 0, float("nan"))
    if not is_probability(prob):
        return nan_fit
    y, p = _clean(y, prob)
    if _degenerate(y):
        return nan_fit

    lo, hi = 1e-6, 1 - 1e-6
    clipped = float(np.mean((p < lo) | (p > hi))) if p.size else float("nan")
    p = np.clip(p, lo, hi)
    x = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    converged = False
    used = 0
    for i in range(max_iter):
        used = i + 1
        eta = X @ beta
        mu = _sigmoid(eta)
        w = np.maximum(mu * (1 - mu), _EPS)
        z = eta + (y - mu) / w
        XtW = X.T * w
        try:
            beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            return CalibrationFit(float("nan"), float("nan"), False, used, clipped)
        if not np.all(np.isfinite(beta_new)):
            return CalibrationFit(float("nan"), float("nan"), False, used, clipped)
        done = np.max(np.abs(beta_new - beta)) < tol
        beta = beta_new
        if done:
            converged = True
            break
    if not converged:
        return CalibrationFit(float("nan"), float("nan"), False, used, clipped)
    return CalibrationFit(float(beta[1]), float(beta[0]), True, used, clipped)


def bootstrap_ci(fn: Callable, y: Sequence, score: Sequence, *,
                 n_boot: int = 200, alpha: float = 0.05, seed: int = 0,
                 stratified: bool = True):
    """Percentile bootstrap CI over INDEPENDENT ROWS. Stratified by class so pos_rate holds.

    WARNING, RECORDED RATHER THAN IMPLIED: this treats variants as independent. They are not
    -- they cluster within genes, transcripts, protein families, submitting laboratories and
    disease groups. Intervals from this function are ANTI-CONSERVATIVE (too narrow) whenever
    that clustering carries signal. For any gene-disjoint claim use `cluster_bootstrap_ci`.
    This function is retained because it is the right estimator when rows genuinely are
    exchangeable, and because it is the naive term in the design effect.

    n_boot defaults to 200: each replicate is O(n log n), so 1000 replicates over a
    1.2M-row split is minutes, not seconds. Raise it for a final publication panel.
    """
    _require_finite_scores(score, metric_name="auprc_gain")
    y, s = _clean(y, score)
    if _degenerate(y):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        if stratified:
            i = np.concatenate([rng.choice(pos, pos.size, replace=True),
                                rng.choice(neg, neg.size, replace=True)])
        else:
            i = rng.integers(0, y.size, y.size)
        v = fn(y[i], s[i])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cluster_bootstrap_ci(fn: Callable, y: Sequence, score: Sequence,
                         clusters: Sequence, *,
                         n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                         two_stage: bool = False,
                         return_design_effect: bool = False):
    """Percentile bootstrap resampling WHOLE CLUSTERS -- the correct estimator here.

    Added 2026-07-20. Variants within a gene are correlated: they share the gene's
    constraint, its network position, its curation history and often its true class.
    Resampling variants independently treats each as a fresh observation, which understates
    variance and produces intervals narrower than the data support. Resampling GENES
    preserves the dependence.

    `two_stage=True` additionally resamples variants WITHIN each drawn cluster, which is the
    more conservative design when within-gene sampling variability also matters.

    `return_design_effect=True` returns (lo, hi, design_effect) where design_effect is the
    ratio of this interval's width to the naive row-level interval's width. That number is
    the point: it says HOW MUCH every previously published interval understated its
    uncertainty, so old panels can be re-read rather than merely replaced. A design effect
    of 1.0 means clustering carried no information; values above 1.0 quantify the correction.
    """
    y_arr = np.asarray(y)
    s_arr = np.asarray(score)
    c_arr = np.asarray(clusters)
    if not (len(y_arr) == len(s_arr) == len(c_arr)):
        raise ValueError(
            f"length mismatch: y={len(y_arr)}, score={len(s_arr)}, clusters={len(c_arr)}"
        )

    c_clean = clean_arrays(y_arr, s_arr)
    if _degenerate(c_clean.y):
        return (float("nan"), float("nan"), float("nan")) if return_design_effect \
            else (float("nan"), float("nan"))

    uniq = np.unique(c_arr)
    index_of = {u: np.flatnonzero(c_arr == u) for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=uniq.size, replace=True)
        parts = []
        for u in drawn:
            idx = index_of[u]
            if two_stage and idx.size > 1:
                idx = rng.choice(idx, idx.size, replace=True)
            parts.append(idx)
        i = np.concatenate(parts) if parts else np.empty(0, dtype=int)
        if i.size == 0:
            continue
        v = fn(y_arr[i], s_arr[i])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"), float("nan")) if return_design_effect \
            else (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if not return_design_effect:
        return float(lo), float(hi)
    n_lo, n_hi = bootstrap_ci(fn, y_arr, s_arr, n_boot=min(n_boot, 500),
                              alpha=alpha, seed=seed)
    naive_width = n_hi - n_lo
    de = float((hi - lo) / naive_width) if np.isfinite(naive_width) and naive_width > 0 \
        else float("nan")
    return float(lo), float(hi), de


# ---------------------------------------------------------------------------
# Canonical bootstrap dispatcher. ONE engine, an EXPLICIT resampling unit.
#
# The resampling unit is part of every confidence interval, never an accidental
# consequence of which caller produced it or whether metadata happened to be
# present. Certified (clinical/release) CIs REQUIRE gene-cluster resampling;
# variant-level resampling is available only when the caller explicitly asks for
# it, is flagged not certification-eligible, and records that it assumes row
# independence. There is NO silent fallback between the two -- that would let two
# structurally identical CIs mean different things.
#
# This wraps the proven bootstrap_ci / cluster_bootstrap_ci above; it does not
# reimplement the resampling mathematics.
# ---------------------------------------------------------------------------

DEFAULT_MIN_VALID_REPLICATES = 100

# A second, RELATIVE floor. An absolute floor alone is satisfied by 100 valid
# replicates out of 100,000 requested -- a run in which 99.9 per cent of
# resamples were degenerate and the surviving 0.1 per cent are a biased
# subsample of the ones that happened to contain both classes. Percentiles taken
# from that set are not the percentiles of the sampling distribution. The
# effective floor is therefore the LARGER of the two, computed in
# `_effective_min_valid`.
DEFAULT_MIN_VALID_FRACTION = 0.5


def _effective_min_valid(n_boot: int, min_valid: int, min_valid_fraction: float) -> int:
    """The binding floor on valid replicates: absolute and relative, whichever is higher."""
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError(
            f"min_valid_fraction must lie in [0, 1], got {min_valid_fraction}")
    return max(int(min_valid), int(math.ceil(min_valid_fraction * n_boot)))


class InsufficientSupportError(ValueError):
    """A certified bootstrap was requested without the support it requires."""


@dataclass(frozen=True)
class BootstrapResult:
    """A confidence interval WITH the design that produced it.

    `ci_width_ratio_vs_row` is the ratio of this interval's width to the naive
    row-level interval's width (the kernel's historical "design effect"); it is a
    CI-WIDTH ratio, not a variance ratio. `variance_ratio_vs_row` is its square,
    the approximate classical survey design effect, reported separately so the
    terminology is unambiguous. Both are diagnostic; the cluster interval is the
    inferential output.

    `stratified` records whether the resampling held the class balance fixed. It
    is False for the gene design, which draws whole clusters and cannot hold a
    row-level quantity constant, and True for the row design, which is the
    kernel's declared behaviour. None means no resampling was attempted.

    `min_valid_effective` is the floor `n_valid` had to clear, recorded so a
    withheld interval explains itself without the reader reconstructing the
    policy from two constants and a call-site argument.

    TWO AXES, DELIBERATELY SEPARATE:

        status                  was an interval successfully produced?
        certification_eligible  is that interval admissible for certified claims?

    They are not the same question, and collapsing them is a defect in both
    directions. An exploratory row-level interval is genuinely PRODUCED --
    status OK -- while being inadmissible for a gene-disjoint claim, so it
    carries certification_eligible False and a finding naming the assumption it
    rests on. Conversely an interval withheld for too few valid replicates is
    not merely uncertified: it does not exist, and status says so.
    """

    estimate: float
    lower: float
    upper: float
    confidence_level: float

    resampling_unit: BootstrapUnit
    stratified: bool | None
    n_observations: int
    n_clusters: int | None
    n_requested: int
    n_valid: int
    n_degenerate: int
    min_valid_effective: int
    random_seed: int

    row_ci_width: float | None
    cluster_ci_width: float | None
    ci_width_ratio_vs_row: float | None
    variance_ratio_vs_row: float | None

    certification_eligible: bool
    status: MetricStatus
    finding: str | None


def _cluster_draw(index_lists):
    """Row indices for one CLUSTER replicate: whole clusters, drawn with replacement.

    Mirrors `cluster_bootstrap_ci`: it draws as many cluster labels as there are
    clusters, with replacement, and concatenates every row of each drawn cluster.
    """
    keys_arr = np.array(list(index_lists.keys()), dtype=object)

    def draw(rng):
        drawn = rng.choice(keys_arr, size=keys_arr.size, replace=True)
        parts = [index_lists[k] for k in drawn]
        return np.concatenate(parts) if parts else np.empty(0, dtype=int)

    return draw


def _stratified_row_draw(pos, neg):
    """Row indices for one STRATIFIED ROW replicate: both strata, resampled within.

    Mirrors `bootstrap_ci(stratified=True)`. It matters that BOTH strata are
    always present. An earlier version of this accounting drew two strata from
    {positive, negative} WITH REPLACEMENT, which yields a single-class resample
    half the time and reported roughly 50 per cent of replicates as degenerate
    (measured 0.506 against the theoretical 0.500 on 2026-07-26). No such
    replicate is ever drawn by `bootstrap_ci`; the accounting was describing a
    sampling scheme the kernel does not use, and it withheld sound intervals.
    """
    def draw(rng):
        return np.concatenate([rng.choice(pos, pos.size, replace=True),
                               rng.choice(neg, neg.size, replace=True)])

    return draw


def _count_valid_replicates(fn, y_arr, s_arr, *, draw, n_boot, seed):
    """Replay one resampling design for accounting: valid versus degenerate.

    `draw` is the design's own index generator, so the count describes the loop
    that actually produced the interval rather than an approximation of it. The
    generator is seeded identically to the kernel's, and `Generator.choice`
    consumes the same stream for a given population size, so the replicates
    counted here are the replicates that were taken.
    """
    rng = np.random.default_rng(seed)
    n_valid = 0
    n_degenerate = 0
    for _ in range(n_boot):
        i = draw(rng)
        if i.size == 0:
            n_degenerate += 1
            continue
        yi = y_arr[i]
        if np.unique(yi[np.isfinite(yi)]).size < 2:
            n_degenerate += 1
            continue
        v = fn(y_arr[i], s_arr[i])
        if np.isfinite(v):
            n_valid += 1
        else:
            n_degenerate += 1
    return n_valid, n_degenerate


def bootstrap_metric(
    fn: Callable,
    y: Sequence,
    score: Sequence,
    *,
    clusters: Sequence | None = None,
    unit: BootstrapUnit = BootstrapUnit.GENE,
    confidence_level: float = 0.95,
    n_boot: int = 1000,
    seed: int = 42,
    min_valid: int = DEFAULT_MIN_VALID_REPLICATES,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
) -> BootstrapResult:
    """Canonical bootstrap CI with an explicit, typed resampling unit.

    unit=GENE (default, certified): requires `clusters`; resamples whole clusters.
      Without clusters this RAISES InsufficientSupportError -- it never silently
      falls back to row resampling.
    unit=VARIANT (exploratory): resamples rows (stratified by class); the interval
      is genuinely produced, so its status is OK, but it is never
      certification-eligible and it records the assumption it rests on.

    STATUS, which is about existence, is set independently of
    CERTIFICATION_ELIGIBLE, which is about admissibility:

        OK                 endpoints are finite and cleared the replicate floor
        UNDEFINED          one class present; the metric has no value to bound
        INSUFFICIENT_DATA  too few valid replicates to take percentiles from

    A caller that reads only `certification_eligible` can never mistake an
    exploratory interval for a certified one; a caller that reads only `status`
    can never mistake a withheld interval for a produced one.
    """
    alpha = 1.0 - confidence_level
    y_arr = np.asarray(y)
    s_arr = np.asarray(score)
    n_obs = int(len(y_arr))
    est = fn(*(_clean(y_arr, s_arr)))
    min_valid_eff = _effective_min_valid(n_boot, min_valid, min_valid_fraction)

    if unit is BootstrapUnit.GENE:
        if clusters is None:
            raise InsufficientSupportError(
                "Gene-cluster bootstrap (the certified design) requires gene clusters, "
                "but none were supplied. Pass clusters=..., or request "
                "unit=BootstrapUnit.VARIANT explicitly for an exploratory row-level CI."
            )
        c_arr = np.asarray(clusters)
        if len(c_arr) != n_obs:
            raise ValueError(f"clusters length {len(c_arr)} != n_observations {n_obs}")
        lo, hi, de = cluster_bootstrap_ci(
            fn, y_arr, s_arr, c_arr, n_boot=n_boot, alpha=alpha, seed=seed,
            return_design_effect=True,
        )
        # accounting loop (same cluster structure)
        uniq = np.unique(c_arr)
        index_of = {u: np.flatnonzero(c_arr == u) for u in uniq}
        n_valid, n_degen = _count_valid_replicates(
            fn, y_arr, s_arr, draw=_cluster_draw(index_of), n_boot=n_boot, seed=seed)
        n_lo, n_hi = bootstrap_ci(fn, y_arr, s_arr, n_boot=min(n_boot, 500), alpha=alpha, seed=seed)
        row_w = (n_hi - n_lo) if (np.isfinite(n_lo) and np.isfinite(n_hi)) else None
        clus_w = (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else None
        var_ratio = (de * de) if (de is not None and np.isfinite(de)) else None

        if clus_w is None:
            # Every replicate was degenerate, or the cohort carries one class.
            # The metric has no value to bound, which is UNDEFINED rather than
            # unsupported: the machinery ran and the mathematics has no answer.
            status = MetricStatus.UNDEFINED
            finding = "cluster_bootstrap_degenerate"
        elif n_valid < min_valid_eff:
            # The cohort is admissible and the machinery ready; there are simply
            # too few surviving replicates to take percentiles from. That is
            # INSUFFICIENT_DATA, distinct from INSUFFICIENT_SUPPORT, which would
            # say the certified design was not available at all.
            status = MetricStatus.INSUFFICIENT_DATA
            finding = (f"only {n_valid} of {n_boot} replicates were valid "
                       f"(floor {min_valid_eff}); certified interval withheld")
        else:
            status = MetricStatus.OK
            finding = None

        return BootstrapResult(
            estimate=float(est), lower=float(lo), upper=float(hi),
            confidence_level=confidence_level, resampling_unit=BootstrapUnit.GENE,
            stratified=False,
            n_observations=n_obs, n_clusters=int(uniq.size),
            n_requested=n_boot, n_valid=n_valid, n_degenerate=n_degen,
            min_valid_effective=min_valid_eff, random_seed=seed,
            row_ci_width=row_w, cluster_ci_width=clus_w,
            ci_width_ratio_vs_row=(float(de) if de is not None and np.isfinite(de) else None),
            variance_ratio_vs_row=var_ratio,
            certification_eligible=(status is MetricStatus.OK),
            status=status, finding=finding,
        )

    # VARIANT (exploratory)
    lo, hi = bootstrap_ci(fn, y_arr, s_arr, n_boot=n_boot, alpha=alpha, seed=seed)
    y_clean, s_clean = _clean(y_arr, s_arr)
    pos = np.flatnonzero(y_clean == 1); neg = np.flatnonzero(y_clean == 0)
    n_valid, n_degen = _count_valid_replicates(
        fn, y_clean, s_clean, draw=_stratified_row_draw(pos, neg),
        n_boot=n_boot, seed=seed)
    row_w = (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else None

    if row_w is None:
        status = MetricStatus.UNDEFINED
        finding = "row_bootstrap_degenerate"
    elif n_valid < min_valid_eff:
        status = MetricStatus.INSUFFICIENT_DATA
        finding = (f"only {n_valid} of {n_boot} replicates were valid "
                   f"(floor {min_valid_eff}); exploratory interval withheld")
    else:
        # The interval EXISTS and is reported. It is simply not admissible for a
        # gene-disjoint claim, which is the certification axis, not the status
        # axis. Marking it INSUFFICIENT_SUPPORT here would say no interval was
        # produced, which is false, and would make a produced interval
        # indistinguishable from a withheld one.
        status = MetricStatus.OK
        finding = "variant_level_resampling_assumes_row_independence"

    return BootstrapResult(
        estimate=float(est), lower=float(lo), upper=float(hi),
        confidence_level=confidence_level, resampling_unit=BootstrapUnit.VARIANT,
        stratified=True,
        n_observations=n_obs, n_clusters=None,
        n_requested=n_boot, n_valid=n_valid, n_degenerate=n_degen,
        min_valid_effective=min_valid_eff, random_seed=seed,
        row_ci_width=row_w, cluster_ci_width=None,
        ci_width_ratio_vs_row=None, variance_ratio_vs_row=None,
        certification_eligible=False,
        status=status, finding=finding,
    )


def evaluate(y: Sequence, score: Sequence, *,
             prob=None, n_boot: int = 0, seed: int = 0, n_bins: int = 10,
             clusters: Sequence | None = None) -> dict:
    """LEGACY SURVIVOR-FILTERING INTERFACE. NOT A CERTIFIABLE PATH.

    This function constructs its own population by calling `clean_arrays` and then
    computes over the SURVIVORS, reporting `n_input`, `n`, `n_dropped` and
    `dropped_fraction` so the narrowing is visible. Visible narrowing is
    population-accounting TRANSPARENCY; it is not fail-closed behaviour, and this
    function must not be cited as evidence that strict kernels tolerate filtering.
    A non-finite predicted probability here still yields a number over the rows
    that survived.

    The certifiable path is the registry, which refuses non-finite predictions
    before dispatch and returns a FAILED result over the full attempted
    population. This function is retained UNCHANGED for compatibility: its
    callers depend on exact dictionary keys and bare-float values. Whether it is
    frozen permanently as historical compatibility or gains a strict mode is a
    deliberate decision for its own commit, not an incidental change.

    Full single-population panel.

    `score` ranks; `prob` (default: score) is used for calibration. Pass both when the
    ranking score is not a probability.

    DEFECT A is fixed here: y, score and prob are now cleaned on ONE joint mask, and the
    panel reports how many rows that mask removed. Pass `clusters` (gene symbols, typically)
    to get gene-cluster confidence intervals and the design effect alongside the naive ones.
    """
    c = clean_arrays(y, score, score if prob is None else prob)
    y_c, s_c, p = c.y, c.score, c.probability
    base = no_skill_auprc(y_c)
    ap = auprc(y_c, s_c)
    # CALIBRATION VALIDITY IS NOT MERELY "ARE THESE PROBABILITIES?".
    #
    # This gated on is_probability(p) alone until 2026-07-21, which let a single-class
    # cohort through. Measured on y = [1,1,1,1], p = [.9,.8,.85,.95]:
    #
    #     auroc NaN   auprc NaN            <- correct, ranking is undefined
    #     cal_slope NaN   cal_intercept NaN
    #     brier 0.01875   ece 0.125        <- NUMBERS
    #     calibration_valid True           <- asserting those numbers are sound
    #
    # That ECE of 0.125 is just 1 - 0.875: the gap between the mean prediction and the
    # only label present. It says nothing about calibration across the probability
    # range, because the reliability diagram has a single occupied row. The flag's own
    # documented invariant -- "False => brier/log_loss/ece/cal_* are NaN by design" --
    # was already violated in the other direction: cal_slope and cal_intercept were NaN
    # while the flag read True, so a reader would take an undefined estimand for a
    # failed computation.
    #
    # Both classes present is a HARD requirement; without it the quantity is not
    # calibration. Thin support is REPORTED, not refused -- refusing what the
    # predecessor accepted is the regression that broke this suite earlier today on a
    # 427-row cohort, and small fixtures must keep working. DEFAULT_MIN_POS/NEG are the
    # same floors stratified_evaluate already applies per subgroup, so identical data
    # was being called insufficient as a stratum and sound on its own.
    n_pos_c = int(y_c.sum())
    n_neg_c = int(y_c.size - n_pos_c)
    # Ordered most specific first. An empty cohort's problem is that it is
    # empty, not that its (nonexistent) values are not probabilities -- and
    # is_probability([]) is correctly False, so a naive ordering reported the
    # less useful of two true reasons.
    if y_c.size == 0:
        cal_ok, cal_support = False, "insufficient_rows"
    elif not is_probability(p):
        cal_ok, cal_support = False, "not_probabilities"
    elif n_pos_c == 0 or n_neg_c == 0:
        cal_ok, cal_support = False, f"single_class:pos={n_pos_c},neg={n_neg_c}"
    elif n_pos_c < DEFAULT_MIN_POS or n_neg_c < DEFAULT_MIN_NEG:
        cal_ok = True
        cal_support = (f"thin:pos={n_pos_c}(min {DEFAULT_MIN_POS}),"
                       f"neg={n_neg_c}(min {DEFAULT_MIN_NEG})")
    else:
        cal_ok, cal_support = True, "sufficient"
    fit = calibration_slope_intercept(y_c, p)
    out = {
        "n": c.n,
        "n_pos": int(y_c.sum()),
        "n_input": c.n_input,
        "n_dropped": c.n_dropped,
        "dropped_fraction": c.dropped_fraction,
        "pos_rate": base,
        "auroc": auroc(y_c, s_c),
        "auprc": ap,
        "auprc_no_skill": base,
        "auprc_lift": (ap / base) if (base and np.isfinite(ap)) else float("nan"),
        "auprc_gain": (ap - base) if np.isfinite(ap) else float("nan"),
        # ENFORCED, not merely documented. Until 2026-07-21 the comment below
        # promised "False => ... NaN by design" while ece and brier came back as
        # numbers for a single-class cohort, because brier_score/ECE only
        # self-guard on is_probability. A flag that does not enforce what it
        # asserts is the same defect it exists to prevent.
        "brier": brier_score(y_c, p) if cal_ok else float("nan"),
        "log_loss": log_loss(y_c, p) if cal_ok else float("nan"),
        "ece": (expected_calibration_error(y_c, p, n_bins=n_bins)
                if cal_ok else float("nan")),
        "cal_slope": fit.slope,
        "cal_intercept": fit.intercept,
        "cal_converged": fit.converged,
        "cal_clipped_fraction": fit.clipped_fraction,
        "calibration_valid": cal_ok,   # False => brier/log_loss/ece/cal_* are NaN by design
        # WHY, machine-readable: "sufficient" | "thin:..." | "single_class:..." |
        # "not_probabilities". A bare boolean cannot separate "these are not
        # probabilities" from "this cohort has one class" from "computed, but on three
        # positives", and a reader must respond differently to each.
        "calibration_support": cal_support,
    }
    if n_boot:
        out["auroc_ci95"] = bootstrap_ci(auroc, y_c, s_c, n_boot=n_boot, seed=seed)
        out["auprc_ci95"] = bootstrap_ci(auprc, y_c, s_c, n_boot=n_boot, seed=seed)
        if clusters is not None:
            cl = np.asarray(clusters).ravel()
            if cl.size != c.n_input:
                raise ValueError(
                    f"clusters has {cl.size} rows, inputs have {c.n_input}"
                )
            cl = cl[c.mask]     # the SAME mask the arrays were cleaned on
            lo, hi, de = cluster_bootstrap_ci(
                auroc, y_c, s_c, cl, n_boot=n_boot, seed=seed,
                return_design_effect=True)
            out["auroc_ci95_cluster"] = (lo, hi)
            out["auroc_design_effect"] = de
            lo, hi, de = cluster_bootstrap_ci(
                auprc, y_c, s_c, cl, n_boot=n_boot, seed=seed,
                return_design_effect=True)
            out["auprc_ci95_cluster"] = (lo, hi)
            out["auprc_design_effect"] = de
    return out


_INSUFFICIENT = {
    "auroc": float("nan"), "auprc": float("nan"), "auprc_no_skill": float("nan"),
    "auprc_lift": float("nan"), "auprc_gain": float("nan"), "brier": float("nan"),
    "log_loss": float("nan"), "ece": float("nan"), "cal_slope": float("nan"),
    "cal_intercept": float("nan"), "cal_converged": False,
    "cal_clipped_fraction": float("nan"), "calibration_valid": False,
    "calibration_support": "insufficient_rows",
}

MISSING_STRATUM = "__MISSING__"


def stratified_evaluate(y: Sequence, score: Sequence, groups: Iterable, *,
                        prob=None, n_boot: int = 0, seed: int = 0,
                        min_n: int = DEFAULT_MIN_N,
                        min_pos: int = DEFAULT_MIN_POS,
                        min_neg: int = DEFAULT_MIN_NEG) -> "pd.DataFrame":
    """One panel per group, plus an ALL row.

    DEFECT F. Rows whose group label is missing previously vanished: the loop iterated
    `g.dropna().unique()`, so they counted in ALL and appeared in NO stratum, and the strata
    did not partition the cohort. They are now reported under `__MISSING__`. In this project
    missingness is informative -- a variant with no consequence annotation is a different
    object from one annotated as missense -- so dropping it destroys a finding.

    DEFECT G. Sufficiency tested total n only, so a stratum of 1,000 rows containing ONE
    positive passed `min_n=30` and was reported as though its AUPRC and calibration slope
    meant something. Positives and negatives now have their own floors, and an insufficient
    stratum is reported with a `status` naming which floor it failed -- never dropped, never
    given numbers.

    The returned frame carries a `status` column: 'ok', or 'insufficient:<reason>'.
    """
    y_arr = np.asarray(y).ravel()
    s_arr = np.asarray(score, dtype=float).ravel()
    p_arr = s_arr if prob is None else np.asarray(prob, dtype=float).ravel()
    g = pd.Series(list(groups), dtype="object")
    if not (len(y_arr) == len(s_arr) == len(g) == len(p_arr)):
        raise ValueError(
            f"length mismatch y={len(y_arr)} s={len(s_arr)} g={len(g)} p={len(p_arr)}"
        )

    g = g.where(g.notna(), MISSING_STRATUM)

    rows = {"ALL": dict(evaluate(y_arr, s_arr, prob=p_arr, n_boot=n_boot, seed=seed),
                        status="ok")}
    for name in sorted(g.unique(), key=str):
        m = (g == name).to_numpy()
        sub_y = np.asarray(y_arr)[m]
        finite = np.isfinite(np.asarray(sub_y, dtype=float))
        yy = np.asarray(sub_y, dtype=float)[finite]
        n = int(m.sum())
        n_pos = int((yy == 1).sum())
        n_neg = int((yy == 0).sum())
        reasons = []
        if n < min_n:
            reasons.append(f"n<{min_n}")
        if n_pos < min_pos:
            reasons.append(f"pos<{min_pos}")
        if n_neg < min_neg:
            reasons.append(f"neg<{min_neg}")
        if reasons:
            rows[str(name)] = dict(
                _INSUFFICIENT,
                n=n, n_pos=n_pos, n_input=n, n_dropped=int(n - yy.size),
                dropped_fraction=float((n - yy.size) / n) if n else float("nan"),
                pos_rate=float(yy.mean()) if yy.size else float("nan"),
                status="insufficient:" + ",".join(reasons),
            )
            continue
        rows[str(name)] = dict(
            evaluate(y_arr[m], s_arr[m], prob=p_arr[m], n_boot=n_boot, seed=seed),
            status="ok")

    df = pd.DataFrame(rows).T

    # The strata must PARTITION the cohort. Before the __MISSING__ fix they did not, and
    # nothing said so. Asserted rather than assumed.
    stratum_n = int(df.drop(index="ALL")["n_input"].astype(int).sum())
    if stratum_n != len(y_arr):
        raise AssertionError(
            f"strata do not partition the cohort: rows across strata = {stratum_n}, "
            f"input rows = {len(y_arr)}. Every row belongs to exactly one stratum."
        )

    for c in df.columns:
        if c not in ("auroc_ci95", "auprc_ci95", "auroc_ci95_cluster",
                     "auprc_ci95_cluster", "calibration_valid", "cal_converged",
                     "status"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


if __name__ == "__main__":
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 200)
    y_proba = 0.6 * y_true + 0.4 * np.random.beta(2, 5, 200)
    y_proba = np.clip(y_proba, 0, 1)
    evaluator = ModelEvaluator(y_true, y_proba)
    print(evaluator.generate_report())
