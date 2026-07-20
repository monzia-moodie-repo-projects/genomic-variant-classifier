"""
Evaluation Module for Genomic Variant Classification
Author: Monzia Moodie

=============================================================================
REVISION 2026-07-08 -- METRIC STACK ADDED **ALONGSIDE** THE ORIGINAL API.

Nothing above the "METRIC STACK" banner is changed. `compute_classification_metrics`
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

def compute_classification_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_proba),
        "auprc": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

class ModelEvaluator:
    def __init__(self, y_true, y_proba, threshold=0.5):
        self.y_true = np.array(y_true)
        self.y_proba = np.array(y_proba)
        self.threshold = threshold
        self.y_pred = (self.y_proba >= threshold).astype(int)

    def get_all_metrics(self):
        return {
            "classification": compute_classification_metrics(
                self.y_true, self.y_pred, self.y_proba
            ),
        }

    def generate_report(self):
        metrics = self.get_all_metrics()
        clf = metrics["classification"]
        lines = [
            "=" * 50,
            "MODEL EVALUATION REPORT",
            "=" * 50,
            f"Samples: {len(self.y_true)} ({self.y_true.sum()} positive)",
            f"AUROC: {clf['auroc']:.4f}",
            f"AUPRC: {clf['auprc']:.4f}",
            f"F1: {clf['f1']:.4f}",
            f"Precision: {clf['precision']:.4f}",
            f"Recall: {clf['recall']:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)


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
# NOT changed here: `compute_classification_metrics` and `ModelEvaluator` above this banner.
# They are unsafe in ways this stack explicitly rejects, and unifying them is a separate,
# separately-measured commit. Nothing above this line is edited by the 2026-07-20 work.
# =============================================================================

from dataclasses import dataclass  # noqa: E402
from typing import Callable, Iterable, Iterator, Sequence  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = [
    # original API -- do not remove
    "compute_classification_metrics", "ModelEvaluator",
    # metric stack
    "auroc", "auprc", "auprc_gain", "no_skill_auprc", "brier_score", "log_loss",
    "expected_calibration_error", "calibration_slope_intercept",
    "bootstrap_ci", "cluster_bootstrap_ci", "evaluate", "stratified_evaluate",
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
    """Rank-based AUROC (Mann-Whitney U). Ties get average ranks. NaN if one class."""
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
    y, p = _clean(y, prob)
    if y.size == 0:
        return float("nan")
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def expected_calibration_error(y: Sequence, prob: Sequence, n_bins: int = 10) -> float:
    """Equal-width binning, TOP BIN CLOSED. |accuracy - confidence| weighted by occupancy.

    NaN if `prob` is not a probability. Note the closed top bin: `evaluator.py`'s
    `_calibration_error` uses `(p >= lo) & (p < hi)` with `hi == 1.0`, so every `p == 1.0`
    -- a pure tree leaf -- falls into no bin and is silently excluded, under-reporting ECE
    (86.5% on a 20%-pure-leaf split). See docs/audits/EVALUATION_STACK_AUDIT_2026-07-08.md.
    """
    if not is_probability(prob):
        return float("nan")
    y, p = _clean(y, prob)
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / y.size) * abs(y[m].mean() - p[m].mean())
    return float(ece)


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


def evaluate(y: Sequence, score: Sequence, *,
             prob=None, n_boot: int = 0, seed: int = 0, n_bins: int = 10,
             clusters: Sequence | None = None) -> dict:
    """Full single-population panel.

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
    cal_ok = is_probability(p)
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
        "brier": brier_score(y_c, p),
        "log_loss": log_loss(y_c, p),
        "ece": expected_calibration_error(y_c, p, n_bins=n_bins),
        "cal_slope": fit.slope,
        "cal_intercept": fit.intercept,
        "cal_converged": fit.converged,
        "cal_clipped_fraction": fit.clipped_fraction,
        "calibration_valid": cal_ok,   # False => brier/log_loss/ece/cal_* are NaN by design
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
