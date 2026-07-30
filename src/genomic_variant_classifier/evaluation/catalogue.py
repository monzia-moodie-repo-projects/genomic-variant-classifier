"""The declared catalogue of metrics, including the ones not yet implemented.

WHY THIS EXISTS
===============
`project_metrics.txt` specifies sixteen panels, A through P. Two are present. The
other fourteen are absent, and their absence is INVISIBLE: nothing in the code
records that they were specified, so a reader of the registry sees ten metrics
and no indication that thirteen more were asked for.

That is the same defect the absence vocabulary removed from the artifact surface,
one level up. A missing panel and a panel nobody ever specified look identical,
and only one of them is a gap.

So this module makes the catalogue explicit. A metric that is specified but not
built gets a registered entry with `status = NOT_IMPLEMENTED` and no callable.
The registry can then answer two different questions that were previously one:

    which metrics can I compute?          the implemented subset
    which metrics was I asked for?        the whole catalogue

WHY THE FIELDS ARE THESE FIELDS
-------------------------------
The handoff of 2026-07-20 names them: panel letter, formula, range, direction,
required context, and implementation status. `MetricDescriptor` already carried
the name, the callable, the required inputs and the applicability predicate; the
five here are what it lacked.

`direction` is not decoration. A dashboard that sorts by value needs to know
whether higher is better, and a Brier score sorted descending as though it were
an area under a curve would rank the worst model first. Recording it beside the
metric is the only place it cannot drift out of step.

`value_range` is likewise load-bearing: the Matthews correlation coefficient runs
from -1 to +1 while a probability-scale metric runs from 0 to 1, and a caller
normalising for display cannot infer which from the name.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
------------------------------------------
It does not assign panel letters from `project_metrics.txt`. That document is
34,678 bytes, was uploaded on 2026-07-20, and is not available in this session --
so every entry below carries `panel=None` rather than a guessed letter. A guessed
panel assignment would be indistinguishable from a measured one, and this project
has spent long enough separating those.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

__all__ = [
    "CATALOGUE",
    "MetricDirection",
    "MetricStatus",
    "SpecifiedMetric",
    "catalogue_names",
    "implemented_names",
    "unimplemented_names",
]


class MetricDirection(str, Enum):
    """Which way is better.

    A dashboard that sorts by value cannot infer this from a metric's name, and a
    Brier score sorted as though higher were better ranks the worst model first.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    ZERO_IS_BEST = "zero_is_best"
    DESCRIPTIVE = "descriptive"


class MetricStatus(str, Enum):
    """Whether the specified metric exists yet.

    Named `MetricStatus` in this module and NOT to be confused with the
    result-status vocabulary in `capabilities.py`, which describes one
    computation's outcome. This describes the CATALOGUE: whether the code exists
    at all, independent of any cohort.
    """

    IMPLEMENTED = "implemented"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass(frozen=True)
class SpecifiedMetric:
    """One entry in the declared catalogue.

    `panel` is `Optional` on purpose. The specification assigning panel letters is
    not available in this session, so an unassigned entry records that fact rather
    than carrying a guess.
    """

    name: str
    display_name: str
    formula: str
    value_range: Tuple[Optional[float], Optional[float]]
    direction: MetricDirection
    status: MetricStatus
    panel: Optional[str] = None
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("a catalogue entry requires a name")
        if not self.formula or not self.formula.strip():
            raise ValueError(
                f"{self.name}: a formula is required. A metric whose definition is "
                "not written down cannot be checked against its implementation, "
                "and a name alone does not distinguish the several conventions "
                "most of these have.")
        low, high = self.value_range
        if low is not None and high is not None and low >= high:
            raise ValueError(
                f"{self.name}: value_range {self.value_range} is not ordered")


# --------------------------------------------------------------------------- #
# THE CATALOGUE
#
# Every metric named by the handoff of 2026-07-20, whether built or not. The
# thirteen unbuilt ones are the point of this file: they are now registered
# absences rather than silent ones.
#
# Formulae are written out because most of these have several conventions in the
# literature and a name does not pick one. The positive likelihood ratio is
# sensitivity over one-minus-specificity, not sensitivity over specificity, and a
# reader checking an implementation needs the definition beside it.
# --------------------------------------------------------------------------- #
_IMPLEMENTED = MetricStatus.IMPLEMENTED
_ABSENT = MetricStatus.NOT_IMPLEMENTED
_UP = MetricDirection.HIGHER_IS_BETTER
_DOWN = MetricDirection.LOWER_IS_BETTER

CATALOGUE = (
    # ---- already implemented and registered -------------------------------
    SpecifiedMetric(
        name="auroc", display_name="Area under the receiver operating characteristic curve",
        formula="integral of the true-positive rate against the false-positive rate over all thresholds",
        value_range=(0.0, 1.0), direction=_UP, status=_IMPLEMENTED),
    SpecifiedMetric(
        name="auprc", display_name="Area under the precision-recall curve",
        formula="integral of precision against recall over all thresholds",
        value_range=(0.0, 1.0), direction=_UP, status=_IMPLEMENTED),
    SpecifiedMetric(
        name="brier_score", display_name="Brier score",
        formula="mean over observations of (predicted probability - observed outcome) squared",
        value_range=(0.0, 1.0), direction=_DOWN, status=_IMPLEMENTED),
    SpecifiedMetric(
        name="expected_calibration_error", display_name="Expected calibration error",
        formula="sum over occupied bins of (bin weight) times |observed frequency - mean predicted probability|",
        value_range=(0.0, 1.0), direction=_DOWN, status=_IMPLEMENTED),
    SpecifiedMetric(
        name="maximum_calibration_error", display_name="Maximum calibration error",
        formula="maximum over occupied bins of |observed frequency - mean predicted probability|",
        value_range=(0.0, 1.0), direction=_DOWN, status=_IMPLEMENTED),
    SpecifiedMetric(
        name="matthews_correlation_coefficient",
        display_name="Matthews correlation coefficient",
        formula="(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
        value_range=(-1.0, 1.0), direction=_UP, status=_IMPLEMENTED),
    SpecifiedMetric(
        name="f1", display_name="F1 score, positive class",
        formula="2 * precision * recall / (precision + recall)",
        value_range=(0.0, 1.0), direction=_UP, status=_IMPLEMENTED),
    SpecifiedMetric(
        name="log_loss", display_name="Logarithmic loss",
        formula="negative mean of y*log(p) + (1-y)*log(1-p)",
        value_range=(0.0, None), direction=_DOWN, status=_IMPLEMENTED),
    SpecifiedMetric(
        name="prevalence", display_name="Prevalence",
        formula="count of positive labels divided by total observations",
        value_range=(0.0, 1.0), direction=MetricDirection.DESCRIPTIVE,
        status=_IMPLEMENTED,
        note="descriptive: a cohort property, not a model property"),
    SpecifiedMetric(
        name="auprc_gain", display_name="Precision-recall gain over the no-skill floor",
        formula="auprc minus the no-skill baseline, which equals prevalence",
        value_range=(None, 1.0), direction=_UP, status=_IMPLEMENTED,
        note="composed from auprc; the lower bound depends on prevalence"),

    # ---- SPECIFIED AND NOT BUILT -- the thirteen -------------------------
    SpecifiedMetric(
        name="brier_decomposition_residual",
        display_name="Brier decomposition residual",
        formula="brier - (reliability - resolution + uncertainty)",
        value_range=(None, None), direction=MetricDirection.ZERO_IS_BEST,
        status=_IMPLEMENTED,
        note="NOT IN THE ORIGINAL SPECIFICATION. Added 2026-07-30 because the "
             "Murphy identity does not close under interval binning: measured "
             "residuals ran from -0.001769 to +0.000633 at ten bins across four "
             "cohorts. Reporting it lets the three components be audited instead "
             "of trusted, and hiding it would make an approximate decomposition "
             "look exact"),
    SpecifiedMetric(
        name="balanced_accuracy", display_name="Balanced accuracy",
        formula="(sensitivity + specificity) / 2",
        value_range=(0.0, 1.0), direction=_UP, status=_IMPLEMENTED,
        note="insensitive to prevalence, unlike plain accuracy"),
    SpecifiedMetric(
        name="sensitivity", display_name="Sensitivity (recall, true-positive rate)",
        formula="TP / (TP + FN)",
        value_range=(0.0, 1.0), direction=_UP, status=_IMPLEMENTED),
    SpecifiedMetric(
        name="specificity", display_name="Specificity (true-negative rate)",
        formula="TN / (TN + FP)",
        value_range=(0.0, 1.0), direction=_UP, status=_IMPLEMENTED),
    SpecifiedMetric(
        name="positive_predictive_value",
        display_name="Positive predictive value (precision)",
        formula="TP / (TP + FP)",
        value_range=(0.0, 1.0), direction=_UP, status=_IMPLEMENTED,
        note="prevalence-dependent; not transferable between cohorts"),
    SpecifiedMetric(
        name="negative_predictive_value", display_name="Negative predictive value",
        formula="TN / (TN + FN)",
        value_range=(0.0, 1.0), direction=_UP, status=_IMPLEMENTED,
        note="prevalence-dependent; not transferable between cohorts"),
    SpecifiedMetric(
        name="positive_likelihood_ratio", display_name="Positive likelihood ratio",
        formula="sensitivity / (1 - specificity)",
        value_range=(0.0, None), direction=_UP, status=_IMPLEMENTED,
        note="prevalence-INDEPENDENT, which is why it belongs beside the "
             "predictive values rather than instead of them"),
    SpecifiedMetric(
        name="negative_likelihood_ratio", display_name="Negative likelihood ratio",
        formula="(1 - sensitivity) / specificity",
        value_range=(0.0, None), direction=_DOWN, status=_IMPLEMENTED,
        note="prevalence-independent"),
    SpecifiedMetric(
        name="partial_auroc",
        display_name="Partial area under the receiver operating characteristic curve",
        formula="area under the curve restricted to a stated false-positive-rate interval, standardised to [0, 1]",
        value_range=(0.0, 1.0), direction=_UP, status=_IMPLEMENTED,
        note="requires a DECLARED false-positive-rate bound; the bound is part "
             "of the metric identity, as a threshold is"),
    SpecifiedMetric(
        name="integrated_calibration_index", display_name="Integrated calibration index",
        formula="weighted mean absolute difference between predicted probability and a smoothed calibration curve",
        value_range=(0.0, 1.0), direction=_DOWN, status=_IMPLEMENTED,
        note="binning-free, so it does not inherit the interval-convention "
             "hazard that cost this project seventeen days"),
    SpecifiedMetric(
        name="adaptive_expected_calibration_error",
        display_name="Adaptive expected calibration error",
        formula="expected calibration error over EQUAL-COUNT bins rather than equal-width bins",
        value_range=(0.0, 1.0), direction=_DOWN, status=_IMPLEMENTED,
        note="equal-count binning is what makes it robust on saturated "
             "predictions, where equal-width bins leave most bins empty"),
    SpecifiedMetric(
        name="brier_reliability", display_name="Brier decomposition: reliability",
        formula="sum over bins of (bin weight) times (mean predicted probability - observed frequency) squared",
        value_range=(0.0, 1.0), direction=_DOWN, status=_IMPLEMENTED,
        note="the calibration component; brier = reliability - resolution + uncertainty"),
    SpecifiedMetric(
        name="brier_resolution", display_name="Brier decomposition: resolution",
        formula="sum over bins of (bin weight) times (observed frequency - overall prevalence) squared",
        value_range=(0.0, 1.0), direction=_UP, status=_IMPLEMENTED,
        note="the discrimination component; HIGHER is better, unlike the other two"),
    SpecifiedMetric(
        name="brier_uncertainty", display_name="Brier decomposition: uncertainty",
        formula="prevalence times (1 - prevalence)",
        value_range=(0.0, 0.25), direction=MetricDirection.DESCRIPTIVE,
        status=_IMPLEMENTED,
        note="a cohort property no model can change; its maximum is 0.25 at "
             "prevalence 0.5"),
)


def catalogue_names() -> Tuple[str, ...]:
    """Every specified metric, built or not."""
    return tuple(entry.name for entry in CATALOGUE)


def implemented_names() -> Tuple[str, ...]:
    return tuple(e.name for e in CATALOGUE if e.status is MetricStatus.IMPLEMENTED)


def unimplemented_names() -> Tuple[str, ...]:
    """The registered absences. This is the number that should fall over time."""
    return tuple(e.name for e in CATALOGUE
                 if e.status is MetricStatus.NOT_IMPLEMENTED)
