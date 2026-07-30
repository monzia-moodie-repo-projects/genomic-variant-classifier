"""The catalogue and the registry must agree, and absences must stay visible.

WHY THIS EXISTS
===============
`project_metrics.txt` specifies sixteen panels. Two are present. The other
fourteen were absent and their absence was INVISIBLE: nothing in the code
recorded that they had been specified, so a reader saw ten registered metrics and
no indication that thirteen more had been asked for.

A missing metric and a metric nobody ever specified look identical. Only one of
them is a gap, and the handoff of 2026-07-20 named the property this file
enforces: Panels I, J and L must "register as unimplemented rather than being
silently absent".

THE TWO DIRECTIONS BOTH MATTER
------------------------------
    catalogued as built but absent from the registry   a false claim
    in the registry but not catalogued                 an undocumented metric

The first tells a reader a metric exists when it does not. The second means the
catalogue has stopped being the catalogue. Both fail here.

AND THE COUNT IS PINNED
-----------------------
The number of unimplemented entries is asserted exactly. It should fall as the
metric programme proceeds, and each fall is a deliberate edit to this file
alongside the implementation -- not a silent drift in either direction.
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.evaluation.catalogue import (
    CATALOGUE,
    MetricDirection,
    MetricStatus,
    SpecifiedMetric,
    catalogue_names,
    implemented_names,
    unimplemented_names,
)
from genomic_variant_classifier.evaluation.registry import by_name, names


# --------------------------------------------------------------------------- #
# 1. The catalogue and the registry agree
# --------------------------------------------------------------------------- #
def test_every_metric_the_catalogue_calls_built_is_actually_registered():
    """A false claim of existence is the worse of the two failures: it sends a
    reader looking for a metric that is not there."""
    missing = sorted(set(implemented_names()) - set(names()))
    assert not missing, (
        f"the catalogue marks {missing} IMPLEMENTED but the registry does not "
        "have them. Either the metric was removed and the catalogue not updated, "
        "or the status was set before the work landed.")


def test_every_registered_metric_appears_in_the_catalogue():
    """Otherwise the catalogue has quietly stopped being the catalogue."""
    undocumented = sorted(set(names()) - set(catalogue_names()))
    assert not undocumented, (
        f"{undocumented} are registered but absent from the catalogue. A metric "
        "that computes without a declared formula, range and direction cannot be "
        "checked against its definition or displayed correctly.")


def test_the_two_sets_are_disjoint_and_exhaustive():
    built = set(implemented_names())
    absent = set(unimplemented_names())
    assert not (built & absent), "a metric cannot be both built and absent"
    assert built | absent == set(catalogue_names())


# --------------------------------------------------------------------------- #
# 2. Absences stay visible
# --------------------------------------------------------------------------- #
def test_thirteen_specified_metrics_remain_unbuilt():
    """PINNED EXACTLY, so the number moves only by deliberate edit.

    Measured 2026-07-30: TWENTY-FOUR specified, TWENTY-ONE built, three absent.
    The Brier decomposition landed in registry commit 2, and the residual was
    ADDED to the specification because the Murphy identity does not close under
    interval binning.
    The seven confusion-matrix metrics landed in this commit; the count fell from
    thirteen to six in the same change that implemented them. Each
    of the thirteen is a metric the specification asked for and the code does not
    have. As the programme proceeds this number falls, and every fall should be a
    change to this assertion made alongside the implementation.
    """
    absent = unimplemented_names()
    assert len(absent) == 3, (
        f"expected three registered absences, found {len(absent)}: "
        f"{sorted(absent)}. If a metric was implemented, flip its status and "
        "lower this count in the same commit; if one was added to the "
        "specification, raise it.")


def test_the_named_clinical_metrics_are_all_accounted_for():
    """The handoff names these explicitly. Each must be in the catalogue with
    SOME status -- present or absent -- rather than unmentioned."""
    required = {
        "balanced_accuracy", "matthews_correlation_coefficient", "sensitivity",
        "specificity", "positive_predictive_value", "negative_predictive_value",
        "positive_likelihood_ratio", "negative_likelihood_ratio",
        "partial_auroc", "integrated_calibration_index",
        "adaptive_expected_calibration_error", "maximum_calibration_error",
        "brier_reliability", "brier_resolution", "brier_uncertainty",
    }
    missing = sorted(required - set(catalogue_names()))
    assert not missing, (
        f"{missing} were named in the 2026-07-20 handoff and appear nowhere in "
        "the catalogue. An unmentioned metric is indistinguishable from one "
        "nobody asked for.")


# --------------------------------------------------------------------------- #
# 3. Every entry is usable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entry", CATALOGUE, ids=lambda e: e.name)
def test_every_entry_carries_a_written_formula(entry):
    """Most of these have several conventions in the literature and a name does
    not pick one. The positive likelihood ratio is sensitivity over
    one-minus-specificity, not sensitivity over specificity."""
    assert entry.formula and len(entry.formula) > 10, entry.name


@pytest.mark.parametrize("entry", CATALOGUE, ids=lambda e: e.name)
def test_every_entry_declares_a_direction(entry):
    """A dashboard sorting by value cannot infer this from a name, and a Brier
    score sorted as though higher were better ranks the worst model first."""
    assert isinstance(entry.direction, MetricDirection)


@pytest.mark.parametrize("entry", CATALOGUE, ids=lambda e: e.name)
def test_every_entry_declares_an_ordered_range(entry):
    low, high = entry.value_range
    if low is not None and high is not None:
        assert low < high, f"{entry.name}: range {entry.value_range} not ordered"


def test_the_matthews_coefficient_range_is_signed():
    """A real case where the range cannot be guessed: this one runs from -1 to
    +1 while every probability-scale metric beside it runs from 0 to 1, and a
    caller normalising for display has no way to tell from the name."""
    entry = next(e for e in CATALOGUE
                 if e.name == "matthews_correlation_coefficient")
    assert entry.value_range == (-1.0, 1.0)


def test_brier_resolution_is_the_one_component_where_higher_is_better():
    """The decomposition is brier = reliability - resolution + uncertainty, so
    resolution enters with a negative sign and a larger value is better. Getting
    this backwards would invert a dashboard column."""
    resolution = next(e for e in CATALOGUE if e.name == "brier_resolution")
    reliability = next(e for e in CATALOGUE if e.name == "brier_reliability")
    assert resolution.direction is MetricDirection.HIGHER_IS_BETTER
    assert reliability.direction is MetricDirection.LOWER_IS_BETTER


# --------------------------------------------------------------------------- #
# 4. The catalogue refuses to be malformed
# --------------------------------------------------------------------------- #
def test_an_entry_without_a_formula_is_refused():
    """A metric whose definition is not written down cannot be checked against
    its implementation."""
    with pytest.raises(ValueError, match="formula is required"):
        SpecifiedMetric(
            name="x", display_name="X", formula="",
            value_range=(0.0, 1.0), direction=MetricDirection.HIGHER_IS_BETTER,
            status=MetricStatus.NOT_IMPLEMENTED)


def test_an_unordered_range_is_refused():
    with pytest.raises(ValueError, match="not ordered"):
        SpecifiedMetric(
            name="x", display_name="X", formula="a well described formula",
            value_range=(1.0, 0.0), direction=MetricDirection.HIGHER_IS_BETTER,
            status=MetricStatus.NOT_IMPLEMENTED)


def test_panel_letters_are_unassigned_and_that_is_recorded():
    """DELIBERATE. `project_metrics.txt` assigns the letters, it is 34,678 bytes,
    and it was not available in the session that wrote this catalogue. Every entry
    therefore carries `panel=None` rather than a guess, because a guessed
    assignment would be indistinguishable from a measured one.

    When that document is re-supplied, this test is the one to change.
    """
    assigned = [e.name for e in CATALOGUE if e.panel is not None]
    assert not assigned, (
        f"{assigned} carry panel letters. If project_metrics.txt has been "
        "supplied, assign them from it and update this test to assert the "
        "mapping rather than its absence.")


# --------------------------------------------------------------------------- #
# 5. The implemented entries describe the real descriptors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", implemented_names())
def test_an_implemented_entry_matches_its_descriptor(name):
    """The catalogue's display name must not drift from the registry's."""
    entry = next(e for e in CATALOGUE if e.name == name)
    descriptor = by_name(name)
    assert descriptor is not None
    assert entry.display_name.lower() == descriptor.display_name.lower(), (
        f"{name}: catalogue says {entry.display_name!r}, registry says "
        f"{descriptor.display_name!r}")
