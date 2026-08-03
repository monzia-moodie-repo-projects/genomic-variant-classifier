"""The population surface on the report: schema version five (POP-1b).

POP-1a made `n_samples` the LABEL-ELIGIBLE count and added no fields, so a reader
of a version-4 artifact could not tell a smaller cohort from a narrowed one.
Version 5 says which, by how much, and what the narrowing narrowed FROM.

FIXTURES ARE DERIVED FROM REAL REPORTS, NEVER HAND-BUILT. `EvaluationReport` has
fifty-six fields, thirty-seven of them required; a first version of this file
constructed reports from a six-key dictionary and six tests died at
construction, before reaching anything they meant to assert. Hand-listing
thirty-seven fields would have produced a fixture that goes stale the moment a
field is added -- the same failure, rebuilt slightly larger.

So the invariant tests use `dataclasses.replace()` on a real report, carrying
everything over unchanged and overriding only what is under test; and the
historical-artifact tests build a payload from a real report and DELETE the five
new keys, which is exactly what a version-4 artifact looks like on disk.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.evaluator import (
    EVALUATION_REPORT_SCHEMA_VERSION_ABSENCE,
    EVALUATION_REPORT_SCHEMA_VERSION_POPULATION,
    SUPPORTED_REPORT_SCHEMA_VERSIONS,
    ClinicalEvaluator,
    EvaluationReport,
)
from genomic_variant_classifier.evaluation.population import EvaluationPopulation

SOURCE = "population-surface-fixture:sha256:0000000000000000"
NAN = float("nan")

WITHHELD_Y = np.array([1.0, 1.0, 0.0, NAN])
WITHHELD_P = np.array([0.9, 0.1, 0.2, 0.8])
FULL_Y = np.array([1.0, 1.0, 0.0, 0.0])
FULL_P = np.array([0.9, 0.6, 0.2, 0.1])

POPULATION_FIELDS = (
    "n_source",
    "n_label_eligible",
    "n_reference_label_withheld",
    "population_scope",
    "population_parent_fingerprint",
)


def _evaluator() -> ClinicalEvaluator:
    """Few replicates: no assertion here depends on the bootstrap count."""
    return ClinicalEvaluator(n_bootstrap=10, random_state=0)


def _real_report(y=None, p=None) -> EvaluationReport:
    """A genuine report. Nothing in this file constructs one by hand."""
    return _evaluator().evaluate(
        FULL_Y if y is None else y,
        FULL_P if p is None else p,
        source_id=SOURCE)


def _historical_payload(version: int) -> dict:
    """What a pre-POP-1b artifact looks like ON DISK.

    A real report, serialised, with the five new keys REMOVED and the version
    set back. Built this way rather than hand-written so it cannot go stale when
    the dataclass grows -- and so it exercises the real payload shape, all
    fifty-six fields of it, rather than a stub that resembles one.
    """
    payload = json.loads(json.dumps(_real_report().to_serializable()))
    for name in POPULATION_FIELDS:
        payload.pop(name, None)
    payload["schema_version"] = version
    return payload


# --------------------------------------------------------------------------- #
# A1 -- the five fields on a narrowed cohort
# --------------------------------------------------------------------------- #

def test_a_narrowed_cohort_names_its_population():
    """THE ACCEPTANCE CRITERION FOR POP-1b.

    Before this commit the artifact carried a population FINGERPRINT with
    nothing beside it saying what the population WAS, so a reader could not
    distinguish a four-row cohort from a four-row cohort narrowed to three.
    """
    report = _real_report(WITHHELD_Y, WITHHELD_P)

    assert report.schema_version == EVALUATION_REPORT_SCHEMA_VERSION_POPULATION
    assert report.n_source == 4
    assert report.n_label_eligible == 3
    assert report.n_reference_label_withheld == 1
    assert report.population_scope == "label_eligible"
    assert report.population_parent_fingerprint is not None


def test_the_parent_fingerprint_is_the_attempted_populations():
    """Constructed independently rather than compared against a literal: a
    hard-coded digest would pin the fixture, not the property."""
    attempted = EvaluationPopulation.full(
        WITHHELD_Y.size, scope="attempted_cohort", source_id=SOURCE)

    report = _real_report(WITHHELD_Y, WITHHELD_P)
    assert report.population_parent_fingerprint == attempted.membership_fingerprint


def test_an_unrestricted_cohort_has_no_parent_to_point_at():
    """None here means "nothing was narrowed", NOT "the population is
    unattributed". The two are different conditions with the same value, and
    collapsing them would make the field unreadable as evidence."""
    report = _real_report()

    assert report.n_source == 4
    assert report.n_label_eligible == 4
    assert report.n_reference_label_withheld == 0
    assert report.population_scope == "attempted_cohort"
    assert report.population_parent_fingerprint is None


# --------------------------------------------------------------------------- #
# A3 -- the invariant, including constructions that must be refused
# --------------------------------------------------------------------------- #

def test_the_three_counts_must_reconcile():
    """A field trio that must sum is exactly the shape that rots silently: any
    one can drift and nothing notices. AN INVARIANT NOTHING CAN VIOLATE IN A
    TEST IS AN INVARIANT NOTHING CHECKS, so this constructs a violation.

    `dataclasses.replace` carries all fifty-six fields over unchanged and
    overrides only the three under test, so the test states exactly what it
    varies -- and it calls `__init__`, so `__post_init__` really runs.
    """
    report = _real_report()
    with pytest.raises(ValueError, match="do not reconcile"):
        dataclasses.replace(report, n_source=4, n_label_eligible=3,
                            n_reference_label_withheld=2)


def test_partially_recorded_counts_are_refused():
    """All three are sentinels, or all three are measurements. A mixture means
    somebody set two and forgot the third, which is the drift the invariant
    exists to catch at its earliest moment."""
    report = _real_report()
    with pytest.raises(ValueError, match="partially recorded"):
        dataclasses.replace(report, n_source=4, n_label_eligible=3,
                            n_reference_label_withheld=-1)


def test_reconciling_counts_are_accepted():
    """The invariant must not simply reject everything."""
    report = _real_report()
    adjusted = dataclasses.replace(report, n_source=9, n_label_eligible=7,
                                   n_reference_label_withheld=2)
    assert adjusted.n_source == 9


# --------------------------------------------------------------------------- #
# A5 -- HISTORICAL ARTIFACTS STILL READ
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("version", [2, 4])
def test_a_pre_population_artifact_still_deserialises(version):
    """THE CRITERION THE WHOLE DESIGN TURNED ON.

    `from_serialized` filters a payload to known field names and calls the
    constructor. A payload written before POP-1b contains none of the five, so
    if they were declared WITHOUT DEFAULTS this would raise TypeError on every
    version-1 through version-4 artifact ever written. The defect is invisible
    in the dataclass and appears only when the deserialisers are read.

    Version 2 routes through `from_serialized_v2`, a DIFFERENT code path from
    version 4. Both are exercised deliberately: if they diverge, that is worth
    knowing.
    """
    payload = _historical_payload(version)
    for name in POPULATION_FIELDS:
        assert name not in payload, f"{name} was not removed from the fixture"

    report = EvaluationReport.from_serialized(payload)

    assert report.n_source == -1
    assert report.n_label_eligible == -1
    assert report.n_reference_label_withheld == -1
    assert report.population_scope is None
    assert report.population_parent_fingerprint is None


def test_the_sentinel_is_negative_and_not_zero():
    """Zero is a legitimate count for a cohort that was attempted and yielded
    nothing. A historical artifact must not be readable as though it recorded a
    measurement it never took, and a negative sentinel cannot be mistaken for
    one."""
    report = EvaluationReport.from_serialized(_historical_payload(4))
    assert report.n_source < 0
    assert report.n_label_eligible < 0
    assert report.n_reference_label_withheld < 0


# --------------------------------------------------------------------------- #
# A4 -- round trip
# --------------------------------------------------------------------------- #

def test_the_five_fields_survive_a_round_trip():
    """`to_serializable` uses `asdict`, so the fields serialise automatically --
    but automatic is a claim, and this is the test that makes it falsifiable."""
    report = _real_report(WITHHELD_Y, WITHHELD_P)
    payload = json.loads(json.dumps(report.to_serializable()))

    for name in POPULATION_FIELDS:
        assert name in payload, f"{name} did not reach the artifact"

    restored = EvaluationReport.from_serialized(payload)
    for name in POPULATION_FIELDS:
        assert getattr(restored, name) == getattr(report, name), name


# --------------------------------------------------------------------------- #
# A7 -- POP-1a is not relitigated
# --------------------------------------------------------------------------- #

def test_n_samples_still_means_the_label_eligible_count():
    """POP-1b adds surface. It does not change what POP-1a decided, and if these
    two ever diverge one of them is wrong."""
    report = _real_report(WITHHELD_Y, WITHHELD_P)
    assert report.n_samples == report.n_label_eligible == 3


def test_the_version_is_the_highest_supported():
    """Formulated so it does NOT need editing at version six. A test that is
    updated at every version asserts nothing durable -- it is a hand-kept number
    wearing a test's clothes.

    The absence maps arrived at version four and every later version still
    carries them, which is why the second assertion is `>=` rather than `==`.
    """
    assert (EVALUATION_REPORT_SCHEMA_VERSION_POPULATION
            == max(SUPPORTED_REPORT_SCHEMA_VERSIONS))
    assert (EVALUATION_REPORT_SCHEMA_VERSION_POPULATION
            >= EVALUATION_REPORT_SCHEMA_VERSION_ABSENCE)
