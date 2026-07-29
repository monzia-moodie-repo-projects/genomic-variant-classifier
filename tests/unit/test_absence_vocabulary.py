"""Absence must be explicit, and its causes must not be interchangeable.

WHY THIS VOCABULARY EXISTS
==========================
`dump_strict_json` refuses a non-finite number, correctly. But the flat report
surface had no way to say a value was absent, so the whole file was rejected
rather than the one field being recorded as missing.

Measured 2026-07-29 at `2a1e7f6`, three of five cohorts produced reports that
could not be written at all:

    healthy               PERSISTS
    all-negative          REFUSED   auroc, tpr_curve[0], tpr_curve[1]
    all-positive          REFUSED   auroc, fpr_curve[0], fpr_curve[1]
    constant classifier   PERSISTS
    non-finite input      REFUSED   auroc, auprc, mcc, f1, brier_score

A scientifically valid evaluation over a degenerate cohort could not produce an
artifact.

WHY CURVE-LEVEL AND NOT ELEMENT-LEVEL
--------------------------------------
Measured on the same tree: NO curve in any degenerate cohort mixes valid and
non-finite entries. Every array is entirely clean, entirely non-finite, or empty.

    all-negative   tpr_curve all-bad 2/2; every other curve clean
    all-positive   fpr_curve all-bad 2/2; every other curve clean
    non-finite     all curves EMPTY -- withheld upstream by the input gates

Element-level absence would be a representation for a state that cannot occur.
The decision is measured, not aesthetic.

AND ABSENCE IS PER-CURVE, NOT PER-REPORT. On an all-negative cohort `tpr_curve`
is absent while `fpr_curve`, `precision_curve` and `recall_curve` are all valid.
Marking the report's curves absent wholesale would discard three usable arrays.
"""
from __future__ import annotations

import math

import pytest

from genomic_variant_classifier.evaluation.absence import (
    AbsenceCause,
    CurveAbsence,
    FieldAbsence,
    absence_for_curve,
    absence_for_value,
)

NAN = float("nan")


# --------------------------------------------------------------------------- #
# 1. The causes are not interchangeable
# --------------------------------------------------------------------------- #
def test_the_cause_vocabulary_is_closed_and_exact():
    """Narrow and semantic. A cause nobody can act on differently from another
    is documentation pretending to be a distinction."""
    assert {c.value for c in AbsenceCause} == {
        "undefined_on_cohort",
        "withheld_by_input_gate",
        "insufficient_support",
        "not_applicable",
    }


def test_undefined_and_withheld_are_different_findings():
    """THE DISTINCTION THIS VOCABULARY EXISTS FOR.

    `UNDEFINED_ON_COHORT` is a property of the DATA -- the quantity has no value
    for this cohort, which is a legitimate scientific finding requiring no fix.
    `WITHHELD_BY_INPUT_GATE` is a property of the MODEL OUTPUT -- the inputs were
    refused before computation, which is a defect to investigate.

    Reporting both as "missing" tells a reader to investigate the wrong thing.
    """
    undefined = FieldAbsence(cause=AbsenceCause.UNDEFINED_ON_COHORT,
                             reason="binary_class_support_required")
    withheld = FieldAbsence(cause=AbsenceCause.WITHHELD_BY_INPUT_GATE,
                            reason="nonfinite_probabilities")

    assert undefined.cause is not withheld.cause
    assert undefined.to_dict()["cause"] != withheld.to_dict()["cause"]


@pytest.mark.parametrize("record", [FieldAbsence, CurveAbsence])
def test_a_free_string_cause_is_refused(record):
    """Two artifacts must not be able to describe one situation differently."""
    with pytest.raises(TypeError, match="closed AbsenceCause"):
        record(cause="undefined_on_cohort")


# --------------------------------------------------------------------------- #
# 2. Scalars
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,absent", [
    (0.87, False),
    (0.0, False),
    (-1.5, False),
    (NAN, True),
    (float("inf"), True),
    (float("-inf"), True),
    (None, True),
])
def test_absence_is_detected_for_exactly_the_unpersistable_values(value, absent):
    """Zero and negative values are PRESENT. A detector that treated a legitimate
    0.0 as absent would erase a real measurement -- worse than the defect it
    was written to fix."""
    result = absence_for_value(value, cause=AbsenceCause.UNDEFINED_ON_COHORT)
    assert (result is not None) is absent


def test_the_caller_chooses_the_cause():
    """Only the caller knows why. The same NaN means "undefined on this cohort"
    when the labels are single-class and "withheld by an input gate" when the
    probabilities were unusable -- and the value itself cannot say which."""
    for cause in AbsenceCause:
        assert absence_for_value(NAN, cause=cause).cause is cause


def test_the_reason_survives_into_the_record():
    absence = absence_for_value(NAN, cause=AbsenceCause.UNDEFINED_ON_COHORT,
                                reason="binary_class_support_required")
    assert absence.reason == "binary_class_support_required"
    assert absence.to_dict()["reason"] == "binary_class_support_required"


# --------------------------------------------------------------------------- #
# 3. Curves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("values,absent", [
    ([0.0, 0.5, 1.0], False),
    ([0.0], False),
    ([NAN, NAN], True),
    ([], True),
    (None, True),
    ([0.1, NAN], True),          # a partial curve is not a curve
])
def test_a_curve_is_absent_when_it_cannot_be_persisted(values, absent):
    result = absence_for_curve(values, cause=AbsenceCause.UNDEFINED_ON_COHORT)
    assert (result is not None) is absent


def test_any_non_finite_element_makes_the_whole_curve_absent():
    """MEASURED: no curve mixes valid and non-finite entries, so "any" and
    "every" coincide in practice. "Any" is the safe rule, because a partial
    curve written as though complete would be read as a complete one."""
    partial = absence_for_curve([0.0, 0.25, NAN, 0.75],
                                cause=AbsenceCause.UNDEFINED_ON_COHORT)
    assert partial is not None
    assert partial.n_expected == 4, (
        "the length the curve WOULD have had must be recorded")


def test_an_absent_curve_is_distinguishable_from_an_empty_cohort():
    """`n_expected` is what separates them. An empty curve over an empty cohort
    is not the same as a withheld curve over two hundred rows, and an artifact
    that cannot tell them apart is the silent absence this removes."""
    withheld = absence_for_curve([], cause=AbsenceCause.WITHHELD_BY_INPUT_GATE,
                                 reason="nonfinite_probabilities", n_expected=200)
    empty_cohort = absence_for_curve([], cause=AbsenceCause.INSUFFICIENT_SUPPORT,
                                     reason="empty_cohort", n_expected=0)

    assert withheld.n_expected == 200
    assert empty_cohort.n_expected == 0
    assert withheld.cause is not empty_cohort.cause


def test_a_valid_curve_yields_no_absence_record():
    """Guards the guard: a detector that flagged everything would pass every
    test above and destroy every curve in the artifact."""
    assert absence_for_curve([0.0, 0.5, 1.0],
                             cause=AbsenceCause.UNDEFINED_ON_COHORT) is None


# --------------------------------------------------------------------------- #
# 4. Serialisation shape
# --------------------------------------------------------------------------- #
def test_both_records_serialise_to_json_safe_dictionaries():
    """These end up in an artifact that `dump_strict_json` will inspect, so the
    records themselves must contain nothing it refuses."""
    import json

    field = FieldAbsence(cause=AbsenceCause.UNDEFINED_ON_COHORT,
                         reason="binary_class_support_required", detail="n=200")
    curve = CurveAbsence(cause=AbsenceCause.WITHHELD_BY_INPUT_GATE,
                         reason="nonfinite_probabilities", n_expected=200)

    for payload in (field.to_dict(), curve.to_dict()):
        restored = json.loads(json.dumps(payload))
        assert restored == payload
        for value in payload.values():
            assert value is None or isinstance(value, (str, int)), (
                f"{value!r} is not a JSON scalar the strict writer accepts")


def test_the_two_record_types_have_different_shapes():
    """A scalar and an array are absent in DIFFERENT ways, and one map with
    mixed semantics would invite a reader to treat them alike."""
    field = FieldAbsence(cause=AbsenceCause.NOT_APPLICABLE).to_dict()
    curve = CurveAbsence(cause=AbsenceCause.NOT_APPLICABLE).to_dict()

    assert "detail" in field and "n_expected" not in field
    assert "n_expected" in curve and "detail" not in curve


# --------------------------------------------------------------------------- #
# 5. The measured premise, pinned
# --------------------------------------------------------------------------- #
def test_no_curve_in_a_degenerate_cohort_mixes_valid_and_absent_entries():
    """THE MEASUREMENT THE DESIGN RESTS ON.

    Curve-level absence is correct only because element-level absence would
    describe a state that does not occur. If that ever stops being true, this
    test fails and the design must be revisited rather than quietly extended.
    """
    import contextlib
    import io

    import numpy as np

    from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator

    rng = np.random.default_rng(5)
    n = 200
    mixed = rng.binomial(1, 0.5, n).astype(float)
    good = np.clip(0.5 + 0.25 * (2 * mixed - 1) + rng.normal(0, 0.15, n), 0, 1)

    cohorts = {
        "all_negative": (np.zeros(n), np.full(n, 0.10)),
        "all_positive": (np.ones(n), np.full(n, 0.90)),
        "nonfinite_probabilities": (mixed, np.where(np.arange(n) < 8, NAN, good)),
    }
    curve_names = ("fpr_curve", "tpr_curve", "precision_curve", "recall_curve",
                   "calibration_frac_pos", "calibration_mean_pred")

    for label, (y, p) in cohorts.items():
        with contextlib.redirect_stdout(io.StringIO()):
            report = ClinicalEvaluator(n_bootstrap=0, random_state=42).evaluate(
                y, p, model_name=label)
        for name in curve_names:
            values = np.asarray(getattr(report, name), dtype=float)
            if values.size == 0:
                continue
            n_bad = int((~np.isfinite(values)).sum())
            assert n_bad in (0, values.size), (
                f"{label}.{name} mixes {n_bad} absent of {values.size} entries. "
                "Curve-level absence assumes this cannot happen; element-level "
                "absence is now required and the design must be revisited.")
