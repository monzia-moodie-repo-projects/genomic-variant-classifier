"""A degenerate cohort must still produce an artifact, and absence must be explicit.

WHAT WAS WRONG
==============
Measured 2026-07-29 at `594a6af`, before this commit:

    healthy               PERSISTS   03b67304e8e76f17a2d56309  15,511 bytes
    all-negative          REFUSED    auroc, tpr_curve[0], tpr_curve[1]
    all-positive          REFUSED    auroc, fpr_curve[0], fpr_curve[1]
    constant classifier   PERSISTS   0d195a97bf43908b205f1989   6,568 bytes
    non-finite input      REFUSED    auprc, auroc, brier_score, calibration_ece,
                                     calibration_mce, f1, mcc

**Three of five cohorts produced reports that could not be written at all.**
`dump_strict_json` refuses a non-finite number -- correctly -- but the flat
surface had no way to say a value was ABSENT, so the whole file was rejected
rather than the one field being recorded as missing.

THE CAUSE COMES FROM WHERE THE REFUSAL HAPPENED
------------------------------------------------
The same NaN means different things. On a single-class cohort the area under the
receiver operating characteristic curve is UNDEFINED_ON_COHORT -- a property of
the data, and a legitimate finding. When CI-t's input gates refuse unusable
probabilities it is WITHHELD_BY_INPUT_GATE -- a property of the model output, and
a defect to investigate.

Inferring which from the value would be exactly the guess the absence vocabulary
exists to replace, so the cause is threaded from the gate verdict.

THE BICONDITIONAL
-----------------
A scalar is `null` IF AND ONLY IF it is declared absent. Both directions are
enforced because each failure is a different defect:

    null with no entry     a SILENT ABSENCE -- indistinguishable from a value
                           that was never computed
    entry with a value     an ORPHANED CLAIM -- the artifact contradicts itself
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.absence import AbsenceCause
from genomic_variant_classifier.evaluation.evaluator import (
    EVALUATION_REPORT_SCHEMA_VERSION_ABSENCE,
    ClinicalEvaluator,
)
from genomic_variant_classifier.evaluation.serialization import dump_strict_json

N = 200


def _cohorts():
    rng = np.random.default_rng(5)
    mixed = rng.binomial(1, 0.5, N).astype(float)
    good = np.clip(0.5 + 0.25 * (2 * mixed - 1) + rng.normal(0, 0.15, N), 0, 1)
    return {
        "healthy": (mixed, good),
        "all_negative": (np.zeros(N), np.full(N, 0.10)),
        "all_positive": (np.ones(N), np.full(N, 0.90)),
        "constant_classifier": (
            np.concatenate([np.zeros(150), np.ones(50)]), np.full(N, 0.10)),
        "nonfinite_prob": (mixed, np.where(np.arange(N) < 8, np.nan, good)),
    }


def _report(label):
    y, p = _cohorts()[label]
    with contextlib.redirect_stdout(io.StringIO()):
        return ClinicalEvaluator(n_bootstrap=0, random_state=42).evaluate(
            y, p, model_name=label)


def _artifact(label):
    return json.loads(dump_strict_json(_report(label).to_serializable(),
                                       artifact=label))


# --------------------------------------------------------------------------- #
# 1. Every cohort now persists
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", sorted(_cohorts()))
def test_every_cohort_produces_an_artifact(label):
    """THE ACCEPTANCE CRITERION. Three of these five could not be written before
    this commit. An evaluation that records why a metric was unavailable is more
    scientifically useful than no artifact at all."""
    payload = _artifact(label)
    assert payload["model_name"] == label
    assert payload["n_samples"] == N


@pytest.mark.parametrize("label", ["healthy", "constant_classifier"])
def test_a_healthy_report_declares_no_absence(label):
    """Guards the guard: a change that marked everything absent would satisfy
    every test above while destroying the artifact."""
    payload = _artifact(label)
    assert payload["field_absence"] == {}
    assert payload["curve_absence"] == {}
    for name in ("auroc", "auprc", "mcc", "f1", "brier_score", "prevalence"):
        assert payload[name] is not None, f"{name} became absent on a valid cohort"


# --------------------------------------------------------------------------- #
# 2. The cause discriminates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,expected", [
    ("all_negative", AbsenceCause.UNDEFINED_ON_COHORT),
    ("all_positive", AbsenceCause.UNDEFINED_ON_COHORT),
    ("nonfinite_prob", AbsenceCause.WITHHELD_BY_INPUT_GATE),
])
def test_the_cause_names_where_the_refusal_happened(label, expected):
    """THE DISTINCTION THIS COMMIT EXISTS FOR.

    A single-class cohort makes the area under the receiver operating
    characteristic curve genuinely undefined -- nothing is broken and there is
    nothing to fix. Unusable probabilities are a defect in the model output. The
    NaN is identical in both cases; only the gate verdict separates them.
    """
    absence = _report(label).field_absence
    assert "auroc" in absence, "auroc must be declared absent"
    assert absence["auroc"].cause is expected


def test_a_single_class_cohort_marks_only_the_undefined_curve():
    """Absence is PER-CURVE. Marking the report's curves absent wholesale would
    discard three usable arrays."""
    absent = set(_report("all_negative").curve_absence)
    assert "tpr_curve" in absent
    assert "fpr_curve" not in absent
    assert "precision_curve" not in absent
    assert "recall_curve" not in absent


def test_a_withheld_input_marks_every_curve():
    """When the input gates refuse, no curve was computed at all -- a different
    situation from one curve being undefined."""
    absent = set(_report("nonfinite_prob").curve_absence)
    for name in ("fpr_curve", "tpr_curve", "precision_curve", "recall_curve",
                 "calibration_frac_pos", "calibration_mean_pred"):
        assert name in absent, name


def test_the_withheld_scalars_are_exactly_the_measured_set():
    """Pinned from measurement, not from expectation. I described this set as
    five fields; it is SEVEN -- the two calibration errors refuse as well."""
    assert set(_report("nonfinite_prob").field_absence) == {
        "auroc", "auprc", "mcc", "f1", "brier_score",
        "calibration_ece", "calibration_mce"}


def test_prevalence_survives_every_degeneracy():
    """The evidence a refusal would have discarded. Prevalence does not depend on
    the predictions at all, and an artifact that records it plus the reasons the
    predictive metrics were unavailable is worth more than no artifact."""
    for label in _cohorts():
        payload = _artifact(label)
        assert payload["prevalence"] is not None
        assert "prevalence" not in payload["field_absence"]


# --------------------------------------------------------------------------- #
# 3. The biconditional, in BOTH directions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", sorted(_cohorts()))
def test_null_if_and_only_if_declared_absent(label):
    payload = _artifact(label)
    declared = set(payload["field_absence"])
    for name in declared:
        assert payload[name] is None, (
            f"{name} is declared absent but carries a value: the artifact "
            "contradicts itself")
    for name in ("auroc", "auprc", "mcc", "f1", "brier_score",
                 "calibration_ece", "calibration_mce", "prevalence"):
        if payload[name] is None:
            assert name in declared, (
                f"{name} is null with no absence entry: a reader cannot tell an "
                "undefined quantity from one that was never computed")


def test_a_silent_null_is_refused():
    """PROVES THE INVARIANT CAN FAIL. A guard never seen to reject anything is
    a guard nobody has tested."""
    from genomic_variant_classifier.evaluation.evaluator import (
        _assert_absence_biconditional)

    with pytest.raises(ValueError, match="no absence record"):
        _assert_absence_biconditional({"auroc": None, "field_absence": {},
                                       "curve_absence": {}})


def test_an_orphaned_absence_entry_is_refused():
    from genomic_variant_classifier.evaluation.evaluator import (
        _assert_absence_biconditional)

    with pytest.raises(ValueError, match="carry a value"):
        _assert_absence_biconditional({
            "auroc": 0.87,
            "field_absence": {"auroc": {"cause": "undefined_on_cohort"}},
            "curve_absence": {}})


def test_an_absent_curve_carrying_data_is_refused():
    from genomic_variant_classifier.evaluation.evaluator import (
        _assert_absence_biconditional)

    with pytest.raises(ValueError, match="carry values"):
        _assert_absence_biconditional({
            "field_absence": {},
            "curve_absence": {"tpr_curve": {"cause": "undefined_on_cohort"}},
            "tpr_curve": [0.0, 1.0]})


def test_a_consistent_payload_passes():
    """Guards the guard: an invariant that rejected everything would pass all
    three tests above and block every artifact."""
    from genomic_variant_classifier.evaluation.evaluator import (
        _assert_absence_biconditional)

    _assert_absence_biconditional({
        "auroc": None,
        "field_absence": {"auroc": {"cause": "undefined_on_cohort"}},
        "curve_absence": {}})


# --------------------------------------------------------------------------- #
# 4. The schema
# --------------------------------------------------------------------------- #
def test_the_schema_version_advances_to_four():
    """Version 4 is what tells a reader the absence maps exist. A version-3
    consumer meeting a null would have no entry to consult."""
    assert EVALUATION_REPORT_SCHEMA_VERSION_ABSENCE == 4
    assert _artifact("healthy")["schema_version"] == 4


def test_no_non_finite_value_reaches_the_artifact():
    """The property `dump_strict_json` exists to enforce, now achievable on
    every cohort rather than only the well-behaved ones."""
    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                yield from walk(value)
        elif isinstance(node, float):
            yield node

    for label in _cohorts():
        for value in walk(_artifact(label)):
            assert np.isfinite(value), f"{label} persisted a non-finite value"


def test_the_invariant_is_actually_wired_into_serialisation():
    """PROVES THE CALL SITE, NOT JUST THE FUNCTION.

    A sabotage replacing `_assert_absence_biconditional(payload)` with `pass`
    survived the first matrix: every test called the function DIRECTLY, so
    deleting its call site broke nothing. The invariant was verified and unused.

    Same lesson as commit CI-q, where a shared population was constructed and
    never handed over. A guard that is never reached is not a guard.
    """
    import ast as _ast
    import inspect as _inspect

    from genomic_variant_classifier.evaluation.evaluator import EvaluationReport

    source = _inspect.getsource(EvaluationReport.to_serializable)
    tree = _ast.parse(source.lstrip())
    called = {node.func.id for node in _ast.walk(tree)
              if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)}
    assert "_assert_absence_biconditional" in called, (
        "to_serializable does not call the biconditional invariant; the guard "
        "exists but nothing reaches it")


def test_a_contradictory_report_is_refused_end_to_end():
    """The behavioural half of the same guard: a report whose absence map
    disagrees with its values must not serialise, through the real path."""
    from genomic_variant_classifier.evaluation.absence import FieldAbsence

    report = _report("healthy")
    tampered = dataclasses.replace(
        report,
        field_absence={"auroc": FieldAbsence(
            cause=AbsenceCause.UNDEFINED_ON_COHORT, reason="fabricated")})

    with pytest.raises(ValueError, match="carry a value"):
        tampered.to_serializable()

