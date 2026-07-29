"""The report path must validate before it dispatches.

WHY THIS EXISTS
===============
Five scikit-learn calls sit in the report path on the same `(y, p)` pair, and
their behaviour under bad input is inconsistent. Measured 2026-07-28:

    defect                      roc_   pr_    calibration_  roc_auc  average_
                                curve  curve  curve         _score   precision
    non-finite probabilities    raise  raise  RETURNS       raise    raise
    outside the unit interval   ok     ok     RAISE         ok       ok
    single-class labels         warn   warn   returns       warn     warn

No consistent rule exists to translate, so the library cannot be allowed to
decide which defect becomes which status. Validation happens BEFORE dispatch.

THREE DEFECTS, EACH WORSE THAN THE LAST
----------------------------------------
1. `roc_curve` RAISES on non-finite input, aborting the whole report after the
   point metrics had already been computed successfully.

2. `calibration_curve` neither raises nor warns -- it returns a degenerate
   one-point curve carrying NaN, down from ten points. The failure surfaces only
   at persistence, where strict JSON refuses it and names the calibration curve
   rather than the corrupt model that caused it.

3. THE OPERATING-POINT SWEEP SHIPPED A WRONG NUMBER. `preds = (p >= t)`
   evaluates FALSE for a NaN, so every unusable prediction silently became a
   PREDICTED NEGATIVE. Measured with 100 of 200 true positives corrupted:

       clean    threshold 0.6366  sensitivity 0.90  specificity 1.00  ppv 1.0000
       corrupt  threshold 0.0000  sensitivity 0.50  specificity 0.00  ppv 0.3333

   An exception is loud. A poisoned curve fails at persistence. This shipped a
   plausible clinical decision threshold.

AND GATING TWO OF THREE WAS NOT ENOUGH
---------------------------------------
`at_high_ppv` comes from a SEPARATE function and kept reporting sensitivity 0.5,
specificity 0.875, positive predictive value 0.8 after both sensitivity targets
were gated. It was found only by checking all three report fields rather than
the two that had just been changed.
"""
from __future__ import annotations

import io
import contextlib

import numpy as np
import pytest

from genomic_variant_classifier.evaluation import evaluator as evaluator_module
from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator
from genomic_variant_classifier.evaluation.input_validation import (
    validate_probabilities,
    validate_ranking_scores,
    validate_reference_labels,
)

LIBRARY_CALLS = ("roc_curve", "precision_recall_curve", "calibration_curve",
                 "roc_auc_score", "average_precision_score")

OPERATING_POINTS = ("at_sensitivity_90", "at_sensitivity_95", "at_high_ppv")


def _cohort(n=400, seed=7):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.ones(n // 2), np.zeros(n - n // 2)])
    p = np.concatenate([rng.uniform(0.6, 0.99, n // 2),
                        rng.uniform(0.01, 0.4, n - n // 2)])
    return y, p


def _report(y, p, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return ClinicalEvaluator(n_bootstrap=0, random_state=42).evaluate(
            y, p, model_name="gate_probe", **kwargs)


# --------------------------------------------------------------------------- #
# 1. The three channels are validated independently
# --------------------------------------------------------------------------- #
def test_a_score_may_leave_the_unit_interval_but_a_probability_may_not():
    """THE CONTRACT THIS COMMIT ESTABLISHES.

    An array outside [0, 1] ranks perfectly well and is not a probability. The
    same array, two channels, two correct answers -- which is what dissolves the
    incoherent contract where one array was an invalid probability for
    calibration and an accepted probability for the receiver operating
    characteristic curve.
    """
    out_of_range = np.array([-4.1, 2.7, 0.3, 8.2])
    assert validate_ranking_scores(out_of_range).ok is True
    probability_verdict = validate_probabilities(out_of_range)
    assert probability_verdict.ok is False
    assert probability_verdict.reason == "probability_out_of_unit_interval"


@pytest.mark.parametrize("values,reason", [
    (np.array([0.1, np.nan, 0.3]), "nonfinite_probabilities"),
    (np.array([0.1, 1.4, 0.3]), "probability_out_of_unit_interval"),
    (np.array([0.1, -0.2, 0.3]), "probability_out_of_unit_interval"),
    (np.array([]), "empty_population"),
])
def test_probability_validation_names_the_defect(values, reason):
    verdict = validate_probabilities(values)
    assert verdict.ok is False
    assert verdict.reason == reason


@pytest.mark.parametrize("values,reason", [
    (np.array([0.0, 1.0, np.nan]), "nonfinite_reference_labels"),
    (np.array([0.0, 2.0, 1.0]), "reference_labels_not_binary"),
    (np.array([]), "empty_population"),
])
def test_reference_label_validation_names_the_defect(values, reason):
    verdict = validate_reference_labels(values)
    assert verdict.ok is False
    assert verdict.reason == reason


def test_misaligned_lengths_are_refused_on_every_channel():
    for validator in (validate_reference_labels, validate_ranking_scores,
                      validate_probabilities):
        verdict = validator(np.array([0.1, 0.2, 0.3]), n_expected=5)
        assert verdict.ok is False
        assert verdict.reason == "length_mismatch"


# --------------------------------------------------------------------------- #
# 2. Refusal is COMPONENT-level, never a report-wide exception
# --------------------------------------------------------------------------- #
def test_a_corrupt_model_still_produces_a_complete_report():
    """A report-wide exception would discard scientifically valid information.
    Prevalence does not depend on the predictions at all."""
    y, p = _cohort()
    corrupt = p.copy()
    corrupt[:100] = np.nan

    report = _report(y, corrupt)

    assert report.n_samples == len(y), "the attempted population must not narrow"
    assert np.isfinite(report.prevalence), "prevalence survives corrupt predictions"
    assert report.model_name == "gate_probe"
    assert report.metric_results, "the typed results must still be present"


def test_a_corrupt_model_withholds_the_curves_rather_than_poisoning_them():
    y, p = _cohort()
    corrupt = p.copy()
    corrupt[:100] = np.nan

    report = _report(y, corrupt)

    assert report.fpr_curve == [] and report.tpr_curve == []
    assert report.calibration_frac_pos == [] and report.calibration_mean_pred == []
    assert not any(np.isnan(v) for v in report.calibration_mean_pred), (
        "a NaN in a persisted curve fails only at serialisation, naming the "
        "curve rather than the model that caused it")


@pytest.mark.parametrize("field", OPERATING_POINTS)
def test_every_operating_point_is_withheld_on_corrupt_predictions(field):
    """ALL THREE. Gating the two sensitivity targets left `at_high_ppv` still
    reporting sensitivity 0.5, specificity 0.875, positive predictive value 0.8
    -- an entirely plausible decision threshold over a cohort nobody declared."""
    y, p = _cohort()
    corrupt = p.copy()
    corrupt[:100] = np.nan
    assert getattr(_report(y, corrupt), field) is None, (
        f"{field} reported a decision threshold despite unusable predictions")


@pytest.mark.parametrize("field", OPERATING_POINTS)
def test_every_operating_point_is_withheld_outside_the_unit_interval(field):
    """The sweep walks thresholds across [0, 1] and therefore assumes the
    probability scale. An array outside it places every row on one side of every
    threshold."""
    y, p = _cohort()
    assert getattr(_report(y, p * 3.0 - 1.0), field) is None


@pytest.mark.parametrize("field", OPERATING_POINTS)
def test_clean_input_still_produces_every_operating_point(field):
    """The gate must refuse defects, not capability."""
    y, p = _cohort()
    assert getattr(_report(y, p), field) is not None


def test_the_certified_interval_is_withheld_rather_than_raising():
    """`roc_auc_score` raises on non-finite input, and it is called from inside
    the bootstrap -- after the point metrics have already succeeded."""
    from genomic_variant_classifier.evaluation.capabilities import MetricStatus

    y, p = _cohort()
    corrupt = p.copy()
    corrupt[:100] = np.nan
    with contextlib.redirect_stdout(io.StringIO()):
        report = ClinicalEvaluator(n_bootstrap=50, random_state=42).evaluate(
            y, corrupt, model_name="interval_probe")

    assert report.auroc_ci_status is MetricStatus.FAILED
    # The reason names the PROBABILITY channel, because that is what was
    # validated: the array arrived as `y_proba`, so an unusable value is invalid
    # model output rather than an unusable ranking score.
    assert report.auroc_ci_finding == "nonfinite_probabilities"
    assert report.auroc_ci_lo is None and report.auroc_ci_hi is None
    assert report.auroc_ci_certification_eligible is False


# --------------------------------------------------------------------------- #
# 3. Counting spies -- on ALL FIVE library calls, not only roc_curve
# --------------------------------------------------------------------------- #
def _spy_all(monkeypatch):
    seen: dict = {}
    for name in LIBRARY_CALLS:
        original = getattr(evaluator_module, name, None)
        if original is None:
            continue

        def make(original=original, name=name):
            def counting(*args, **kwargs):
                seen[name] = seen.get(name, 0) + 1
                return original(*args, **kwargs)
            return counting

        monkeypatch.setattr(evaluator_module, name, make())
    return seen


def test_no_library_call_is_reached_after_failed_validation(monkeypatch):
    """The point of validating BEFORE dispatch. If a call is still reached, the
    gate is documentation rather than a gate."""
    seen = _spy_all(monkeypatch)
    y, p = _cohort()
    corrupt = p.copy()
    corrupt[:100] = np.nan
    _report(y, corrupt)

    reached = {k: v for k, v in seen.items() if v}
    assert not reached, (
        f"library call(s) {sorted(reached)} were reached despite failed "
        "validation; the gate did not run before dispatch")


def test_clean_input_does_reach_the_library(monkeypatch):
    """Guards the guard. A gate that refuses everything would pass the test
    above and be useless."""
    seen = _spy_all(monkeypatch)
    y, p = _cohort()
    _report(y, p)
    assert seen, "no library call was reached on clean input; the spy is not wired"
    assert seen.get("roc_curve", 0) >= 1
    assert seen.get("calibration_curve", 0) >= 1


def test_out_of_range_values_never_reach_any_library_call(monkeypatch):
    """An out-of-range array explicitly supplied as a probability must not be
    quietly reinterpreted as a ranking score."""
    seen = _spy_all(monkeypatch)
    y, p = _cohort()
    _report(y, p * 3.0 - 1.0)
    reached = {k: v for k, v in seen.items() if v}
    assert not reached, f"out-of-range probabilities reached {sorted(reached)}"


# --------------------------------------------------------------------------- #
# 4. The attempted population is never narrowed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutate", [
    lambda p: np.where(np.arange(p.size) < 40, np.nan, p),
    lambda p: p * 3.0 - 1.0,
])
def test_the_attempted_population_is_unchanged_by_refusal(mutate):
    """Refusing to compute is not the same as removing rows. Every count in the
    report must still describe the cohort that was submitted."""
    y, p = _cohort()
    report = _report(y, mutate(p))
    assert report.n_samples == len(y)
    assert report.n_pathogenic + report.n_benign == len(y)


# --------------------------------------------------------------------------- #
# 5. The seam, and the scores channel
#
# BOTH ADDED AFTER SABOTAGE SURVIVORS. These cover the two pieces written last,
# which were the two pieces with no tests -- the coverage was written against an
# earlier design and never caught up.
# --------------------------------------------------------------------------- #
def test_an_invalid_probability_is_not_ranked_as_a_score():
    """THE SEAM. Restoring `ranking_values = p` unconditionally survived the
    first sabotage matrix.

    Before the fallback was gated, an out-of-range array supplied as `y_proba`
    reached the registry through `y_score`, where ranking is scale-free, and was
    reported as `auroc 1.0` -- while the curve computed from the same values was
    withheld. One input, two layers, opposite verdicts.

    The caller who meant a ranking score has `scores`. A caller who put one in
    `y_proba` supplied invalid model output, and every layer must say so.
    """
    y, p = _cohort()
    out_of_range = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    report = _report(y, out_of_range)

    assert not np.isfinite(report.auroc), (
        "the flat area under the receiver operating characteristic curve was "
        "computed from an array that is not a probability")
    typed = report.metric_results["auroc"]
    assert typed.status is not None and typed.status.value != "ok", (
        f"the typed result reported {typed.status.value}; the registry ranked an "
        "invalid probability as though it were a score")
    assert report.fpr_curve == []


def test_the_scores_channel_accepts_a_genuine_ranking_score():
    """The capability the gate must not cost. A log-odds is a perfectly good
    ordering and belongs somewhere."""
    y, p = _cohort()
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    report = _report(y, p, scores=logit)

    assert np.isfinite(report.auroc)
    assert report.fpr_curve, "ranking quantities must compute from scores"
    assert report.calibration_frac_pos, "probabilities must still calibrate"


@pytest.mark.parametrize("bad_scores,reason", [
    (lambda n: np.concatenate([[np.nan], np.zeros(n - 1)]), "nonfinite_ranking_scores"),
    (lambda n: np.zeros(n - 1), "length_mismatch"),
])
def test_the_scores_channel_refuses_unusable_scores(bad_scores, reason):
    """Also a sabotage survivor: the new channel had no negative test, so
    removing its finiteness check broke nothing observable."""
    y, p = _cohort()
    verdict = validate_ranking_scores(bad_scores(len(y)), n_expected=len(y))
    assert verdict.ok is False
    assert verdict.reason == reason

    report = _report(y, p, scores=bad_scores(len(y)))
    assert report.fpr_curve == [], "ranking quantities computed from unusable scores"
    assert report.calibration_frac_pos, (
        "the probability channel is independent and must still calibrate")


# --------------------------------------------------------------------------- #
# 6. THE STANDING ENUMERATION
#
# ADDED AFTER A MISS. CI-t enumerated ten call sites by hand and declared the
# class closed. It was not: `_consequence_breakdown` calls `roc_auc_score` and
# `average_precision_score` directly, and that path is reached ONLY when `meta`
# is supplied -- which every corrupt-model test in this module did not do. The
# fixture shape hid it, exactly as it hid the calibration binning defect for
# seventeen days, and it was found by a measurement that happened to pass `meta`
# for an unrelated reason.
#
# A hand count is a claim. This is a check: every scikit-learn metric call in the
# module must sit inside a function that consults a validator. A new call site
# added tomorrow fails here rather than waiting for a corrupt cohort to reach it.
# --------------------------------------------------------------------------- #
import ast as _ast
import inspect as _inspect

SKLEARN_METRIC_CALLS = {
    "roc_auc_score", "average_precision_score", "roc_curve",
    "precision_recall_curve", "calibration_curve", "brier_score_loss",
    "f1_score", "matthews_corrcoef",
}

VALIDATOR_NAMES = {
    "validate_probabilities", "validate_ranking_scores",
    "validate_reference_labels", "ranking_usable", "probability_usable",
}


def _metric_calls_by_function():
    """Every scikit-learn metric call in evaluator.py, with its enclosing
    function. Parsed, not grepped: a call inside a docstring or a comment is not
    a call, and a hand count is not a check."""
    source = _inspect.getsource(evaluator_module)
    tree = _ast.parse(source)

    bodies = {node.name: _ast.unparse(node)
              for node in _ast.walk(tree) if isinstance(node, _ast.FunctionDef)}

    found = []

    class Walker(_ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            func = node.func
            name = (func.id if isinstance(func, _ast.Name)
                    else func.attr if isinstance(func, _ast.Attribute) else None)
            if name in SKLEARN_METRIC_CALLS:
                found.append((node.lineno, name,
                              self.stack[-1] if self.stack else "<module>"))
            self.generic_visit(node)

    Walker().visit(tree)
    return found, bodies


def _validation_governs_a_branch(body: str) -> bool:
    """Does a validator RESULT control a branch, or is the name merely present?

    STRENGTHENED AFTER THE GUARD FAILED TO FIRE. The first version asked whether
    a validator name appeared anywhere in the function. Disabling a gate with
    `if False:` leaves the name in place, so a dead gate satisfied it -- the same
    vocabulary-not-structure weakness already found once in the carried-item
    register's predicate.

    This walks the tree: a validator call must be assigned to a name, and that
    name must appear in the TEST of an `if`. A gate whose condition has been
    replaced by a constant no longer references it, and fails here.
    """
    tree = _ast.parse(body)

    validated_names = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign) and isinstance(node.value, _ast.Call):
            func = node.value.func
            called = (func.id if isinstance(func, _ast.Name)
                      else func.attr if isinstance(func, _ast.Attribute) else None)
            if called in {"validate_probabilities", "validate_ranking_scores",
                          "validate_reference_labels"}:
                for target in node.targets:
                    if isinstance(target, _ast.Name):
                        validated_names.add(target.id)

    if not validated_names:
        return False

    # PROPAGATE THROUGH ASSIGNMENTS. In `evaluate` the chain is three hops:
    #
    #     probability_check = validate_probabilities(...)
    #     probability_usable = label_check.ok and probability_check.ok
    #     if probability_usable: ...
    #
    # A one-hop check flags that as ungated, which is a FALSE POSITIVE -- and a
    # guard that cries wolf on correct code gets weakened until it catches
    # nothing. Iterate to a fixed point so a derived condition still counts.
    changed = True
    while changed:
        changed = False
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Assign):
                continue
            referenced = {n.id for n in _ast.walk(node.value)
                          if isinstance(n, _ast.Name)}
            if not (referenced & validated_names):
                continue
            for target in node.targets:
                if isinstance(target, _ast.Name) and target.id not in validated_names:
                    validated_names.add(target.id)
                    changed = True

    for node in _ast.walk(tree):
        if isinstance(node, _ast.If):
            referenced = {n.id for n in _ast.walk(node.test)
                          if isinstance(n, _ast.Name)}
            if referenced & validated_names:
                return True
    return False

def test_every_metric_call_sits_inside_a_gated_function():
    """THE GUARD THAT WOULD HAVE CAUGHT THE MISS."""
    found, bodies = _metric_calls_by_function()
    assert found, "no metric calls found at all; the enumeration is not wired"

    ungated = []
    for lineno, name, function in found:
        body = bodies.get(function, "")
        if not _validation_governs_a_branch(body):
            ungated.append(f"{function}:{lineno} calls {name}")

    assert not ungated, (
        "scikit-learn metric call(s) reachable without an input gate:\n  "
        + "\n  ".join(ungated)
        + "\n\nA hand count is a claim; this is the check. Add the validator to "
          "the enclosing function, or move the call behind one that has it.")


def test_the_enumeration_covers_the_functions_it_should():
    """Guards the guard: an enumeration that silently found nothing would make
    the assertion above vacuous."""
    found, _ = _metric_calls_by_function()
    functions = {f for _, _, f in found}
    assert "evaluate" in functions
    assert "_consequence_breakdown" in functions, (
        "the subgroup breakdown is the path the hand count missed; if it no "
        "longer appears, the enumeration has stopped seeing it")
    assert len(found) >= 5


def test_the_subgroup_breakdown_is_withheld_on_corrupt_predictions():
    """The specific miss, pinned with `meta` supplied -- which is the condition
    that makes this path reachable and which no earlier test met."""
    import pandas as pd

    y, p = _cohort()
    corrupt = p.copy()
    corrupt[:60] = np.nan
    meta = pd.DataFrame({
        "consequence": ["missense"] * len(y),
        "gene_symbol": [f"G{i % 40}" for i in range(len(y))],
    })

    with contextlib.redirect_stdout(io.StringIO()):
        report = ClinicalEvaluator(n_bootstrap=0, random_state=42).evaluate(
            y, corrupt, meta=meta, model_name="breakdown_probe")

    assert report.consequence_breakdown == [], (
        "a subgroup area under the curve was computed over unusable predictions")
    assert report.n_samples == len(y), "the attempted population must not narrow"


def test_the_enumeration_rejects_a_dead_gate():
    """GUARD-THE-GUARD. Weakening the check is invisible on clean code.

    A sabotage that reverted `_validation_governs_a_branch` to a substring test
    survived, because on a correct tree the weak and strong checks agree. The
    difference only appears on a body where the validator is CALLED but its
    result governs nothing -- which is exactly what `if False:` produces, and
    exactly the shape the first version of this guard accepted.

    Asserted on synthetic bodies rather than by disabling a real gate, because
    the point is the check's discrimination, not any particular call site.
    """
    dead = """
def probe(y, p):
    validation = validate_probabilities(p, n_expected=len(y))
    if False:
        return []
    return roc_auc_score(y, p)
"""
    live = """
def probe(y, p):
    validation = validate_probabilities(p, n_expected=len(y))
    if not validation.ok:
        return []
    return roc_auc_score(y, p)
"""
    chained = """
def probe(y, p):
    check = validate_probabilities(p, n_expected=len(y))
    usable = check.ok and len(y) > 0
    if usable:
        return roc_auc_score(y, p)
    return []
"""
    absent = """
def probe(y, p):
    return roc_auc_score(y, p)
"""
    assert _validation_governs_a_branch(dead) is False, (
        "a validator whose result governs nothing was accepted; the check is "
        "testing vocabulary rather than structure")
    assert _validation_governs_a_branch(absent) is False
    assert _validation_governs_a_branch(live) is True
    assert _validation_governs_a_branch(chained) is True, (
        "a derived condition must still count, or the guard cries wolf on "
        "correct code and gets weakened until it catches nothing")


def test_the_enumeration_uses_the_structural_check_and_not_a_substring():
    """CLOSES THE LAST SABOTAGE SURVIVOR.

    The guard-the-guard above proves `_validation_governs_a_branch` discriminates
    correctly. It does NOT prove the enumeration calls it: a mutation replacing
    the call site with `any(v in body for v in VALIDATOR_NAMES)` left the
    function intact, so every test of the function still passed while the
    enumeration silently reverted to a substring test.

    Parsed rather than grepped, for the same reason the enumeration itself is.
    """
    source = _inspect.getsource(test_every_metric_call_sits_inside_a_gated_function)
    tree = _ast.parse(source.lstrip())

    called = {node.func.id for node in _ast.walk(tree)
              if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)}
    assert "_validation_governs_a_branch" in called, (
        "the enumeration no longer calls the structural check; a substring test "
        "would accept a validator whose result governs nothing")

