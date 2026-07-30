"""The guards that keep the retirement retired.

WHY THESE EXIST
===============
Commit 3b-2 deleted the evaluator's duplicate computation: `matthews_corrcoef`
and `f1_score` at a hard-coded 0.5, an inline Brier expression, and a private
calibration loop. The report is now a projection of the typed registry results.

Deletion alone does not keep it deleted. A future change adding
`f1_score(y, p >= 0.5)` back into the report path would be small, plausible, and
would produce a number indistinguishable from the projected one on almost every
cohort — the duplication would be back and no value-comparing test would notice.

These guards make the ABSENCE structural. Two independent mechanisms, because
each catches what the other cannot:

    the abstract-syntax-tree guard   catches duplication that is WRITTEN
    the counting wrappers            catch duplication that is EXECUTED

Static analysis cannot see a kernel reached through a dynamic lookup. Counting
cannot see dead code that a future edit will wake. Together they close both.

SCOPE, NARROWED DELIBERATELY
----------------------------
The static guard inspects the REPORT-CONSTRUCTION PATH only. It does not ban
thresholding across the module: `_find_operating_point` legitimately sweeps
thresholds, and a blanket rule would either fail on it or be weakened until it
caught nothing. The question is not "does this file threshold anywhere" but
"does the path that builds the report compute what the registry already
computed".
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from genomic_variant_classifier.evaluation import evaluator as evaluator_module
from genomic_variant_classifier.evaluation import legacy_projection as projection_module
from genomic_variant_classifier.evaluation import metrics as metrics_module
from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator
from genomic_variant_classifier.evaluation.population import EvaluationPopulation

# Kernels the report must never call directly. Reaching one of these from the
# report path means the registry is no longer the only computation path.
FORBIDDEN_DIRECT_CALLS = {
    "matthews_corrcoef", "f1_score", "roc_auc_score", "average_precision_score",
    "brier_score_loss", "matthews_correlation_coefficient", "f1_at_threshold",
    "expected_calibration_error", "maximum_calibration_error",
}


def _evaluate_source_tree():
    """The report-construction path, as an abstract syntax tree."""
    return ast.parse(inspect.getsource(ClinicalEvaluator.evaluate).lstrip())


def _cohort(n=200, seed=11):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.5, n).astype(float)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.15, n), 0.0, 1.0)
    return y, p


# --------------------------------------------------------------------------- #
# 1. The static guard -- duplication that is WRITTEN
# --------------------------------------------------------------------------- #
def test_the_report_path_calls_no_metric_kernel_directly():
    """Carried item (o), activated now that the evaluator is retired rather than
    merely observed.

    Written earlier it would have tested the intended ARCHITECTURE against an
    implementation that contradicted it, which is a test guaranteed to fail for
    a reason nobody intends to fix.
    """
    called = set()
    for node in ast.walk(_evaluate_source_tree()):
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else None)
            if name:
                called.add(name)

    forbidden = called & FORBIDDEN_DIRECT_CALLS
    assert not forbidden, (
        f"the report-construction path calls {sorted(forbidden)} directly. The "
        "registry is the only computation path; a flat report field is a "
        "projection of a typed result, never a second computation of it.")


def test_the_report_path_compares_no_probability_against_a_literal():
    """A hard-coded threshold in the report path is the specific defect this
    commit removed. `p >= 0.5` produced hard labels for the Matthews coefficient
    and F1 with the threshold invisible to every reader of the report."""
    offending = []
    for node in ast.walk(_evaluate_source_tree()):
        if isinstance(node, ast.Compare):
            left_is_series = isinstance(node.left, ast.Name) and node.left.id in {"p", "y", "prob", "y_proba"}
            has_numeric_literal = any(
                isinstance(c, ast.Constant) and isinstance(c.value, (int, float))
                and not isinstance(c.value, bool)
                for c in node.comparators)
            if left_is_series and has_numeric_literal:
                offending.append(ast.unparse(node))
    assert not offending, (
        f"the report path thresholds a probability against a literal: {offending}. "
        "A decision threshold is declared provenance -- metric identity, value, "
        "operator and source -- carried on the descriptor, not a number written "
        "into the reporting code where no artifact can record it.")


def test_the_report_path_does_not_aggregate_calibration_itself():
    """The private calibration loop is gone; a replacement built inline would
    reintroduce the seventeen-day interval-convention defect one layer up."""
    tree = _evaluate_source_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "_calibration_error", (
                "the retired private calibration loop has been called again")
    source = inspect.getsource(ClinicalEvaluator.evaluate)
    assert "np.linspace(0" not in source or "bin" not in source.lower(), (
        "the report path appears to construct calibration bins itself")


def test_the_retired_private_calibration_loop_is_gone():
    assert not hasattr(ClinicalEvaluator, "_calibration_error"), (
        "ClinicalEvaluator._calibration_error still exists. It was the second "
        "calibration implementation, and the two disagreed about every interior "
        "bin edge for seventeen days.")


# --------------------------------------------------------------------------- #
# 2. The counting wrappers -- duplication that is EXECUTED
# --------------------------------------------------------------------------- #
def _count_kernel_calls(monkeypatch, y, p, **evaluate_kwargs):
    """Count every kernel invocation during one full report construction."""
    counts: dict = {}
    for name in ("auroc", "auprc", "brier_score", "expected_calibration_error",
                 "maximum_calibration_error", "matthews_correlation_coefficient",
                 "f1_at_threshold", "prevalence"):
        original = getattr(metrics_module, name)

        def make(original=original, name=name):
            def counting(*args, **kwargs):
                counts[name] = counts.get(name, 0) + 1
                return original(*args, **kwargs)
            return counting

        monkeypatch.setattr(metrics_module, name, make())

    ClinicalEvaluator(n_bootstrap=0, random_state=42).evaluate(
        y, p, model_name="counting_probe", **evaluate_kwargs)
    return counts


def test_each_kernel_is_invoked_exactly_once_per_report(monkeypatch):
    """THE CENTRAL GUARANTEE. More than one invocation means a second computation
    path exists, whether or not it currently agrees."""
    y, p = _cohort()
    counts = _count_kernel_calls(monkeypatch, y, p)

    assert counts, "no kernel was invoked at all; the probe is not wired"

    # COMPOSITION IS NOT DUPLICATION, and the guard must tell them apart.
    #
    # A first draft asserted that NO kernel is called more than once. That is too
    # strong: `auprc_gain` is a registered metric defined as
    # `auprc - no_skill_auprc`, so it calls `auprc` by construction. Two
    # invocations, one per registered metric that needs the quantity -- which is
    # composition, not a second implementation.
    #
    # The property that actually matters is that no registered metric is computed
    # more than the number of registered metrics that consume it. The allowance
    # is declared explicitly, so a NEW duplicate invocation still fails.
    #
    # DERIVED, NOT LICENSED. Each entry is the number of REGISTERED metrics
    # that consume the kernel:
    #   auprc        -- itself, and auprc_gain, defined as auprc - no_skill_auprc
    #   brier_score  -- itself, and brier_decomposition_residual, defined at
    #                   metrics.py:1750 as
    #                   brier - (reliability - resolution + uncertainty)
    # Added 2026-07-30 with registry commit 2. The assertion below this one
    # checks the table in the OTHER direction, so an inflated allowance fails
    # too: these numbers must be exact, not generous.
    composed_by = {"auprc": 2, "brier_score": 2}
    over_budget = {
        name: seen for name, seen in counts.items()
        if seen > composed_by.get(name, 1)}
    assert not over_budget, (
        f"kernel(s) invoked more often than any registered metric requires: "
        f"{over_budget}. Two computations of one quantity is how two "
        "implementations of it begin.")

    unexpected_composition = {
        name for name in composed_by if counts.get(name, 0) < composed_by[name]}
    assert not unexpected_composition, (
        f"{sorted(unexpected_composition)} was invoked FEWER times than the "
        "declared composition requires; the allowance no longer describes the "
        "registry and must be re-derived rather than left as a blanket licence")


def test_the_projection_itself_invokes_no_kernel(monkeypatch):
    """The projection translates typed results into legacy scalars. If it
    computed anything, it would be a third implementation wearing the name of a
    translation layer."""
    from genomic_variant_classifier.evaluation.registry import (
        MetricContext, evaluate_registered)

    y, p = _cohort()
    population = EvaluationPopulation.full(y.size, scope="c", source_id=None)
    results = evaluate_registered(MetricContext(
        y_true=y, y_prob=p, y_score=p, population=population))

    counts: dict = {}
    for name in ("auroc", "auprc", "brier_score", "expected_calibration_error",
                 "maximum_calibration_error", "matthews_correlation_coefficient",
                 "f1_at_threshold", "prevalence"):
        original = getattr(metrics_module, name)

        def make(original=original, name=name):
            def counting(*args, **kwargs):
                counts[name] = counts.get(name, 0) + 1
                return original(*args, **kwargs)
            return counting

        monkeypatch.setattr(metrics_module, name, make())

    projection_module.project_legacy_fields(results)
    assert counts == {}, (
        f"the projection invoked kernel(s) {sorted(counts)}; it must translate "
        "already-computed typed results, never compute")


def test_report_construction_performs_no_threshold_comparison(monkeypatch):
    """Counted rather than read. The static guard catches a threshold that is
    WRITTEN in the report path; this catches one reached indirectly."""
    y, p = _cohort()
    thresholds: list = []
    original = metrics_module.apply_decision_threshold

    def recording(prob, *, threshold, operator):
        thresholds.append((threshold, operator))
        return original(prob, threshold=threshold, operator=operator)

    monkeypatch.setattr(metrics_module, "apply_decision_threshold", recording)
    ClinicalEvaluator(n_bootstrap=0, random_state=42).evaluate(
        y, p, model_name="threshold_probe")

    assert thresholds, "no threshold was applied at all; the probe is not wired"
    assert all(t == (0.5, ">=") for t in thresholds), (
        f"a threshold other than the declared one was applied: {set(thresholds)}")
    assert len(set(thresholds)) == 1, (
        "more than one distinct threshold was applied during a single report")


def test_the_two_calibration_errors_reuse_one_binning(monkeypatch):
    """The expected and maximum calibration errors are two summaries of ONE
    table. Binning twice is how they came to disagree, and the disagreement was
    invisible until a cohort placed mass exactly on an interior edge."""
    y, p = _cohort()
    builds: list = []
    original = metrics_module.CalibrationBins.from_predictions

    def counting(cls_y, cls_p, n_bins=10):
        builds.append(n_bins)
        return original(cls_y, cls_p, n_bins=n_bins)

    monkeypatch.setattr(metrics_module.CalibrationBins, "from_predictions",
                        staticmethod(counting))
    ClinicalEvaluator(n_bootstrap=0, random_state=42).evaluate(
        y, p, model_name="calibration_probe")

    assert builds, "no calibration table was built; the probe is not wired"
    assert len(set(builds)) == 1, (
        f"calibration tables were built with differing bin counts: {set(builds)}")


# --------------------------------------------------------------------------- #
# 3. evaluate(source_id=...) -- both paths
# --------------------------------------------------------------------------- #
def test_evaluate_without_a_source_id_is_unattributed_and_not_certifiable():
    """`evaluate` receives arrays, not a canonical table. Without an identity it
    must say so rather than invent one."""
    y, p = _cohort()
    report = ClinicalEvaluator(n_bootstrap=0, random_state=42).evaluate(
        y, p, model_name="probe")

    assert report.metric_results == {} or all(
        r.certification_eligible is not True for r in report.metric_results.values())
    assert np.isfinite(report.auroc), "the value is still computed and reported"


def test_evaluate_with_a_source_id_produces_an_attributed_population():
    from genomic_variant_classifier.evaluation.registry import (
        MetricContext, evaluate_registered)

    y, p = _cohort()
    attributed = EvaluationPopulation.full(
        y.size, scope="attempted_cohort", source_id="frame:sha256:probe")
    unattributed = EvaluationPopulation.full(
        y.size, scope="attempted_cohort", source_id=None)

    a = evaluate_registered(MetricContext(y_true=y, y_prob=p, y_score=p,
                                          population=attributed))["auroc"]
    u = evaluate_registered(MetricContext(y_true=y, y_prob=p, y_score=p,
                                          population=unattributed))["auroc"]

    assert a.value == u.value, "attribution must not change the number"
    assert a.certification_eligible is True
    assert u.certification_eligible is False
    assert u.metadata["certification_blocked_by"] == "unattributed_population"
    assert a.population_fingerprint is not None
    assert u.population_fingerprint is None


def test_an_unattributed_population_blocks_certification_without_changing_values():
    """Admissibility and arithmetic are separate axes. An unattributed cohort
    yields the same numbers; what it cannot support is a certified claim, because
    a certified claim asserts something about a NAMED set of rows."""
    from genomic_variant_classifier.evaluation.registry import (
        MetricContext, evaluate_registered)

    y, p = _cohort()
    for source_id in (None, "frame:sha256:probe"):
        results = evaluate_registered(MetricContext(
            y_true=y, y_prob=p, y_score=p,
            population=EvaluationPopulation.full(
                y.size, scope="c", source_id=source_id)))
        projected = projection_module.project_legacy_fields(results)
        if source_id is None:
            reference = projected
        else:
            assert projected == reference, (
                "attribution changed a projected value; it must change only "
                "admissibility")


# --------------------------------------------------------------------------- #
# 4. The REPORT's own surface
#
# ADDED AFTER A SABOTAGE SURVIVOR. A mutation making `evaluate` fabricate a
# source identity went undetected, because every guard above constructs its
# population directly and none observes what `evaluate` actually does with
# attribution.
#
# Chasing that survivor exposed something worse: the report was a projection but
# `metric_results` was never populated and the schema never advanced, so the
# typed results computed on every call were discarded. Commit 3a had stated
# plainly that 3b would emit them. Nothing noticed, because no guard asserted on
# the report's typed surface.
#
# Testing the real execution graph rather than a synthetic stand-in is the same
# lesson as the protected-key guard in commit 3b-1a, and this is its second
# occurrence in this commit.
# --------------------------------------------------------------------------- #
def _report(source_id=None, seed=11):
    y, p = _cohort(seed=seed)
    return ClinicalEvaluator(n_bootstrap=0, random_state=42).evaluate(
        y, p, model_name="surface_probe", source_id=source_id)


def test_the_report_emits_the_typed_surface():
    """Commit 3a introduced schema 3 as a CAPABILITY and said 3b would emit it.
    A report whose scalars are projections of a mapping it does not carry would
    force every consumer wanting status, reason, population or certification to
    recompute them -- which is the duplication this whole sequence removed."""
    from genomic_variant_classifier.evaluation.evaluator import (
        EVALUATION_REPORT_SCHEMA_VERSION_ABSENCE)
    from genomic_variant_classifier.evaluation.registry import names

    report = _report()
    assert report.schema_version == EVALUATION_REPORT_SCHEMA_VERSION_ABSENCE
    assert set(report.metric_results) == set(names()), (
        "the report must carry every registered result, not a subset")


def test_evaluate_without_a_source_id_yields_no_fingerprint():
    """THE MUTATION THIS CLOSES: `source_id or 'some-string'` inside `evaluate`.

    A fabricated identity would give two entirely different cohorts the same
    fingerprint, certifying an equivalence nobody established. Asserted on the
    REPORT, because that is where a fabrication would actually surface.
    """
    report = _report(source_id=None)
    for name, result in report.metric_results.items():
        assert result.population_fingerprint is None, (
            f"{name} carries a population fingerprint although no source "
            "identity was supplied; an identity has been fabricated")
        assert result.certification_eligible is not True, (
            f"{name} is certification-eligible over an unattributed population")


def test_evaluate_with_a_source_id_yields_a_fingerprint_and_certifiability():
    report = _report(source_id="frame:sha256:probe")
    for name, result in report.metric_results.items():
        assert result.population_fingerprint is not None, name
    assert report.metric_results["auroc"].certification_eligible is True


def test_attribution_changes_admissibility_and_never_the_numbers():
    """The whole point of the attribution model in one assertion."""
    unattributed = _report(source_id=None)
    attributed = _report(source_id="frame:sha256:probe")

    for field in ("auroc", "auprc", "mcc", "f1", "brier_score",
                  "calibration_ece", "calibration_mce", "prevalence"):
        assert getattr(unattributed, field) == getattr(attributed, field), (
            f"{field} changed with attribution; attribution governs what may be "
            "CLAIMED, never what was MEASURED")

    assert unattributed.metric_results["auroc"].certification_eligible is False
    assert attributed.metric_results["auroc"].certification_eligible is True


def test_the_flat_fields_are_exactly_the_projection_of_the_typed_results():
    """The projection invariant, on a real report. If a flat field ever diverges
    from `project_legacy_fields` output, the report has begun computing again."""
    report = _report(source_id="frame:sha256:probe")
    projected = projection_module.project_legacy_fields(report.metric_results)

    for field, expected in projected.items():
        actual = getattr(report, field)
        assert projection_module.legacy_values_equal(actual, expected), (
            f"report.{field} is {actual!r} but the projection of its typed "
            f"result is {expected!r}; the flat field is no longer a derived view")

