"""Contract and sabotage tests for the metric registry.

THE DEFECT CLASS THIS GUARDS
============================
A metric that returns a bare float cannot say "I am not a value", so a caller
must remember to check. `metrics.evaluate`'s docstring records what that cost, on
`y = [1,1,1,1]` with `p = [.9,.8,.85,.95]`: `brier 0.01875` and `ece 0.125` came
back as NUMBERS with `calibration_valid True`, where the 0.125 is merely
`1 - 0.875` -- the gap between the mean prediction and the only label present,
from a reliability diagram with one occupied row.

That specific case was fixed inside `evaluate()` on 2026-07-21. The general
problem was not: every new metric must remember to return NaN, and every new
caller must remember to check. The registry makes status structural.

WHY APPLICABILITY IS DECIDED BEFORE THE KERNEL RUNS
---------------------------------------------------
A post-hoc rule -- `status = UNDEFINED if isnan(value) else OK` -- cannot catch
that case, because 0.125 is not NaN. So these tests assert the ORDER, not just
the outcome: an inapplicable metric must never be invoked at all, proven with a
callable that raises if called.

BOTH FALSE-POSITIVE DIRECTIONS ARE COVERED
------------------------------------------
  * finite but scientifically unsupported must not be reported OK-and-certified;
  * NaN from an APPLICABLE metric must be FAILED, not UNDEFINED -- that is an
    implementation defect, and calling it UNDEFINED would blame the cohort.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation import registry as reg
from genomic_variant_classifier.evaluation.capabilities import MetricResult, MetricStatus
from genomic_variant_classifier.evaluation.registry import (
    Applicability,
    MetricContext,
    MetricDescriptor,
    MetricInput,
    compute,
    evaluate_registered,
)


def _two_class(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    p = np.clip(rng.uniform(0, 1, n) * 0.5 + y * 0.4, 0, 1)
    return MetricContext(y_true=y, y_score=p, y_prob=p,
                         population_scope="synthetic_two_class")


def _single_class():
    return MetricContext(y_true=np.array([1., 1., 1., 1.]),
                         y_score=np.array([.9, .8, .85, .95]),
                         y_prob=np.array([.9, .8, .85, .95]),
                         population_scope="synthetic_single_class")


# --------------------------------------------------------------------------- #
# 1. The declaration is immutable and self-consistent
# --------------------------------------------------------------------------- #
def test_the_registry_is_a_frozen_tuple_not_a_mutable_table():
    assert isinstance(reg.all_metrics(), tuple)
    with pytest.raises((AttributeError, TypeError)):
        reg.all_metrics().append(None)          # type: ignore[attr-defined]


def test_names_are_unique_and_accessors_preserve_order():
    names = reg.names()
    assert len(names) == len(set(names))
    assert names == tuple(d.name for d in reg.all_metrics())


def test_every_descriptor_requires_labels_and_a_policy():
    for d in reg.all_metrics():
        assert MetricInput.LABELS in d.required_inputs, d.name
        assert callable(d.applicability), d.name
        assert callable(d.function), d.name
        assert d.name == d.name.strip().lower(), d.name


def test_by_name_raises_for_an_unregistered_metric():
    with pytest.raises(KeyError, match="no registered metric"):
        reg.by_name("not_a_metric")


@pytest.mark.parametrize("bad,fragment", [
    (dict(name="", fragment=None), "nonempty and lower-case"),
    (dict(name="AUROC", fragment=None), "nonempty and lower-case"),
])
def test_import_time_validation_rejects_a_malformed_name(bad, fragment):
    d = MetricDescriptor(name=bad["name"], function=lambda ctx: 1.0,
                         required_inputs=frozenset({MetricInput.LABELS}),
                         applicability=lambda ctx: reg.APPLICABLE)
    with pytest.raises(ValueError, match=fragment):
        reg._validate_registry([d])


def test_validation_rejects_a_duplicate_name():
    d = reg.by_name("auroc")
    with pytest.raises(ValueError, match="duplicate metric name"):
        reg._validate_registry([d, d])


def test_validation_rejects_requires_clusters_without_the_cluster_input():
    """The two would disagree about what the metric needs."""
    d = MetricDescriptor(name="x", function=lambda ctx: 1.0,
                         required_inputs=frozenset({MetricInput.LABELS}),
                         applicability=lambda ctx: reg.APPLICABLE,
                         requires_clusters=True)
    with pytest.raises(ValueError, match="requires_clusters=True but CLUSTERS"):
        reg._validate_registry([d])


# --------------------------------------------------------------------------- #
# 2. The context is aligned ONCE
# --------------------------------------------------------------------------- #
def test_a_misaligned_context_is_refused_at_construction():
    """CleanArrays exists because independent masks produced two arrays of the
    same length describing DIFFERENT ROWS. The context refuses the mismatch
    before any descriptor can compute over it."""
    with pytest.raises(ValueError, match="the context is aligned ONCE"):
        MetricContext(y_true=np.array([0., 1., 0.]), y_score=np.array([0.1, 0.2]),
                      population_scope="misaligned")


def test_derived_class_facts_are_computed_once_from_the_context():
    ctx = _single_class()
    assert ctx.n_classes_observed == 1 and ctx.classes_observed == (1.0,)
    assert not ctx.has_both_classes
    assert _two_class().has_both_classes


# --------------------------------------------------------------------------- #
# 3. SABOTAGE -- an inapplicable metric must never be invoked
# --------------------------------------------------------------------------- #
def test_an_inapplicable_metric_is_NOT_invoked():
    """Proves the ORDER, not merely the outcome. If applicability were checked
    after computing, this kernel would raise and the test would error."""
    def explode(ctx):
        raise AssertionError("the kernel was invoked for an inapplicable metric")

    d = MetricDescriptor(
        name="never_runs", function=explode,
        required_inputs=frozenset({MetricInput.LABELS}),
        applicability=lambda ctx: Applicability(
            applicable=False, status=MetricStatus.NOT_APPLICABLE, reason="by_design"))
    r = compute(d, _two_class())
    assert r.status is MetricStatus.NOT_APPLICABLE and r.reason == "by_design"


def test_a_metric_with_missing_inputs_is_NOT_invoked():
    def explode(ctx):
        raise AssertionError("invoked despite missing inputs")

    d = MetricDescriptor(
        name="needs_clusters", function=explode,
        required_inputs=frozenset({MetricInput.LABELS, MetricInput.CLUSTERS}),
        applicability=lambda ctx: reg.APPLICABLE, requires_clusters=True)
    r = compute(d, _two_class())
    assert r.status is MetricStatus.NOT_APPLICABLE
    assert r.reason == "required_inputs_missing"
    assert r.metadata["missing_inputs"] == ["clusters"]


# --------------------------------------------------------------------------- #
# 4. SABOTAGE -- finite but scientifically unsupported
# --------------------------------------------------------------------------- #
def test_single_class_expected_calibration_error_is_not_reported_ok():
    """The original defect class. A finite ECE on one class is the gap between
    the mean prediction and the only label present."""
    r = evaluate_registered(_single_class())["expected_calibration_error"]
    assert r.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert r.reason == "calibration_requires_class_support"
    assert r.metadata["n_classes_observed"] == 1


def test_single_class_ranking_metrics_are_undefined_not_failed():
    """The estimand does not exist for one class. That is a property of the
    cohort, so UNDEFINED, not FAILED."""
    out = evaluate_registered(_single_class())
    for name in ("auroc", "auprc", "auprc_gain"):
        assert out[name].status is MetricStatus.UNDEFINED, name
        assert out[name].reason == "binary_class_support_required", name


def test_a_finite_value_does_not_confer_certification_eligibility():
    """Numeric computability, scientific interpretability and certification
    eligibility are three different things. A Brier score on one class is a
    correct proper-score calculation AND inadmissible for a certified claim.

    An earlier version hard-coded certification_eligible=True for every OK
    result, collapsing the third axis into the first."""
    r = evaluate_registered(_single_class())["brier_score"]
    assert r.status is MetricStatus.OK
    assert np.isfinite(r.value)
    assert r.metadata["certification_eligible"] is False
    assert r.metadata["certification_blocked_by"] == "single_class_cohort"


def test_a_healthy_cohort_is_certification_eligible():
    for name, r in evaluate_registered(_two_class()).items():
        assert r.status is MetricStatus.OK, (name, r.reason)
        assert r.metadata["certification_eligible"] is True, name
        assert "certification_blocked_by" not in r.metadata, name


# --------------------------------------------------------------------------- #
# 5. SABOTAGE -- NaN from an APPLICABLE metric is FAILED, not UNDEFINED
# --------------------------------------------------------------------------- #
def test_an_applicable_metric_returning_nan_is_FAILED_not_undefined():
    """Otherwise an implementation defect is misclassified as a property of the
    cohort, and the fix is looked for in the wrong place."""
    d = MetricDescriptor(
        name="broken", function=lambda ctx: float("nan"),
        required_inputs=frozenset({MetricInput.LABELS}),
        applicability=lambda ctx: reg.APPLICABLE)
    r = compute(d, _two_class())
    assert r.status is MetricStatus.FAILED
    assert r.reason == "applicable_metric_returned_non_finite"
    assert r.metadata["metric_name"] == "broken"


def test_a_kernel_exception_is_FAILED_and_preserves_the_metric_identity():
    d = MetricDescriptor(
        name="raiser", function=lambda ctx: 1 / 0,
        required_inputs=frozenset({MetricInput.LABELS}),
        applicability=lambda ctx: reg.APPLICABLE)
    r = compute(d, _two_class())
    assert r.status is MetricStatus.FAILED
    assert r.reason == "metric_computation_failed"
    assert r.metadata["metric_name"] == "raiser"
    assert r.metadata["exception_type"] == "ZeroDivisionError"


def test_the_machine_readable_reason_is_stable_not_the_exception_text():
    """Exception messages change between library versions. The reason must not."""
    d = MetricDescriptor(
        name="raiser", function=lambda ctx: (_ for _ in ()).throw(RuntimeError("x" * 500)),
        required_inputs=frozenset({MetricInput.LABELS}),
        applicability=lambda ctx: reg.APPLICABLE)
    r = compute(d, _two_class())
    assert r.reason == "metric_computation_failed"
    assert len(r.metadata["exception_message"]) <= 200


# --------------------------------------------------------------------------- #
# 6. The Applicability type refuses to be ambiguous
# --------------------------------------------------------------------------- #
def test_an_inapplicable_verdict_requires_a_non_ok_status_and_a_reason():
    with pytest.raises(ValueError, match="requires a non-OK status"):
        Applicability(applicable=False, status=MetricStatus.OK, reason="r")
    with pytest.raises(ValueError, match="requires a non-OK status"):
        Applicability(applicable=False, reason="r")
    with pytest.raises(ValueError, match="requires a nonempty reason"):
        Applicability(applicable=False, status=MetricStatus.UNDEFINED)


def test_an_applicable_verdict_carries_no_status_or_reason():
    with pytest.raises(ValueError, match="carries no status or reason"):
        Applicability(applicable=True, reason="why")


# --------------------------------------------------------------------------- #
# 7. Every metric is always reported
# --------------------------------------------------------------------------- #
def test_no_metric_is_ever_silently_omitted():
    """An absent key and a refused metric are different facts, and a caller
    cannot tell them apart."""
    assert set(evaluate_registered(_single_class())) == set(reg.names())
    assert set(evaluate_registered(_two_class())) == set(reg.names())


def test_results_are_returned_in_registry_order():
    assert tuple(evaluate_registered(_two_class())) == reg.names()


def test_only_selects_a_subset_without_changing_semantics():
    out = evaluate_registered(_two_class(), only=["auroc", "brier_score"])
    assert tuple(out) == ("auroc", "brier_score")
    assert all(isinstance(r, MetricResult) for r in out.values())


def test_every_returned_object_is_a_MetricResult():
    for r in evaluate_registered(_single_class()).values():
        assert isinstance(r, MetricResult)


# --------------------------------------------------------------------------- #
# 8. The legacy path is untouched
# --------------------------------------------------------------------------- #
def test_metrics_evaluate_still_returns_the_legacy_untyped_dict():
    """The compatibility boundary. Its callers may depend on exact keys and
    bare-float values, so this commit does not touch it."""
    from genomic_variant_classifier.evaluation import metrics
    d = metrics.evaluate(np.array([0., 1., 0., 1.]), np.array([.2, .8, .3, .9]))
    assert isinstance(d, dict)
    assert isinstance(d["auroc"], float)
    assert not isinstance(d["auroc"], MetricResult)


def test_the_registry_does_not_call_metrics_evaluate():
    """The registry registers INDIVIDUAL kernels. It must not wrap the composite
    `metrics.evaluate()`, whose five metrics have five different applicability
    rules that one capability decision cannot honestly govern.

    Checked on the ABSTRACT SYNTAX TREE, not on source text: an earlier version
    of this test grepped the file and failed on the module docstring, which
    discusses `evaluate()` at length. A textual guard cannot distinguish a
    reference from a call.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(reg))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute):
                called.add(f.attr)
    assert "evaluate" not in called, (
        "the registry calls evaluate(); it must register individual kernels so "
        "each carries its own applicability rule")
    # and the individual kernels ARE reached
    assert {"auroc", "auprc", "brier_score", "log_loss",
            "expected_calibration_error"} <= called


# --------------------------------------------------------------------------- #
# 9. Support attachment -- how much evidence stood behind every result
# --------------------------------------------------------------------------- #
def test_support_is_attached_to_a_computed_value():
    r = evaluate_registered(_two_class(n=400))["auroc"]
    assert r.metadata["n_observations"] == 400
    assert r.metadata["n_classes_observed"] == 2
    assert "n_clusters" not in r.metadata, "absent when no clusters were supplied"


def test_support_is_attached_to_a_REFUSAL_not_only_to_a_value():
    """An INSUFFICIENT_SUPPORT on 3 rows and one on 300,000 point at different
    problems. Without this the artifact cannot tell them apart."""
    r = evaluate_registered(_single_class())["expected_calibration_error"]
    assert r.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert r.metadata["n_observations"] == 4
    assert r.metadata["n_classes_observed"] == 1


def test_support_is_attached_to_a_FAILURE():
    d = MetricDescriptor(name="raiser", function=lambda ctx: 1 / 0,
                         required_inputs=frozenset({MetricInput.LABELS}),
                         applicability=lambda ctx: reg.APPLICABLE)
    r = compute(d, _two_class(n=250))
    assert r.status is MetricStatus.FAILED
    assert r.metadata["n_observations"] == 250


def test_support_is_attached_when_required_inputs_are_missing():
    d = MetricDescriptor(name="needs_clusters", function=lambda ctx: 1.0,
                         required_inputs=frozenset({MetricInput.LABELS,
                                                    MetricInput.CLUSTERS}),
                         applicability=lambda ctx: reg.APPLICABLE,
                         requires_clusters=True)
    r = compute(d, _two_class(n=99))
    assert r.reason == "required_inputs_missing"
    assert r.metadata["n_observations"] == 99


def test_cluster_support_counts_DISTINCT_clusters():
    ctx = MetricContext(y_true=np.array([0., 1., 0., 1.]),
                        y_score=np.array([.1, .9, .2, .8]),
                        clusters=np.array(["G1", "G1", "G2", "G2"]),
                        population_scope="two_gene_clusters")
    assert ctx.n_clusters == 2
    r = evaluate_registered(ctx)["auroc"]
    assert r.metadata["n_clusters"] == 2
    assert r.metadata["n_observations"] == 4


def test_support_applies_no_threshold_of_its_own():
    """Whether a minimum observation or cluster count blocks certification is a
    scientific policy decision. Inventing one here silently is the class of guess
    this project removes -- so a two-row cohort still reports OK, and records
    that it had two rows."""
    ctx = MetricContext(y_true=np.array([0., 1.]), y_score=np.array([.2, .8]),
                        y_prob=np.array([.2, .8]), population_scope="two_row_cohort")
    r = evaluate_registered(ctx)["auroc"]
    assert r.status is MetricStatus.OK
    assert r.metadata["n_observations"] == 2
    assert r.metadata["certification_eligible"] is True


def test_effective_sample_size_is_NOT_duplicated_here():
    """n_clusters is a count, not an effective sample size. Effective sample size
    under clustering is a property of a resampling design and lives in
    BootstrapResult beside the design effect. A second, weaker answer to a
    question already answered properly is how two numbers come to disagree."""
    ctx = MetricContext(y_true=np.array([0., 1., 0., 1.]),
                        y_score=np.array([.1, .9, .2, .8]),
                        clusters=np.array(["G1", "G1", "G2", "G2"]),
                        population_scope="two_gene_clusters")
    keys = set(ctx.support())
    assert keys == {"population_scope", "n_observations", "n_classes_observed",
                    "n_clusters"}
    assert not any("effective" in k or "design_effect" in k for k in keys)


def test_a_verdicts_own_metadata_survives_alongside_support():
    """The applicability verdict's metadata must not be overwritten by support,
    nor overwrite it."""
    r = evaluate_registered(_single_class())["auroc"]
    assert r.metadata["n_observations"] == 4          # from support
    assert r.metadata["classes_observed"] == [1.0]    # from the verdict
    assert r.metadata["metric_name"] == "auroc"


# --------------------------------------------------------------------------- #
# 10. population_scope -- the denominator must travel with the number
# --------------------------------------------------------------------------- #
def test_a_context_cannot_be_built_without_naming_its_population():
    """Support counts alone do not identify the denominator. This session
    produced two cases where correct numbers described different populations and
    the difference was invisible: 53 and 63 were both right over universes that
    differed by ten variants, and 85 beside 107 was printed as a breakdown of
    107 when 85 + 107 = 192."""
    with pytest.raises(TypeError):
        MetricContext(y_true=np.array([0., 1.]))            # type: ignore[call-arg]


@pytest.mark.parametrize("bad", ["", "   ", None, 7])
def test_an_empty_or_non_string_population_scope_is_refused(bad):
    with pytest.raises(ValueError, match="population_scope is REQUIRED"):
        MetricContext(y_true=np.array([0., 1.]), population_scope=bad)  # type: ignore[arg-type]


def test_population_scope_is_carried_on_every_result_including_refusals():
    for name, r in evaluate_registered(_single_class()).items():
        assert r.metadata["population_scope"] == "synthetic_single_class", name
    for name, r in evaluate_registered(_two_class()).items():
        assert r.metadata["population_scope"] == "synthetic_two_class", name


def test_two_results_with_equal_counts_but_different_populations_are_distinguishable():
    """The defect this field exists to prevent: identical support, different
    denominator semantics, and nothing in the artifact to tell them apart."""
    y = np.array([0., 1., 0., 1.]); p = np.array([.2, .8, .3, .9])
    a = evaluate_registered(MetricContext(y_true=y, y_score=p,
                                          population_scope="all_test_variants"))["auroc"]
    b = evaluate_registered(MetricContext(y_true=y, y_score=p,
                                          population_scope="variants_with_representative_row"))["auroc"]
    assert a.value == b.value
    assert a.metadata["n_observations"] == b.metadata["n_observations"]
    assert a.metadata["population_scope"] != b.metadata["population_scope"]
