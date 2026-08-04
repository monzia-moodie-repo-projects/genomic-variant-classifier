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
from genomic_variant_classifier.evaluation.capabilities import (
    MetricMetadataKey, MetricResult, MetricStatus)
from genomic_variant_classifier.evaluation.registry import (
    Applicability,
    MetricContext,
    MetricDescriptor,
    MetricInput,
    compute,
    evaluate_registered,
)
from genomic_variant_classifier.evaluation.population import EvaluationPopulation
from genomic_variant_classifier.evaluation.registry import ResultKind

# --- population scaffolding (2026-07-27) ------------------------------------
# `MetricContext` requires an `EvaluationPopulation`, not a bare scope string: a
# scope is a NAME and two different row sets may share one. Tests therefore build
# a real population. The source identity is fixed so fingerprints are comparable
# across cases within a file, and differs from production identities so a test
# fixture can never be mistaken for a real cohort.
_TEST_SOURCE_ID = "unit-test-frame:sha256:0000000000000000"


def _pop(n, scope):
    return EvaluationPopulation.full(n, scope=scope, source_id=_TEST_SOURCE_ID)



def _two_class(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    p = np.clip(rng.uniform(0, 1, n) * 0.5 + y * 0.4, 0, 1)
    return MetricContext(y_true=y, y_score=p, y_prob=p,
                         population=_pop(n, "synthetic_two_class"))


def _single_class():
    return MetricContext(y_true=np.array([1., 1., 1., 1.]),
                         y_score=np.array([.9, .8, .85, .95]),
                         y_prob=np.array([.9, .8, .85, .95]),
                         population=_pop(4, "synthetic_single_class"))


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
                         applicability=lambda ctx: reg.APPLICABLE,
                         result_kind=ResultKind.PREDICTION_METRIC,
                         display_name="probe", description="probe")
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
                         requires_clusters=True,
                         result_kind=ResultKind.PREDICTION_METRIC,
                         display_name="probe", description="probe")
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
                      population=_pop(3, "misaligned"))


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
            applicable=False, status=MetricStatus.NOT_APPLICABLE, reason="by_design"),
                         result_kind=ResultKind.PREDICTION_METRIC,
                         display_name="probe", description="probe")
    r = compute(d, _two_class())
    assert r.status is MetricStatus.NOT_APPLICABLE and r.reason == "by_design"


def test_a_metric_with_missing_inputs_is_NOT_invoked():
    def explode(ctx):
        raise AssertionError("invoked despite missing inputs")

    d = MetricDescriptor(
        name="needs_clusters", function=explode,
        required_inputs=frozenset({MetricInput.LABELS, MetricInput.CLUSTERS}),
        applicability=lambda ctx: reg.APPLICABLE, requires_clusters=True,
                         result_kind=ResultKind.PREDICTION_METRIC,
                         display_name="probe", description="probe")
    r = compute(d, _two_class())
    assert r.status is MetricStatus.NOT_APPLICABLE
    assert r.reason == "required_inputs_missing"
    assert r.metadata["missing_inputs"] == ["clusters"]


# --------------------------------------------------------------------------- #
# 4. SABOTAGE -- finite but scientifically unsupported
# --------------------------------------------------------------------------- #
def test_single_class_calibration_is_canonically_defined():
    """REVERSED 2026-07-28. This test previously asserted the opposite.

    It was named `test_single_class_expected_calibration_error_is_not_reported_ok`
    and required INSUFFICIENT_SUPPORT, on the reasoning that "a finite expected
    calibration error on one class is the gap between the mean prediction and
    the only label present" -- correctly identifying the value, then concluding
    it was meaningless.

    That conflated two estimands. DISCRIMINATION asks whether predictions rank
    one class against another and a single-class cohort cannot support it.
    CALIBRATION asks whether predicted probabilities match observed event
    frequencies, and a single-class cohort can: the gap IS the measurement, of
    systematic over- or under-prediction in that population.

    The interpretive limitation is real and is recorded -- as neutral metadata
    and as a certification block -- rather than as a refusal to compute.
    """
    r = evaluate_registered(_single_class())["expected_calibration_error"]
    assert r.status is MetricStatus.OK
    assert r.reason is None
    assert np.isfinite(r.value)
    assert r.metadata["n_classes_observed"] == 1
    assert r.metadata["reference_class_support"] == "single_class"


def test_single_class_calibration_is_recorded_but_not_certifiable():
    """Computability, interpretability and admissibility remain three separate
    axes. The value is correct; the claim is not admissible."""
    r = evaluate_registered(_single_class())["expected_calibration_error"]
    assert r.status is MetricStatus.OK
    assert r.certification_eligible is False
    assert r.metadata["certification_blocked_by"] == "single_class_cohort"


def test_certification_is_derived_from_cohort_facts_not_from_the_diagnostic():
    """The diagnostic must be DESCRIPTIVE, never CAUSAL.

    `reference_class_support` records a structural fact. If certification were
    keyed on the presence of that token rather than on the cohort itself, a
    future refactor renaming or dropping the diagnostic would silently unblock
    certification. The blocker is asserted to come from the class support that
    `_certification_eligibility` reads directly.
    """
    single = evaluate_registered(_single_class())["expected_calibration_error"]
    both = evaluate_registered(_two_class())["expected_calibration_error"]

    assert single.metadata["certification_blocked_by"] == "single_class_cohort"
    assert single.metadata["n_classes_observed"] == 1
    assert both.certification_eligible is True
    assert "certification_blocked_by" not in both.metadata
    assert "reference_class_support" not in both.metadata, (
        "the diagnostic must appear only where the structure warrants it")

    # The blocker names the COHORT PROPERTY, not the diagnostic key.
    assert "single_class_cohort" == single.metadata["certification_blocked_by"]
    assert single.metadata["certification_blocked_by"] != \
        single.metadata["reference_class_support"], (
            "blocker and diagnostic must not be the same token, or a reader "
            "cannot tell which one drove the decision")


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


# Metrics whose applicability depends on the CLASSIFIER rather than the cohort.
# The likelihood ratios refuse when specificity reaches exactly 1.0, because the
# ratio is then infinite and infinity is not a value an artifact can carry. A
# perfectly specific classifier is not an unhealthy cohort.
_CLASSIFIER_DEPENDENT = frozenset({
    "positive_likelihood_ratio", "negative_likelihood_ratio"})


def test_a_healthy_cohort_is_certification_eligible():
    """UPDATED 2026-07-29. This asserted that EVERY registered metric is OK on a
    healthy cohort -- true of the original ten, and false once the confusion
    family arrived.

    The fixture's classifier is perfectly specific, so the positive likelihood
    ratio is genuinely unbounded and refuses with `specificity: 1.0`. That is a
    property of the CLASSIFIER, not of the cohort, and the premise "healthy cohort
    implies every metric computes" does not hold for metrics whose applicability
    reads the predictions. The exclusion is named rather than the assertion
    weakened.
    """
    checked = 0
    for name, r in evaluate_registered(_two_class()).items():
        if name in _CLASSIFIER_DEPENDENT:
            continue
        assert r.status is MetricStatus.OK, (name, r.reason)
        assert r.metadata["certification_eligible"] is True, name
        assert "certification_blocked_by" not in r.metadata, name
        checked += 1
    assert checked >= 10, (
        f"only {checked} metrics were checked; the exclusion set has grown to "
        "swallow the assertion")


def test_the_excluded_metrics_refuse_for_the_stated_reason():
    """Guards the exclusion above: those two must refuse BECAUSE the ratio is
    unbounded, not for some unrelated reason that the exclusion would hide."""
    results = evaluate_registered(_two_class())
    for name in _CLASSIFIER_DEPENDENT:
        r = results[name]
        assert r.status is not MetricStatus.OK, name
        assert r.reason == "likelihood_ratio_unbounded", (name, r.reason)
        assert r.metadata["specificity"] == 1.0, name


# --------------------------------------------------------------------------- #
# 5. SABOTAGE -- NaN from an APPLICABLE metric is FAILED, not UNDEFINED
# --------------------------------------------------------------------------- #
def test_an_applicable_metric_returning_nan_is_FAILED_not_undefined():
    """Otherwise an implementation defect is misclassified as a property of the
    cohort, and the fix is looked for in the wrong place."""
    d = MetricDescriptor(
        name="broken", function=lambda ctx: float("nan"),
        required_inputs=frozenset({MetricInput.LABELS}),
        applicability=lambda ctx: reg.APPLICABLE,
                         result_kind=ResultKind.PREDICTION_METRIC,
                         display_name="probe", description="probe")
    r = compute(d, _two_class())
    assert r.status is MetricStatus.FAILED
    assert r.reason == "applicable_metric_returned_non_finite"
    assert r.metadata["metric_name"] == "broken"


def test_a_kernel_exception_is_FAILED_and_preserves_the_metric_identity():
    d = MetricDescriptor(
        name="raiser", function=lambda ctx: 1 / 0,
        required_inputs=frozenset({MetricInput.LABELS}),
        applicability=lambda ctx: reg.APPLICABLE,
                         result_kind=ResultKind.PREDICTION_METRIC,
                         display_name="probe", description="probe")
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
        applicability=lambda ctx: reg.APPLICABLE,
                         result_kind=ResultKind.PREDICTION_METRIC,
                         display_name="probe", description="probe")
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
    # Uses a RANKING metric, 2026-07-28. This test previously used single-class
    # calibration as its example of a refusal; calibration is no longer refused
    # there, so the example had to become one that still is. The property under
    # test -- that support accompanies a refusal -- is unchanged.
    r = evaluate_registered(_single_class())["auroc"]
    assert r.status is MetricStatus.UNDEFINED
    assert r.metadata["n_observations"] == 4
    assert r.metadata["n_classes_observed"] == 1


def test_support_is_attached_to_a_FAILURE():
    d = MetricDescriptor(name="raiser", function=lambda ctx: 1 / 0,
                         required_inputs=frozenset({MetricInput.LABELS}),
                         applicability=lambda ctx: reg.APPLICABLE,
                         result_kind=ResultKind.PREDICTION_METRIC,
                         display_name="probe", description="probe")
    r = compute(d, _two_class(n=250))
    assert r.status is MetricStatus.FAILED
    assert r.metadata["n_observations"] == 250


def test_support_is_attached_when_required_inputs_are_missing():
    d = MetricDescriptor(name="needs_clusters", function=lambda ctx: 1.0,
                         required_inputs=frozenset({MetricInput.LABELS,
                                                    MetricInput.CLUSTERS}),
                         applicability=lambda ctx: reg.APPLICABLE,
                         requires_clusters=True,
                         result_kind=ResultKind.PREDICTION_METRIC,
                         display_name="probe", description="probe")
    r = compute(d, _two_class(n=99))
    assert r.reason == "required_inputs_missing"
    assert r.metadata["n_observations"] == 99


def test_cluster_support_counts_DISTINCT_clusters():
    ctx = MetricContext(y_true=np.array([0., 1., 0., 1.]),
                        y_score=np.array([.1, .9, .2, .8]),
                        clusters=np.array(["G1", "G1", "G2", "G2"]),
                        population=_pop(4, "two_gene_clusters"))
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
                        y_prob=np.array([.2, .8]), population=_pop(2, "two_row_cohort"))
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
                        population=_pop(4, "two_gene_clusters"))
    keys = set(ctx.support())
    assert keys == {"population_scope", "population_fingerprint", "n_observations",
                    "n_classes_observed", "n_clusters"}, (
        "the support key set is pinned EXACTLY so nothing can be added without a "
        "deliberate decision. population_fingerprint was added 2026-07-27 with "
        "EvaluationPopulation and is NOT an effective sample size: it identifies "
        "WHICH rows, never how many independent ones. The prohibition this test "
        "enforces -- no second, weaker answer to a question BootstrapResult "
        "already answers properly -- is untouched.")
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


@pytest.mark.parametrize("bad", ["", "   ", None, 7, "attempted_cohort"])
def test_a_bare_scope_string_is_refused_in_place_of_a_population(bad):
    """REWRITTEN 2026-07-27. This previously accepted any non-empty string.

    A scope NAMES a population; it does not identify one. Two different row sets
    may both be called "attempted_cohort", so the last parameter -- a perfectly
    well-formed scope string -- must be refused just as firmly as the empty one.
    """
    with pytest.raises(ValueError, match="population is REQUIRED"):
        MetricContext(y_true=np.array([0., 1.]), population=bad)  # type: ignore[arg-type]


def test_population_scope_is_carried_on_every_result_including_refusals():
    for name, r in evaluate_registered(_single_class()).items():
        assert r.metadata["population_scope"] == "synthetic_single_class", name
    for name, r in evaluate_registered(_two_class()).items():
        assert r.metadata["population_scope"] == "synthetic_two_class", name


def test_two_results_with_equal_counts_but_different_populations_are_distinguishable():
    """The defect this field exists to prevent: identical support, different
    denominator semantics, and nothing in the artifact to tell them apart.

    STRENGTHENED 2026-07-27. Scope and fingerprint answer DIFFERENT questions and
    the test now separates them, because conflating the two would hide exactly
    the case each is for.
    """
    y = np.array([0., 1., 0., 1.]); p = np.array([.2, .8, .3, .9])
    a = evaluate_registered(MetricContext(
        y_true=y, y_score=p, population=_pop(4, "all_test_variants")))["auroc"]
    b = evaluate_registered(MetricContext(
        y_true=y, y_score=p,
        population=_pop(4, "variants_with_representative_row")))["auroc"]

    # Same rows, different semantic claim: the NAMES must differ so an artifact
    # can tell the claims apart...
    assert a.value == b.value
    assert a.metadata["n_observations"] == b.metadata["n_observations"]
    assert a.metadata["population_scope"] != b.metadata["population_scope"]
    # ...and the FINGERPRINTS must match, because membership is identical.
    # A fingerprint that changed with the name would be measuring the label
    # rather than the rows, and could not support the report invariant that
    # several metrics describe the same population.
    assert a.population_fingerprint == b.population_fingerprint


def test_equal_counts_over_genuinely_different_rows_are_distinguishable():
    """The case CARDINALITY CANNOT REACH, and which scope alone cannot either.

    Two four-row populations drawn from the same eight-row frame have the same
    n_observations, and could easily be given the same scope by a careless
    caller. Only membership separates them.
    """
    source = "unit-test-frame:sha256:0000000000000000"
    frame = EvaluationPopulation.full(8, scope="frame", source_id=source)
    first = frame.restrict(np.arange(8) < 4, scope="cohort", reason="split")
    second = frame.restrict(np.arange(8) >= 4, scope="cohort", reason="split")

    y = np.array([0., 1., 0., 1.]); p = np.array([.2, .8, .3, .9])
    a = evaluate_registered(MetricContext(y_true=y, y_score=p, population=first))["auroc"]
    b = evaluate_registered(MetricContext(y_true=y, y_score=p, population=second))["auroc"]

    assert a.metadata["n_observations"] == b.metadata["n_observations"] == 4
    assert a.metadata["population_scope"] == b.metadata["population_scope"]
    assert a.population_fingerprint != b.population_fingerprint, (
        "two disjoint four-row populations produced the same fingerprint; "
        "equal counts and an identical scope would then be indistinguishable")


def test_descriptor_metadata_may_extend_registry_metadata_never_replace_it():
    """REBUILT 2026-07-28 on the REAL execution graph.

    The first version constructed a synthetic descriptor and asserted that the
    protected-key check raised. Disabling the check in `compute()` therefore
    broke nothing observable, because no REGISTERED descriptor exercised the
    path -- the guard was real and its test did not reach it.

    This version drives a registered descriptor through `compute()` and asserts
    the SURVIVAL of every registry-owned key, which is stronger than asserting
    that an exception occurred: the invariant is that descriptor metadata may
    EXTEND registry metadata, never replace it.
    """
    import genomic_variant_classifier.evaluation.registry as registry_module

    ctx = _single_class()
    descriptor = registry_module.by_name("expected_calibration_error")
    original = descriptor.applicability

    def hijacking(context):
        verdict = original(context)
        return registry_module.Applicability(
            applicable=True, status=None, reason=None,
            metadata={MetricMetadataKey.POPULATION_SCOPE: "hijacked",
                      MetricMetadataKey.N_OBSERVATIONS: 999_999,
                      MetricMetadataKey.CERTIFICATION_ELIGIBLE: True})

    object.__setattr__(descriptor, "applicability", hijacking)
    try:
        with pytest.raises(registry_module.RegistryInvariantError) as caught:
            registry_module.compute(descriptor, ctx)
        message = str(caught.value)
        for key in ("POPULATION_SCOPE", "N_OBSERVATIONS", "CERTIFICATION_ELIGIBLE"):
            assert key in message, f"{key} was not reported as a collision"
    finally:
        object.__setattr__(descriptor, "applicability", original)

    # And the registry is unchanged afterwards: the attempt neither succeeded
    # nor left residue.
    survivor = evaluate_registered(ctx)["expected_calibration_error"]
    assert survivor.metadata["population_scope"] != "hijacked"
    assert survivor.metadata["n_observations"] == 4
    assert survivor.certification_eligible is False


def test_a_descriptor_may_still_add_its_own_diagnostics():
    """The invariant is EXTEND-not-replace, so a non-colliding key must pass."""
    result = evaluate_registered(_single_class())["expected_calibration_error"]
    assert result.metadata["reference_class_support"] == "single_class"
    assert result.metadata["metric_name"] == "expected_calibration_error"
# --------------------------------------------------------------------------- #
# REG-1 (2026-08-03): the refusal path owns less than the OK path
# --------------------------------------------------------------------------- #

def test_a_refusal_may_not_forge_the_population_fingerprint():
    """THE SHARP CASE FOR REG-1.

    `MetricContext.support` is attached to EVERY result, refusals included,
    because an INSUFFICIENT_SUPPORT on 3 rows and one on 300,000 point at
    different problems. Until 2026-08-03 the refusal branch merged
    `verdict.metadata` LAST with no collision check, so a descriptor could set
    POPULATION_FINGERPRINT and win -- a refusal claiming a membership it never
    examined, on the branch whose whole purpose is stating its evidence base.

    POPULATION_FINGERPRINT exists precisely because cardinality cannot carry
    this: n=980 beside n=980 says nothing about WHICH 980.
    """
    import genomic_variant_classifier.evaluation.registry as registry_module

    ctx = _single_class()
    descriptor = registry_module.by_name("auroc")
    original = descriptor.applicability

    def forging(context):
        return registry_module.Applicability(
            applicable=False,
            status=registry_module.MetricStatus.UNDEFINED,
            reason="binary_class_support_required",
            metadata={MetricMetadataKey.POPULATION_FINGERPRINT: "sha256:forged"})

    object.__setattr__(descriptor, "applicability", forging)
    try:
        with pytest.raises(registry_module.RegistryInvariantError) as caught:
            registry_module.compute(descriptor, ctx)
        assert "POPULATION_FINGERPRINT" in str(caught.value)
    finally:
        object.__setattr__(descriptor, "applicability", original)

    # And the registry is unchanged afterwards: the attempt neither succeeded nor
    # left residue. The real refusal still carries the REAL fingerprint.
    survivor = evaluate_registered(ctx)["auroc"]
    assert survivor.metadata["population_fingerprint"] != "sha256:forged"
    assert survivor.metadata["population_fingerprint"] == \
        ctx.population.membership_fingerprint


def test_a_refusal_may_still_report_the_class_count_that_justifies_it():
    """THE ASSERTION REG-1 v1 FAILED, AND THE REASON THE TWO PATHS DIFFER.

    A first version of REG-1 derived ONE protected set and applied it to both
    branches. It turned 29 tests red, because `auroc` refusing a single-class
    cohort reports N_CLASSES_OBSERVED as THE GROUND of its refusal -- "there is
    one class, therefore this metric is undefined". Seven registered metrics do
    the same.

    On the OK path that key IS registry-owned: the registry has computed the
    cohort and a descriptor claiming otherwise contradicts an established fact.
    On the refusal path it is the descriptor's ARGUMENT. Same key, opposite
    ownership, because the paths genuinely differ.

    Without this test, a future simplification collapsing the two sets back into
    one would go unnoticed until the suite exploded.
    """
    result = evaluate_registered(_single_class())["auroc"]

    assert result.status is MetricStatus.UNDEFINED
    assert result.reason == "binary_class_support_required"
    assert result.metadata["n_classes_observed"] == 1, (
        "the descriptor's own justification for refusing did not survive; the "
        "refusal path is protecting a key it does not own")

    # AND THE REGISTRY'S OWN KEYS ARE STILL THE REGISTRY'S. Compared against the
    # POPULATION, not against the result itself: an earlier draft of this line
    # read `result.metadata["population_scope"] == helper(result)` where the
    # helper returned that same value -- a comparison of a thing with itself,
    # which is the `_ABSENCE == _ABSENCE` tautology repaired in
    # test_bootstrap_reconciliation.py earlier the same day, reproduced hours
    # later by its own author.
    population = _single_class().population
    assert result.metadata["population_scope"] == population.scope
    assert result.metadata["population_fingerprint"] ==         population.membership_fingerprint


def test_the_refusal_ownership_exception_is_still_exhaustive():
    """`_DESCRIPTOR_OWNED_ON_REFUSAL` IS THE ONE STORED THING REG-1 INTRODUCES.

    A stored set rots exactly as a stored number does. This re-derives the
    2026-08-03 probe over the LIVE descriptor graph: across four cohort shapes,
    NO registered descriptor may claim a `support()` key on a refusal unless that
    key is named in the exception.

    On 2026-08-03 this measured 27 refusals, with N_CLASSES_OBSERVED claimed by
    seven metrics and N_OBSERVATIONS, POPULATION_FINGERPRINT and POPULATION_SCOPE
    claimed by none. An eighth metric adopting a different key fails HERE, loudly,
    rather than silently widening the frozenset -- which is the difference
    between a gate and a comment.
    """
    import numpy as np

    import genomic_variant_classifier.evaluation.registry as registry_module
    from genomic_variant_classifier.evaluation.population import (
        EvaluationPopulation)

    cohorts = {
        "single_class_positive": (np.array([1.0, 1.0, 1.0, 1.0]),
                                  np.array([0.9, 0.8, 0.85, 0.95])),
        "single_class_negative": (np.array([0.0, 0.0, 0.0, 0.0]),
                                  np.array([0.1, 0.2, 0.3, 0.4])),
        "two_class": (np.array([1.0, 0.0, 1.0, 0.0]),
                      np.array([0.9, 0.2, 0.8, 0.1])),
        "degenerate_probabilities": (np.array([1.0, 0.0, 1.0, 0.0]),
                                     np.array([0.5, 0.5, 0.5, 0.5])),
    }

    refusals = 0
    unexpected = {}
    for label, (y, p) in cohorts.items():
        population = EvaluationPopulation.full(
            y.size, scope="ownership_probe",
            source_id="ownership-probe:sha256:0000000000000000")
        ctx = registry_module.MetricContext(
            y_true=y, y_prob=p, y_score=p, population=population)
        registry_owned = set(ctx.support())
        for descriptor in registry_module.all_metrics():
            verdict = descriptor.applicability(ctx)
            if verdict.applicable:
                continue
            refusals += 1
            claimed = registry_owned & set(verdict.metadata)
            for key in claimed - set(
                    registry_module._DESCRIPTOR_OWNED_ON_REFUSAL):
                unexpected.setdefault(str(key), set()).add(
                    f"{descriptor.name}/{label}")

    assert refusals > 0, (
        "no refusals were observed at all; the probe cohorts no longer reach the "
        "refusal path and this test proves nothing")
    assert not unexpected, (
        "a registered descriptor claims a registry-owned key on a REFUSAL that "
        "_DESCRIPTOR_OWNED_ON_REFUSAL does not name: "
        f"{ {k: sorted(v) for k, v in unexpected.items()} }. Decide whether the "
        "descriptor is wrong or the exception should grow -- do not widen the "
        "frozenset without recording which.")
# --------------------------------------------------------------------------- #
# REG-1 closure (2026-08-03): the two gates the baseline mutation run exposed
# --------------------------------------------------------------------------- #

def test_an_applicable_verdict_may_not_forge_n_classes_observed():
    """THE SUCCESS HALF OF THE OWNERSHIP ASYMMETRY.

    N_CLASSES_OBSERVED is DESCRIPTOR-OWNED on a refusal -- it is the ground of
    the refusal, "there is one class, therefore this metric is undefined", and
    seven registered metrics report it that way. It is REGISTRY-OWNED on success,
    because by then the registry has established the cohort's class count and a
    descriptor claiming otherwise would contradict an established fact.

    ADDED AFTER MUTATION M05 WENT UNDETECTED on 2026-08-03. The battery's own
    rationale had claimed the pre-existing hijack test covered this key; it sets
    POPULATION_SCOPE, N_OBSERVATIONS and CERTIFICATION_ELIGIBLE instead. M05 was
    a real missing test, not an equivalent mutant.

    Given its own name rather than appended to the generic hijack test, because
    "the same key, opposite ownership on the two paths" is the whole of REG-1 and
    should not be buried inside an assertion list.
    """
    import genomic_variant_classifier.evaluation.registry as registry_module

    ctx = _two_class()
    descriptor = registry_module.by_name("brier_score")
    original = descriptor.applicability

    # An APPLICABLE decision carries no status and no reason -- Applicability
    # refuses one that does, and the test would then pass for the wrong reason.
    def forging(context):
        return registry_module.Applicability(
            applicable=True,
            metadata={MetricMetadataKey.N_CLASSES_OBSERVED: 99})

    object.__setattr__(descriptor, "applicability", forging)
    try:
        with pytest.raises(registry_module.RegistryInvariantError,
                           match="N_CLASSES_OBSERVED"):
            registry_module.compute(descriptor, ctx)
    finally:
        object.__setattr__(descriptor, "applicability", original)

    # RESTORATION IS PROVED, not merely attempted: the real metric still computes
    # and still reports the registry's own count, asserted against the cohort
    # rather than a literal so the test cannot go stale if the fixture changes.
    survivor = registry_module.compute(descriptor, ctx)
    assert survivor.status is MetricStatus.OK
    assert survivor.metadata["n_classes_observed"] == \
        len(set(ctx.y_true.tolist()))


def test_refusal_protected_keys_are_derived_from_ctx_support():
    """A STRUCTURAL GATE: refusal protection must EVOLVE with `support()`.

    ADDED AFTER MUTATION M06 WENT UNDETECTED on 2026-08-03. Hand-listing the
    refusal protected set is behaviourally identical TODAY -- the literal happens
    to match the current `support()` vocabulary -- and diverges only when a key is
    added there. The baseline report called this "the one thing a test cannot
    catch". THAT WAS WRONG: a BEHAVIOURAL test cannot catch it; a STRUCTURAL one
    can, because the derivation is a property of the source.

    The requirement, from `compute`'s own comment: "a future key added to
    `support()` is protected the moment it exists rather than the moment somebody
    remembers to add it here."

    The refusal branch is located SEMANTICALLY -- `if not verdict.applicable` --
    rather than by line number or formatting, so ordinary edits above it do not
    silently break this guard.
    """
    import ast
    import inspect
    import textwrap

    import genomic_variant_classifier.evaluation.registry as registry_module

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(registry_module.compute)))

    refusal_branch = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.UnaryOp)
                and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Attribute)
                and isinstance(test.operand.value, ast.Name)
                and test.operand.value.id == "verdict"
                and test.operand.attr == "applicable"):
            refusal_branch = node
            break
    assert refusal_branch is not None, (
        "no `if not verdict.applicable` branch found in compute(); the refusal "
        "path has been restructured and this guard no longer knows where to look")

    guard_calls = [node for node in ast.walk(refusal_branch)
                   if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name)
                   and node.func.id == "_reject_registry_owned_keys"]
    assert len(guard_calls) == 1, (
        f"the refusal branch calls the ownership guard {len(guard_calls)} times; "
        "exactly one call is expected")

    protected_expression = guard_calls[0].args[3]

    support_calls = [node for node in ast.walk(protected_expression)
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and isinstance(node.func.value, ast.Name)
                     and node.func.value.id == "ctx"
                     and node.func.attr == "support"]
    assert len(support_calls) == 1, (
        "the refusal protected set was HAND-LISTED instead of derived from "
        "ctx.support(). It may be behaviourally identical today and it will not "
        "be the moment a key is added to support(): the point of deriving it is "
        "that a new key is protected when it EXISTS, not when somebody remembers")

    subtractions = [node for node in ast.walk(protected_expression)
                    if isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Sub)]
    assert len(subtractions) == 1, (
        f"expected exactly one subtraction in the refusal protected set, found "
        f"{len(subtractions)}; the descriptor-owned exception must be removed "
        "once and only once")

    right = subtractions[0].right
    assert isinstance(right, ast.Name) and \
        right.id == "_DESCRIPTOR_OWNED_ON_REFUSAL", (
            "the refusal set subtracts something other than the named exception; "
            "an inline literal here would be a second copy of the ownership rule")
