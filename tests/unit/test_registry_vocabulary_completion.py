"""Commit 2b-2: completion of the canonical metric descriptor vocabulary.

WHAT THIS COMMIT IS
===================
Not "more descriptors". What is completed is the VOCABULARY every descriptor
speaks, so that later additions cannot produce a second dialect in which some
descriptors declare their classification, parameters and provenance while others
leave them implicit.

    ResultKind              what kind of quantity a descriptor produces
    ThresholdParameters     the typed, canonical threshold declaration
    parameters              its immutable, JSON-validated serialisation
    REGISTRY_SCHEMA_VERSION 1 -> 2, enforced at import for EVERY descriptor
    four new descriptors    maximum calibration error, Matthews correlation
                            coefficient, F1, prevalence

THREE LAYERS, AND THIS COMMIT TOUCHES ONLY THE FIRST
-----------------------------------------------------
    Layer 1  metric semantics       descriptor, kernel, threshold provenance
    Layer 2  registry orchestration execution, applicability, certification
    Layer 3  report projection      compatibility report, legacy flat fields

2b-2 completes Layer 1. `ClinicalEvaluator` keeps its own threshold computation
until commit 3 turns Layer 3 into a pure projection. That divergence is
DELIBERATE and temporary, which is why the registry's rules are called CANONICAL
here: a reader must not mistake the temporary difference for an accident.

THE ACCEPTANCE CRITERION
------------------------
Not "the new descriptors produce the expected values" but:

    every result that already existed is byte-identical afterwards.

Proved against a snapshot frozen on the 2b-1 tree BEFORE any of this was
written. A baseline captured from a frozen implementation and expectations
written by the author of the change are different scientific standards: only the
first can detect a movement the author did not anticipate.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.metrics import (
    apply_decision_threshold,
    f1_at_threshold,
    matthews_correlation_coefficient,
    prevalence,
)
from genomic_variant_classifier.evaluation.population import EvaluationPopulation
from genomic_variant_classifier.evaluation.registry import (
    REGISTRY_SCHEMA_VERSION,
    REPORT_METRIC_NAMES,
    MetricContext,
    MetricInput,
    ResultKind,
    ThresholdOperator,
    ThresholdParameters,
    ThresholdSource,
    all_metrics,
    by_name,
    evaluate_registered,
)
from genomic_variant_classifier.evaluation.capabilities import MetricStatus

SOURCE = "unit-test-frame:sha256:0000000000000000"
SNAPSHOT = Path(__file__).parent.parent / "fixtures" / "registry_snapshot_2b1.json"


def _ctx(y, prob=None, score=None, clusters=None, scope="unit_test_cohort"):
    y_arr = np.asarray(y, dtype=float)
    pop = EvaluationPopulation.full(y_arr.size, scope=scope, source_id=SOURCE)
    return MetricContext(y_true=y_arr,
                         y_prob=None if prob is None else np.asarray(prob, dtype=float),
                         y_score=None if score is None else np.asarray(score, dtype=float),
                         clusters=clusters, population=pop)


# --------------------------------------------------------------------------- #
# 1. ResultKind
# --------------------------------------------------------------------------- #
def test_result_kind_has_exact_controlled_vocabulary():
    """Narrow and semantic. Categories are not added pre-emptively: a
    classification nobody dispatches on is documentation pretending to be a type."""
    assert {k.value for k in ResultKind} == {"prediction_metric", "population_statistic"}


def test_prevalence_is_a_population_statistic():
    assert by_name("prevalence").result_kind is ResultKind.POPULATION_STATISTIC


def test_other_report_results_are_prediction_metrics():
    """Calibration metrics are PREDICTION metrics: they assess predictions
    against observed outcomes, however differently from the ranking metrics."""
    for name in REPORT_METRIC_NAMES:
        if name == "prevalence":
            continue
        assert by_name(name).result_kind is ResultKind.PREDICTION_METRIC, name


def test_result_kind_does_not_appear_in_result_metadata():
    """It lives on the DESCRIPTOR until schema version 3.

    Putting it in metadata would perturb every already-serialised result, and the
    acceptance criterion for this commit is byte-identity with no carve-outs.
    """
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    for name, result in evaluate_registered(_ctx(y, prob=p, score=p)).items():
        keys = {str(k.value) if hasattr(k, "value") else str(k) for k in result.metadata}
        assert "result_kind" not in keys, name


def test_a_population_statistic_may_not_require_predictions():
    """Enforced at import. A population statistic describes the cohort, not the
    predictions; requiring a probability would make it un-computable before
    scoring and would couple it to model provenance that belongs elsewhere."""
    d = by_name("prevalence")
    assert MetricInput.PROBABILITIES not in d.required_inputs
    assert MetricInput.SCORES not in d.required_inputs
    assert d.required_inputs == frozenset({MetricInput.LABELS})


# --------------------------------------------------------------------------- #
# 2. Descriptor schema
# --------------------------------------------------------------------------- #
def test_the_registry_schema_version_was_raised():
    assert REGISTRY_SCHEMA_VERSION == 2


@pytest.mark.parametrize("descriptor", list(all_metrics()), ids=lambda d: d.name)
def test_registry_descriptor_schema_complete(descriptor):
    """EVERY descriptor, not merely the new ones. Checking only additions is how
    two dialects arise: the old ones keep their implicit blanks."""
    assert descriptor.name and descriptor.name == descriptor.name.strip().lower()
    assert callable(descriptor.function)
    assert isinstance(descriptor.result_kind, ResultKind)
    assert descriptor.required_inputs
    assert MetricInput.LABELS in descriptor.required_inputs
    assert callable(descriptor.applicability)
    assert descriptor.display_name and descriptor.display_name.strip()
    assert descriptor.description and descriptor.description.strip()
    assert isinstance(descriptor.parameters, MappingProxyType)


def test_descriptor_parameters_are_immutable():
    d = by_name("matthews_correlation_coefficient")
    with pytest.raises(TypeError):
        d.parameters["decision_threshold"] = 0.9      # type: ignore[index]
    with pytest.raises(TypeError):
        d.parameters["new_key"] = 1                    # type: ignore[index]


def test_descriptor_parameters_are_json_safe():
    """A descriptor is a frozen declaration destined for an artifact."""
    for d in all_metrics():
        round_tripped = json.loads(json.dumps(dict(d.parameters)))
        assert round_tripped == dict(d.parameters), d.name


@pytest.mark.parametrize("bad", [
    {"values": [1, 2, 3]},                 # list: mutable
    {"array": np.arange(3)},               # array
    {"fn": len},                           # callable
    {"nan": float("nan")},                 # does not survive JSON
])
def test_non_serialisable_parameters_are_refused(bad):
    from genomic_variant_classifier.evaluation.registry import MetricDescriptor

    with pytest.raises((TypeError, ValueError)):
        MetricDescriptor(
            name="probe", function=lambda ctx: 0.0,
            required_inputs=frozenset({MetricInput.LABELS}),
            applicability=lambda ctx: None, result_kind=ResultKind.PREDICTION_METRIC,
            display_name="Probe", description="probe", parameters=bad)


def test_empty_parameters_are_an_empty_immutable_mapping():
    d = by_name("prevalence")
    assert dict(d.parameters) == {}
    assert isinstance(d.parameters, MappingProxyType)


# --------------------------------------------------------------------------- #
# 3. Threshold provenance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["matthews_correlation_coefficient", "f1"])
def test_threshold_metrics_declare_full_provenance(name):
    p = by_name(name).parameters
    assert p["decision_threshold"] == 0.5
    assert p["threshold_operator"] == ">="
    assert p["threshold_source"] == "fixed_default"


@pytest.mark.parametrize("name", ["matthews_correlation_coefficient", "f1"])
def test_the_typed_object_is_the_semantics_and_the_mapping_its_serialisation(name):
    d = by_name(name)
    tp = d.threshold_parameters
    assert isinstance(tp, ThresholdParameters)
    assert tp.threshold == d.parameters["decision_threshold"]
    assert tp.operator.value == d.parameters["threshold_operator"]
    assert tp.source.value == d.parameters["threshold_source"]
    assert tp.to_mapping()["decision_threshold"] == tp.threshold


@pytest.mark.parametrize("name", ["matthews_correlation_coefficient", "f1"])
def test_the_kernel_and_the_applicability_share_ONE_threshold_object(name):
    """Identity, not equality. Two thresholds that merely happen to be equal
    today is exactly how they come to differ tomorrow."""
    d = by_name(name)
    assert d.function._threshold_parameters is d.threshold_parameters
    assert d.applicability._threshold_parameters is d.threshold_parameters


@pytest.mark.parametrize("bad,exc", [
    (float("nan"), ValueError), (float("inf"), ValueError),
    (-0.1, ValueError), (1.5, ValueError), ("0.5", TypeError), (True, TypeError),
])
def test_threshold_must_be_finite_and_in_the_unit_interval(bad, exc):
    with pytest.raises(exc):
        ThresholdParameters(threshold=bad, operator=ThresholdOperator.GREATER_OR_EQUAL,
                            source=ThresholdSource.FIXED_DEFAULT)


def test_operator_and_source_must_be_controlled_members():
    with pytest.raises(TypeError):
        ThresholdParameters(0.5, ">=", ThresholdSource.FIXED_DEFAULT)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ThresholdParameters(0.5, ThresholdOperator.GREATER_OR_EQUAL, "fixed")  # type: ignore[arg-type]


def test_probability_equal_to_the_threshold_obeys_the_declared_operator():
    """The reason an operator is provenance rather than pedantry: the two differ
    at exactly `prob == threshold`, which is what a maximally uncertain model
    emits and what a two-model average produces whenever the pair disagrees."""
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert apply_decision_threshold(p, threshold=0.5, operator=">=").tolist() == [True] * 4
    assert apply_decision_threshold(p, threshold=0.5, operator=">").tolist() == [False] * 4
    with pytest.raises(ValueError, match="unsupported decision operator"):
        apply_decision_threshold(p, threshold=0.5, operator="≥")


@pytest.mark.parametrize("name,kernel", [
    ("matthews_correlation_coefficient", matthews_correlation_coefficient),
    ("f1", f1_at_threshold),
])
def test_the_registered_metric_uses_the_declared_threshold(name, kernel):
    rng = np.random.default_rng(19)
    n = 300
    y = rng.binomial(1, 0.5, n).astype(float)
    p = np.clip(0.5 + 0.2 * (2 * y - 1) + rng.normal(0, 0.2, n), 0, 1)

    tp = by_name(name).threshold_parameters
    direct = kernel(y, p, threshold=tp.threshold, operator=tp.operator.value)
    registered = evaluate_registered(_ctx(y, prob=p, score=p))[name]
    assert registered.status is MetricStatus.OK
    assert registered.value == direct

    at_other = kernel(y, p, threshold=0.7, operator=tp.operator.value)
    assert registered.value != at_other, (
        "this cohort cannot distinguish 0.5 from 0.7, so the assertion above "
        "would hold whichever threshold the registry actually used")


# --------------------------------------------------------------------------- #
# 4. Canonical registry semantics for degenerate cohorts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["matthews_correlation_coefficient", "f1"])
def test_a_zero_denominator_is_undefined_not_zero(name):
    """CANONICAL REGISTRY SEMANTICS.

    scikit-learn returns 0.0 and raises UndefinedMetricWarning while doing so --
    its own warning is the evidence that the 0.0 is a fabrication. Reporting it
    as observed performance would make a constant classifier that never
    discriminated indistinguishable from one that discriminated and found no
    correlation.

    The refusal comes from APPLICABILITY, before dispatch, so the status is
    UNDEFINED -- a property of the cohort. `compute` already rules that an
    applicable metric returning a non-finite value is FAILED, "an implementation
    defect, not a property of the cohort", so letting the kernel's NaN carry this
    meaning would blame the code for the data.
    """
    y = np.zeros(20)
    p = np.full(20, 0.1)                      # nothing predicted positive
    r = evaluate_registered(_ctx(y, prob=p, score=p))[name]
    assert r.status is MetricStatus.UNDEFINED
    # DISTINCT REASONS, 2026-07-28. A shared reason would let the legacy
    # compatibility projection substitute the Matthews value for an F1 undefined
    # for a different cause. The substitution must be authorised by metric
    # identity AND exact reason, so a test accepting either would defeat it.
    expected_reason = ("zero_confusion_margin"
                       if name == "matthews_correlation_coefficient"
                       else "zero_f1_denominator")
    assert r.reason == expected_reason, (
        f"{name} reported {r.reason!r}; the two degenerate conditions -- a "
        "vanishing confusion-matrix margin and a vanishing F1 denominator -- "
        "must be nameable apart")
    assert not np.isfinite(r.value)
    assert r.status is not MetricStatus.FAILED, (
        "a degenerate cohort is not an implementation defect")


@pytest.mark.parametrize("name,kernel", [
    ("matthews_correlation_coefficient", matthews_correlation_coefficient),
    ("f1", f1_at_threshold),
])
def test_the_kernels_agree_with_scikit_learn_where_it_is_defined(name, kernel):
    """The canonical semantics differ from scikit-learn ONLY where scikit-learn
    itself warns that its answer is ill-defined."""
    from sklearn.metrics import f1_score, matthews_corrcoef

    rng = np.random.default_rng(23)
    n = 400
    y = rng.binomial(1, 0.45, n).astype(float)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.18, n), 0, 1)
    hard = (p >= 0.5).astype(int)
    reference = matthews_corrcoef(y, hard) if "matthews" in name else f1_score(y, hard)
    assert kernel(y, p, threshold=0.5, operator=">=") == pytest.approx(
        reference, rel=1e-12, abs=1e-15)


# --------------------------------------------------------------------------- #
# 5. Calibration descriptors share one definition
# --------------------------------------------------------------------------- #
def test_expected_and_maximum_calibration_error_declare_identical_parameters():
    """Two summaries of one table must not be able to declare different tables."""
    ece = by_name("expected_calibration_error").parameters
    mce = by_name("maximum_calibration_error").parameters
    assert dict(ece) == dict(mce)
    assert ece["n_bins"] == 10
    assert ece["binning"] == "equal_width"
    assert ece["interval_convention"] == "[lo,hi);final_closed"


def test_the_maximum_calibration_error_reads_the_shared_table():
    """ON AN INTERIOR-EDGE COHORT, and that is the whole point.

    An earlier draft used a random continuous cohort, where no probability sits
    exactly on an interior decade edge. A second binning loop using the
    SUPERSEDED left-open convention therefore produced the identical answer, and
    the sabotage matrix walked straight through the assertion. That is precisely
    the fixture-blindness that let the two calibration implementations disagree
    for seventeen days, reproduced inside the test written to prevent it.

    The cohort below places mass exactly on 0.5, so a second binning is
    observable.
    """
    from genomic_variant_classifier.evaluation.metrics import CalibrationBins

    p = np.array([0.42] * 40 + [0.47] * 40 + [0.50] * 120 + [0.53] * 40 + [0.58] * 40)
    y = np.array([1.0] * 20 + [0.0] * 20 + [1.0] * 20 + [0.0] * 20 +
                 [1.0] * 110 + [0.0] * 10 + [1.0] * 4 + [0.0] * 36 +
                 [1.0] * 4 + [0.0] * 36)

    # Prove the cohort can distinguish the conventions before relying on it.
    edges = np.linspace(0.0, 1.0, 11)
    left_open = np.clip(np.digitize(p, edges[1:-1], right=True), 0, 9)
    documented = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    assert not np.array_equal(left_open, documented), (
        "the cohort no longer separates the binning conventions, so this "
        "assertion would hold whichever table the metric read")

    bins = CalibrationBins.from_predictions(y, p, n_bins=10)
    result = evaluate_registered(_ctx(y, prob=p, score=p))["maximum_calibration_error"]
    assert result.status is MetricStatus.OK
    assert result.value == bins.maximum, (
        "the registered maximum calibration error did not come from the shared "
        "CalibrationBins table")


@pytest.mark.parametrize("kernel,name", [
    (matthews_correlation_coefficient, "matthews_correlation_coefficient"),
    (f1_at_threshold, "f1"),
])
def test_the_kernel_itself_returns_nan_on_a_zero_denominator(kernel, name):
    """CLOSES A GAP THE SABOTAGE MATRIX FOUND.

    The registry refuses a degenerate cohort through APPLICABILITY, before
    dispatch, so no registry-level test ever reaches the kernel's zero-denominator
    branch. That branch was therefore unguarded: replacing its NaN with 0.0 broke
    nothing. The kernel is a public function with its own contract and is tested
    directly here.

    0.0 would state that the classifier was measured and found uncorrelated. NaN
    states that there was nothing to measure. scikit-learn returns 0.0 and raises
    UndefinedMetricWarning while doing so; its own warning is the evidence.
    """
    y = np.zeros(20)
    p = np.full(20, 0.1)                      # nothing predicted positive
    value = kernel(y, p, threshold=0.5, operator=">=")
    assert np.isnan(value), (
        f"{name} returned {value!r} on a cohort with a vanishing denominator; "
        "0.0 there is indistinguishable from a measured absence of correlation")

    y2 = np.ones(20)
    p2 = np.full(20, 0.9)                     # everything predicted positive, all positive
    if name == "matthews_correlation_coefficient":
        assert np.isnan(kernel(y2, p2, threshold=0.5, operator=">=")), (
            "a single-class cohort has a vanishing margin and no defined "
            "correlation coefficient")


def test_empty_bins_do_not_contribute_a_perfect_gap():
    """An empty bin is an absence of observations, not a perfectly calibrated
    one. Counting it as zero would drag the maximum downward."""
    from genomic_variant_classifier.evaluation.metrics import CalibrationBins

    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.95, 0.95, 0.05, 0.05])       # only bins 0 and 9 occupied
    bins = CalibrationBins.from_predictions(y, p, n_bins=10)
    assert bins.n_occupied == 2
    assert bins.maximum == pytest.approx(0.95, abs=1e-12)


# --------------------------------------------------------------------------- #
# 6. Prevalence
# --------------------------------------------------------------------------- #
def test_prevalence_does_not_require_predictions():
    y = np.array([0.0, 1.0, 1.0, 1.0])
    r = evaluate_registered(_ctx(y))["prevalence"]
    assert r.status is MetricStatus.OK
    assert r.value == pytest.approx(0.75)


@pytest.mark.parametrize("labels,expected", [
    (np.zeros(10), 0.0),
    (np.ones(10), 1.0),
    (np.array([0.0, 1.0, 0.0, 1.0]), 0.5),
])
def test_prevalence_is_valid_on_a_single_class_population(labels, expected):
    """Deliberately NOT inheriting the ranking metrics' both-classes rule: an
    all-negative cohort has a prevalence of 0.0, which is a measurement rather
    than a refusal."""
    r = evaluate_registered(_ctx(labels))["prevalence"]
    assert r.status is MetricStatus.OK
    assert r.value == pytest.approx(expected)


def test_prevalence_does_not_narrow_the_population_it_was_handed():
    """Label eligibility is an upstream population decision. Filtering here would
    describe a different denominator than the result names."""
    with pytest.raises(ValueError, match="must not narrow"):
        prevalence(np.array([0.0, 1.0, np.nan]))


# --------------------------------------------------------------------------- #
# 7. Registry completeness
# --------------------------------------------------------------------------- #
def test_the_registry_contains_every_report_quantity():
    """Stronger than testing each new descriptor: it prevents a future report
    field from being added anywhere but the registry, where it would have no
    applicability policy, no certification rule and no declared parameters."""
    declared = {d.name for d in all_metrics() if d.include_in_evaluation_report}
    assert declared == set(REPORT_METRIC_NAMES)


def test_every_report_name_resolves_to_a_descriptor():
    for name in REPORT_METRIC_NAMES:
        assert by_name(name).name == name


# --------------------------------------------------------------------------- #
# 8. THE ACCEPTANCE CRITERION -- nothing moved
# --------------------------------------------------------------------------- #
def _encode(value):
    import math
    if isinstance(value, float):
        if math.isnan(value):
            return "__nan__"
        if math.isinf(value):
            return "__inf__" if value > 0 else "__-inf__"
        return repr(value)
    if isinstance(value, np.floating):
        return _encode(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, bool) or isinstance(value, (int, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if hasattr(value, "value"):
        return f"__enum__:{value.value}"
    return f"__repr__:{value!r}"


def _rebuild_fixture_context(label, snapshot_entry):
    """Regenerate the exact cohorts the snapshot was captured from."""
    rng = np.random.default_rng(20260727)
    cohorts = {}

    n = 500
    y = rng.binomial(1, 0.5, n).astype(float)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.15, n), 0.0, 1.0)
    cohorts["balanced_ok"] = (y, p, p, None)

    n = 400
    y = rng.binomial(1, 0.08, n).astype(float)
    p = np.clip(0.2 + 0.4 * y + rng.normal(0, 0.2, n), 0.0, 1.0)
    cohorts["imbalanced_ok"] = (y, p, p, None)

    cohorts["single_class_positive"] = (np.ones(60), np.linspace(0.01, 0.99, 60),
                                        np.linspace(0.01, 0.99, 60), None)
    cohorts["single_class_negative"] = (np.zeros(60), np.linspace(0.01, 0.99, 60),
                                        np.linspace(0.01, 0.99, 60), None)

    n = 200
    y = rng.binomial(1, 0.5, n).astype(float)
    p = np.clip(rng.random(n), 0.0, 1.0)
    p_bad = p.copy()
    p_bad[:7] = np.nan
    cohorts["nonfinite_probabilities_failed"] = (y, p_bad, p_bad, None)

    feature = np.array([-0.4, 2.1, 0.3, 4.8])
    cohorts["out_of_range_scores"] = (np.array([0.0, 1.0, 0.0, 1.0]), feature, feature, None)

    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    cohorts["clustered_ok"] = (y, p, p, np.array(["G1", "G1", "G2", "G2",
                                                  "G3", "G3", "G4", "G4"]))

    y_full = np.concatenate([rng.binomial(1, 0.5, 100).astype(float), np.full(20, np.nan)])
    p_full = np.clip(rng.random(120), 0.0, 1.0)
    cohorts["label_restricted"] = (y_full, p_full, p_full, None)

    y, prob, score, clusters = cohorts[label]
    y_arr = np.asarray(y, dtype=float)
    attempted = EvaluationPopulation.full(y_arr.size, scope="attempted_cohort",
                                          source_id="snapshot-frame:sha256:0000000000000000")
    mask = np.isfinite(y_arr)
    pop = attempted if mask.all() else attempted.restrict(
        mask, scope="label_eligible", reason="reference_label_withheld")
    return MetricContext(
        y_true=pop.take(y_arr), y_prob=pop.take(np.asarray(prob, dtype=float)),
        y_score=pop.take(np.asarray(score, dtype=float)),
        clusters=None if clusters is None else pop.take(np.asarray(clusters)),
        population=pop)


def test_existing_registry_results_do_not_move():
    """THE ACCEPTANCE CRITERION FOR THIS COMMIT.

    Compared against a snapshot frozen on the 2b-1 tree BEFORE any of this was
    written. Eight fields per result -- status, value with NaN semantics, reason,
    certification eligibility, support, population scope, population fingerprint
    and the WHOLE metadata mapping. No carve-outs and no expected-change list: a
    test carrying an exemption is weaker than one that cannot.
    """
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    frozen_names = set(snapshot["registered_metric_names"])
    assert frozen_names, "the snapshot fixture is empty"

    movements = []
    compared = 0
    for label, entry in snapshot["fixtures"].items():
        results = evaluate_registered(_rebuild_fixture_context(label, entry))
        for metric_name, recorded in entry.items():
            assert metric_name in results, (
                f"{metric_name} disappeared from the registry; the snapshot is "
                "the record of what already existed")
            r = results[metric_name]
            observed = {
                "status": r.status.value,
                "value": _encode(r.value),
                "reason": r.reason,
                "certification_eligible": _encode(r.certification_eligible),
                "n_observations": _encode(r.n_observations),
                "population_scope": r.population_scope,
                "population_fingerprint": r.population_fingerprint,
                "metadata": {str(_encode(k)): _encode(v)
                             for k, v in sorted(r.metadata.items(),
                                                key=lambda kv: str(kv[0]))},
            }
            for field, was in recorded.items():
                compared += 1
                if observed[field] != was:
                    movements.append(f"{label}/{metric_name}.{field}: "
                                     f"{was!r} -> {observed[field]!r}")
    assert compared > 0

    # THE DECLARED MOVEMENT SET, added 2026-07-28 by commit 3b-1a.
    #
    # This oracle was frozen on the 2b-1 tree and had shown ZERO movements
    # through 2b-2, 2b-3, 3a and 3b-0. Commit 3b-1a is the first change that
    # legitimately moves a value in it, because it REVERSES a scientific
    # judgement: single-class calibration was refused as INSUFFICIENT_SUPPORT and
    # is now computed as OK, on the grounds that calibration and discrimination
    # are different estimands and only the latter requires two classes.
    #
    # The fixture is NOT regenerated. Regenerating it would destroy the only
    # record of what the registry produced before the correction, and the
    # self-validating header exists precisely so that regeneration is detectable.
    # The exceptions are declared here instead, BY IDENTITY: a count alone would
    # accept the wrong ten.
    #
    # The LEGACY REPORT oracle is unaffected and still shows zero movements
    # across all 480 of its values -- the two surfaces are checked independently
    # and, on this commit, correctly disagree.
    declared = {
        (fixture, "expected_calibration_error", field)
        for fixture in ("single_class_negative", "single_class_positive")
        for field in ("status", "reason", "value", "certification_eligible",
                      "metadata")
    }
    observed = {tuple(m.split(":")[0].split("/", 1)[0:1] + m.split(":")[0].split("/", 1)[1].rsplit(".", 1))
                for m in movements}
    undeclared = observed - declared
    missing = declared - observed
    assert not undeclared, (
        f"{len(undeclared)} UNDECLARED movement(s) in the typed registry oracle:"
        f"\n  " + "\n  ".join(sorted(f"{a}/{b}.{c}" for a, b, c in undeclared)))
    assert not missing, (
        "the declared movement set expects changes that did not occur, so it no "
        f"longer describes this commit: {sorted(missing)}")


def test_exactly_four_result_names_were_added():
    from genomic_variant_classifier.evaluation.registry import names

    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    added = set(names()) - set(snapshot["registered_metric_names"])
    removed = set(snapshot["registered_metric_names"]) - set(names())
    assert added == {"maximum_calibration_error", "matthews_correlation_coefficient",
                     "f1", "prevalence"}
    assert not removed


def test_the_snapshot_fixture_is_not_silently_empty():
    """Guards the guard: a snapshot that regenerated itself to nothing would make
    every assertion above vacuous."""
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    n_results = sum(len(v) for v in snapshot["fixtures"].values())
    assert len(snapshot["fixtures"]) >= 8
    assert n_results >= 48
    statuses = {r["status"] for f in snapshot["fixtures"].values() for r in f.values()}
    assert {"ok", "undefined", "failed", "not_applicable"} <= statuses, (
        "the snapshot must exercise refusals and failures, not only successes")


def test_the_snapshot_is_self_validating():
    """A fixture must not be able to validate a registry it did not come from.

    The decisive check is the LAST one. The snapshot records the schema version
    in force when it was captured. If that ever equals the CURRENT schema
    version, the fixture was regenerated on the current tree -- at which point it
    is a photograph of the thing it was supposed to be checking, and every
    identity comparison in `test_existing_registry_results_do_not_move` passes
    for the only reason that guarantees nothing.

    A stale fixture is a visible failure. A silently regenerated one is not, and
    is the more dangerous of the two.
    """
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    assert snapshot["snapshot_version"] == "2b-1"
    assert snapshot["captured_from_commit"] == "683b514"
    assert snapshot["n_metrics"] == len(snapshot["registered_metric_names"]) == 6
    assert snapshot["n_results"] == sum(len(v) for v in snapshot["fixtures"].values()) == 48

    assert snapshot["registry_schema_at_capture"] < REGISTRY_SCHEMA_VERSION, (
        f"the snapshot records schema {snapshot['registry_schema_at_capture']} "
        f"and the registry is at {REGISTRY_SCHEMA_VERSION}. If these are equal "
        "the fixture was regenerated on the current tree and proves nothing; if "
        "the snapshot is ahead, the fixture belongs to a later registry than the "
        "one under test.")


# --------------------------------------------------------------------------- #
# 9. Descriptor immutability audit
#
# Descriptors are now the semantic authority: the threshold a metric applies, the
# inputs it consumes, the classification it carries. Evaluation must therefore be
# incapable of altering them. An in-place edit -- `descriptor.parameters[...] = x`
# in a kernel, a mutated `required_inputs` -- would change what every LATER
# evaluation means, and an ordinary numerical test would not notice because the
# first evaluation of a run would still be correct.
#
# The audit deep-copies the whole descriptor graph, runs every metric over a
# range of cohorts including degenerate and failing ones, and then compares the
# graph field by field.
# --------------------------------------------------------------------------- #
def _descriptor_fingerprint(descriptor):
    """Everything about a descriptor that evaluation must not be able to change."""
    tp = descriptor.threshold_parameters
    return {
        "name": descriptor.name,
        "required_inputs": tuple(sorted(i.value for i in descriptor.required_inputs)),
        "result_kind": descriptor.result_kind.value,
        "display_name": descriptor.display_name,
        "description": descriptor.description,
        "requires_clusters": descriptor.requires_clusters,
        "output_kind": descriptor.output_kind.value,
        "include_in_evaluation_report": descriptor.include_in_evaluation_report,
        "parameters": tuple(sorted((k, repr(v)) for k, v in descriptor.parameters.items())),
        "threshold": None if tp is None else (tp.threshold, tp.operator.value, tp.source.value),
        "function_id": id(descriptor.function),
        "applicability_id": id(descriptor.applicability),
    }


def test_evaluation_never_mutates_the_descriptor_graph():
    """The audit. Nothing an evaluation does may alter the semantic authority."""
    before = {d.name: _descriptor_fingerprint(d) for d in all_metrics()}

    rng = np.random.default_rng(41)
    cohorts = [
        (rng.binomial(1, 0.5, 200).astype(float), np.clip(rng.random(200), 0, 1)),
        (np.zeros(30), np.full(30, 0.1)),                       # degenerate margin
        (np.ones(30), np.full(30, 0.9)),                        # single class
        (np.array([0.0, 1.0, 0.0, 1.0]), np.array([-0.4, 2.1, 0.3, 4.8])),  # not probabilities
    ]
    nonfinite = np.clip(rng.random(50), 0, 1)
    nonfinite[:5] = np.nan
    cohorts.append((rng.binomial(1, 0.5, 50).astype(float), nonfinite))

    for y, p in cohorts:
        evaluate_registered(_ctx(y, prob=p, score=p))

    after = {d.name: _descriptor_fingerprint(d) for d in all_metrics()}
    assert after == before, (
        "evaluation mutated the descriptor graph; descriptors are the semantic "
        "authority and an in-place edit would silently change what every LATER "
        "evaluation means")


# SHARED function and predicate objects. An earlier draft built a fresh lambda
# inside `_probe_descriptor`, so `id(function)` differed on every call and every
# fingerprint comparison was trivially unequal -- the guard passed whichever
# field the fingerprint had stopped covering. Holding these fixed means the ONLY
# difference between two probes is the field under test.
def _PROBE_FN(ctx):
    return 0.0


def _PROBE_APPLICABILITY(ctx):
    return None


def _probe_descriptor(**overrides):
    """A real descriptor, so the fingerprint is exercised rather than simulated."""
    from genomic_variant_classifier.evaluation.registry import MetricDescriptor

    spec = dict(
        name="probe", function=_PROBE_FN,
        required_inputs=frozenset({MetricInput.LABELS}),
        applicability=_PROBE_APPLICABILITY,
        result_kind=ResultKind.PREDICTION_METRIC,
        display_name="Probe", description="probe descriptor",
    )
    spec.update(overrides)
    return MetricDescriptor(**spec)


def _with_threshold(tp):
    def adapter(ctx):
        return 0.0
    adapter._threshold_parameters = tp
    return adapter


def test_the_immutability_audit_would_notice_a_mutation():
    """Guards the guard -- REWRITTEN 2026-07-27 because the first version could
    not fail.

    It built `{**base, "field": other}` as a dict literal, which ADDS the key
    even when `_descriptor_fingerprint` has stopped emitting it, so the
    inequality held whether or not the field was covered. Removing three fields
    from the fingerprint left the suite green. The sabotage matrix caught it.

    The correct form compares the fingerprints of two REAL descriptors differing
    in exactly one field, which exercises the function instead of simulating it.
    """
    from genomic_variant_classifier.evaluation.registry import MetricOutputKind

    base = _probe_descriptor()
    base_print = _descriptor_fingerprint(base)

    variants = {
        "parameters": _probe_descriptor(parameters={"n_bins": 20}),
        "result_kind": _probe_descriptor(result_kind=ResultKind.POPULATION_STATISTIC),
        "required_inputs": _probe_descriptor(
            required_inputs=frozenset({MetricInput.LABELS, MetricInput.PROBABILITIES})),
        "display_name": _probe_descriptor(display_name="Different"),
        "description": _probe_descriptor(description="different description"),
        "include_in_evaluation_report": _probe_descriptor(include_in_evaluation_report=False),
        "output_kind": _probe_descriptor(output_kind=MetricOutputKind.INTERVAL),
        "function": _probe_descriptor(function=lambda ctx: 1.0),
        "applicability": _probe_descriptor(applicability=lambda ctx: True),
        # these two intentionally DO introduce new objects, which is the
        # difference under test for the identity fields
        "threshold": _probe_descriptor(
            function=_with_threshold(ThresholdParameters(
                0.7, ThresholdOperator.GREATER, ThresholdSource.CALIBRATED))),
    }
    for field, variant in variants.items():
        assert _descriptor_fingerprint(variant) != base_print, (
            f"_descriptor_fingerprint does not cover {field!r}; the immutability "
            "audit would not notice a mutation of it")

    # THE THRESHOLD NEEDS A DIRECT STRUCTURAL ASSERTION, not a differential one.
    # A threshold lives ON the kernel adapter, so any descriptor carrying a
    # different threshold necessarily carries a different function object, and
    # `function_id` alone would make the two fingerprints differ. The
    # differential form therefore cannot isolate this field: it would report
    # coverage that does not exist. Asserted structurally instead.
    mcc_print = _descriptor_fingerprint(by_name("matthews_correlation_coefficient"))
    assert "threshold" in mcc_print, (
        "_descriptor_fingerprint does not emit the threshold at all; a silently "
        "altered decision threshold would pass the immutability audit")
    assert mcc_print["threshold"] == (0.5, ">=", "fixed_default"), (
        f"the fingerprint reports {mcc_print['threshold']!r} rather than the "
        "descriptor's declared threshold provenance")
    assert _descriptor_fingerprint(_probe_descriptor())["threshold"] is None, (
        "a descriptor without a threshold must report None rather than omitting "
        "the key, so absence and 'not covered' stay distinguishable")


def test_descriptor_parameters_resist_in_place_edits_during_evaluation():
    """The specific defect the audit exists to catch, attempted directly."""
    d = by_name("f1")
    with pytest.raises(TypeError):
        d.parameters["decision_threshold"] = 0.9      # type: ignore[index]
    with pytest.raises(AttributeError):
        d.required_inputs.add(MetricInput.SCORES)     # type: ignore[attr-defined]
    assert d.parameters["decision_threshold"] == 0.5
