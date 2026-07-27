"""Commit 2a: the fail-closed prediction-input contract, and the transitional
boundary between prediction validation and label selection.

THE RULING, 2026-07-27
======================
    No numerical kernel may select, filter, normalise or redefine its
    evaluation population. Population construction is an explicit upstream
    operation, and every result must describe exactly that population.

WHAT WAS WRONG
--------------
Every kernel in metrics.py routed through `clean_arrays`, which dropped
non-finite rows on one joint mask covering labels, scores and probabilities
alike. A metric therefore returned a value computed over a silently narrowed
population while `MetricContext.support()` named the wider one. Measured:
twenty non-finite probabilities in a thousand rows produced a Brier score over
980 rows, reported as n_observations = 1000, status ok, certification_eligible
True.

THE DISTINCTION THAT MAKES THIS COHERENT
-----------------------------------------
A non-finite predicted probability is a MODEL-OUTPUT FAILURE. A withheld
reference label is an ORDINARY MISSING OBSERVATION and is first-class in this
project by design. The two must not share a mask:

    labels        selected upstream, by a named transitional selector,
                  pending EvaluationPopulation
    predictions   never selected; validated, and failed closed
    kernels       assert their prediction inputs
    registry      owns the refusal and its diagnostics

THE TRANSITIONAL STATE IS DELIBERATE AND IS TRIPWIRED
------------------------------------------------------
Label selection has NOT yet moved to EvaluationPopulation. That is a staged
decision, not an oversight. `test_the_transitional_label_boundary_is_documented`
and `test_the_transitional_label_selector_is_a_named_deletion_target` exist so
the transitional state cannot quietly become permanent: both fail the moment the
documentation or the named selector is removed without the replacement arriving.

WHAT IS EXPLICITLY NOT CLAIMED
-------------------------------
`metrics.evaluate` is NOT fail-closed and this module proves it rather than
glossing it. It reports `n_input`, `n` and `n_dropped`, which is
population-accounting TRANSPARENCY, and it still computes over the survivors.
Transparency is not validity. It is retained unchanged as a compatibility
interface and is not a certifiable path.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from genomic_variant_classifier.evaluation import canonical as canonical_module
from genomic_variant_classifier.evaluation import metrics as metrics_module
from genomic_variant_classifier.evaluation import registry as registry_module
from genomic_variant_classifier.evaluation.capabilities import (
    MetricMetadataKey,
    MetricStatus,
)
from genomic_variant_classifier.evaluation.metrics import (
    auroc,
    brier_score,
    evaluate,
    expected_calibration_error,
    log_loss,
    select_finite_reference_labels,
)
from genomic_variant_classifier.evaluation.registry import (
    MetricContext,
    evaluate_registered,
)

N_ROWS = 1000
N_BAD = 20


@pytest.fixture
def cohort_with_nonfinite_predictions():
    """A thousand rows of which twenty carry a non-finite predicted probability."""
    rng = np.random.default_rng(20260727)
    y = rng.binomial(1, 0.5, N_ROWS).astype(float)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.15, N_ROWS), 0.0, 1.0)
    p[:N_BAD] = np.nan
    return y, p


def _ctx(y, prob=None, score=None, scope="prediction_contract_cohort"):
    return MetricContext(y_true=y, y_prob=prob, y_score=score,
                         population_scope=scope)


# --------------------------------------------------------------------------- #
# 1. The kernels assert; they do not repair
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel,arg_is_probability", [
    (auroc, False),
    (brier_score, True),
    (log_loss, True),
    (expected_calibration_error, True),
])
def test_nonfinite_predictions_are_not_silently_dropped(kernel, arg_is_probability):
    y = np.array([0.0, 1.0, 0.0, 1.0])
    predictions = np.array([0.1, 0.9, np.nan, 0.8])
    with pytest.raises(ValueError, match="non-finite model outputs"):
        kernel(y, predictions)


def test_validation_is_metric_specific_not_universal():
    """A probability-only metric must not fail because an unrelated score array
    carries a non-finite value. A universal assertion over every supplied array
    would contradict the descriptor-specific accounting the registry keeps."""
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    score = prob.copy()
    score[:3] = np.nan

    results = evaluate_registered(_ctx(y, prob=prob, score=score))

    assert results["brier_score"].status is MetricStatus.OK, (
        "brier_score reads probabilities only; a corrupt SCORE array must not "
        "fail it")
    assert results["auroc"].status is MetricStatus.FAILED, (
        "auroc reads scores, so it must fail on non-finite scores")
    assert results["auroc"].reason == "nonfinite_predicted_scores"
    assert results["auroc"].n_nonfinite_scores == 3


def test_range_and_finiteness_are_different_categories():
    """A vector outside [0, 1] was never a probability vector, and the SAME
    array remains a valid score for a ranking metric on the same rows. Raising
    on range would conflate 'not a probability' with 'no prediction' and break a
    landed contract."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    feature = np.array([-0.4, 2.1, 0.3, 4.8])          # StandardScaler output
    assert np.isnan(brier_score(y, feature))
    assert np.isnan(expected_calibration_error(y, feature))
    assert np.isfinite(auroc(y, feature))


# --------------------------------------------------------------------------- #
# 2. The registry refuses before dispatch, over the ATTEMPTED population
# --------------------------------------------------------------------------- #
def test_nonfinite_probability_support_reports_raw_population(
        cohort_with_nonfinite_predictions):
    y, p = cohort_with_nonfinite_predictions
    r = evaluate_registered(_ctx(y, prob=p, score=p))["brier_score"]
    assert r.n_observations == N_ROWS, (
        f"support reported {r.n_observations}; it must name the ATTEMPTED "
        f"population of {N_ROWS}, not the {N_ROWS - N_BAD} rows that happened "
        "to carry a finite prediction")
    assert r.n_nonfinite_probabilities == N_BAD
    assert r.n_finite_probabilities == N_ROWS - N_BAD


def test_brier_is_not_finite_after_implicit_row_deletion(
        cohort_with_nonfinite_predictions):
    y, p = cohort_with_nonfinite_predictions
    r = evaluate_registered(_ctx(y, prob=p, score=p))["brier_score"]
    assert r.status is MetricStatus.FAILED
    assert not np.isfinite(r.value), (
        "a finite Brier score here would be a value over 980 undisclosed rows")
    assert r.reason == "nonfinite_predicted_probabilities"


def test_nonfinite_probabilities_block_certification(
        cohort_with_nonfinite_predictions):
    y, p = cohort_with_nonfinite_predictions
    for name, r in evaluate_registered(_ctx(y, prob=p, score=p)).items():
        assert r.certification_eligible is not True, (
            f"{name} is certification-eligible despite non-finite model output")


def test_registry_kernel_population_matches_reported_support():
    """On a clean cohort the kernel computes over exactly the rows named. The
    equality is the whole contract, so it is asserted directly."""
    rng = np.random.default_rng(5)
    n = 400
    y = rng.binomial(1, 0.5, n).astype(float)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.15, n), 0.0, 1.0)

    seen = {}
    originals = {}
    for name in ("auroc", "brier_score", "expected_calibration_error"):
        originals[name] = getattr(metrics_module, name)

        def spy(y_arg, pred_arg, *a, _name=name, **kw):
            seen[_name] = int(np.asarray(y_arg, dtype=float).ravel().size)
            return originals[_name](y_arg, pred_arg, *a, **kw)

        setattr(metrics_module, name, spy)
    try:
        results = evaluate_registered(_ctx(y, prob=p, score=p))
    finally:
        for name, fn in originals.items():
            setattr(metrics_module, name, fn)

    for name, rows_seen in seen.items():
        assert rows_seen == results[name].n_observations == n, (
            f"{name} computed over {rows_seen} rows but reported "
            f"{results[name].n_observations}")


def test_a_clean_cohort_carries_no_failure_diagnostics():
    """The diagnostics belong to a failure. An OK result computed over every row
    it was given and must not carry counts implying otherwise."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    r = evaluate_registered(_ctx(y, prob=p, score=p))["brier_score"]
    assert r.status is MetricStatus.OK
    assert r.n_nonfinite_probabilities is None
    assert r.n_finite_probabilities is None
    assert MetricMetadataKey.N_NONFINITE_PROBABILITIES not in r.metadata


# --------------------------------------------------------------------------- #
# 3. The legacy composite is NOT fail-closed, and this is proven, not glossed
# --------------------------------------------------------------------------- #
def test_legacy_evaluate_projects_before_strict_kernels():
    """`evaluate` builds its own population and reports the narrowing. That is
    what lets it keep calling kernels that now refuse non-finite input."""
    result = evaluate(
        np.array([0.0, 1.0, np.nan, 0.0, 1.0]),
        np.array([0.1, 0.9, 0.4, 0.2, 0.8]),
        prob=np.array([0.1, 0.9, 0.4, 0.2, 0.8]),
    )
    assert result["n_input"] == 5
    assert result["n"] == 4
    assert result["n_dropped"] == 1


def test_legacy_evaluate_does_not_hide_nonfinite_predictions():
    """THE POINT OF THIS TEST IS THAT IT DOCUMENTS A NON-CONFORMING PATH.

    `evaluate` still computes over survivors when a PREDICTION is non-finite. It
    discloses the narrowing rather than concealing it, which is transparency, not
    fail-closed behaviour. Asserting the disclosure pins the difference so that
    `evaluate` can never be cited as evidence that strict kernels tolerate
    filtering.
    """
    result = evaluate(
        np.array([0.0, 1.0, 0.0, 1.0, 1.0]),
        np.array([0.1, 0.9, np.nan, 0.2, 0.8]),
        prob=np.array([0.1, 0.9, np.nan, 0.2, 0.8]),
    )
    assert result["n_input"] == 5
    assert result["n"] == 4
    assert result["n_dropped"] == 1, (
        "the dropped prediction must be DISCLOSED, not concealed")
    assert np.isfinite(result["auroc"]), (
        "this path computes over survivors; if it ever stops doing so, the "
        "compatibility contract has changed and that is a deliberate decision "
        "for its own commit")

    doc = evaluate.__doc__ or ""
    assert "NOT A CERTIFIABLE PATH" in doc, (
        "the non-certifiable status must be stated where a caller reads it")


def test_the_registry_never_routes_through_the_legacy_composite():
    """Structural, on the abstract syntax tree, not a text grep -- a grep cannot
    tell a call from a docstring, which tripped an earlier guard in this stack."""
    import ast

    source = inspect.getsource(registry_module)
    tree = ast.parse(source)
    imported, called = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "metrics":
            imported.update(a.name for a in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    for banned in ("evaluate", "stratified_evaluate"):
        assert banned not in imported, (
            f"the registry imports metrics.{banned}; the legacy composite is a "
            "survivor-filtering path and the registry must not route through it")
        assert banned not in called, f"the registry calls {banned}"


# --------------------------------------------------------------------------- #
# 4. The transitional boundary, tripwired
# --------------------------------------------------------------------------- #
def test_the_transitional_label_boundary_is_documented():
    """Fails the moment the seam stops declaring the transitional state, so the
    temporary arrangement cannot silently become permanent."""
    module_doc = canonical_module.__doc__ or ""
    assert "EvaluationPopulation" in module_doc, (
        "the seam's MODULE docstring -- where the contract is stated -- must "
        "name what supersedes the transitional label mask. An occurrence "
        "elsewhere in the file does not state the contract, and a tripwire "
        "satisfied by any occurrence anywhere is one the sabotage matrix "
        "walked straight through.")
    assert "TRANSITIONAL" in module_doc.upper()
    assert "PREDICTIONS are validated, never selected" in module_doc, (
        "the seam must state the prediction contract positively, not merely "
        "omit the false claim")

    source = inspect.getsource(canonical_module)
    assert "joint mask" not in source.lower(), (
        "the seam still claims one joint mask over labels and predictions; "
        "predictions are no longer selected, so that statement is false")


def test_the_transitional_label_selector_is_a_named_deletion_target():
    """The residual population decision must be a named function, not an
    anonymous clause inside `clean_arrays`, so the next commit has one precise
    thing to delete."""
    assert callable(select_finite_reference_labels)
    mask = select_finite_reference_labels(np.array([0.0, 1.0, np.nan, 1.0]))
    assert mask.dtype == bool
    assert mask.tolist() == [True, True, False, True]

    doc = select_finite_reference_labels.__doc__ or ""
    assert "TRANSITIONAL" in doc.upper()
    assert "EvaluationPopulation" in doc

    clean_source = inspect.getsource(metrics_module.clean_arrays)
    assert "select_finite_reference_labels" in clean_source, (
        "clean_arrays must delegate the label decision to the named selector "
        "rather than owning it")


def test_the_label_selector_is_not_applied_to_predictions():
    """Labels are selected; predictions are validated. Applying the selector to
    a prediction array would reintroduce the defect under a new name."""
    source = inspect.getsource(metrics_module)
    kernel_bodies = [
        inspect.getsource(fn)
        for fn in (auroc, brier_score, log_loss, expected_calibration_error)
    ]
    for body in kernel_bodies:
        assert "select_finite_reference_labels" not in body, (
            "a kernel applies the label selector to its own inputs; kernels "
            "must assert their prediction contract, never select on it")
    assert "def select_finite_reference_labels" in source


@pytest.mark.parametrize("metric_name", list(registry_module.names()))
def test_every_metric_refuses_at_the_gate_not_by_raising(
        metric_name, cohort_with_nonfinite_predictions):
    """CLOSES A GAP THE SABOTAGE MATRIX FOUND.

    Removing the gate from one applicability predicate still produced a FAILED
    result, because the strict kernel raised and `compute` caught it. Status
    alone therefore cannot distinguish "refused before dispatch" from "blew up
    during dispatch" -- and only the first carries the diagnostics or leaves the
    population untouched. Asserting the REASON is what separates them.
    """
    y, p = cohort_with_nonfinite_predictions
    r = evaluate_registered(_ctx(y, prob=p, score=p))[metric_name]

    assert r.status is MetricStatus.FAILED
    assert r.reason in {"nonfinite_predicted_probabilities",
                        "nonfinite_predicted_scores"}, (
        f"{metric_name} failed with reason {r.reason!r}. "
        "'metric_computation_failed' means the kernel raised because the gate "
        "did not fire: the refusal must happen BEFORE dispatch.")
    assert r.n_observations == N_ROWS
    diagnostics = (r.n_nonfinite_probabilities, r.n_nonfinite_scores)
    assert N_BAD in diagnostics, (
        f"{metric_name} carries no non-finite count; a gate refusal always does")


def test_a_probability_metric_is_unaffected_by_nonfinite_scores_at_the_gate():
    """The registry-level counterpart of metric-specific validation. Catches a
    gate widened to check every prediction array regardless of descriptor."""
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    score = prob.copy()
    score[:2] = np.nan
    results = evaluate_registered(_ctx(y, prob=prob, score=score))
    for name in ("brier_score", "log_loss", "expected_calibration_error"):
        assert results[name].status is MetricStatus.OK, (
            f"{name} consumes probabilities only and must not be failed by a "
            "corrupt score array")


def test_the_probability_range_guard_runs_before_the_finiteness_assertion():
    """ORDER IS THE CONTRACT, and it is not observable from the values alone.

    `is_probability` must be consulted first so an out-of-range vector returns
    NaN, per the landed contract. If the finiteness assertion were moved ahead of
    it, an out-of-range vector would RAISE instead, silently converting a
    documented NaN into an exception for every caller passing a raw feature.
    """
    y = np.array([0.0, 1.0, 0.0, 1.0])
    out_of_range = np.array([-0.4, 2.1, 0.3, 4.8])
    assert np.isnan(brier_score(y, out_of_range))

    both = np.array([-0.4, 2.1, np.nan, 4.8])
    assert np.isnan(brier_score(y, both)), (
        "out-of-range must win over non-finite: the vector was never a "
        "probability vector, so there is no model output to call failed")
