"""Population wiring: every consumer describes the rows the report names.

WHY THIS FILE EXISTS
====================
Ruled 2026-07-27:

    No numerical kernel may select, filter, normalise or redefine its evaluation
    population. Population construction is an explicit upstream operation, and
    every result must describe exactly that population.

Commit 2a enforced that half of it -- predicted scores and probabilities fail
closed rather than being silently filtered -- and DELIBERATELY left the label
half standing, because withheld labels are first-class in this project and are
carried as NaN by `CanonicalVariantTable`, so selecting on them is a POPULATION
decision. `population.py` was written as the replacement and, until POP-1a, had
no call site in production.

WHAT WAS MEASURED BEFORE THIS FILE WAS WRITTEN
----------------------------------------------
On 2026-08-01, against the installed package at 960f807, `registry.compute`
returned this for `positive_predictive_value` on y = [1, 1, 0, nan]:

    value 1.0    status ok    reason None
    CERTIFICATION_ELIGIBLE True
    N_OBSERVATIONS 4
    POPULATION_FINGERPRINT sha256:9ff577fc...

A value computed over THREE rows, carrying the FOUR-row population's size and
fingerprint, certified eligible, with no reason and no diagnostic. The narrowing
happened inside `metrics.clean_arrays`, where nothing downstream could see it.
`registry.py:530-535` records the same defect shape on the probability axis --
a Brier score over 980 rows reported as n_observations = 1000 -- which is what
the 2026-07-27 ruling was written to eliminate.

Every expectation below is a number this suite's subject actually produced after
POP-1a was applied, on 2026-08-01. None is predicted.

WHAT IS DELIBERATELY NOT ASSERTED HERE
--------------------------------------
Membership fingerprints are not hard-coded. A fingerprint is a function of the
source identity and the surviving indices, so pinning a literal would fail the
day a fixture's `source_id` changed and would say nothing about the property
under test. What is asserted is the RELATIONSHIP: that the attempted and
label-eligible fingerprints differ, and that the one a report carries equals the
one a directly constructed restricted population produces.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.evaluation.capabilities import (
    MetricMetadataKey, MetricStatus)
from genomic_variant_classifier.evaluation.evaluator import (
    ClinicalEvaluator, compare_models)
from genomic_variant_classifier.evaluation.population import EvaluationPopulation

SOURCE = "population-wiring-fixture:sha256:0000000000000000"
NAN = float("nan")

# Four rows, the last withheld. Chosen so that the label-eligible cohort is
# still BINARY -- [1, 1, 0] -- because a single-class cohort would refuse the
# ranking family for a reason unrelated to the population, and the test would
# then pass for the wrong reason.
WITHHELD_Y = np.array([1.0, 1.0, 0.0, NAN])
WITHHELD_P = np.array([0.9, 0.1, 0.2, 0.8])

FULL_Y = np.array([1.0, 1.0, 0.0, 0.0])
FULL_P = np.array([0.9, 0.6, 0.2, 0.1])


def _evaluator() -> ClinicalEvaluator:
    """Few bootstrap replicates: no interval here depends on the count, and the
    default of 1000 would make every test in this file slow for no signal."""
    return ClinicalEvaluator(n_bootstrap=10, random_state=0)


def _metadata_of(report, metric_name: str) -> dict:
    result = report.metric_results[metric_name]
    return dict(result.metadata)


def _scope_of(report, metric_name: str = "prevalence"):
    return _metadata_of(report, metric_name)[MetricMetadataKey.POPULATION_SCOPE]


def _observations_of(report, metric_name: str = "prevalence"):
    return _metadata_of(report, metric_name)[MetricMetadataKey.N_OBSERVATIONS]


def _fingerprint_of(report, metric_name: str = "prevalence"):
    return _metadata_of(report, metric_name)[MetricMetadataKey.POPULATION_FINGERPRINT]


# --------------------------------------------------------------------------- #
# A1 -- the reported population is the one the metrics were computed over
# --------------------------------------------------------------------------- #

def test_a_withheld_label_narrows_the_reported_population():
    """THE ACCEPTANCE CRITERION FOR POP-1a.

    Before this commit the same cohort produced n_observations = 4 beside a
    value computed over three rows. The count and the value must now agree.
    """
    report = _evaluator().evaluate(WITHHELD_Y, WITHHELD_P, source_id=SOURCE)

    assert report.n_samples == 3
    assert report.n_pathogenic == 2
    assert report.n_benign == 1
    assert _observations_of(report) == 3
    assert _scope_of(report) == "label_eligible"


def test_the_prevalence_describes_the_label_eligible_rows_not_the_attempted_ones():
    """Two of three eligible rows are pathogenic, not two of four.

    Prevalence is the field where the two denominators are easiest to confuse,
    because 2/4 and 2/3 are both plausible-looking numbers for this cohort.
    """
    report = _evaluator().evaluate(WITHHELD_Y, WITHHELD_P, source_id=SOURCE)
    assert report.prevalence == pytest.approx(2 / 3, abs=1e-4)


# --------------------------------------------------------------------------- #
# A2 -- EVERY consumer, named individually
# --------------------------------------------------------------------------- #

def test_every_registered_metric_reports_the_same_population():
    """Named per metric rather than sampled.

    A projection that reaches thirteen of fourteen consumers produces a
    fingerprint that is ACTIVELY WRONG rather than merely over-broad, and the
    one it missed is the one a spot check will not visit.
    """
    report = _evaluator().evaluate(WITHHELD_Y, WITHHELD_P, source_id=SOURCE)

    scopes, counts, fingerprints = set(), set(), set()
    for name, result in report.metric_results.items():
        metadata = dict(result.metadata)
        assert MetricMetadataKey.POPULATION_SCOPE in metadata, name
        scopes.add(metadata[MetricMetadataKey.POPULATION_SCOPE])
        counts.add(metadata[MetricMetadataKey.N_OBSERVATIONS])
        fingerprints.add(metadata[MetricMetadataKey.POPULATION_FINGERPRINT])

    assert scopes == {"label_eligible"}
    assert counts == {3}
    assert len(fingerprints) == 1


def test_the_operating_points_are_computed_over_the_projected_rows():
    """The three operating points take `y` and `p` positionally and have no
    length check of their own, so nothing but this would catch them being handed
    the attempted arrays."""
    report = _evaluator().evaluate(WITHHELD_Y, WITHHELD_P, source_id=SOURCE)

    for field in ("at_sensitivity_90", "at_sensitivity_95", "at_high_ppv"):
        point = getattr(report, field)
        assert point is not None, field
        total = point.n_tp + point.n_fp + point.n_fn + point.n_tn
        assert total == 3, f"{field} counted {total} rows, not 3"


def test_the_metadata_frame_is_projected_with_the_arrays():
    """`_gene_error_analysis` writes columns into `meta` from `y` and `p`. A
    four-row frame beside three-row arrays is either a broadcasting error or,
    worse, a silent misalignment that produces plausible per-gene counts for the
    wrong rows."""
    meta = pd.DataFrame({
        "gene_symbol": ["BRCA1", "BRCA1", "TP53", "TP53"],
        "consequence": ["missense_variant"] * 4,
    })
    report = _evaluator().evaluate(WITHHELD_Y, WITHHELD_P, meta=meta,
                                   source_id=SOURCE)

    assert report.n_samples == 3
    counted = sum(row.n_variants for row in report.gene_errors)
    assert counted == 3, (
        f"the per-gene analysis covered {counted} rows for a population of 3; "
        "the metadata frame was not projected with the arrays")


# --------------------------------------------------------------------------- #
# A3 -- a fully labelled cohort must acquire nothing
# --------------------------------------------------------------------------- #

def test_a_fully_labelled_cohort_reports_the_attempted_scope():
    """The `mask.all()` guard is not a workaround for the strict-narrowing rule.

    `EvaluationPopulation.restrict` refuses a mask that removes nothing, because
    `label_eligible(n=4)` beneath `attempted_cohort(n=4)` would assert a
    restriction that never happened. The guard is what keeps a clean cohort
    honest, and this is the test that would fail if it were removed as
    redundant.
    """
    report = _evaluator().evaluate(FULL_Y, FULL_P, source_id=SOURCE)

    assert report.n_samples == 4
    assert _observations_of(report) == 4
    assert _scope_of(report) == "attempted_cohort"


def test_the_two_cohorts_carry_different_fingerprints():
    """A fingerprint that did not move when the row set moved would be the
    original defect wearing a different mask."""
    evaluator = _evaluator()
    full = evaluator.evaluate(FULL_Y, FULL_P, source_id=SOURCE)
    withheld = evaluator.evaluate(WITHHELD_Y, WITHHELD_P, source_id=SOURCE)

    assert _fingerprint_of(full) != _fingerprint_of(withheld)


def test_the_reported_fingerprint_is_the_one_the_surviving_rows_produce():
    """Constructed independently rather than compared against a literal: a
    hard-coded digest would pin the fixture, not the property."""
    attempted = EvaluationPopulation.full(
        WITHHELD_Y.size, scope="attempted_cohort", source_id=SOURCE)
    eligible = attempted.restrict(
        np.isfinite(WITHHELD_Y), scope="label_eligible",
        reason="reference_label_withheld")

    report = _evaluator().evaluate(WITHHELD_Y, WITHHELD_P, source_id=SOURCE)
    assert _fingerprint_of(report) == eligible.membership_fingerprint


# --------------------------------------------------------------------------- #
# A4 -- the narrowing is recorded, not merely performed
# --------------------------------------------------------------------------- #

def test_the_lineage_states_what_was_removed_and_why():
    """The point of the migration, not merely that it happened. The retired
    selector `metrics.select_finite_reference_labels` returned a bare mask: it
    could say which rows survived and nothing about why."""
    attempted = EvaluationPopulation.full(
        WITHHELD_Y.size, scope="attempted_cohort", source_id=SOURCE)
    eligible = attempted.restrict(
        np.isfinite(WITHHELD_Y), scope="label_eligible",
        reason="reference_label_withheld")

    assert eligible.n == 3
    assert eligible.n_excluded_from_parent == 1
    assert eligible.restriction_reason == "reference_label_withheld"
    assert [step["scope"] for step in eligible.lineage()] == [
        "attempted_cohort", "label_eligible"]
    assert "reference_label_withheld" in eligible.describe()


# --------------------------------------------------------------------------- #
# A5 -- no kernel narrows further
# --------------------------------------------------------------------------- #

def test_no_kernel_narrows_the_population_a_second_time():
    """`metrics.clean_arrays` drops non-finite rows on a joint mask. After
    POP-1a its mask must be all-true on the registry path, because the
    population is now the ONLY narrowing operation -- which is the ruling."""
    report = _evaluator().evaluate(WITHHELD_Y, WITHHELD_P, source_id=SOURCE)

    result = report.metric_results["positive_predictive_value"]
    assert result.status is MetricStatus.OK

    # Computed directly on the surviving rows: predicted positive at 0.5 is row
    # 0 alone, which is a true positive, so the value is 1.0 over ONE flagged
    # row. If a kernel had narrowed again, or had seen the attempted array, the
    # flagged set would differ.
    assert result.value == pytest.approx(1.0)
    assert dict(result.metadata)[MetricMetadataKey.N_OBSERVATIONS] == 3


# --------------------------------------------------------------------------- #
# The guard, and the defect it made unreachable
# --------------------------------------------------------------------------- #

def test_an_all_withheld_cohort_is_refused_with_its_lineage():
    """Three failures lie downstream of an empty population -- a division by
    zero in the logger, a `LegacyProjectionError` from the single-class
    area-under-the-precision-recall-curve rule, and an uncharacterised
    multiplication in `print_report`. Guarding them one at a time would be
    patchwork; refusing the cohort makes all three unreachable.

    This is NOT the component-level refusal the input gates perform. Those
    withhold ONE quantity from a cohort that exists. Here there is no cohort.
    """
    with pytest.raises(ValueError, match="label-eligible population is empty"):
        _evaluator().evaluate(np.array([NAN, NAN, NAN]),
                              np.array([0.9, 0.1, 0.2]), source_id=SOURCE)


def test_the_refusal_names_the_count_and_the_lineage():
    """An unexplained refusal is indistinguishable from a defect."""
    with pytest.raises(ValueError) as caught:
        _evaluator().evaluate(np.array([NAN, NAN, NAN]),
                              np.array([0.9, 0.1, 0.2]), source_id=SOURCE)
    message = str(caught.value)
    assert "all 3 attempted row(s)" in message
    assert "label_eligible(n=0, -3 reference_label_withheld)" in message


def test_a_withheld_label_in_a_pandas_series_no_longer_raises():
    """REGRESSION. Before POP-1a, `n_pos = int(y.sum())` ran at line 817 --
    above every input gate -- and raised `ValueError: cannot convert float NaN
    to integer`. Measured 2026-08-01 on six of nine probed dtypes, including a
    plain float array and a pandas Series, which is exactly what the signature
    advertises. `evaluate` died there rather than reaching the gates built for
    that input.
    """
    report = _evaluator().evaluate(pd.Series([1.0, 1.0, 0.0, NAN]),
                                   FULL_P, source_id=SOURCE)
    assert report.n_samples == 3
    assert report.n_pathogenic == 2


# --------------------------------------------------------------------------- #
# A7 -- compare_models restricts once
# --------------------------------------------------------------------------- #

def test_compare_models_reports_the_label_eligible_population():
    """`compare_models` records the shared population's fingerprint and count
    into the comparison artifact while hard-coding
    `comparison_is_like_for_like=True`. If `evaluate` restricted per call, the
    artifact would carry the ATTEMPTED figures while every report it summarises
    described the narrower set -- the POP-1 defect, one layer up.
    """
    y = np.array([1.0, 1.0, 0.0, 0.0, NAN])
    comparison = compare_models(
        y,
        {"model_one": np.array([0.9, 0.6, 0.2, 0.1, 0.5]),
         "model_two": np.array([0.8, 0.7, 0.3, 0.2, 0.4])},
        n_bootstrap=10,
        output_csv=os.devnull,
        source_id=SOURCE)

    assert comparison.population_n == 4
    assert comparison.comparison_is_like_for_like is True
    assert len(comparison.table) == 2


def test_every_model_in_a_comparison_describes_the_same_rows():
    """Claim 1 -- these models were evaluated on the SAME ROWS -- is established
    STRUCTURALLY by handing one object to every model. The restriction must
    therefore happen once, above the loop, not once per call."""
    y = np.array([1.0, 1.0, 0.0, 0.0, NAN])
    attempted = EvaluationPopulation.full(
        y.size, scope="model_comparison_attempted_cohort", source_id=SOURCE)
    eligible = attempted.restrict(
        np.isfinite(y), scope="label_eligible",
        reason="reference_label_withheld")

    comparison = compare_models(
        y,
        {"model_one": np.array([0.9, 0.6, 0.2, 0.1, 0.5]),
         "model_two": np.array([0.8, 0.7, 0.3, 0.2, 0.4])},
        n_bootstrap=10,
        output_csv=os.devnull,
        source_id=SOURCE)

    assert comparison.population_fingerprint == eligible.membership_fingerprint


# --------------------------------------------------------------------------- #
# A8 -- the defence that makes the registry path unfalsifiable
# --------------------------------------------------------------------------- #

def test_an_unprojected_array_is_still_refused_by_the_context():
    """POP-1a relies on this. Once the population is restricted, handing the
    registry an unprojected array raises immediately -- which is why the
    registry path cannot be half-wired. The other consumers have no such check,
    which is why they are named individually above.
    """
    from genomic_variant_classifier.evaluation.registry import MetricContext

    attempted = EvaluationPopulation.full(4, scope="attempted_cohort",
                                          source_id=SOURCE)
    eligible = attempted.restrict(
        np.isfinite(WITHHELD_Y), scope="label_eligible",
        reason="reference_label_withheld")

    with pytest.raises(ValueError, match="ALREADY\\s+PROJECTED|already\\s+projected"):
        MetricContext(y_true=WITHHELD_Y, y_score=WITHHELD_P, population=eligible)

    projected = eligible.take(WITHHELD_Y)
    context = MetricContext(y_true=projected, y_score=eligible.take(WITHHELD_P),
                            population=eligible)
    assert context.n == 3
    assert context.population_scope == "label_eligible"


# --------------------------------------------------------------------------- #
# The ranking-scores channel across a narrowing (POP-1a-fix, 2026-08-01)
# --------------------------------------------------------------------------- #

def test_a_mis_sized_scores_array_is_refused_rather_than_raising(
        monkeypatch):
    """THE CASE THAT SEPARATES THE BROKEN AND FIXED IMPLEMENTATIONS.

    Four rows, one label withheld, and THREE scores. So `n_source` is 4 and `n`
    is 3, and the supplied array happens to match the narrower count.

    Validated against `n` -- which is what POP-1a did until this fix -- the array
    passes, and `population.take` then RAISES `PopulationError`, converting a
    refusal this gate exists to make graceful into an exception. That is the
    defect recorded at the gate on 2026-07-28, reintroduced one layer earlier.

    Validated against `n_source` it is correctly refused: the caller supplied
    three scores for four rows.

    `test_report_input_gates.py::test_the_scores_channel_refuses_unusable_scores`
    cannot catch this, because its cohort has no withheld label and therefore
    runs where the two checks agree.
    """
    from genomic_variant_classifier.evaluation import evaluator as evaluator_module

    observed_n_expected: list[int] = []
    original = evaluator_module.validate_ranking_scores

    def recording_validate_ranking_scores(scores, *, n_expected=None):
        observed_n_expected.append(n_expected)
        return original(scores, n_expected=n_expected)

    monkeypatch.setattr(evaluator_module, "validate_ranking_scores",
                        recording_validate_ranking_scores)

    report = _evaluator().evaluate(
        WITHHELD_Y, WITHHELD_P,
        scores=np.array([0.9, 0.1, 0.2]),
        source_id=SOURCE)

    # The report is produced rather than the call exploding.
    assert report.n_samples == 3

    # The ranking channel was refused, so nothing derived from it was computed.
    assert len(report.fpr_curve) == 0
    assert len(report.tpr_curve) == 0

    # AND IT WAS REFUSED FOR THE RIGHT REASON. Empty curves alone are consistent
    # with a refusal on the wrong grounds. The check must have been made against
    # the SOURCE length of 4; against the label-eligible count of 3 this
    # three-element array would wrongly have passed, which is the defect.
    assert observed_n_expected == [4], (
        f"validate_ranking_scores was called with n_expected="
        f"{observed_n_expected}, expected [4] -- the source length")


def test_a_correctly_sized_scores_array_is_projected_and_used():
    """The companion case, so the fix cannot be satisfied by refusing everything.

    A source-aligned array of four scores is valid -- the caller supplied one per
    attempted row -- and must be projected to the three label-eligible rows AND
    USED, exactly as `y` and `p` are.

    THE FIXTURE IS CHOSEN TO SEPARATE THE TWO CHANNELS. An earlier version used
    [0.95, 0.05, 0.15, 0.75], which projects to [0.95, 0.05, 0.15] and yields an
    area under the receiver operating characteristic curve of 0.5 on labels
    [1, 1, 0] -- the SAME value the probability channel [0.9, 0.1, 0.2] yields.
    That test would have passed even if the score array were validated,
    projected, and then ignored. It asserted that something happened, not that
    the right thing did.

    [0.95, 0.85, 0.15, 0.75] projects to [0.95, 0.85, 0.15] and ranks the cohort
    perfectly, giving 1.0. The smoke run of 2026-08-01 confirms the baseline: the
    same cohort with NO scores reported 0.5, because `ranking_values` falls back
    to `p`. A reading of 1.0 can arise only from the score array.
    """
    report = _evaluator().evaluate(
        WITHHELD_Y, WITHHELD_P,
        scores=np.array([0.95, 0.85, 0.15, 0.75]),
        source_id=SOURCE)

    assert report.n_samples == 3
    assert _observations_of(report) == 3
    assert _scope_of(report) == "label_eligible"

    # THE DISCRIMINATING ASSERTION. 1.0 from the scores, 0.5 from the
    # probabilities. Anything but 1.0 means the score array did not reach the
    # ranking channel.
    assert report.auroc == pytest.approx(1.0)
    assert len(report.fpr_curve) > 0
    assert len(report.tpr_curve) > 0


def test_every_kernel_receives_a_population_that_needs_no_further_cleaning(
        monkeypatch):
    """THE RULING, ASSERTED AS AN INVARIANT RATHER THAN AT ONE OBSERVED POINT.

    Ruled 2026-07-27: no numerical kernel may select, filter, normalise or
    redefine its evaluation population.

    `test_no_kernel_narrows_the_population_a_second_time` checks one value and
    one count. That proves the observed case is right; it does not prove that
    `metrics.clean_arrays` never narrowed some OTHER eligible input on the way
    through. Before POP-1a it demonstrably did, silently, inside the kernel.

    This instruments the cleaner itself and asserts that on the registry path
    every call receives three rows and returns three -- the joint finiteness mask
    is all true, so the population is the ONLY narrowing operation.

    `clean_arrays` returns a `CleanArrays` dataclass rather than a tuple, so the
    recorder reads `.y`. `metrics._clean` resolves `clean_arrays` as a module
    global, so patching the module attribute reaches every kernel that cleans.
    """
    from genomic_variant_classifier.evaluation import metrics

    original = metrics.clean_arrays
    observed: list[tuple[int, int]] = []

    def recording_clean_arrays(y, score, probability=None):
        cleaned = original(y, score, probability)
        observed.append((int(np.asarray(y).shape[0]),
                         int(np.asarray(cleaned.y).shape[0])))
        return cleaned

    monkeypatch.setattr(metrics, "clean_arrays", recording_clean_arrays)

    report = _evaluator().evaluate(WITHHELD_Y, WITHHELD_P, source_id=SOURCE)

    assert report.n_samples == 3
    assert observed, (
        "clean_arrays was never called; the instrumentation did not reach the "
        "kernels and this test proves nothing")
    for received, returned in observed:
        assert received == 3, (
            f"a kernel was handed {received} rows for a population of 3; the "
            "arrays reaching it were not projected")
        assert returned == 3, (
            f"clean_arrays narrowed {received} rows to {returned}; the "
            "population must be the only narrowing operation")
