"""The model comparison must prove it compared like for like.

WHAT WAS WRONG
==============
Measured 2026-07-28, before this commit: `compare_models` scored several models
against one shared `y_true` and produced a complete, ordered table. Nothing in
that table could demonstrate the models had seen the same rows. Both results were
unattributed, so `compare_membership` returned UNKNOWN, and the artifact recorded
a ranking whose premise it could not support.

For a model comparison, same-population is not a refinement. It is the ENTIRE
PREMISE: a ranking of models scored over different cohorts is not a ranking of
models.

And with one corrupt model the table read:

    good     0.99937
    fair     0.74253
    corrupt  NaN

A ranking was presented. The corrupt model sorted last on a NaN comparison, and a
reader could not distinguish "evaluated and worst" from "never evaluated".

TWO CLAIMS, KEPT APART
----------------------
    SHARED_BY_CONSTRUCTION    one population object was handed to every model
    VERIFIED_BY_FINGERPRINT   those rows are externally identified

The first needs no identity and is proved by construction history. The second
requires an attributed cohort. `compare_membership` is NOT used for the first,
because UNKNOWN is the correct answer to the question it asks, and teaching it
otherwise would destroy the only honest answer it has.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.evaluator import compare_models
from genomic_variant_classifier.evaluation.model_comparison import (
    COMPARISON_SCHEMA_VERSION,
    ComparisonBlocker,
    ComparisonPopulationRelation,
    ModelComparison,
)
from genomic_variant_classifier.evaluation.population import (
    EvaluationPopulation,
    PopulationComparison,
)

ATTRIBUTED = "canonical-variant-table:sha256:probe"


def _cohort(n=300, seed=11):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.5, n).astype(float)
    good = np.clip(0.5 + 0.30 * (2 * y - 1) + rng.normal(0, 0.12, n), 0, 1)
    fair = np.clip(0.5 + 0.15 * (2 * y - 1) + rng.normal(0, 0.25, n), 0, 1)
    return y, good, fair


def _compare(y, models, tmp_path, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return compare_models(y, models, n_bootstrap=0,
                              output_csv=str(Path(tmp_path) / "cmp.csv"), **kwargs)


# --------------------------------------------------------------------------- #
# 1. The shared population
# --------------------------------------------------------------------------- #
def test_all_models_share_one_population_by_construction(tmp_path):
    y, good, fair = _cohort()
    comparison = _compare(y, {"good": good, "fair": fair}, tmp_path)

    assert comparison.comparison_is_like_for_like is True
    assert comparison.population_relation is (
        ComparisonPopulationRelation.SHARED_BY_CONSTRUCTION)
    assert comparison.population_n == len(y)


def test_every_model_receives_THE_SAME_POPULATION_OBJECT(monkeypatch, tmp_path):
    """BY IDENTITY, not by equal fingerprints.

    A first version of this test counted how many populations were built with
    the comparison scope, and a sabotage removing the hand-over survived it: the
    shared population was constructed for the metadata while each model quietly
    built its own. Fingerprints still matched, because the same `source_id` was
    passed to each -- equal by coincidence, not shared by construction.

    Equal fingerprints show that two independently built populations HAPPENED to
    agree. Object identity shows there is only one.
    """
    from genomic_variant_classifier.evaluation import evaluator as module

    received = []
    original = module.ClinicalEvaluator.evaluate

    def recording(self, y_true, y_proba, meta=None, model_name="model",
                  source_id=None, *, scores=None, population=None):
        received.append(population)
        return original(self, y_true, y_proba, meta=meta, model_name=model_name,
                        source_id=source_id, scores=scores, population=population)

    monkeypatch.setattr(module.ClinicalEvaluator, "evaluate", recording)
    y, good, fair = _cohort()
    _compare(y, {"good": good, "fair": fair}, tmp_path)

    assert len(received) == 2, "the probe is not wired"
    assert all(p is not None for p in received), (
        "a model was evaluated without the shared population; sameness would be "
        "true in fact and unprovable from the artifacts")
    assert received[0] is received[1], (
        "the models received DIFFERENT population objects; equal fingerprints "
        "would not distinguish that from sharing one")


def test_a_supplied_population_must_describe_this_cohort(tmp_path):
    """The hand-over must not become a way to attach an unrelated frame."""
    from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator

    y, good, _ = _cohort()
    wrong_size = EvaluationPopulation.full(
        len(y) - 1, scope="attempted_cohort", source_id=None)
    with pytest.raises(ValueError, match="rows but"):
        with contextlib.redirect_stdout(io.StringIO()):
            ClinicalEvaluator(n_bootstrap=0).evaluate(
                y, good, model_name="probe", population=wrong_size)


def test_attribution_promotes_the_relation_and_yields_a_fingerprint(tmp_path):
    y, good, fair = _cohort()
    comparison = _compare(y, {"good": good, "fair": fair}, tmp_path,
                          source_id=ATTRIBUTED)

    assert comparison.population_relation is (
        ComparisonPopulationRelation.VERIFIED_BY_FINGERPRINT)
    assert comparison.population_is_attributed is True
    assert comparison.population_source_id == ATTRIBUTED
    assert comparison.population_fingerprint is not None


def test_an_unattributed_comparison_has_no_fingerprint(tmp_path):
    """Absence, not a sentinel. Minting an identity so the rows compare equal
    would prove sameness WITHIN a call while asserting a frame identity nobody
    established -- the error already ruled out one layer down."""
    y, good, fair = _cohort()
    comparison = _compare(y, {"good": good, "fair": fair}, tmp_path)

    assert comparison.population_is_attributed is False
    assert comparison.population_source_id is None
    assert comparison.population_fingerprint is None


def test_compare_membership_is_not_used_for_intra_call_sameness():
    """It correctly answers UNKNOWN for two unattributed populations, and that
    must not change. The comparison layer knows something stronger from
    construction; they are different evidence channels, not a contradiction."""
    first = EvaluationPopulation.full(4, scope="c", source_id=None)
    second = EvaluationPopulation.full(4, scope="c", source_id=None)
    assert first.compare_membership(second) is PopulationComparison.UNKNOWN


# --------------------------------------------------------------------------- #
# 2. Ranking admissibility
# --------------------------------------------------------------------------- #
def test_a_valid_comparison_is_ranked(tmp_path):
    y, good, fair = _cohort()
    comparison = _compare(y, {"good": good, "fair": fair}, tmp_path)

    assert comparison.comparison_rankable is True
    assert comparison.comparison_blocked_by is None
    assert comparison.blocked_models == ()
    assert comparison.table["rank"].tolist() == [1, 2]
    assert comparison.table["model"].tolist() == ["good", "fair"]


def test_one_invalid_model_refuses_the_whole_ranking(tmp_path):
    """NOT filtered. A ranking that silently excludes a submitted model is not a
    ranking of the models submitted."""
    y, good, fair = _cohort()
    corrupt = good.copy()
    corrupt[:30] = np.nan

    comparison = _compare(y, {"good": good, "fair": fair, "corrupt": corrupt},
                          tmp_path)

    assert comparison.comparison_rankable is False
    assert comparison.comparison_blocked_by is ComparisonBlocker.INVALID_RANKING_METRIC
    assert comparison.blocked_models == ("corrupt",)
    assert len(comparison.table) == 3, "every submitted model keeps a row"


def test_a_refused_ranking_carries_no_order_at_all(tmp_path):
    """Sorting with a NaN present places it LAST, which visually implies worst
    rather than not-evaluated. So no sort runs, and submission order is kept."""
    y, good, fair = _cohort()
    corrupt = good.copy()
    corrupt[:30] = np.nan

    comparison = _compare(y, {"good": good, "corrupt": corrupt, "fair": fair},
                          tmp_path)

    assert comparison.table["model"].tolist() == ["good", "corrupt", "fair"], (
        "submission order must be preserved when no ranking is asserted")
    assert all(r is None for r in comparison.table["rank"].tolist())


def test_admissibility_reads_the_typed_result_not_the_interval(tmp_path):
    """MEASURED 2026-07-28: `format_ci` renders an unavailable interval and a
    FAILED one identically, and the certification Boolean is False in all four
    interval states. Neither is evidence about the model.

    Here every interval is unavailable -- `n_bootstrap=0` -- and the comparison
    is still rankable, because the TYPED point results are valid.
    """
    y, good, fair = _cohort()
    comparison = _compare(y, {"good": good, "fair": fair}, tmp_path)

    assert comparison.comparison_rankable is True
    assert all(c is False for c in comparison.table["auroc_ci_certified"]), (
        "the fixture must have unavailable intervals, or it cannot separate "
        "interval state from ranking admissibility")


# --------------------------------------------------------------------------- #
# 3. The three certification axes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source_id,corrupt,expected", [
    (None,       False, (True, False, False)),
    (ATTRIBUTED, False, (True, True,  True)),
    (None,       True,  (True, False, False)),
    (ATTRIBUTED, True,  (True, True,  False)),
])
def test_the_three_axes_are_never_collapsed(source_id, corrupt, expected, tmp_path):
    """like_for_like, attributed, certifiable.

    An unattributed shared comparison is (True, False, False): internally valid,
    externally unreproducible. One Boolean would report it as invalid, which is
    false and would discourage a sound exploratory analysis.

    The last row is the one that matters most: an ATTRIBUTED comparison with a
    corrupt model keeps its population premise -- the rows really were shared --
    while the ranking, and therefore certification, is refused.
    """
    y, good, fair = _cohort()
    models = {"good": good, "fair": fair}
    if corrupt:
        bad = good.copy()
        bad[:30] = np.nan
        models["corrupt"] = bad

    comparison = _compare(y, models, tmp_path, source_id=source_id)
    observed = (comparison.comparison_is_like_for_like,
                comparison.population_is_attributed,
                comparison.comparison_certification_eligible)
    assert observed == expected


@pytest.mark.parametrize("kwargs,match", [
    (dict(comparison_certification_eligible=True, population_is_attributed=False,
          population_source_id=None), "unattributed"),
    (dict(comparison_certification_eligible=True, comparison_rankable=False,
          comparison_blocked_by=ComparisonBlocker.INVALID_RANKING_METRIC),
     "ranking is refused"),
    (dict(comparison_rankable=False, comparison_blocked_by=None), "must name WHY"),
    (dict(comparison_rankable=True,
          comparison_blocked_by=ComparisonBlocker.INVALID_RANKING_METRIC),
     "must not name a blocker"),
    (dict(population_is_attributed=True, population_source_id=None),
     "attribution and the source identity must agree"),
])
def test_contradictory_comparisons_are_refused_at_construction(kwargs, match):
    """The artifact must not be able to contradict itself."""
    import pandas as pd

    base = dict(
        table=pd.DataFrame([{"model": "m"}]),
        ranking_metric="auroc",
        comparison_rankable=True,
        comparison_blocked_by=None,
        blocked_models=(),
        population_relation=ComparisonPopulationRelation.SHARED_BY_CONSTRUCTION,
        comparison_population_key="population_0",
        population_source_id=ATTRIBUTED,
        population_fingerprint="sha256:x",
        comparison_is_like_for_like=True,
        population_is_attributed=True,
        comparison_certification_eligible=False,
        n_models=1,
        population_n=1,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ModelComparison(**base)


# --------------------------------------------------------------------------- #
# 4. The artifact
# --------------------------------------------------------------------------- #
def test_the_legacy_columns_are_preserved(tmp_path):
    """MEASURED: the comparison artifact has NO consumers -- the only reference
    to `output_csv` outside `compare_models` is a test passing the null device.
    The eleven columns are kept on grounds of CHURN, not compatibility."""
    y, good, fair = _cohort()
    comparison = _compare(y, {"good": good, "fair": fair}, tmp_path)

    for column in ("model", "auroc", "auroc_95ci", "auroc_ci_certified", "auprc",
                   "mcc", "f1", "brier", "ece", "sens_at_90_spec", "ppv_at_90_sens"):
        assert column in comparison.table.columns, column


def test_the_metadata_sidecar_carries_the_comparison_level_facts(tmp_path):
    """Comparison-level fields duplicated across every model row is not a schema:
    it invites a reader to believe they could differ between rows."""
    y, good, fair = _cohort()
    comparison = _compare(y, {"good": good, "fair": fair}, tmp_path,
                          source_id=ATTRIBUTED)

    sidecar = Path(tmp_path) / "cmp.csv.metadata.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    assert payload["comparison_schema_version"] == COMPARISON_SCHEMA_VERSION
    assert payload["ranking_metric"] == "auroc"
    assert payload["comparison_rankable"] is True
    assert payload["population_relation"] == "verified_by_fingerprint"
    assert payload["comparison_population_key"] == "population_0"
    assert payload["population_source_id"] == ATTRIBUTED
    assert payload["population_fingerprint"] is not None


def test_the_sidecar_records_a_refusal(tmp_path):
    y, good, fair = _cohort()
    corrupt = good.copy()
    corrupt[:30] = np.nan
    _compare(y, {"good": good, "corrupt": corrupt}, tmp_path)

    payload = json.loads(
        (Path(tmp_path) / "cmp.csv.metadata.json").read_text(encoding="utf-8"))
    assert payload["comparison_rankable"] is False
    assert payload["comparison_blocked_by"] == "invalid_ranking_metric"
    assert payload["blocked_models"] == ["corrupt"]


def test_the_comparison_population_key_is_deterministic(tmp_path):
    """An artifact-local grouping key, NOT a source identity. Deterministic
    rather than a unique identifier, so serialisation stays byte-stable and no
    false cross-artifact identity is implied."""
    y, good, fair = _cohort()
    first = _compare(y, {"good": good}, tmp_path)
    second = _compare(y, {"fair": fair}, tmp_path)

    assert first.comparison_population_key == second.comparison_population_key
    assert first.comparison_population_key == "population_0"
    assert first.comparison_population_key != first.population_source_id


# --------------------------------------------------------------------------- #
# 5. The null device
#
# ADDED AFTER A DEFECT THAT THE SUITE COULD NOT SEE. `test_evaluator_phase5`
# passes `output_csv=os.devnull`, which on Windows is `nul` -- a RESERVED DEVICE
# NAME with no suffix. `with_suffix(".metadata.json")` therefore produced
# `nul.metadata.json` in the working directory: an entry that appears in a
# directory listing and CANNOT BE OPENED.
#
# The full suite passed, 3610 tests. Only `git add -A` caught it, failing with
# "unable to index file 'nul.metadata.json'". A test that writes to the null
# device never reads back what it wrote, so nothing in the suite could notice.
# --------------------------------------------------------------------------- #
def test_the_null_device_takes_no_sidecar(tmp_path):
    """A caller discarding the table is asking for no artifact at all, and a
    metadata file beside a discarded table is meaningless."""
    y, good, fair = _cohort()
    with contextlib.redirect_stdout(io.StringIO()):
        comparison = compare_models(y, {"good": good, "fair": fair},
                                    n_bootstrap=0, output_csv=os.devnull)

    table_path, sidecar = comparison.write_csv(os.devnull)
    assert sidecar is None, (
        "a sidecar beside the null device is meaningless, and on Windows it "
        "creates an unopenable directory entry that breaks version control")


@pytest.mark.parametrize("name,expected", [
    ("nul", True), ("NUL", True), ("con", True), ("com1", True),
    ("nul.metadata.json", True),
    ("models/model_comparison.csv", False),
    ("data/nulls.csv", False),
    ("annulment.csv", False),
])
def test_the_null_device_detector_does_not_over_match(name, expected):
    """A detector that swallowed `nulls.csv` would silently discard a real
    artifact -- worse than the defect it was written to fix."""
    from genomic_variant_classifier.evaluation.model_comparison import _is_null_device

    assert _is_null_device(Path(name)) is expected


def test_a_real_path_still_gets_its_sidecar(tmp_path):
    """Guards the guard: a fix that suppressed every sidecar would pass the test
    above and destroy the artifact schema."""
    y, good, fair = _cohort()
    comparison = _compare(y, {"good": good, "fair": fair}, tmp_path)
    table_path, sidecar = comparison.write_csv(Path(tmp_path) / "real.csv")
    assert sidecar is not None and sidecar.exists()

