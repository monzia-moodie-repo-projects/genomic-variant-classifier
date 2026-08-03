"""Commit 3a: the typed report surface and schema version 3.

WHAT THIS COMMIT IS, AND IS NOT
===============================
It introduces the typed surface and the schema that carries it. It retires
NOTHING. `ClinicalEvaluator.evaluate` still computes the Matthews correlation
coefficient, F1 and the calibration errors itself, still emits schema version 2,
and still leaves `metric_results` empty.

That separation is deliberate. Schema introduction and computational retirement
have different failure modes: a schema defect corrupts artifacts, a retirement
defect corrupts numbers. Landing them together would leave any regression with
two plausible causes.

    3a   the typed surface exists          acceptance: NOTHING moves
    3b   the report becomes a projection   acceptance: exactly four declared
                                                       field-cohort movements

THE ACCEPTANCE CRITERION FOR 3a
--------------------------------
Every one of the 480 report field values across ten cohorts, frozen on the 2b-3
tree BEFORE this was written, must be byte-identical. There is no declared
movement set at all, because a commit that adds a surface without touching a
computation has no business changing a number.

WHY `result_kind` IS SERIALISED BUT NOT STORED
-----------------------------------------------
Commit 2b-2 ruled that `result_kind` lives on the descriptor and never in result
metadata: putting it in metadata would perturb every already-serialised result.
But an artifact that cannot say what kind of quantity it recorded is not
self-describing, and a future registry revision could reinterpret it silently.

So it is written into the artifact from the descriptor at serialisation time and
VERIFIED on read. A disagreement is raised as a version conflict, never resolved
by preferring today's registry -- the artifact is the evidence, the registry is
only the interpreter.

`asdict()` cannot carry it: it walks the dataclass and bypasses `to_dict()`
entirely, so anything added there would never reach a file. That is why
`to_serializable()` exists.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.capabilities import (
    MetricResult,
    MetricStatus,
)
from genomic_variant_classifier.evaluation.evaluator import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    EVALUATION_REPORT_SCHEMA_VERSION_ABSENCE,
    EVALUATION_REPORT_SCHEMA_VERSION_TYPED,
    SUPPORTED_REPORT_SCHEMA_VERSIONS,
    ClinicalEvaluator,
    EvaluationReport,
    deserialize_metric_results,
    serialize_metric_results,
)
from genomic_variant_classifier.evaluation.registry import ResultKind, by_name
from genomic_variant_classifier.evaluation.serialization import dump_strict_json

SNAPSHOT = Path(__file__).parent.parent / "fixtures" / "report_snapshot_2b3.json"


def _minimal_fields(**overrides):
    """The smallest field set a report needs, so schema behaviour can be tested
    without dragging in a full evaluation."""
    base = dict(
        model_name="probe", n_samples=100, n_pathogenic=40, n_benign=60,
        prevalence=0.4, auroc=0.9, auprc=0.8,
        mcc=0.5, brier_score=0.1, f1=0.7,
        calibration_ece=0.05, calibration_mce=0.09,
    )
    # The twenty interval-provenance fields are REQUIRED and cross-validated by
    # `_validate_ci_fields`, so they cannot be invented: an incoherent set --
    # say a finite endpoint beside a non-OK status -- is refused, correctly. The
    # values below are the ones a real evaluation with `n_bootstrap=0` produces,
    # taken from the frozen oracle rather than guessed, so this helper exercises
    # a configuration the code actually emits.
    for metric in ("auroc", "auprc"):
        base.update({
            f"{metric}_ci_lo": None,
            f"{metric}_ci_hi": None,
            f"{metric}_ci_status": MetricStatus.INSUFFICIENT_SUPPORT,
            f"{metric}_ci_resampling_unit": None,
            f"{metric}_ci_stratified": None,
            f"{metric}_ci_cluster_source": None,
            f"{metric}_ci_partition_verified": False,
            f"{metric}_ci_certification_eligible": False,
            f"{metric}_ci_n_requested": 0,
            f"{metric}_ci_n_valid": 0,
            f"{metric}_ci_n_degenerate": 0,
            f"{metric}_ci_finding": "gene_cluster_identifier_required",
        })
    known = {f.name for f in dataclasses.fields(EvaluationReport)}
    base = {k: v for k, v in base.items() if k in known}
    base.update(overrides)
    return base


def _ok_result(value=0.9):
    return MetricResult(value, MetricStatus.OK, None, {})


# --------------------------------------------------------------------------- #
# 1. Schema and the typed mapping must agree, in both directions
# --------------------------------------------------------------------------- #
def test_the_typed_schema_version_is_declared_and_readable():
    assert EVALUATION_REPORT_SCHEMA_VERSION_TYPED == 3
    assert EVALUATION_REPORT_SCHEMA_VERSION == 2
    # POP-1b (2026-08-02) added version five: the report names its evaluation
    # population. Asserted as an EXACT SET on purpose -- this test's job is to
    # notice when the READABLE RANGE changes, which is a different question from
    # whether the emitted version is recent enough. The two assertions above are
    # untouched: POP-1b adds five, it does not renumber three or two.
    assert set(SUPPORTED_REPORT_SCHEMA_VERSIONS) == {1, 2, 3, 4, 5}


def test_a_version_three_report_requires_a_non_empty_typed_mapping():
    """The version asserts the typed surface is present. A version-3 report with
    an empty mapping would claim a surface it does not have."""
    with pytest.raises(ValueError, match="requires a non-empty"):
        EvaluationReport(schema_version=3, metric_results={}, **_minimal_fields())


@pytest.mark.parametrize("version", [1, 2])
def test_a_historical_report_must_have_an_empty_typed_mapping(version):
    """Versions 1 and 2 predate the typed surface, so a populated mapping on one
    is either a mislabelled artifact or synthesised provenance."""
    with pytest.raises(ValueError, match="must have an EMPTY"):
        EvaluationReport(schema_version=version,
                         metric_results={"auroc": _ok_result()},
                         **_minimal_fields())


def test_an_unsupported_schema_version_is_refused():
    with pytest.raises(ValueError, match="unsupported report schema version"):
        EvaluationReport(schema_version=99, metric_results={}, **_minimal_fields())


def test_a_bare_float_cannot_masquerade_as_a_typed_result():
    """Admitting one would reintroduce the untyped surface this layer replaces."""
    with pytest.raises(TypeError, match="must be a MetricResult"):
        EvaluationReport(schema_version=3, metric_results={"auroc": 0.9},
                         **_minimal_fields())


def test_the_typed_mapping_must_be_a_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):
        EvaluationReport(schema_version=2, metric_results=[], **_minimal_fields())


# --------------------------------------------------------------------------- #
# 2. Construction
# --------------------------------------------------------------------------- #
def test_from_metric_results_builds_a_version_three_report():
    report = EvaluationReport.from_metric_results(
        metric_results={"auroc": _ok_result(0.93)}, **_minimal_fields())
    assert report.schema_version == EVALUATION_REPORT_SCHEMA_VERSION_TYPED
    assert set(report.metric_results) == {"auroc"}
    assert report.metric_results["auroc"].value == 0.93


def test_from_metric_results_refuses_an_empty_mapping():
    with pytest.raises(ValueError, match="requires at least one typed result"):
        EvaluationReport.from_metric_results(metric_results={}, **_minimal_fields())


def test_direct_construction_still_works_for_historical_data():
    """Making the fields `init=False` would break every existing caller and every
    historical deserialisation path. Consistency is enforced in `__post_init__`
    rather than by removing the door."""
    report = EvaluationReport(schema_version=2, **_minimal_fields())
    assert report.schema_version == 2
    assert dict(report.metric_results) == {}


# --------------------------------------------------------------------------- #
# 3. Version-2 artifacts are never given synthesised provenance
# --------------------------------------------------------------------------- #
def test_a_version_two_artifact_does_not_gain_synthetic_typed_results():
    """THE RULING THIS COMMIT MOST NEEDS TO HOLD.

    An `OK` result manufactured from a bare float would assert a population
    scope, a support count, an applicability verdict, a threshold provenance and
    a certification eligibility that the artifact never recorded. That is not
    recovery; it is fabrication, and it would be indistinguishable downstream
    from provenance that was genuinely measured.
    """
    payload = {"schema_version": 2, **_minimal_fields()}
    report = EvaluationReport.from_serialized_v2(payload)
    assert report.schema_version == 2
    assert dict(report.metric_results) == {}
    assert report.auroc == payload["auroc"], "flat scalars must survive exactly"
    assert report.mcc == payload["mcc"]


def test_from_serialized_v2_refuses_a_typed_artifact():
    payload = {"schema_version": 3, **_minimal_fields()}
    with pytest.raises(ValueError, match="refuses schema version 3"):
        EvaluationReport.from_serialized_v2(payload)


def test_from_serialized_dispatches_on_the_recorded_version():
    """On what the artifact says it is, not on what the reader hopes to find."""
    v2 = EvaluationReport.from_serialized({"schema_version": 2, **_minimal_fields()})
    assert v2.schema_version == 2 and dict(v2.metric_results) == {}

    typed = EvaluationReport.from_metric_results(
        metric_results={"auroc": _ok_result()}, **_minimal_fields())
    v3 = EvaluationReport.from_serialized(typed.to_serializable())
    assert v3.schema_version == 3 and set(v3.metric_results) == {"auroc"}


def test_a_serialized_report_without_a_version_is_refused():
    with pytest.raises(ValueError, match="carries no schema_version"):
        EvaluationReport.from_serialized(_minimal_fields())


# --------------------------------------------------------------------------- #
# 4. result_kind: written from the descriptor, verified on read
# --------------------------------------------------------------------------- #
def test_result_kind_is_written_into_the_artifact():
    payload = serialize_metric_results({"auroc": _ok_result(),
                                        "prevalence": _ok_result(0.4)})
    assert payload["auroc"]["result_kind"] == ResultKind.PREDICTION_METRIC.value
    assert payload["prevalence"]["result_kind"] == ResultKind.POPULATION_STATISTIC.value


def test_result_kind_is_not_stored_on_the_result_itself():
    """2b-2's ruling: it lives on the descriptor. Storing it on the result would
    perturb every already-serialised result and force the acceptance test to
    carry an exemption."""
    result = _ok_result()
    assert "result_kind" not in result.to_dict()
    assert not any("result_kind" in str(k) for k in result.metadata)


def test_a_result_kind_conflict_is_raised_and_never_overwritten():
    """A disagreement between artifact and registry is a version conflict
    requiring an explicit decision. Preferring today's registry would let a
    registry revision silently reinterpret old evidence."""
    payload = serialize_metric_results({"auroc": _ok_result()})
    payload["auroc"]["result_kind"] = ResultKind.POPULATION_STATISTIC.value
    with pytest.raises(ValueError, match="result_kind conflict"):
        deserialize_metric_results(payload)


def test_a_typed_artifact_must_be_self_describing():
    payload = serialize_metric_results({"auroc": _ok_result()})
    del payload["auroc"]["result_kind"]
    with pytest.raises(ValueError, match="carries no result_kind"):
        deserialize_metric_results(payload)


def test_an_unregistered_metric_cannot_be_serialised_or_read():
    with pytest.raises(ValueError, match="no descriptor is registered"):
        serialize_metric_results({"not_a_metric": _ok_result()})
    with pytest.raises(ValueError, match="no descriptor in the current registry"):
        deserialize_metric_results(
            {"not_a_metric": {**_ok_result().to_dict(), "result_kind": "prediction_metric"}})


# --------------------------------------------------------------------------- #
# 5. Serialisation round trip
# --------------------------------------------------------------------------- #
def test_asdict_alone_would_not_carry_result_kind():
    """The reason `to_serializable()` exists. `asdict` walks the dataclass and
    bypasses `to_dict()`, so anything added there never reaches a file."""
    report = EvaluationReport.from_metric_results(
        metric_results={"auroc": _ok_result()}, **_minimal_fields())
    assert "result_kind" not in dataclasses.asdict(report)["metric_results"]["auroc"]
    assert "result_kind" in report.to_serializable()["metric_results"]["auroc"]


def test_a_typed_report_survives_strict_json_and_a_round_trip():
    original = EvaluationReport.from_metric_results(
        metric_results={"auroc": _ok_result(0.93),
                        "prevalence": MetricResult(0.4, MetricStatus.OK, None, {}),
                        "f1": MetricResult(float("nan"), MetricStatus.UNDEFINED,
                                           "zero_f1_denominator", {})},
        **_minimal_fields())
    text = dump_strict_json(original.to_serializable(), artifact="round_trip")
    restored = EvaluationReport.from_serialized(json.loads(text))

    assert restored.schema_version == 3
    assert set(restored.metric_results) == set(original.metric_results)
    for name, was in original.metric_results.items():
        now = restored.metric_results[name]
        assert now.status is was.status
        assert now.reason == was.reason
        if math.isnan(was.value):
            assert math.isnan(now.value), f"{name}: NaN did not survive"
        else:
            assert now.value == was.value


def test_a_refusal_survives_the_round_trip_as_a_refusal():
    """NaN does not survive strict JSON as a number, so a refused metric must be
    restored as a refusal rather than as a value of zero or a missing entry."""
    original = EvaluationReport.from_metric_results(
        metric_results={"matthews_correlation_coefficient": MetricResult(
            float("nan"), MetricStatus.UNDEFINED, "zero_confusion_margin", {})},
        **_minimal_fields())
    restored = EvaluationReport.from_serialized(
        json.loads(dump_strict_json(original.to_serializable(), artifact="probe")))
    result = restored.metric_results["matthews_correlation_coefficient"]
    assert result.status is MetricStatus.UNDEFINED
    assert result.reason == "zero_confusion_margin"
    assert math.isnan(result.value)


# --------------------------------------------------------------------------- #
# 6. THE ACCEPTANCE CRITERION -- nothing moved
# --------------------------------------------------------------------------- #
def _encode(value):
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
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if hasattr(value, "value"):
        return f"__enum__:{value.value}"
    return f"__repr__:{type(value).__name__}"


def _rebuild_cohorts():
    rng = np.random.default_rng(20260728)
    out = {}
    n = 600
    y = rng.binomial(1, 0.5, n).astype(float)
    out["balanced"] = (y, np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.15, n), 0, 1))
    n = 500
    y = rng.binomial(1, 0.08, n).astype(float)
    out["imbalanced"] = (y, np.clip(0.2 + 0.4 * y + rng.normal(0, 0.2, n), 0, 1))
    n = 400
    y = rng.binomial(1, 0.5, n).astype(float)
    out["pure_leaves"] = (y, np.where(rng.random(n) < 0.3, y.copy(), rng.random(n)))
    n = 300
    y = rng.binomial(1, 0.5, n).astype(float)
    out["coarse_probabilities"] = (y, np.round(np.clip(rng.random(n), 0, 1), 1))
    n = 200
    y = rng.binomial(1, 0.5, n).astype(float)
    out["near_random"] = (y, np.clip(rng.random(n), 0, 1))
    n = 700
    y = np.concatenate([np.ones(333), np.zeros(n - 333)])
    out["prevalence_separates_rounding_a"] = (
        y, np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.15, n), 0, 1))
    n = 900
    y = np.concatenate([np.ones(401), np.zeros(n - 401)])
    out["prevalence_separates_rounding_b"] = (
        y, np.clip(0.5 + 0.20 * (2 * y - 1) + rng.normal(0, 0.18, n), 0, 1))
    out["degenerate_all_negative"] = (np.zeros(80), np.full(80, 0.10))
    out["degenerate_all_positive"] = (np.ones(80), np.full(80, 0.90))
    out["constant_classifier"] = (np.concatenate([np.zeros(60), np.ones(20)]),
                                  np.full(80, 0.10))
    return out


def test_the_report_snapshot_is_self_validating():
    """A fixture must not be able to validate a tree it was regenerated on. The
    decisive check is the schema: the snapshot was captured under version 2, and
    if that ever equals what `evaluate()` emits today the fixture is a photograph
    of the thing it was meant to check."""
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert snapshot["captured_from"] == "commit 2b-3"
    assert snapshot["captured_from_commit"] == "15ad3f0"

    # THE DECISIVE CHECK, and it is NOT the one 2b-3 used.
    #
    # There the fixture was captured under registry schema 1 against a current 2,
    # so "the recorded version must differ from the current one" worked. It does
    # not transfer here: this oracle was captured under report schema 2 and
    # `evaluate()` still emits 2 throughout 3a, so an inequality assertion would
    # fail for an entirely legitimate fixture.
    #
    # The invariant that DOES hold is that the oracle predates the typed
    # emission. A fixture regenerated once `evaluate()` emits version 3 -- which
    # is precisely what commit 3b makes it do -- would record 3 and be caught.
    # Without this, the sabotage matrix walked straight through a regenerated
    # oracle: the field count and cohort count were unchanged, so nothing else
    # noticed.
    assert snapshot["report_schema_at_capture"] == EVALUATION_REPORT_SCHEMA_VERSION, (
        "the oracle records a schema version other than the one evaluate() "
        "emitted when it was captured")
    assert snapshot["report_schema_at_capture"] < EVALUATION_REPORT_SCHEMA_VERSION_TYPED, (
        "the oracle was regenerated on a tree whose evaluate() already emits the "
        "typed schema; it is then a photograph of the thing it is supposed to be "
        "checking, and every comparison against it passes for the one reason "
        "that guarantees nothing")
    assert snapshot["n_report_fields"] == 48
    assert len(snapshot["cohorts"]) == 10
    assert snapshot["n_bootstrap"] == 0, (
        "the capture must be deterministic, or interval fields would be "
        "compared against random draws")


def test_no_report_field_moved():
    """THE ACCEPTANCE CRITERION FOR 3a. Ten cohorts, 48 frozen fields, 480 values,
    and NO declared movement set: a commit that adds a surface without touching a
    computation has no business changing a number.
    """
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    evaluator = ClinicalEvaluator(n_bootstrap=snapshot["n_bootstrap"], random_state=42)
    cohorts = _rebuild_cohorts()

    movements, compared = [], 0
    for label, frozen_row in snapshot["cohorts"].items():
        y, p = cohorts[label]
        report = evaluator.evaluate(y, p, model_name="snapshot_model")
        for field_name, was in frozen_row.items():
            compared += 1
            now = _encode(getattr(report, field_name))
            if now != was:
                movements.append(f"{label}/{field_name}: {was!r} -> {now!r}")

    assert compared == 480, f"expected 480 comparisons, made {compared}"

    # THE DECLARED MOVEMENT SET, added 2026-07-28 by commit 3b-2.
    #
    # This oracle showed ZERO movements through 3a, 3b-0, 3b-1a and 3b-1b.
    # Commit 3b-2 moves exactly one field, and moves it in every cohort:
    # `schema_version` advances from 2 to 3 because the report now CARRIES the
    # typed results rather than merely being able to. That is not a numerical
    # change; it is the report stating what it contains, which is the one thing
    # a schema version is for.
    #
    # Declared BY IDENTITY. A count alone would accept ten movements in the
    # wrong fields, and the whole purpose of this commit is that no measured
    # value changes when the report becomes a projection.
    declared = {f"{cohort}/schema_version" for cohort in snapshot["cohorts"]}
    observed = {m.split(":")[0] for m in movements}
    undeclared = observed - declared
    assert not undeclared, (
        f"{len(undeclared)} UNDECLARED report field movement(s):\n  "
        + "\n  ".join(sorted(undeclared)))
    assert observed == declared, (
        "the declared movement set expects changes that did not occur: "
        f"{sorted(declared - observed)}")


def test_no_report_field_appeared_unannounced():
    """A CUMULATIVE growth guard against the 2b-3 snapshot, not a claim about
    one commit.

    RENAMED 2026-08-02 (POP-1b). It was called
    `test_exactly_one_report_field_was_added` while asserting THREE: it had been
    extended once and the name was left behind. That stale name cost a full turn
    of the POP-1b session, reading as a historical claim about a single commit
    when it is a running record. The new name does not need renaming again.

    THE SNAPSHOT FIXTURE IS NEVER REGENERATED. Rebasing it onto today would make
    this guard permanently blind to everything added before now -- green while
    guarding nothing, which is worse than red. Fields are APPENDED here, in the
    order `dataclasses.fields` returns them, which was measured from the live
    class rather than assumed.

    POP-1b (2026-08-02) added five: the report now names its evaluation
    population, because POP-1a made `n_samples` the label-eligible count and a
    reader could not otherwise tell a smaller cohort from a narrowed one.
    """
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    current = [f.name for f in dataclasses.fields(EvaluationReport)]
    added = [f for f in current if f not in snapshot["report_field_names"]]
    removed = [f for f in snapshot["report_field_names"] if f not in current]
    assert added == [
        # commit 3b-2 (2026-07-28)
        "metric_results",
        # CI-u-3 (2026-07-29)
        "field_absence",
        "curve_absence",
        # POP-1b (2026-08-02)
        "n_source",
        "n_label_eligible",
        "n_reference_label_withheld",
        "population_scope",
        "population_parent_fingerprint",
    ]
    assert removed == []


def test_evaluate_now_emits_the_typed_schema_version():
    """INVERTED 2026-07-28 by commit 3b-2. This asserted the opposite.

    Commit 3a introduced version 3 as a CAPABILITY and required that `evaluate`
    NOT emit it, so that schema introduction and computational retirement stayed
    independently falsifiable. That separation did its work: 3a moved nothing,
    and 3b-2 moves exactly one field.

    Now that the report is a pure projection of the typed results, withholding
    them would force every consumer wanting status, reason, population or
    certification to recompute what the report already has.
    """
    from genomic_variant_classifier.evaluation.registry import names

    evaluator = ClinicalEvaluator(n_bootstrap=0, random_state=42)
    y, p = _rebuild_cohorts()["balanced"]
    report = evaluator.evaluate(y, p, model_name="probe")
    # THRESHOLD, NOT A LITERAL (POP-1b, 2026-08-02). This read `== 4` and broke
    # when POP-1b began emitting five. What the docstring above claims is that
    # `evaluate` EMITS the typed surface, and every version from four onward
    # does. A literal here needs editing at six and at seven.
    assert (report.schema_version
            >= EVALUATION_REPORT_SCHEMA_VERSION_ABSENCE)
    assert set(report.metric_results) == set(names()), (
        "a version-3 report must carry every registered typed result")
