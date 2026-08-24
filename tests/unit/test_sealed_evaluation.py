"""A metric's origin is part of the metric, enforced. Commit C.

Created 2026-08-24, against two committed censuses read in full.

WHAT THIS GUARDS
----------------
`EvaluationEvidence.metrics` is a flat `Mapping[str, float]` with
`EvaluationProtocol` beside it rather than inside each entry, so one evidence
object can hold a figure computed on held-out predictions and a figure scraped
from a training log with nothing to tell them apart.

MEASURED, `MEASUREMENT_2026-08-08_metricorigin-census.md`: Run 14's manifest
holds FOUR figures spanning 0.9975 to 0.9985 that a careless reader would call
"Run 14's AUROC", distinguished only by a key suffix. The census names the
consequence -- "That is precisely the mechanism by which 0.9847 came to be
published as a holdout metric" -- and BASELINE-1 establishes that `0.9847` is
UNATTRIBUTABLE, its earliest appearance a commit subject line.

WHY THE FIXTURES USE RUN 14 AND RUN 10b AND NOTHING ELSE
--------------------------------------------------------
`MEASUREMENT_2026-08-08_baseline1-provenance-census.md` section 9 bounds the
scope, and it is quoted rather than paraphrased:

    Commit C's scope is bounded by section 9: seal Run 14, represent Run 10b's
    partiality honestly, and attribute nothing else.

So the worked example is Run 14's four real figures, and the partial case is Run
10b's three real lost outputs. Inventing a tidier fixture would test a type
against data the project does not have.

NEGATIVE CONTROLS ASSERT THE MESSAGE, NOT MERELY THAT SOMETHING RAISED
----------------------------------------------------------------------
Measured 2026-08-23 while building `test_archive_manifest.py`: three adversarial
cases were constructed by reusing an object, so an earlier guard fired first.
They refused, and proved nothing about the invariant they named. `refuses()`
asserts the message text for that reason.

Author: Monzia Moodie
"""
from __future__ import annotations

import json

import pytest

from genomic_variant_classifier.evaluation.sealed_evaluation import (
    SCHEMA,
    SCHEMA_VERSION,
    Coercion,
    MetricOrigin,
    SealCompleteness,
    SealedEvaluation,
    SealedMetric,
    SealError,
    evidence_from_seal,
)
from genomic_variant_classifier.monitoring.model_registry import (
    EvaluationEvidence,
    EvaluationProtocol,
    TrainingLineage,
)

#: Run 14's real digest-shaped placeholders. Sixty-four lowercase hexadecimal
#: characters, because that is what the seal requires and what
#: `artifact_sha256` already carries in the manifest.
RUN14_ENSEMBLE = "a" * 64
RUN14_SCALER = "b" * 64
ROSTER_13 = "sha256:13:catboost,cnn_1d,deep_ensemble,gradient_boosting,kan"
ROSTER_12 = "sha256:12:catboost,cnn_1d,deep_ensemble,gradient_boosting"

#: The four figures, from the census section 1 table. Real values.
RUN14_FIGURES = (
    ("auroc", 0.9975, MetricOrigin.COMPUTED_FROM_PREDICTIONS),
    ("stacker_metrics_test.auroc", 0.9975, MetricOrigin.COMPUTED_FROM_PREDICTIONS),
    ("lr_stacker_auroc_from_log", 0.9984, MetricOrigin.SCRAPED_FROM_TRAINING_LOG),
    ("oof_blend_auroc_from_log", 0.9985, MetricOrigin.SCRAPED_FROM_TRAINING_LOG),
)


def a_protocol(**over) -> EvaluationProtocol:
    kw = {"protocol_id": "run14-test", "split_kind": "test",
          "population_scope": "clinvar-grch38-acmg5", "n_observations": 349067,
          "label_policy": "acmg-5-binary", "population_fingerprint": "fp-run14"}
    kw.update(over)
    return EvaluationProtocol(**kw)


def run14_metrics() -> tuple:
    return tuple(SealedMetric(name=n, value=v, origin=o)
                 for n, v, o in RUN14_FIGURES)


def a_seal(**over) -> SealedEvaluation:
    kw = {"seal_id": "seal-run14-test",
          "lineage": TrainingLineage(
              run_id="run14",
              source_commit="80ac62ca7e83d35638274a01170d4c8f4f62c418"),
          "protocol": a_protocol(),
          "metrics": run14_metrics(),
          "artifact_sha256": {"ensemble": RUN14_ENSEMBLE,
                              "scaler": RUN14_SCALER},
          "roster_fingerprint": ROSTER_13}
    kw.update(over)
    return SealedEvaluation(**kw)


def refuses(fn, fragment):
    """Assert the refusal fires on the invariant it CLAIMS to test."""
    with pytest.raises(SealError) as exc:
        fn()
    assert fragment in str(exc.value), (
        "refused, but on the WRONG check.\n  expected the message to contain: "
        "{!r}\n  actual: {}".format(fragment, exc.value))


# ---------------------------------------------------------------------------
# 1. THE DEFECT THIS TYPE CLOSES
# ---------------------------------------------------------------------------

def test_run14s_four_figures_separate_by_origin():
    """The census's section 1 table, expressed as a type.

    Two computed figures agreeing exactly, two scraped figures describing a
    different quantity: out-of-fold performance during training, not held-out
    performance after it.
    """
    seal = a_seal()
    computed = seal.metrics_by_origin(MetricOrigin.COMPUTED_FROM_PREDICTIONS)
    scraped = seal.metrics_by_origin(MetricOrigin.SCRAPED_FROM_TRAINING_LOG)
    assert [m.name for m in computed] == [
        "auroc", "stacker_metrics_test.auroc"]
    assert [m.name for m in scraped] == [
        "lr_stacker_auroc_from_log", "oof_blend_auroc_from_log"]
    assert {m.value for m in computed} == {0.9975}
    assert {m.value for m in scraped} == {0.9984, 0.9985}


def test_the_spread_is_between_two_named_kinds_not_inside_one_mapping():
    """METRICORIGIN-1, stated as a property.

    The largest computed figure and the largest scraped figure differ by
    0.0010 -- "small enough to look like rounding, large enough to change which
    model appears best". The point is not that the gap exists; it is that the
    record can now say the two are different KINDS.
    """
    seal = a_seal()
    computed = max(m.value for m in seal.metrics_by_origin(
        MetricOrigin.COMPUTED_FROM_PREDICTIONS))
    scraped = max(m.value for m in seal.metrics_by_origin(
        MetricOrigin.SCRAPED_FROM_TRAINING_LOG))
    assert round(scraped - computed, 4) == 0.0010
    assert seal.has_mixed_origins


def test_a_flat_mapping_cannot_express_what_this_seal_expresses():
    """The projection is LOSSY, and that loss IS the defect.

    `evidence_from_seal` is one-directional on purpose: a seal reduces to
    evidence; evidence can never be promoted to a seal, because the information
    to do so was never there.
    """
    seal = a_seal()
    evidence = evidence_from_seal(seal)
    assert isinstance(evidence, EvaluationEvidence)
    assert set(evidence.metrics) == {m.name for m in seal.metrics}
    assert all(isinstance(v, float) for v in evidence.metrics.values())
    # Nothing in the projection carries origin. That is the point.
    assert not any(hasattr(v, "origin") for v in evidence.metrics.values())


# ---------------------------------------------------------------------------
# 2. THE SUFFIX AND THE FIELD MUST AGREE
# ---------------------------------------------------------------------------

def test_a_from_log_key_may_not_declare_a_computed_origin():
    """The suffix is the ARTEFACT's own statement of origin.

    "Whoever wrote this manifest understood that the origin of a number belongs
    with the number, and had nowhere to put it except the key." A seal may
    replace that convention; it must not contradict it, or one ambiguity
    becomes two.
    """
    refuses(lambda: SealedMetric("oof_blend_auroc_from_log", 0.9985,
                                 MetricOrigin.COMPUTED_FROM_PREDICTIONS),
            "must not contradict")


def test_a_scraped_metric_need_not_carry_the_suffix():
    """The field REPLACES the convention, so a plain name may be scraped."""
    m = SealedMetric("oof_blend_auroc", 0.9985,
                     MetricOrigin.SCRAPED_FROM_TRAINING_LOG)
    assert m.origin is MetricOrigin.SCRAPED_FROM_TRAINING_LOG


# ---------------------------------------------------------------------------
# 3. WHAT attribution.py ASKS
# ---------------------------------------------------------------------------

def test_a_seal_is_evidence_only_for_the_digest_and_roster_it_names():
    """`api/attribution.py:387`, quoted:

        Even a linked sealed_evaluation_id is NOT enough here. Commit C
        requires the evaluation to name this artifact digest AND this served
        roster fingerprint.
    """
    seal = a_seal()
    assert seal.is_evidence_for(RUN14_ENSEMBLE, ROSTER_13)
    assert seal.is_evidence_for(RUN14_SCALER, ROSTER_13)
    assert not seal.is_evidence_for("f" * 64, ROSTER_13)


def test_a_twelve_model_projection_is_not_evidence_for_a_thirteen_model_seal():
    """ROSTER_MISMATCH, which the enumeration calls "the status this project
    needs most":

        A metric measured on a thirteen-model ensemble is not automatically
        evidence for a twelve-model serving projection of it, however
        intentional the projection. Resolving a digest authorises IDENTITY,
        not EVIDENCE.
    """
    assert not a_seal().is_evidence_for(RUN14_ENSEMBLE, ROSTER_12)


def test_a_seal_must_name_a_roster():
    refuses(lambda: a_seal(roster_fingerprint="   "),
            "IDENTITY, not EVIDENCE")


def test_a_seal_must_name_at_least_one_artefact():
    """"Sealing does not need to introduce artifact_sha256; it needs to make it
    mandatory." Run 14's manifest already carries two entries."""
    refuses(lambda: a_seal(artifact_sha256={}),
            "mandatory rather than introducing")


def test_an_artefact_digest_must_look_like_a_digest():
    refuses(lambda: a_seal(artifact_sha256={"ensemble": "80ac62c"}),
            "64 lowercase hexadecimal")


def test_an_uppercase_digest_is_refused():
    """Digest comparison is exact. A case difference would silently fail
    `is_evidence_for` rather than refuse at construction."""
    refuses(lambda: a_seal(artifact_sha256={"ensemble": "A" * 64}),
            "64 lowercase hexadecimal")


# ---------------------------------------------------------------------------
# 4. RUN 10b: HONESTLY PARTIAL
# ---------------------------------------------------------------------------

def test_run10b_is_representable_as_partial():
    """Its artefact declares its own incompleteness:

        "status": "partial -- Run 10b instance destroyed mid-pipeline",
        "lost": ["deep_ensemble.joblib", "GNN", "cloud-computed test AUROC"]

    "An artefact recording its own incompleteness is exactly what the type must
    be able to represent WITHOUT PRETENDING OTHERWISE."
    """
    seal = a_seal(
        seal_id="seal-run10b",
        lineage=TrainingLineage(run_id="run10b"),
        protocol=a_protocol(protocol_id="run10b-partial", n_observations=1000),
        metrics=(SealedMetric("auprc", 0.97,
                              MetricOrigin.COMPUTED_FROM_PREDICTIONS),),
        completeness=SealCompleteness.PARTIAL,
        lost_outputs=("deep_ensemble.joblib", "GNN",
                      "cloud-computed test AUROC"))
    assert seal.completeness is SealCompleteness.PARTIAL
    assert len(seal.lost_outputs) == 3


def test_a_partial_seal_must_say_what_was_lost():
    refuses(lambda: a_seal(completeness=SealCompleteness.PARTIAL),
            "must name what was lost")


def test_a_complete_seal_may_not_name_losses():
    refuses(lambda: a_seal(lost_outputs=("GNN",)), "Declare it PARTIAL")


def test_duplicate_lost_outputs_are_refused():
    refuses(lambda: a_seal(completeness=SealCompleteness.PARTIAL,
                           lost_outputs=("GNN", "GNN")),
            "duplicate entry in lost_outputs")


# ---------------------------------------------------------------------------
# 5. COERCION IS DECLARED, NEVER SILENT
# ---------------------------------------------------------------------------

def test_a_string_metric_is_coerced_only_with_a_declaration():
    """Run 14's per-model metrics are stored as `str`, and
    `EvaluationEvidence.from_dict` calls `float(v)` silently at
    model_registry.py:260.

    "A string that looks like a number may be a rounded rendering of one, and
    rounding is a transformation a sealed record should declare."
    """
    m = SealedMetric.from_manifest_value(
        "f1_macro", "0.9775", MetricOrigin.COMPUTED_FROM_PREDICTIONS)
    assert m.value == 0.9775
    assert m.coercion == Coercion(original="0.9775")


def test_a_numeric_metric_carries_no_coercion():
    m = SealedMetric.from_manifest_value(
        "mcc", 0.955, MetricOrigin.COMPUTED_FROM_PREDICTIONS)
    assert m.coercion is None


def test_a_string_that_is_not_a_number_is_refused_not_guessed():
    refuses(lambda: SealedMetric.from_manifest_value(
        "auroc", "n/a", MetricOrigin.COMPUTED_FROM_PREDICTIONS),
        "refuses rather than guessing")


def test_an_empty_coercion_original_is_refused():
    refuses(lambda: Coercion(original="   "), "record the original text")


# ---------------------------------------------------------------------------
# 6. THE REFUSALS EvaluationEvidence ALREADY MAKES, KEPT
# ---------------------------------------------------------------------------

def test_a_seal_with_no_metrics_is_not_evidence():
    refuses(lambda: a_seal(metrics=()), "is not evidence")


def test_a_boolean_is_not_a_real_number():
    """`EvaluationEvidence.__post_init__` excludes bool explicitly, because
    `isinstance(True, int)` is True. The seal keeps that exclusion."""
    refuses(lambda: SealedMetric("auroc", True,
                                 MetricOrigin.COMPUTED_FROM_PREDICTIONS),
            "must be a real number")


def test_a_nan_metric_is_refused():
    """A metric that could not be computed is not sealed evidence. The metric
    stack already returns NaN and says so; a seal records the omission
    instead."""
    refuses(lambda: SealedMetric("auroc", float("nan"),
                                 MetricOrigin.COMPUTED_FROM_PREDICTIONS),
            "is NaN")


def test_duplicate_metric_names_are_refused():
    m = run14_metrics()
    refuses(lambda: a_seal(metrics=(m[0], m[0])), "flat-mapping defect")


def test_a_metric_requires_a_name():
    refuses(lambda: SealedMetric("  ", 0.5,
                                 MetricOrigin.COMPUTED_FROM_PREDICTIONS),
            "requires a name")


def test_a_seal_requires_an_identifier():
    refuses(lambda: a_seal(seal_id=" "), "requires an identifier")


# ---------------------------------------------------------------------------
# 7. SERIALIZATION
# ---------------------------------------------------------------------------

def test_render_is_deterministic_and_diffable():
    """Sorted keys and fixed indentation are fully deterministic; a compact
    separator form is no more so and turns a durable record into one
    unreadable line."""
    seal = a_seal()
    first = seal.render()
    assert first == seal.render()
    assert first.count(b"\n") > 20
    assert first.endswith(b"\n")
    assert not any(b > 0x7F for b in first)


def test_the_rendered_payload_declares_its_schema():
    payload = json.loads(a_seal().render().decode("utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["schema_version"] == SCHEMA_VERSION


def test_every_rendered_metric_carries_its_origin():
    """The whole purpose, surviving serialization."""
    payload = json.loads(a_seal().render().decode("utf-8"))
    origins = {m["name"]: m["origin"] for m in payload["metrics"]}
    assert origins["auroc"] == "computed_from_predictions"
    assert origins["oof_blend_auroc_from_log"] == "scraped_from_training_log"


def test_a_coercion_survives_serialization():
    seal = a_seal(metrics=(SealedMetric.from_manifest_value(
        "f1_macro", "0.9775", MetricOrigin.COMPUTED_FROM_PREDICTIONS),))
    payload = json.loads(seal.render().decode("utf-8"))
    assert payload["metrics"][0]["coercion"] == {
        "original": "0.9775", "parsed_as": "float"}


# ---------------------------------------------------------------------------
# 8. THE SEAL IS NOT THE DEPLOYMENT RECORD
# ---------------------------------------------------------------------------

def test_a_seal_composes_the_existing_types_rather_than_replacing_them():
    """"The indicated design is a thin sealing layer over them, not a parallel
    hierarchy" -- the same ruling GATE-1 took when it extended the registry's
    promotion policy rather than building a second one."""
    seal = a_seal()
    assert isinstance(seal.protocol, EvaluationProtocol)
    assert isinstance(seal.lineage, TrainingLineage)
    assert seal.protocol.n_observations == 349067
    assert seal.lineage.run_id == "run14"


def test_the_protocol_still_refuses_what_it_always_refused():
    """The seal adds requirements; it removes none. An unnamed protocol makes
    its metrics incomparable, and `EvaluationProtocol` says so."""
    from genomic_variant_classifier.monitoring.model_registry import (
        RegistryInvariantError)
    with pytest.raises(RegistryInvariantError):
        a_protocol(protocol_id="")
