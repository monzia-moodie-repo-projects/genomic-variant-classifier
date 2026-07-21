"""The capability-and-evidence contract.

WHY THIS FILE EXISTS
====================
A validation gate over a capability that does not exist does not block. It
PASSES VACUOUSLY, because there is nothing to check and therefore nothing to
fail. A green Panel H would then be cited as evidence a disease head was
validated, when no disease head exists.

That is the defect removed five times on 2026-07-21: `assert_data_usable`
called from nowhere; a stale enumerated role cap; a calibrator fitted on genes
the models trained on; `calibration_valid` asserting soundness it never checked;
and `n_groups_with_multiple_covariate_values` recorded with exactly one reader
in the whole repository, a test asserting it equalled zero.

    A check that cannot fail is worse than no check, because it manufactures
    confidence.

WHAT THIS CONTRACT DOES DIFFERENTLY
------------------------------------
The invariant lives in `CapabilityEvidence.__post_init__`, not in the gate. An
OK capability that is not validated, or whose target is not admissible, or which
names no artifact, CANNOT BE CONSTRUCTED. There is no path to the object that
avoids the check, so no caller can forget to consult it -- which is exactly how
a well-tested `assert_data_usable` came to be called from nowhere.

THE THREE STATES THAT MADE A FOURTH AXIS NECESSARY
---------------------------------------------------
Measured against the repository on 2026-07-21:

  regression / conformal quantile regression
      data/external/gtex, data/rnaseq and data/external/functional_assays are
      all ABSENT. SpliceAI and AlphaMissense supply scores, but those are
      another model's predictions used as INPUT FEATURES, not measured
      outcomes.  -> IMPLEMENTED_NO_OUTPUT + ABSENT

  multi-label disease
      ClinVar carries a disease field and phenotype identifiers and the pipeline
      reads NEITHER; OMIM contributes only a gene-level disease COUNT.
      -> NOT_IMPLEMENTED + ABSENT

  gene ranking
      Gene-level ground truth EXISTS -- clingen_validity_score, omim_n_diseases
      -- but all four gene-disease annotations are INPUT FEATURES in the
      91-feature contract. Ranking genes with a model handed ClinGen's verdict
      and scoring that ranking against ClinGen is circular.
      -> OUTPUT_AVAILABLE + CONTAMINATED

The third is why TargetState exists as a typed axis rather than a reason string:
the capability works, the target is present, and the result is still
inadmissible.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.evaluation.capabilities import (
    REASON_NO_DISEASE_LABEL_INGESTION,
    REASON_NO_REGRESSION_TARGETS,
    REASON_TARGET_IS_AN_INPUT_FEATURE,
    CapabilityEvidence,
    CapabilityState,
    MetricStatus,
    TargetState,
    release_gate_satisfied,
    summarize_release,
)


def _validated(name="panel", artifact="out.parquet"):
    return CapabilityEvidence(
        capability_name=name,
        capability_state=CapabilityState.VALIDATED,
        target_state=TargetState.ADMISSIBLE,
        output_artifact=artifact,
        target_manifest="targets.json",
        status=MetricStatus.OK,
        reason=None)


# --------------------------------------------------------------------------- #
# 1. a false green cannot be CONSTRUCTED
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state", [s for s in CapabilityState
                                   if s is not CapabilityState.VALIDATED])
def test_ok_requires_a_validated_capability(state):
    """'It did not fail' is not 'it was tested'."""
    with pytest.raises(ValueError, match="requires CapabilityState.VALIDATED"):
        CapabilityEvidence(
            capability_name="panel", capability_state=state,
            target_state=TargetState.ADMISSIBLE, output_artifact="out.parquet",
            target_manifest=None, status=MetricStatus.OK, reason=None)


@pytest.mark.parametrize("target", [t for t in TargetState
                                    if t is not TargetState.ADMISSIBLE])
def test_ok_requires_an_admissible_target(target):
    """A result measured against an absent, contaminated or unverified target is
    not evidence, however good the number."""
    with pytest.raises(ValueError, match="requires TargetState.ADMISSIBLE"):
        CapabilityEvidence(
            capability_name="panel", capability_state=CapabilityState.VALIDATED,
            target_state=target, output_artifact="out.parquet",
            target_manifest=None, status=MetricStatus.OK, reason=None)


@pytest.mark.parametrize("artifact", [None, ""])
def test_ok_requires_a_named_output_artifact(artifact):
    """A pass with nothing to point at cannot be reproduced or audited."""
    with pytest.raises(ValueError, match="requires a named output artifact"):
        CapabilityEvidence(
            capability_name="panel", capability_state=CapabilityState.VALIDATED,
            target_state=TargetState.ADMISSIBLE, output_artifact=artifact,
            target_manifest=None, status=MetricStatus.OK, reason=None)


@pytest.mark.parametrize("status", [s for s in MetricStatus
                                    if s is not MetricStatus.OK])
@pytest.mark.parametrize("reason", [None, ""])
def test_every_non_ok_status_requires_a_reason(status, reason):
    """Parametrized over MetricStatus ITSELF, so a status added tomorrow is
    covered without anyone editing this test."""
    with pytest.raises(ValueError, match="requires a machine-readable reason"):
        CapabilityEvidence(
            capability_name="panel",
            capability_state=CapabilityState.IMPLEMENTED_NO_OUTPUT,
            target_state=TargetState.ABSENT, output_artifact=None,
            target_manifest=None, status=status, reason=reason)


def test_a_fully_validated_capability_can_be_constructed():
    """The invariant must not be so strict that nothing can ever pass -- a gate
    that cannot be satisfied is as useless as one that cannot fail."""
    e = _validated()
    assert e.status is MetricStatus.OK
    assert release_gate_satisfied(e) is True


# --------------------------------------------------------------------------- #
# 2. type discipline
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,value,match", [
    ("capability_state", "validated", "must be a CapabilityState"),
    ("target_state", "admissible", "must be a TargetState"),
    ("status", "ok", "must be a MetricStatus"),
])
def test_bare_strings_are_refused(field, value, match):
    """A bare string cannot be checked against the enum and would let an unknown
    state pass -- and because these enums inherit from str, `==` comparisons
    would even appear to work."""
    kwargs = dict(capability_name="panel",
                  capability_state=CapabilityState.VALIDATED,
                  target_state=TargetState.ADMISSIBLE,
                  output_artifact="out.parquet", target_manifest=None,
                  status=MetricStatus.OK, reason=None)
    kwargs[field] = value
    with pytest.raises(TypeError, match=match):
        CapabilityEvidence(**kwargs)


def test_an_empty_capability_name_is_refused():
    with pytest.raises(ValueError, match="capability_name"):
        CapabilityEvidence(
            capability_name="", capability_state=CapabilityState.NOT_IMPLEMENTED,
            target_state=TargetState.ABSENT, output_artifact=None,
            target_manifest=None, status=MetricStatus.NOT_IMPLEMENTED,
            reason=REASON_NO_DISEASE_LABEL_INGESTION)


def test_the_evidence_is_frozen():
    """Evidence that can be edited after construction escapes its own
    invariant."""
    e = _validated()
    with pytest.raises(Exception):
        e.status = MetricStatus.FAILED       # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 3. the release gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [s for s in MetricStatus
                                    if s is not MetricStatus.OK])
def test_no_non_ok_status_satisfies_the_release_gate(status):
    """Not skipped, not waived, not passed. UNSATISFIED."""
    e = CapabilityEvidence(
        capability_name="panel",
        capability_state=CapabilityState.IMPLEMENTED_NO_OUTPUT,
        target_state=TargetState.ABSENT, output_artifact=None,
        target_manifest=None, status=status, reason="some_reason")
    assert release_gate_satisfied(e) is False


def test_the_gate_refuses_a_dict():
    """A dict would compare a string to an enum and quietly return False for a
    PASSING capability -- a false red, which erodes trust in the gate as surely
    as a false green."""
    with pytest.raises(TypeError, match="expects CapabilityEvidence"):
        release_gate_satisfied({"status": "ok"})


def test_the_summary_cannot_report_green_while_a_panel_is_unsatisfied():
    """A caller must not be able to say 'all panels green' by iterating only the
    ones that passed."""
    items = [_validated("panel_a"),
             CapabilityEvidence(
                 capability_name="panel_h",
                 capability_state=CapabilityState.NOT_IMPLEMENTED,
                 target_state=TargetState.ABSENT, output_artifact=None,
                 target_manifest=None, status=MetricStatus.NOT_IMPLEMENTED,
                 reason=REASON_NO_DISEASE_LABEL_INGESTION)]
    out = summarize_release(items)
    assert out["release_complete"] is False
    assert out["n_unsatisfied"] == 1
    assert out["unsatisfied"][0]["capability_name"] == "panel_h"
    assert out["unsatisfied"][0]["reason"] == REASON_NO_DISEASE_LABEL_INGESTION


def test_an_empty_release_is_not_complete():
    """Zero panels is not 'everything passed'. Vacuous truth is the exact
    failure mode this module exists to prevent."""
    assert summarize_release([])["release_complete"] is False


def test_a_release_of_only_validated_panels_is_complete():
    assert summarize_release([_validated("a"), _validated("b")])["release_complete"] is True


# --------------------------------------------------------------------------- #
# 4. THE SABOTAGE THAT MUST FAIL
# --------------------------------------------------------------------------- #
def test_sabotage_a_contaminated_gene_ranking_cannot_report_ok():
    """The fixture that MUST fail. Gene ranking has a working capability, an
    available output AND an available target. Only TargetState.CONTAMINATED
    stops it, and that state exists precisely because the earlier draft encoded
    this in a reason STRING, where nothing could enforce it.

    A validation suite needs at least one fixture that must fail for every gate,
    or the project again risks a guard no possible input can trigger."""
    with pytest.raises(ValueError, match="requires TargetState.ADMISSIBLE"):
        CapabilityEvidence(
            capability_name="gene_ranking",
            capability_state=CapabilityState.VALIDATED,
            target_state=TargetState.CONTAMINATED,
            output_artifact="gene_rankings.parquet",
            target_manifest="clingen_current.json",
            status=MetricStatus.OK,
            reason=None)


def test_the_contaminated_gene_ranking_is_representable_as_not_ok():
    """The honest form of the same situation: everything real is recorded, and
    the status says why it does not count."""
    e = CapabilityEvidence(
        capability_name="gene_ranking",
        capability_state=CapabilityState.OUTPUT_AVAILABLE,
        target_state=TargetState.CONTAMINATED,
        output_artifact="gene_rankings.parquet",
        target_manifest="clingen_current.json",
        status=MetricStatus.INSUFFICIENT_SUPPORT,
        reason=REASON_TARGET_IS_AN_INPUT_FEATURE)
    assert release_gate_satisfied(e) is False
    assert e.to_dict()["target_state"] == "contaminated"


# --------------------------------------------------------------------------- #
# 5. the three measured capability states
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,cap,target,status,reason", [
    ("regression", CapabilityState.IMPLEMENTED_NO_OUTPUT, TargetState.ABSENT,
     MetricStatus.INSUFFICIENT_SUPPORT, REASON_NO_REGRESSION_TARGETS),
    ("conformal_quantile_regression", CapabilityState.IMPLEMENTED_NO_OUTPUT,
     TargetState.ABSENT, MetricStatus.INSUFFICIENT_SUPPORT,
     REASON_NO_REGRESSION_TARGETS),
    ("multilabel_disease", CapabilityState.NOT_IMPLEMENTED, TargetState.ABSENT,
     MetricStatus.NOT_IMPLEMENTED, REASON_NO_DISEASE_LABEL_INGESTION),
    ("gene_ranking", CapabilityState.OUTPUT_AVAILABLE, TargetState.CONTAMINATED,
     MetricStatus.INSUFFICIENT_SUPPORT, REASON_TARGET_IS_AN_INPUT_FEATURE),
])
def test_the_measured_states_are_representable_and_unsatisfied(
        name, cap, target, status, reason):
    e = CapabilityEvidence(capability_name=name, capability_state=cap,
                           target_state=target, output_artifact=None,
                           target_manifest=None, status=status, reason=reason)
    assert release_gate_satisfied(e) is False
    assert e.reason == reason


# --------------------------------------------------------------------------- #
# 6. ONE status vocabulary, and a Python floor that cannot drift
# --------------------------------------------------------------------------- #
def test_there_is_exactly_one_metric_status_class():
    """Two enums sharing a name is the divergence problem removed in b8275a0,
    where the legacy evaluator was DELETED rather than wrapped because two
    evaluation contracts in one codebase invite drift."""
    from genomic_variant_classifier.evaluation import capabilities, clustering_metrics
    assert clustering_metrics.MetricStatus is capabilities.MetricStatus


def test_the_original_status_values_are_frozen():
    """Existing run manifests on disk contain these exact strings. Changing one
    would silently orphan every historical record."""
    for name, value in (("OK", "ok"), ("UNDEFINED", "undefined"),
                        ("INSUFFICIENT_SUPPORT", "insufficient_support"),
                        ("DEPENDENCY_UNAVAILABLE", "dependency_unavailable"),
                        ("COMPUTATIONALLY_DEFERRED", "computationally_deferred"),
                        ("FAILED", "failed")):
        assert getattr(MetricStatus, name).value == value


@pytest.mark.parametrize("enum_cls", [MetricStatus, CapabilityState, TargetState])
def test_the_enums_are_json_serialisable_without_a_custom_encoder(enum_cls):
    import json
    for member in enum_cls:
        assert json.loads(json.dumps({"v": member}))["v"] == member.value


def test_no_module_uses_strenum_which_would_break_the_declared_python_floor():
    """StrEnum arrived in Python 3.11. pyproject declares
    requires-python = ">=3.10", and the continuous integration matrix runs only
    3.11 and 3.12 -- so a 3.10 installation would fail at IMPORT time with
    nothing in the pipeline to catch it. This test is the thing that catches
    it."""
    import pathlib
    import genomic_variant_classifier as pkg
    root = pathlib.Path(pkg.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "StrEnum" in stripped and "`StrEnum`" not in stripped:
                offenders.append(f"{path.name}: {stripped[:70]}")
    assert not offenders, (
        "StrEnum requires Python 3.11 but pyproject declares >=3.10: " +
        "; ".join(offenders))
