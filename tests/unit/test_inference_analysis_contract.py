"""The analysis contract (owner ruling 2026-10-03 L226-248; independence states 2026-10-03b; exposure-failure policy 2026-10-08g).

Author: Monzia Moodie
"""
from __future__ import annotations

import dataclasses

import pytest

from genomic_variant_classifier.inference.analysis_contract import (
    Amendment, DraftContract, Evaluation, Execution, ExposureFailurePolicy, Families, FailureDiagnostic, FailureEligibility,
    FailureExecution, FailurePrimary, FailureRevision, Independence, InputSource, Overlap, Relationship, Target)
from genomic_variant_classifier.inference.exact_confirmation import InferenceError

TARGET = Target("asthma (FinnGen J10_ASTHMA, provisional)", "FinnGen R13 standalone", "trans evidence adds to burden",
                "known-positive recovery at 20")
INPUT = InputSource("burden GCST90085447", "2021", "a" * 64, 1024, "pLoF + likely deleterious missense, MAF <= 1%",
                    "decimal text p-values")
FAMILIES = Families(("GENE1", "GENE2"), (("GENE1", ("e1", "e2")), ("GENE2", ("e1",))), "unmeasured pairs: bound 1, state kept")
EXECUTION = Execution("f471153bfa3c0069cd68a67565000889c7cdf5d1", "R/DANDELION.R (root)", "R 4.x + pinned lockfile",
                      "record the actual backend per exposure")
EVALUATION = Evaluation(20, "DANDELION", "burden-only", "frozen order; ties by stable gene identifier", "reference-v1",
                        "assay-v1", ("trans-only", "conservative conjunction"))


def overlap(state, relationship=Relationship.PARTICIPANT_OVERLAP):
    return Independence("FinnGen R13", "eQTLGen Phase I trans", relationship, state, "Phase I study documentation")


def draft(**kw):
    args = dict(analysis_id="asthma-extension", unresolved=(), target=TARGET, inputs=(INPUT,), families=FAMILIES,
                execution=EXECUTION, evaluation=EVALUATION, exposure_failure_policy=ExposureFailurePolicy())
    args.update(kw)
    return DraftContract(**args)


def code_of(call):
    with pytest.raises(InferenceError) as exc:
        call()
    return exc.value.code


def test_a_complete_draft_seals_with_a_content_identity():
    sealed = draft().seal()
    assert len(sealed.contract_id) == 64 and sealed.render().endswith(b"\n")
    assert sealed.contract_id == draft().seal().contract_id          # deterministic


def test_named_unresolved_choices_refuse_sealing():
    assert code_of(lambda: draft(unresolved=("reference inclusion rule",)).seal()) == "contract_unresolved"


@pytest.mark.parametrize("missing", ["target", "families", "execution", "evaluation", "exposure_failure_policy"])
def test_a_missing_element_refuses_sealing(missing):
    assert code_of(lambda: draft(**{missing: None}).seal()) == "contract_incomplete"


@pytest.mark.parametrize("state", [Overlap.UNRESOLVED, Overlap.DOCUMENTED_OVERLAP])
def test_an_independence_claim_needs_documented_disjointness(state):
    assert code_of(lambda: draft(independence=(overlap(state),), independence_claimed=True).seal()) == \
        "independence_claim_unsupported"
    assert draft(independence=(overlap(state),), independence_claimed=False).seal()       # the CONTRACT still seals


def test_an_independence_claim_with_no_recorded_relationship_is_refused():
    assert code_of(lambda: draft(independence_claimed=True).seal()) == "independence_claim_unsupported"


def test_a_documented_disjoint_claim_seals_and_relationship_kinds_are_distinct():
    sealed = draft(independence=(overlap(Overlap.DOCUMENTED_DISJOINT),
                                 overlap(Overlap.DOCUMENTED_DISJOINT, Relationship.EVIDENCE_REUSE_IN_ASCERTAINMENT)),
                   independence_claimed=True).seal()
    assert len(sealed.independence) == 2


def test_any_changed_element_changes_the_identity():
    base = draft().seal().contract_id
    assert draft(evaluation=dataclasses.replace(EVALUATION, k=50)).seal().contract_id != base
    assert draft(inputs=(dataclasses.replace(INPUT, sha256="b" * 64),)).seal().contract_id != base


def test_an_amendment_is_a_new_identity_naming_the_superseded_one():
    first = draft().seal()
    amended = draft(amendment=Amendment(first.contract_id, "reference rule v2 after adjudication")).seal()
    assert amended.contract_id != first.contract_id and amended.amendment.supersedes == first.contract_id
    assert code_of(lambda: Amendment("not-a-digest", "x")) == "amendment_supersedes"


@pytest.mark.parametrize("make, code", [
    (lambda: dataclasses.replace(EVALUATION, k=True), "k_invalid"),
    (lambda: dataclasses.replace(EVALUATION, primary_comparator="DANDELION"), "evaluation_comparator"),
    (lambda: dataclasses.replace(INPUT, size_bytes=0), "input_size"),
    (lambda: dataclasses.replace(EXECUTION, code_commit="f471153"), "code_commit"),
    (lambda: Families(("G1", "G2"), (("G2", ("e",)), ("G1", ("e",))), "p"), "exposure_families"),
    (lambda: Families(("G1", "G1"), (("G1", ("e",)), ("G1", ("e",))), "p"), "gene_family"),
])
def test_elements_refuse_malformations(make, code):
    assert code_of(make) == code


def test_a_sealed_contract_is_immutable():
    sealed = draft().seal()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sealed.analysis_id = "changed"


# ------------------------------------------------------------------------------------------------ exposure-failure policy (2026-10-08g)
def test_the_ruled_policy_is_the_default_and_is_bound_into_the_contract_identity():
    import json
    sealed = draft().seal()
    doc = json.loads(sealed.render())
    assert doc["exposure_failure_policy"] == {
        "eligibility": {"definition": "frozen_before_execution", "mixture_failure_is_ineligibility": False},
        "execution": {"continue_after_exposure_failure": True, "preserve_successful_outputs": True,
                      "require_one_status_per_planned_exposure": True},
        "primary": {"require_all_eligible_exposures_scored": True, "on_failure": "withhold_primary_ranking_and_delta_h20"},
        "diagnostic": {"partial_ranking_allowed": True, "status": "exploratory_conditional_on_estimability",
                       "retain_planned_gene_universe": True, "missing_score_representation": "unscored"},
        "policy_revision": {"automatic_relaxation": False, "require_new_contract_version": True}}
    stricter = ExposureFailurePolicy(diagnostic=FailureDiagnostic(partial_ranking_allowed=False))
    assert draft(exposure_failure_policy=stricter).seal().contract_id != sealed.contract_id


@pytest.mark.parametrize("make, code", [
    (lambda: FailureEligibility(mixture_failure_is_ineligibility=True), "policy_mixture_failure_relabelled"),
    (lambda: FailureEligibility(definition="after_execution"), "policy_eligibility_definition"),
    (lambda: FailureExecution(continue_after_exposure_failure=False), "policy_execution_continue_after_exposure_failure"),
    (lambda: FailureExecution(preserve_successful_outputs=1), "policy_execution_preserve_successful_outputs"),
    (lambda: FailurePrimary(require_all_eligible_exposures_scored=False), "policy_primary_completeness"),
    (lambda: FailurePrimary(on_failure="rank_what_was_scored"), "policy_primary_on_failure"),
    (lambda: FailureDiagnostic(missing_score_representation="1"), "policy_diagnostic_missing_score"),
    (lambda: FailureDiagnostic(retain_planned_gene_universe=False), "policy_diagnostic_universe"),
    (lambda: FailureDiagnostic(status="primary"), "policy_diagnostic_status"),
    (lambda: FailureRevision(automatic_relaxation=True), "policy_automatic_relaxation"),
    (lambda: FailureRevision(require_new_contract_version=False), "policy_revision_version"),
    (lambda: ExposureFailurePolicy(primary=FailureDiagnostic()), "policy_section"),
    (lambda: draft(exposure_failure_policy="ruled").seal(), "exposure_failure_policy"),
])
def test_a_weaker_policy_cannot_be_constructed_or_sealed(make, code):
    assert code_of(make) == code
