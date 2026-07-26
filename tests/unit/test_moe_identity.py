"""S0 Commit 2: typed contracts for Panel S0 expert identity and admissibility.

WHAT IS PINNED
==============
  1. Causal boundary: BIOLOGICAL_MEDIATION can never be admissible through S0.
  2. The four routing quantities are distinct and mechanism evidence never enters
     the allocation.
  3. Claim-ledger invariants: admissible requires held-out scientific validation;
     non-admissible requires a finding.
  4. Report-label invariant: opaque key below ADMISSIBLE_EXTERNAL; biological name
     only for an externally admissible anchored mechanism.
  5. Highest-supported-claim summary, not a hidden minimum.
  6. Lineage resets admissibility except on a pure continuation.
  7. Anchor independence: a computational teacher cannot establish mechanism alone.
  8. Only an ANCHORED_MECHANISM expert may reach an admissible state.
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.evaluation.r3_capability import (
    MethodValidationState, ScientificValidationState)
from genomic_variant_classifier.evaluation.moe_identity import (
    InsufficientSupportError, ExpertIdentityClass, MechanisticAdmissibilityState,
    MechanisticClaimType, AnchorEvidenceTier, RoutingQuantity,
    AnchorIndependenceProfile, ExpertLineageEvent, ExpertLineage, ClaimLedgerEntry,
    ExpertIdentityEvidence, S0_INADMISSIBLE_CLAIMS, OPAQUE_LABEL_PREFIX)
from genomic_variant_classifier.evaluation.moe_identity import (
    _assert_biological_mediation_is_unreachable, _mechanism_evidence_may_enter_allocation)

MV = MethodValidationState
SV = ScientificValidationState


def _lineage(event=ExpertLineageEvent.CREATED, uuid="expert_003"):
    return ExpertLineage(uuid, "release_1", (), event, "a" * 64, "b" * 64)


def _finding_entry(claim=MechanisticClaimType.ANCHOR_PREDICTION):
    return ClaimLedgerEntry(claim, MV.NOT_EVALUATED, SV.NOT_EVALUATED, False,
                            "pending", ("not yet validated",))


# --------------------------------------------------------------------------- #
# 1. causal boundary
# --------------------------------------------------------------------------- #
def test_biological_mediation_can_never_be_admissible():
    with pytest.raises(InsufficientSupportError):
        ClaimLedgerEntry(MechanisticClaimType.BIOLOGICAL_MEDIATION,
                         MV.PASSED_SYNTHETIC, SV.PASSED_HELDOUT, True, None, ())


def test_mediation_guard_raises_and_other_claims_pass():
    with pytest.raises(InsufficientSupportError):
        _assert_biological_mediation_is_unreachable(
            MechanisticClaimType.BIOLOGICAL_MEDIATION)
    # a non-mediation claim passes the guard silently
    _assert_biological_mediation_is_unreachable(
        MechanisticClaimType.ANCHOR_PREDICTION)


def test_mediation_is_in_the_inadmissible_set():
    assert MechanisticClaimType.BIOLOGICAL_MEDIATION in S0_INADMISSIBLE_CLAIMS


# --------------------------------------------------------------------------- #
# 2. routing quantities
# --------------------------------------------------------------------------- #
def test_mechanism_evidence_never_enters_allocation():
    assert _mechanism_evidence_may_enter_allocation() is False


def test_four_distinct_routing_quantities():
    assert len(set(RoutingQuantity)) == 4
    assert RoutingQuantity.MECHANISM_EVIDENCE is not RoutingQuantity.ALLOCATION


# --------------------------------------------------------------------------- #
# 3. claim-ledger invariants
# --------------------------------------------------------------------------- #
def test_admissible_claim_requires_heldout_scientific_validation():
    with pytest.raises(ValueError, match="held-out or stronger"):
        ClaimLedgerEntry(MechanisticClaimType.ANCHOR_PREDICTION,
                         MV.PASSED_SYNTHETIC, SV.NOT_EVALUATED, True, None, ())


def test_non_admissible_claim_requires_a_finding():
    with pytest.raises(ValueError, match="at least one finding"):
        ClaimLedgerEntry(MechanisticClaimType.ANCHOR_PREDICTION,
                         MV.NOT_EVALUATED, SV.NOT_EVALUATED, False, "x", ())


def test_admissible_claim_construction_succeeds():
    e = ClaimLedgerEntry(MechanisticClaimType.ANCHOR_PREDICTION,
                         MV.PASSED_SYNTHETIC, SV.PASSED_HELDOUT, True, None, ())
    assert e.admissible


# --------------------------------------------------------------------------- #
# 4. report-label invariant
# --------------------------------------------------------------------------- #
def test_external_label_is_opaque_below_admissible_external():
    ev = ExpertIdentityEvidence(
        "expert_003", ExpertIdentityClass.ANCHORED_MECHANISM, _lineage(),
        MechanisticAdmissibilityState.NOT_ADMISSIBLE, (_finding_entry(),),
        "splice-compatible expert", ("not yet validated",))
    assert ev.external_report_label() == "expert_003"
    assert ev.internal_candidate_label() == "expert_003"


def test_external_label_shows_biology_only_at_admissible_external():
    ev = ExpertIdentityEvidence(
        "expert_003", ExpertIdentityClass.ANCHORED_MECHANISM, _lineage(),
        MechanisticAdmissibilityState.ADMISSIBLE_EXTERNAL,
        (ClaimLedgerEntry(MechanisticClaimType.ANCHOR_PREDICTION,
                          MV.PASSED_SYNTHETIC, SV.PASSED_HELDOUT, True, None, ()),),
        "splice-compatible expert")
    assert ev.external_report_label() == "splice-compatible expert"
    assert "candidate" in ev.internal_candidate_label()


def test_expert_uuid_must_be_opaque_key():
    with pytest.raises(ValueError, match="opaque key"):
        ExpertIdentityEvidence(
            "the_splice_expert", ExpertIdentityClass.ANCHORED_MECHANISM, _lineage("created", "the_splice_expert"),
            MechanisticAdmissibilityState.NOT_ADMISSIBLE, (_finding_entry(),),
            None, ("x",))


# --------------------------------------------------------------------------- #
# 5. highest-supported claim
# --------------------------------------------------------------------------- #
def test_highest_supported_claim_is_not_a_hidden_minimum():
    # TWO admissible claims at different order positions, so the summary must pick the
    # HIGHER one (a min would pick ANCHOR_PREDICTION). ANCHOR_PREDICTION is earliest in
    # the enum; ROUTING_RELEVANCE is later, so the highest supported is ROUTING_RELEVANCE.
    ev = ExpertIdentityEvidence(
        "expert_003", ExpertIdentityClass.ANCHORED_MECHANISM, _lineage(),
        MechanisticAdmissibilityState.ADMISSIBLE_EXTERNAL,
        (ClaimLedgerEntry(MechanisticClaimType.ANCHOR_PREDICTION,
                          MV.PASSED_SYNTHETIC, SV.PASSED_HELDOUT, True, None, ()),
         ClaimLedgerEntry(MechanisticClaimType.ROUTING_RELEVANCE,
                          MV.PASSED_SYNTHETIC, SV.PASSED_HELDOUT, True, None, ()),
         ClaimLedgerEntry(MechanisticClaimType.CLINICAL_ACTIONABILITY,
                          MV.NOT_EVALUATED, SV.NOT_EVALUATED, False, "not run", ("x",))),
        "splice-compatible expert")
    # highest of {ANCHOR_PREDICTION, ROUTING_RELEVANCE} by enum order is ROUTING_RELEVANCE
    assert ev.highest_supported_claim() is MechanisticClaimType.ROUTING_RELEVANCE
    assert MechanisticClaimType.CLINICAL_ACTIONABILITY in ev.claims_not_established()


def test_no_admissible_claim_yields_no_highest():
    ev = ExpertIdentityEvidence(
        "expert_003", ExpertIdentityClass.PREDICTIVE_UNNAMED, _lineage(),
        MechanisticAdmissibilityState.NOT_ADMISSIBLE, (_finding_entry(),),
        None, ("nothing validated",))
    assert ev.highest_supported_claim() is None


# --------------------------------------------------------------------------- #
# 6. lineage
# --------------------------------------------------------------------------- #
def test_lineage_event_resets_admissibility_except_continuation():
    assert _lineage(ExpertLineageEvent.CREATED).resets_admissibility() is True
    assert _lineage(ExpertLineageEvent.SPLIT).resets_admissibility() is True
    assert _lineage(ExpertLineageEvent.MERGED).resets_admissibility() is True
    assert _lineage(ExpertLineageEvent.CONTINUED).resets_admissibility() is False


# --------------------------------------------------------------------------- #
# 7. anchor independence
# --------------------------------------------------------------------------- #
def test_computational_teacher_cannot_establish_mechanism_alone():
    prof = AnchorIndependenceProfile(
        tier=AnchorEvidenceTier.COMPUTATIONAL_TEACHER,
        in_training_signal=False, in_router_input=False,
        shares_generating_process=False, covers_claim_population=True)
    assert prof.is_independent_enough_for_mechanism() is False


def test_leaked_experimental_anchor_is_not_independent():
    prof = AnchorIndependenceProfile(
        tier=AnchorEvidenceTier.DIRECT_EXPERIMENTAL,
        in_training_signal=True, in_router_input=False,
        shares_generating_process=False, covers_claim_population=True)
    assert prof.is_independent_enough_for_mechanism() is False


def test_clean_experimental_anchor_is_independent():
    prof = AnchorIndependenceProfile(
        tier=AnchorEvidenceTier.DIRECT_EXPERIMENTAL,
        in_training_signal=False, in_router_input=False,
        shares_generating_process=False, covers_claim_population=True)
    assert prof.is_independent_enough_for_mechanism() is True


# --------------------------------------------------------------------------- #
# 8. only anchored mechanism may reach admissible
# --------------------------------------------------------------------------- #
def test_only_anchored_mechanism_may_reach_admissible_state():
    with pytest.raises(ValueError, match="only an ANCHORED_MECHANISM"):
        ExpertIdentityEvidence(
            "expert_003", ExpertIdentityClass.GENERALIST, _lineage(),
            MechanisticAdmissibilityState.ADMISSIBLE_INTERNAL,
            (ClaimLedgerEntry(MechanisticClaimType.ANCHOR_PREDICTION,
                              MV.PASSED_SYNTHETIC, SV.PASSED_HELDOUT, True, None, ()),),
            None)


def test_biological_mediation_cannot_appear_admissible_in_a_ledger():
    # constructing the entry already raises; confirm the composite also guards by
    # building a legal ledger and asserting mediation is absent from established claims
    ev = ExpertIdentityEvidence(
        "expert_003", ExpertIdentityClass.ANCHORED_MECHANISM, _lineage(),
        MechanisticAdmissibilityState.NOT_ADMISSIBLE, (_finding_entry(),),
        None, ("x",))
    assert MechanisticClaimType.BIOLOGICAL_MEDIATION in ev.claims_not_established()
