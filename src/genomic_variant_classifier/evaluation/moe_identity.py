"""Typed contracts for Panel S0 -- expert identity, routing quantities, and
mechanistic interpretive admissibility.

WHY THIS EXISTS
===============
docs/specifications/PANEL_S0_ROUTING_IDENTIFIABILITY.md specifies when a Mixture-of-
Experts routing quantity may be read as evidence about biology. This module turns the
parts of that specification that are pure contract -- the identity classes, the four
routing quantities, the anchor evidence hierarchy, the claim ledger, the
admissibility states, expert lineage, and the causal boundary -- into enforced Python
types. No production model exists; nothing here trains, routes, or scores. Every
record is constructed NON-ADMISSIBLE, and the invariants make the inadmissible states
the only expressible default.

Five specification commitments are enforced structurally here, not merely documented:

  1. Model reliance is not biological causation. BIOLOGICAL_MEDIATION is a claim type
     that Panel S0 can NEVER establish; requesting it through an S0 admissibility path
     raises InsufficientSupportError. (S0.17)
  2. The four routing quantities are distinct. Mechanism evidence is the only one
     eligible for a mechanistic reading, and it must not enter the allocation. (S0.2)
  3. Admissibility is claim-specific. An expert admissible for one claim is not
     thereby admissible for a stronger one; the expert-global summary is the HIGHEST
     supported claim, never a hidden minimum. (S0.15, S0.17, S0.19)
  4. The report-label invariant. Below ADMISSIBLE_EXTERNAL an expert's external label
     is its opaque key (expert_003), never a biological name. (S0.19)
  5. Lineage resets admissibility. A split, merge, anchor change, or architecture
     change resets the claim ledger to require re-validation. (S0.14)

The two validation axes (method, scientific) are REUSED from r3_capability.py rather
than redefined, so Panel S0 and Panel R speak the same validation language.

Author: written for Monzia Moodie, 2026-07-22.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .r3_capability import MethodValidationState, ScientificValidationState

__all__ = [
    "InsufficientSupportError",
    "ExpertIdentityClass",
    "MechanisticAdmissibilityState",
    "MechanisticClaimType",
    "AnchorEvidenceTier",
    "RoutingQuantity",
    "AnchorIndependenceProfile",
    "ExpertLineageEvent",
    "ExpertLineage",
    "ClaimLedgerEntry",
    "ExpertIdentityEvidence",
    "OPAQUE_LABEL_PREFIX",
    "S0_INADMISSIBLE_CLAIMS",
]

OPAQUE_LABEL_PREFIX = "expert_"


class InsufficientSupportError(RuntimeError):
    """Raised when a claim is requested that the available evidence -- or the panel's
    scope -- cannot establish. Used for the causal boundary: biological mediation is
    outside Panel S0 entirely."""


class ExpertIdentityClass(str, Enum):
    """S0.3. The mechanistic-admissibility gate applies only to ANCHORED_MECHANISM."""

    ANCHORED_MECHANISM = "anchored_mechanism"
    PREDICTIVE_UNNAMED = "predictive_unnamed"
    GENERALIST = "generalist"
    RESIDUAL = "residual"


class MechanisticAdmissibilityState(str, Enum):
    """S0.19. An expert's mechanistic admissibility, on its own axis."""

    NOT_EVALUATED = "not_evaluated"
    NOT_ADMISSIBLE = "not_admissible"
    PROVISIONAL = "provisional"
    ADMISSIBLE_INTERNAL = "admissible_internal"
    ADMISSIBLE_EXTERNAL = "admissible_external"


class MechanisticClaimType(str, Enum):
    """S0.17. The claim ledger records admissibility PER claim type. BIOLOGICAL_
    MEDIATION is present but is never admissible through Panel S0 -- see
    S0_INADMISSIBLE_CLAIMS and the ClaimLedgerEntry invariant."""

    ANCHOR_PREDICTION = "anchor_prediction"
    ROUTING_RELEVANCE = "routing_relevance"
    EXPERT_COMPUTATIONAL_NECESSITY = "expert_computational_necessity"
    PATHOGENICITY_RELEVANCE = "pathogenicity_relevance"
    CROSS_GENE_TRANSFER = "cross_gene_transfer"
    CROSS_ASSAY_TRANSFER = "cross_assay_transfer"
    CLINICAL_ACTIONABILITY = "clinical_actionability"
    # Never admissible through Panel S0; requires an independent causal panel.
    BIOLOGICAL_MEDIATION = "biological_mediation"


# Claims Panel S0 can never establish. Deferred to a future Panel T.
S0_INADMISSIBLE_CLAIMS = frozenset({MechanisticClaimType.BIOLOGICAL_MEDIATION})


class AnchorEvidenceTier(str, Enum):
    """S0.4. Evidence tier, strongest first. A COMPUTATIONAL_TEACHER (e.g. SpliceAI)
    cannot establish mechanistic admissibility on its own."""

    DIRECT_EXPERIMENTAL = "direct_experimental"
    ORTHOGONAL_EXPERIMENTAL = "orthogonal_experimental"
    HUMAN_ASSOCIATION = "human_association"
    CURATED_MECHANISM = "curated_mechanism"
    COMPUTATIONAL_TEACHER = "computational_teacher"
    HEURISTIC_PRIOR = "heuristic_prior"


# Tiers that can, on their own, support a mechanistic (not merely distillation) claim.
_MECHANISM_CAPABLE_TIERS = frozenset({
    AnchorEvidenceTier.DIRECT_EXPERIMENTAL,
    AnchorEvidenceTier.ORTHOGONAL_EXPERIMENTAL,
})


class RoutingQuantity(str, Enum):
    """S0.2. The four distinct routing quantities. Only MECHANISM_EVIDENCE is eligible
    for a mechanistic reading; ALLOCATION is engineering and simplex-constrained."""

    ALLOCATION = "allocation"                    # a_e -- engineering, simplex-constrained
    MECHANISM_EVIDENCE = "mechanism_evidence"    # m_e -- the only scientific quantity
    EXPERT_UTILITY = "expert_utility"            # u_e -- marginal predictive benefit
    RELIABILITY = "reliability"                  # q_e -- runtime evidence trust


# Quantities that may be combined into the softmax allocation. Mechanism evidence is
# deliberately excluded so the scientific quantity never competes for mixing mass.
_ALLOCATION_INPUT_QUANTITIES = frozenset({
    RoutingQuantity.EXPERT_UTILITY,
    RoutingQuantity.RELIABILITY,
})


def _mechanism_evidence_may_enter_allocation() -> bool:
    return RoutingQuantity.MECHANISM_EVIDENCE in _ALLOCATION_INPUT_QUANTITIES


@dataclass(frozen=True)
class AnchorIndependenceProfile:
    """S0.4. Tier and independence are separate axes. `in_training_signal` or
    `in_router_input` true means the anchor leaked into what it is meant to validate."""

    tier: AnchorEvidenceTier
    in_training_signal: bool
    in_router_input: bool
    shares_generating_process: bool
    covers_claim_population: bool

    def is_independent_enough_for_mechanism(self) -> bool:
        """A mechanistic (not distillation) claim needs a mechanism-capable tier that
        did not leak into training or routing and covers the claim population."""
        if self.tier not in _MECHANISM_CAPABLE_TIERS:
            return False
        if self.in_training_signal or self.in_router_input:
            return False
        if self.shares_generating_process:
            return False
        return self.covers_claim_population


class ExpertLineageEvent(str, Enum):
    """S0.14. A lineage event resets the claim ledger to require re-validation."""

    CREATED = "created"
    CONTINUED = "continued"
    SPLIT = "split"
    MERGED = "merged"
    RETIRED = "retired"
    IDENTITY_REVISED = "identity_revised"


# Events that do NOT reset admissibility (a pure continuation preserves it).
_ADMISSIBILITY_PRESERVING_EVENTS = frozenset({ExpertLineageEvent.CONTINUED})


@dataclass(frozen=True)
class ExpertLineage:
    """S0.14. Immutable identity and lineage. Mechanistic admissibility does not
    silently transfer across a split, merge, anchor change, or architecture change."""

    expert_uuid: str
    model_release: str
    parent_expert_uuids: tuple[str, ...]
    lineage_event: ExpertLineageEvent
    anchor_manifest_sha256: str
    expert_spec_sha256: str

    def resets_admissibility(self) -> bool:
        return self.lineage_event not in _ADMISSIBILITY_PRESERVING_EVENTS


@dataclass(frozen=True)
class ClaimLedgerEntry:
    """S0.17. One claim's admissibility, on the two shipped validation axes. A
    BIOLOGICAL_MEDIATION entry can never be admissible; a non-admissible entry needs a
    finding; an admissible entry needs held-out-or-stronger scientific validation."""

    claim_type: MechanisticClaimType
    method_validation: MethodValidationState
    scientific_validation: ScientificValidationState
    admissible: bool
    reason: str | None
    findings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.claim_type in S0_INADMISSIBLE_CLAIMS and self.admissible:
            raise InsufficientSupportError(
                f"claim {self.claim_type.value!r} can never be admissible through "
                "Panel S0; it requires independent causal evidence (a future Panel T)")
        if self.admissible:
            if self.method_validation not in {
                MethodValidationState.PASSED_SYNTHETIC,
                MethodValidationState.PASSED_INTERNAL_EMPIRICAL,
            }:
                raise ValueError("an admissible claim requires method validation")
            if self.scientific_validation not in {
                ScientificValidationState.PASSED_HELDOUT,
                ScientificValidationState.PASSED_EXTERNAL,
                ScientificValidationState.PASSED_TEMPORAL,
            }:
                raise ValueError(
                    "an admissible claim requires held-out or stronger scientific "
                    "validation")
            if self.reason is not None:
                raise ValueError("an admissible claim carries no blocking reason")
        if not self.admissible and not self.findings:
            raise ValueError("a non-admissible claim requires at least one finding")


# Admissibility states that permit an internal candidate label (never external).
_INTERNAL_LABEL_STATES = frozenset({
    MechanisticAdmissibilityState.ADMISSIBLE_INTERNAL,
    MechanisticAdmissibilityState.ADMISSIBLE_EXTERNAL,
})


@dataclass(frozen=True)
class ExpertIdentityEvidence:
    """S0.19. The per-expert record: identity class, lineage, the claim ledger, the
    overall admissibility state, and the permitted labels. Invariants enforce the
    report-label rule, the highest-supported-claim summary, and the requirement that
    a non-admissible expert records a finding."""

    expert_uuid: str
    identity_class: ExpertIdentityClass
    lineage: ExpertLineage
    admissibility: MechanisticAdmissibilityState
    claim_ledger: tuple[ClaimLedgerEntry, ...]
    proposed_internal_label: str | None
    findings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.expert_uuid.startswith(OPAQUE_LABEL_PREFIX):
            raise ValueError(
                f"expert_uuid must be an opaque key beginning {OPAQUE_LABEL_PREFIX!r}")
        # A biological-mediation claim must not even appear as admissible in the ledger
        # (ClaimLedgerEntry already enforces this, but the composite re-checks).
        for entry in self.claim_ledger:
            if entry.claim_type in S0_INADMISSIBLE_CLAIMS and entry.admissible:
                raise InsufficientSupportError(
                    "biological mediation cannot be admissible through Panel S0")
        # Only the mechanism class is ever eligible for admissibility above NONE.
        if (self.admissibility in _INTERNAL_LABEL_STATES
                and self.identity_class is not ExpertIdentityClass.ANCHORED_MECHANISM):
            raise ValueError(
                "only an ANCHORED_MECHANISM expert may reach an admissible state")
        if (self.admissibility in {
                MechanisticAdmissibilityState.NOT_EVALUATED,
                MechanisticAdmissibilityState.NOT_ADMISSIBLE,
                MechanisticAdmissibilityState.PROVISIONAL,
        } and not self.findings):
            raise ValueError(
                "a non-admissible or provisional expert requires at least one finding")

    def external_report_label(self) -> str:
        """S0.19 report-label invariant -- the SINGLE SOURCE OF TRUTH for the external
        label (no separate __post_init__ assertion, which would be unreachable). Below
        ADMISSIBLE_EXTERNAL the external label
        is the opaque key. Only an externally admissible anchored mechanism may show a
        biological name, and only if one was proposed."""
        if (self.admissibility is MechanisticAdmissibilityState.ADMISSIBLE_EXTERNAL
                and self.identity_class is ExpertIdentityClass.ANCHORED_MECHANISM
                and self.proposed_internal_label is not None):
            return self.proposed_internal_label
        return self.expert_uuid

    def internal_candidate_label(self) -> str:
        """S0.19. After ADMISSIBLE_INTERNAL a candidate label is permitted for internal
        scientific reports, with the word 'candidate' mandatory and status visible."""
        if (self.admissibility in _INTERNAL_LABEL_STATES
                and self.proposed_internal_label is not None):
            return (f"{self.expert_uuid} -- candidate {self.proposed_internal_label} "
                    f"[{self.admissibility.value}]")
        return self.expert_uuid

    def highest_supported_claim(self) -> MechanisticClaimType | None:
        """S0.19. The expert-global summary is the highest admissible claim, never a
        hidden minimum that buries valid evidence behind one unevaluated claim."""
        order = list(MechanisticClaimType)
        admissible = [e.claim_type for e in self.claim_ledger if e.admissible]
        if not admissible:
            return None
        return max(admissible, key=order.index)

    def claims_not_established(self) -> tuple[MechanisticClaimType, ...]:
        established = {e.claim_type for e in self.claim_ledger if e.admissible}
        return tuple(c for c in MechanisticClaimType if c not in established)


def _assert_biological_mediation_is_unreachable(
    claim_type: MechanisticClaimType,
) -> None:
    """Guard for any S0 admissibility pathway: a request to establish biological
    mediation through Panel S0 raises, because model-component interventions establish
    reliance, not biological causation."""
    if claim_type in S0_INADMISSIBLE_CLAIMS:
        raise InsufficientSupportError(
            "Panel S0 cannot establish biological mediation; independent causal "
            "evidence is required and belongs to a future Panel T")
