"""Two-dimensional validation evidence for the R3a/R3b sub-capabilities.

WHY THIS EXISTS
===============
CapabilityState answers ONE question: how far has a capability progressed
(implemented -> output -> validated)? It does not answer a SECOND, orthogonal
question: what KIND of evidence supports it -- controlled synthetic conditions,
internal held-out data, or external replication? Collapsing both into one enum
(e.g. inserting METHOD_VALIDATED between OUTPUT_AVAILABLE and VALIDATED) would
make the state carry two concepts at once.

This module keeps the base CapabilityState enum untouched and adds two orthogonal
axes -- MethodValidationState and ScientificValidationState -- composed into an
R3-specific evidence record. The governing distinction:

    the method works under controlled known conditions
        is NOT
    the metric is scientifically admissible on the project's genomic data

R3 splits into two sub-capabilities that this record holds side by side:

  R3a angular dispersion    -- implemented, held-out FAILS, alignment-BLIND under
                               the matched-spectrum null, non-admissible as
                               evidence of covariance alignment. A preserved
                               negative finding, not erased.
  R3b whitening alignment   -- implemented, held-out, TRAIN-fitted, matched-null
                               calibrated on SYNTHETIC fixtures (method validated),
                               scientific validation NOT yet evaluated. Eligible
                               for VALIDATED only after real held-out genomic
                               representations, gene-cluster uncertainty, and
                               prespecified thresholds pass.

Author: written for Monzia Moodie, 2026-07-22.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "MethodValidationState",
    "ScientificValidationState",
    "R3SubCapability",
    "R3CapabilityEvidence",
    "REASON_ALIGNMENT_BLIND_STATISTIC",
    "REASON_GENOMIC_VALIDATION_NOT_YET_RUN",
    "r3a_angular_dispersion_evidence",
    "r3b_whitening_alignment_evidence",
]

REASON_ALIGNMENT_BLIND_STATISTIC = "alignment_blind_statistic"
REASON_GENOMIC_VALIDATION_NOT_YET_RUN = "genomic_validation_not_yet_run"


class MethodValidationState(str, Enum):
    """What kind of METHOD evidence exists (controlled conditions)."""

    NOT_EVALUATED = "not_evaluated"
    FAILED = "failed"
    PASSED_SYNTHETIC = "passed_synthetic"
    PASSED_INTERNAL_EMPIRICAL = "passed_internal_empirical"


class ScientificValidationState(str, Enum):
    """What kind of SCIENTIFIC evidence exists (the project's real data)."""

    NOT_EVALUATED = "not_evaluated"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    FAILED = "failed"
    PASSED_HELDOUT = "passed_heldout"
    PASSED_EXTERNAL = "passed_external"
    PASSED_TEMPORAL = "passed_temporal"


class R3SubCapability(str, Enum):
    ANGULAR_DISPERSION = "R3a.angular_dispersion"
    WHITENING_ALIGNMENT = "R3b.whitening_alignment_recovery"


@dataclass(frozen=True)
class R3CapabilityEvidence:
    """An R3 sub-capability record carrying BOTH validation axes. admissible is a
    conjunction enforced at construction: a claim of admissibility requires method
    validation AND scientific held-out (or stronger) validation; a non-admissible
    record requires at least one finding. This mirrors the base CapabilityEvidence
    invariant while adding the second axis."""

    sub_capability: R3SubCapability
    method_validation: MethodValidationState
    scientific_validation: ScientificValidationState
    output_artifact: str | None
    validation_artifact: str | None
    admissible: bool
    reason: str | None
    findings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.method_validation, MethodValidationState):
            raise TypeError("method_validation must be a MethodValidationState")
        if not isinstance(self.scientific_validation, ScientificValidationState):
            raise TypeError(
                "scientific_validation must be a ScientificValidationState")

        if self.admissible:
            if self.method_validation not in {
                MethodValidationState.PASSED_SYNTHETIC,
                MethodValidationState.PASSED_INTERNAL_EMPIRICAL,
            }:
                raise ValueError(
                    "admissibility requires method validation to have passed")
            if self.scientific_validation not in {
                ScientificValidationState.PASSED_HELDOUT,
                ScientificValidationState.PASSED_EXTERNAL,
                ScientificValidationState.PASSED_TEMPORAL,
            }:
                raise ValueError(
                    "admissibility requires held-out or stronger scientific "
                    "validation -- synthetic method validation is not enough")
            if self.reason is not None:
                raise ValueError(
                    "an admissible capability carries no blocking reason")

        if not self.admissible and not self.findings:
            raise ValueError(
                "a non-admissible capability requires at least one finding")

    def to_dict(self) -> dict[str, object]:
        return {
            "sub_capability": self.sub_capability.value,
            "method_validation": self.method_validation.value,
            "scientific_validation": self.scientific_validation.value,
            "output_artifact": self.output_artifact,
            "validation_artifact": self.validation_artifact,
            "admissible": self.admissible,
            "reason": self.reason,
            "findings": list(self.findings),
        }


def r3a_angular_dispersion_evidence() -> R3CapabilityEvidence:
    """R3a after Commit 5: the angular statistic is implemented and its method is
    understood (synthetic), but held-out transfer FAILED and it is alignment-blind
    under the matched null, so it is non-admissible as alignment evidence. The
    negative finding is preserved, not erased."""
    return R3CapabilityEvidence(
        sub_capability=R3SubCapability.ANGULAR_DISPERSION,
        method_validation=MethodValidationState.PASSED_SYNTHETIC,
        scientific_validation=ScientificValidationState.FAILED,
        output_artifact="r3a_angular_dispersion.json",
        validation_artifact="r3a_negative_validation.json",
        admissible=False,
        reason=REASON_ALIGNMENT_BLIND_STATISTIC,
        findings=(
            "partition_specific_centering_explains_most_apparent_recovery",
            "heldout_transfer_failed_under_train_fitted_center",
            "matched_spectrum_alignment_power_structurally_near_zero",
            "unsuitable_as_evidence_of_covariance_aligned_recoverability",
        ),
    )


def r3b_whitening_alignment_evidence() -> R3CapabilityEvidence:
    """R3b after synthetic calibration: the alignment method is implemented and
    passes matched-null Type-I control and power on synthetic fixtures (method
    validated), but scientific validation on real genomic data has NOT been run,
    so it stays non-admissible. Promotion to admissible requires held-out genomic
    representations, gene-cluster uncertainty, and prespecified thresholds."""
    return R3CapabilityEvidence(
        sub_capability=R3SubCapability.WHITENING_ALIGNMENT,
        method_validation=MethodValidationState.PASSED_SYNTHETIC,
        scientific_validation=ScientificValidationState.NOT_EVALUATED,
        output_artifact="r3b_alignment_recovery_method.json",
        validation_artifact="r3b_synthetic_calibration.json",
        admissible=False,
        reason=REASON_GENOMIC_VALIDATION_NOT_YET_RUN,
        findings=(
            "synthetic_type_i_control_passed",
            "synthetic_power_requirement_passed",
            "heldout_genomic_admissibility_not_established",
        ),
    )
