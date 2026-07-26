"""The validated recovery protocol, and the contract by which later Panel R
conditioning stages inherit it.

WHY THIS EXISTS
===============
Panel R stage R3b established a leakage-safe, matched-null-calibrated protocol for
measuring whether a TRAIN-fitted transform recovers held-out covariance structure:
fit the whitening transform on TRAIN only, never use the held-out partition's own
mean, measure the trace-normalised (alignment-SENSITIVE) covariance-shape recovery
rather than the alignment-BLIND angular statistic, and compare against a matched-
spectrum null whose statistic/null pairing is checked for compatibility. That
protocol was calibrated (Type-I controlled, power established) on synthetic
fixtures before any real representation was admitted.

Stage R4 (conditioning recoverability) asks a closely related question with the
SAME leakage hazards. R4 has no real output yet -- no genomic representation
exists to run it on -- so this module does NOT run R4 or promote it. What it does
is bind R4, by contract, to REUSE the R3b-validated protocol when it is eventually
implemented, rather than re-derive a fresh (and possibly leaky, or alignment-
blind) protocol of its own. The contract is a typed record with an enforced
invariant; a stage that tries to inherit a non-TRAIN fit or the alignment-blind
statistic is refused at construction.

This is the honest completion of the R3 matched-null sequence: the validated
protocol is HANDED to the next ladder rung as a binding contract, not fabricated
into output R4 does not have.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .alignment_recovery import (
    RecoveryStatisticSpecification, StatisticSensitivity, NullTarget,
    WHITENING_ALIGNMENT_SPEC, PartitionRole)
from .null_family import NullKind

__all__ = [
    "RecoveryProtocol",
    "ProtocolInheritance",
    "VALIDATED_ALIGNMENT_PROTOCOL",
    "R4_CONDITIONING_INHERITANCE",
    "ProtocolLeakageError",
    "assert_stage_inherits_validated_protocol",
]


class ProtocolLeakageError(ValueError):
    """Raised when a stage attempts to inherit a recovery protocol that is not
    leakage-safe -- a fit partition other than TRAIN, or an alignment-blind
    statistic paired with a null that cannot test it."""


@dataclass(frozen=True)
class RecoveryProtocol:
    """A named, leakage-safe recovery protocol. The invariant refuses a protocol
    that fits on a non-TRAIN partition or pairs a statistic with a null target the
    statistic's own compatibility list does not permit. That second check reuses
    the R3b compatibility contract, so an alignment-blind statistic cannot be
    smuggled in against a gain-assignment null."""

    protocol_name: str
    fit_partition: PartitionRole
    statistic: RecoveryStatisticSpecification
    null_kinds: tuple[NullKind, ...]
    null_target: NullTarget
    calibrated_on_synthetic: bool

    def __post_init__(self) -> None:
        if self.fit_partition is not PartitionRole.TRAIN:
            raise ProtocolLeakageError(
                f"protocol {self.protocol_name!r} fits on "
                f"{self.fit_partition.value}; a recovery protocol must fit on "
                "TRAIN only -- fitting on a held-out partition leaks its geometry")
        if self.null_target not in self.statistic.compatible_null_targets:
            raise ProtocolLeakageError(
                f"protocol {self.protocol_name!r} pairs statistic "
                f"{self.statistic.name!r} (sensitivity "
                f"{self.statistic.sensitivity.value!r}) with null target "
                f"{self.null_target.value!r}, which the statistic cannot inform; "
                f"compatible targets are "
                f"{[t.value for t in self.statistic.compatible_null_targets]}")
        if not self.null_kinds:
            raise ProtocolLeakageError(
                f"protocol {self.protocol_name!r} declares no null family")


# The protocol R3b validated: TRAIN-fit, alignment-sensitive shape recovery,
# matched-spectrum null family, calibrated on synthetic fixtures.
VALIDATED_ALIGNMENT_PROTOCOL = RecoveryProtocol(
    protocol_name="r3b_whitening_alignment_recovery",
    fit_partition=PartitionRole.TRAIN,
    statistic=WHITENING_ALIGNMENT_SPEC,
    null_kinds=(NullKind.EIGENVALUE_PERMUTATION,
                NullKind.MATCHED_SPECTRUM_ORIENTATION),
    null_target=NullTarget.GAIN_ASSIGNMENT,
    calibrated_on_synthetic=True,
)


@dataclass(frozen=True)
class ProtocolInheritance:
    """A record that a later Panel R stage inherits a validated protocol. It does
    NOT assert the stage has produced output -- inheritance binds the PROTOCOL the
    stage must use when it runs, leaving the stage's capability state untouched.
    The invariant refuses inheritance of a protocol not calibrated on synthetic
    fixtures (the R3 discipline: the null must earn its operating characteristics
    before it touches real data) and records why the stage has no output yet."""

    stage_name: str
    inherits: RecoveryProtocol
    stage_has_output: bool
    reason_no_output: str | None
    findings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.inherits.calibrated_on_synthetic:
            raise ProtocolLeakageError(
                f"stage {self.stage_name!r} may not inherit an un-calibrated "
                "protocol; the null's Type-I and power must be established on "
                "synthetic fixtures before the protocol is handed on")
        if not self.stage_has_output and not self.reason_no_output:
            raise ValueError(
                f"stage {self.stage_name!r} has no output and must record why "
                "(a stage without output and without a reason is indistinguishable "
                "from an unreported failure)")


# R4 conditioning recoverability inherits the R3b-validated protocol. R4 has no
# real output (no genomic representation exists to run it on), so this binds the
# protocol without promoting the stage.
R4_CONDITIONING_INHERITANCE = ProtocolInheritance(
    stage_name="panel_r4_conditioning_recoverability",
    inherits=VALIDATED_ALIGNMENT_PROTOCOL,
    stage_has_output=False,
    reason_no_output=(
        "no genomic representation is exposed to run the conditioning ladder on; "
        "R4 stays IMPLEMENTED_NO_OUTPUT and inherits the R3b-validated matched-"
        "null protocol for when a representation becomes available"),
    findings=(
        "inherits_train_fitted_leakage_safe_whitening",
        "inherits_alignment_sensitive_shape_recovery_statistic",
        "inherits_matched_spectrum_null_family",
        "no_output_pending_real_representation",
    ),
)


def assert_stage_inherits_validated_protocol(
    inheritance: ProtocolInheritance,
) -> None:
    """Guard for use at the point R4 (or any later stage) is wired to run: confirm
    the stage inherits a TRAIN-fitted, compatibility-checked, synthetically-
    calibrated protocol. Raises ProtocolLeakageError otherwise. Construction
    already enforces these, so this is the explicit gate a caller invokes before
    trusting the inheritance."""
    protocol = inheritance.inherits
    if protocol.fit_partition is not PartitionRole.TRAIN:
        raise ProtocolLeakageError(
            f"{inheritance.stage_name}: inherited protocol is not TRAIN-fitted")
    if protocol.null_target not in protocol.statistic.compatible_null_targets:
        raise ProtocolLeakageError(
            f"{inheritance.stage_name}: inherited statistic/null pairing is "
            "incompatible")
    if not protocol.calibrated_on_synthetic:
        raise ProtocolLeakageError(
            f"{inheritance.stage_name}: inherited protocol is not calibrated")
