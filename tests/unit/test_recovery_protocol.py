"""Commit 7: the R4 conditioning ladder inherits the R3b-validated recovery
protocol -- a binding contract, not a fabricated output.

WHAT IS PINNED
==============
  1. The validated protocol is TRAIN-fitted, alignment-sensitive, matched-null,
     synthetically calibrated.
  2. A protocol that fits on a non-TRAIN partition is refused.
  3. A protocol that pairs an alignment-blind statistic with a gain-assignment
     null is refused (reuses the R3b compatibility contract).
  4. R4 inherits the validated protocol WITHOUT being promoted: it stays
     IMPLEMENTED_NO_OUTPUT and records why it has no output.
  5. The R4 ladder OUTPUT_AVAILABLE description cites the inheritance.
  6. A stage cannot inherit an un-calibrated protocol.
  7. A stage without output must record a reason.

Author: written for Monzia Moodie, 2026-07-22.
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.evaluation.alignment_recovery import (
    PartitionRole, NullTarget, WHITENING_ALIGNMENT_SPEC, ANGULAR_DISPERSION_SPEC)
from genomic_variant_classifier.evaluation.null_family import NullKind
from genomic_variant_classifier.evaluation.recovery_protocol import (
    RecoveryProtocol, ProtocolInheritance, VALIDATED_ALIGNMENT_PROTOCOL,
    R4_CONDITIONING_INHERITANCE, ProtocolLeakageError,
    assert_stage_inherits_validated_protocol)
from genomic_variant_classifier.evaluation.capability_lifecycle import (
    panel_r_expected_ladder)


# --------------------------------------------------------------------------- #
# 1. the validated protocol is what R3b established
# --------------------------------------------------------------------------- #
def test_validated_protocol_is_train_fitted_alignment_sensitive_matched_null():
    p = VALIDATED_ALIGNMENT_PROTOCOL
    assert p.fit_partition is PartitionRole.TRAIN
    assert p.statistic is WHITENING_ALIGNMENT_SPEC
    assert NullKind.EIGENVALUE_PERMUTATION in p.null_kinds
    assert p.calibrated_on_synthetic is True


# --------------------------------------------------------------------------- #
# 2-3. leakage-safety of a protocol
# --------------------------------------------------------------------------- #
def test_non_train_protocol_is_refused():
    with pytest.raises(ProtocolLeakageError, match="must fit on"):
        RecoveryProtocol(
            protocol_name="leaky", fit_partition=PartitionRole.TEST,
            statistic=WHITENING_ALIGNMENT_SPEC,
            null_kinds=(NullKind.EIGENVALUE_PERMUTATION,),
            null_target=NullTarget.GAIN_ASSIGNMENT, calibrated_on_synthetic=True)


def test_alignment_blind_statistic_with_gain_null_is_refused():
    with pytest.raises(ProtocolLeakageError, match="cannot inform"):
        RecoveryProtocol(
            protocol_name="blind", fit_partition=PartitionRole.TRAIN,
            statistic=ANGULAR_DISPERSION_SPEC,
            null_kinds=(NullKind.EIGENVALUE_PERMUTATION,),
            null_target=NullTarget.GAIN_ASSIGNMENT, calibrated_on_synthetic=True)


def test_protocol_requires_a_null_family():
    with pytest.raises(ProtocolLeakageError, match="no null family"):
        RecoveryProtocol(
            protocol_name="empty", fit_partition=PartitionRole.TRAIN,
            statistic=WHITENING_ALIGNMENT_SPEC, null_kinds=(),
            null_target=NullTarget.GAIN_ASSIGNMENT, calibrated_on_synthetic=True)


# --------------------------------------------------------------------------- #
# 4. R4 inherits WITHOUT promotion
# --------------------------------------------------------------------------- #
def test_r4_inherits_validated_protocol_without_output():
    inh = R4_CONDITIONING_INHERITANCE
    assert inh.stage_name == "panel_r4_conditioning_recoverability"
    assert inh.inherits is VALIDATED_ALIGNMENT_PROTOCOL
    assert inh.stage_has_output is False
    assert inh.reason_no_output is not None
    assert_stage_inherits_validated_protocol(inh)


def test_r4_stays_implemented_no_output_in_the_ladder():
    """The inheritance must NOT promote R4. Its ladder still runs from
    NOT_IMPLEMENTED through IMPLEMENTED_NO_OUTPUT, OUTPUT_AVAILABLE, VALIDATED --
    R4 has produced nothing; it is only contractually bound to a protocol."""
    ladder = panel_r_expected_ladder()["panel_r4_conditioning_recoverability"]
    states = [rung[0].value for rung in ladder]
    assert "implemented_no_output" in states
    assert "output_available" in states
    assert "validated" in states


def test_r4_ladder_output_rung_cites_protocol_inheritance():
    ladder = panel_r_expected_ladder()["panel_r4_conditioning_recoverability"]
    output_rung = next(r for r in ladder if r[0].value == "output_available")
    # the descriptive element cites that the probe inherits the validated protocol
    assert "INHERITS" in output_rung[2]
    assert "recovery_protocol.py" in output_rung[2]


# --------------------------------------------------------------------------- #
# 6-7. inheritance invariants
# --------------------------------------------------------------------------- #
def test_uncalibrated_protocol_cannot_be_inherited():
    uncalibrated = RecoveryProtocol(
        protocol_name="uncalibrated", fit_partition=PartitionRole.TRAIN,
        statistic=WHITENING_ALIGNMENT_SPEC,
        null_kinds=(NullKind.EIGENVALUE_PERMUTATION,),
        null_target=NullTarget.GAIN_ASSIGNMENT, calibrated_on_synthetic=False)
    with pytest.raises(ProtocolLeakageError, match="un-calibrated"):
        ProtocolInheritance(
            stage_name="panel_r4_conditioning_recoverability",
            inherits=uncalibrated, stage_has_output=False,
            reason_no_output="x")


def test_stage_without_output_must_record_a_reason():
    with pytest.raises(ValueError, match="must record why"):
        ProtocolInheritance(
            stage_name="panel_r4_conditioning_recoverability",
            inherits=VALIDATED_ALIGNMENT_PROTOCOL, stage_has_output=False,
            reason_no_output=None)
