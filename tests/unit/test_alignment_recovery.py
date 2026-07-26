"""Commit 5: alignment-sensitive whitening recovery, calibrated against the
matched-spectrum null, with the angular statistic preserved as a documented
alignment-blind negative result.

WHAT IS PINNED
==============
  1. shape vs scale -- trace-normalisation makes shape_error exactly scale-
     invariant; scale is reported separately, so a global variance contraction
     cannot masquerade as alignment recovery.
  2. leakage -- the recovery API accepts only a TRAIN-fitted transform; a non-
     TRAIN fit raises LeakageError with a stable reason.
  3. compatibility contract -- an alignment-blind statistic cannot be paired with
     a gain-assignment null; an alignment-sensitive one can.
  4. Type-I control -- under no transferable structure, the matched null admits at
     or below the nominal rate.
  5. power -- under genuine shared-covariance transfer, the alignment statistic
     beats the matched null.
  6. alignment blindness -- the angular statistic cannot beat the matched null
     even where the alignment statistic has power.
  7. CalibrationSummary -- a rate cannot diverge from its numerator/denominator.
  8. R3a/R3b capability records -- two-axis validation; admissibility requires
     held-out scientific validation, never synthetic alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.representation_artifact import (
    extract_focal_embeddings)
from genomic_variant_classifier.evaluation.norm_angle_probe import (
    fit_whitening, angular_concentration, LeakageError)
from genomic_variant_classifier.evaluation.null_family import (
    NullKind, eigenvalue_assignment_null)
from genomic_variant_classifier.evaluation.alignment_recovery import (
    covariance_identity_error, whitening_alignment_recovery,
    validate_alignment_recovery_matched, FittedWhiteningTransform, PartitionRole,
    CalibrationSummary, validate_statistic_null_compatibility,
    WHITENING_ALIGNMENT_SPEC, ANGULAR_DISPERSION_SPEC, NullTarget,
    REASON_NON_TRAIN_FIT)
from genomic_variant_classifier.evaluation.r3_capability import (
    r3a_angular_dispersion_evidence, r3b_whitening_alignment_evidence,
    R3CapabilityEvidence, R3SubCapability, MethodValidationState,
    ScientificValidationState)


class _MockOut:
    def __init__(self, e):
        self.focal_embeddings = e

    @property
    def has_embeddings(self):
        return self.focal_embeddings is not None


def _artifact(emb, role):
    return extract_focal_embeddings(
        _MockOut(emb), [f"v{i}" for i in range(len(emb))],
        representation_name="r", partition_role=role, model_class="M", git_sha="t")


_SHARED_COV = None


def _shared_cov():
    global _SHARED_COV
    if _SHARED_COV is None:
        q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(16, 16)))
        _SHARED_COV = q @ np.diag(np.linspace(1, 10, 16)) @ q.T
    return _SHARED_COV


def _isotropic_fixture(seed):
    rng = np.random.default_rng(seed)
    return (_artifact(rng.normal(size=(500, 16)), "TRAIN"),
            _artifact(rng.normal(size=(500, 16)), "TEST"))


def _transfer_fixture(seed):
    cs = _shared_cov()
    rng = np.random.default_rng(1000 + seed)
    return (_artifact(rng.normal(size=(800, 16)) @ cs, "TRAIN"),
            _artifact(rng.normal(size=(800, 16)) @ cs, "TEST"))


# --------------------------------------------------------------------------- #
# 1. shape vs scale
# --------------------------------------------------------------------------- #
def test_global_rescaling_does_not_fake_shape_alignment_recovery():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(500, 12))
    x[:, :3] *= 3.0
    covariance = np.cov(x, rowvar=False)
    scaled = 25.0 * covariance
    original = covariance_identity_error(covariance, effective_sample_size=500)
    rescaled = covariance_identity_error(scaled, effective_sample_size=500)
    assert rescaled.shape_error == pytest.approx(original.shape_error, rel=1e-10, abs=1e-10)
    assert not np.isclose(rescaled.scale_error, original.scale_error)


def test_covariance_identity_error_rejects_tiny_trace():
    with pytest.raises(ValueError, match="trace is too small"):
        covariance_identity_error(np.zeros((4, 4)), effective_sample_size=100)


# --------------------------------------------------------------------------- #
# 2. leakage
# --------------------------------------------------------------------------- #
def test_alignment_recovery_rejects_non_train_transform():
    x = np.random.default_rng(0).normal(size=(100, 8))
    bad = FittedWhiteningTransform(
        train_mean=np.zeros(8), operator=np.eye(8),
        fitted_partition=PartitionRole.TEST)
    with pytest.raises(LeakageError) as exc:
        whitening_alignment_recovery(x_heldout=x, transform=bad)
    assert exc.value.reason == REASON_NON_TRAIN_FIT


def test_raw_covariance_uses_train_mean_not_heldout_mean():
    """The raw held-out covariance must be centred with the TRAIN mean, never the
    held-out partition's own mean. Construct a fixture where the two means DIFFER
    substantially and confirm the reported raw shape error matches the TRAIN-mean
    computation, not the own-mean one. This locks the leakage-safe centring that
    the delta's relative structure would otherwise hide."""
    rng = np.random.default_rng(5)
    d = 8
    x_heldout = rng.normal(size=(400, d))
    x_heldout[:, 0] += 5.0  # held-out own mean is far from a zero TRAIN mean
    train_mean = np.zeros(d)  # deliberately different from x_heldout.mean(0)
    transform = FittedWhiteningTransform(
        train_mean=train_mean, operator=np.eye(d),
        fitted_partition=PartitionRole.TRAIN)
    result = whitening_alignment_recovery(x_heldout=x_heldout, transform=transform)

    # recompute the raw shape error BOTH ways
    from genomic_variant_classifier.evaluation.alignment_recovery import (
        empirical_covariance, covariance_identity_error)
    cov_train = empirical_covariance(x_heldout, center=train_mean)
    cov_own = empirical_covariance(x_heldout, center=x_heldout.mean(axis=0))
    err_train = covariance_identity_error(cov_train, effective_sample_size=400).shape_error
    err_own = covariance_identity_error(cov_own, effective_sample_size=400).shape_error

    # the means differ enough that the two shape errors differ
    assert not np.isclose(err_train, err_own)
    # and the reported raw uses the TRAIN-mean one
    assert result.raw.shape_error == pytest.approx(err_train)
    assert not np.isclose(result.raw.shape_error, err_own)


def test_alignment_validation_refuses_non_train_partition():
    tr = _artifact(np.random.default_rng(0).normal(size=(200, 16)), "TEST")
    te = _artifact(np.random.default_rng(1).normal(size=(200, 16)), "TEST")
    with pytest.raises(LeakageError):
        validate_alignment_recovery_matched(tr, te, n_null=10)


# --------------------------------------------------------------------------- #
# 3. compatibility contract
# --------------------------------------------------------------------------- #
def test_angular_statistic_incompatible_with_gain_assignment_null():
    with pytest.raises(ValueError, match="not compatible"):
        validate_statistic_null_compatibility(
            statistic=ANGULAR_DISPERSION_SPEC, null_target=NullTarget.GAIN_ASSIGNMENT)


def test_alignment_statistic_compatible_with_matched_nulls():
    # both matched null targets are valid for the alignment-sensitive statistic
    validate_statistic_null_compatibility(
        statistic=WHITENING_ALIGNMENT_SPEC, null_target=NullTarget.GAIN_ASSIGNMENT)
    validate_statistic_null_compatibility(
        statistic=WHITENING_ALIGNMENT_SPEC, null_target=NullTarget.OPERATOR_ORIENTATION)


# --------------------------------------------------------------------------- #
# 4-5. Type-I and power calibration
# --------------------------------------------------------------------------- #
def test_matched_null_controls_false_alignment_recovery_rate():
    n = 40
    admitted = sum(
        validate_alignment_recovery_matched(
            *_isotropic_fixture(s), n_null=40, seed=s).beats_matched_null
        for s in range(n))
    summary = CalibrationSummary(
        n_simulations=n, n_admitted=admitted, observed_rate=admitted / n)
    assert summary.observed_rate <= 0.075


def test_matched_null_detects_transferable_alignment():
    n = 40
    admitted = sum(
        validate_alignment_recovery_matched(
            *_transfer_fixture(s), n_null=40, seed=s).beats_matched_null
        for s in range(n))
    summary = CalibrationSummary(
        n_simulations=n, n_admitted=admitted, observed_rate=admitted / n)
    assert summary.observed_rate >= 0.80


# --------------------------------------------------------------------------- #
# 6. alignment blindness (the scientific crux)
# --------------------------------------------------------------------------- #
def test_angular_statistic_is_alignment_blind_under_matched_spectrum():
    """On a transferable fixture where the alignment statistic has power, the
    angular statistic does NOT beat the matched null -- it responds to spectrum
    magnitude (preserved by the null), not gain-to-direction alignment."""
    cs = _shared_cov()
    rng = np.random.default_rng(17)
    tr = rng.normal(size=(800, 16)) @ cs
    tr[:, 0] += 4.0
    te = rng.normal(size=(800, 16)) @ cs
    te[:, 0] += 4.0
    t = fit_whitening(_artifact(tr, "TRAIN"))
    xc = te - t.mean

    def ang_drop(op):
        return angular_concentration(te).value - angular_concentration(xc @ op).value

    observed = ang_drop(t.W)
    g = np.random.default_rng(0)
    nulls = np.array([ang_drop(eigenvalue_assignment_null(t, g).matrix) for _ in range(40)])
    assert observed <= float(np.quantile(nulls, 0.95))


def test_alignment_recovery_beats_matched_null_on_transfer_fixture():
    """The complement: the alignment-sensitive statistic DOES beat the matched
    null on the same class of fixture -- proving the null has a real power target."""
    rv = validate_alignment_recovery_matched(*_transfer_fixture(23), n_null=60, seed=23)
    assert rv.beats_matched_null
    assert rv.observed_shape_recovery > rv.null_p95
    assert rv.p_value <= 0.05


# --------------------------------------------------------------------------- #
# 7. CalibrationSummary invariant
# --------------------------------------------------------------------------- #
def test_calibration_summary_rejects_rate_mismatched_to_counts():
    with pytest.raises(ValueError, match="does not match"):
        CalibrationSummary(n_simulations=40, n_admitted=2, observed_rate=0.067)


def test_calibration_summary_accepts_consistent_counts():
    s = CalibrationSummary(n_simulations=40, n_admitted=3, observed_rate=3 / 40)
    assert s.observed_rate == 0.075


# --------------------------------------------------------------------------- #
# 8. R3a/R3b capability records
# --------------------------------------------------------------------------- #
def test_r3a_is_non_admissible_and_preserves_negative_finding():
    ev = r3a_angular_dispersion_evidence()
    assert ev.sub_capability is R3SubCapability.ANGULAR_DISPERSION
    assert ev.admissible is False
    assert ev.scientific_validation is ScientificValidationState.FAILED
    assert any("alignment" in f for f in ev.findings)


def test_r3b_is_method_validated_but_not_scientifically_admissible():
    ev = r3b_whitening_alignment_evidence()
    assert ev.method_validation is MethodValidationState.PASSED_SYNTHETIC
    assert ev.scientific_validation is ScientificValidationState.NOT_EVALUATED
    assert ev.admissible is False


def test_admissibility_requires_scientific_not_just_synthetic():
    with pytest.raises(ValueError, match="held-out or stronger scientific"):
        R3CapabilityEvidence(
            sub_capability=R3SubCapability.WHITENING_ALIGNMENT,
            method_validation=MethodValidationState.PASSED_SYNTHETIC,
            scientific_validation=ScientificValidationState.NOT_EVALUATED,
            output_artifact="o", validation_artifact="v",
            admissible=True, reason=None, findings=())


def test_non_admissible_record_requires_a_finding():
    with pytest.raises(ValueError, match="requires at least one finding"):
        R3CapabilityEvidence(
            sub_capability=R3SubCapability.WHITENING_ALIGNMENT,
            method_validation=MethodValidationState.NOT_EVALUATED,
            scientific_validation=ScientificValidationState.NOT_EVALUATED,
            output_artifact=None, validation_artifact=None,
            admissible=False, reason="x", findings=())
