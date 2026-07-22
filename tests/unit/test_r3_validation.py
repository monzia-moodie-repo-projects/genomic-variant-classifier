"""R3 validation: implementation, transfer, and utility are separate verdicts,
and the validator refuses to promote a metric that fails held-out transfer.

WHAT IS PINNED
==============
  1. the three-concept ValidationArtifact enforces admissible == implementation
     AND transfer, and demands a finding when non-admissible.
  2. validate_recovery is held-out and null-calibrated, fit on TRAIN only.
  3. decompose_recovery_source attributes recovery to centering, exposing the
     per-partition mean artifact.
  4. THE PERMANENT REGRESSION TEST (Monzia, 2026-07-21): a representation with a
     common direction + noise + no transferable signal shows in-sample recovery
     but fails held-out; the validator must NOT promote it. This permanently
     prevents this class of false discovery.

No torch. Artifacts are built from numpy via the real extraction boundary.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.representation_artifact import (
    extract_focal_embeddings)
from genomic_variant_classifier.evaluation.norm_angle_probe import (
    angular_concentration, fit_whitening, LeakageError)
from genomic_variant_classifier.evaluation.r3_validation import (
    ValidationArtifact, RecoveryValidation, validate_recovery,
    decompose_recovery_source, build_r3_validation_record, random_orthogonal)


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


def _cone(n=2000, d=32, seed=0):
    rng = np.random.default_rng(seed)
    cone = rng.normal(size=(n, d)) * 0.03
    cone[:, 0] += 1.0
    cone /= np.linalg.norm(cone, axis=1, keepdims=True)
    return cone * np.where(np.arange(n) < n // 2, 1.0, 3.0)[:, None]


# --------------------------------------------------------------------------- #
# 1. the ValidationArtifact contract
# --------------------------------------------------------------------------- #
def test_admissible_requires_both_implementation_and_transfer():
    a = ValidationArtifact("m", True, True, True, (), "ok")
    assert a.admissible


def test_admissible_cannot_contradict_its_components():
    with pytest.raises(ValueError):
        ValidationArtifact("m", True, False, True, ("x",), "r")  # can't be admissible


def test_non_admissible_requires_a_finding():
    with pytest.raises(ValueError):
        ValidationArtifact("m", True, False, False, (), "r")  # no finding


def test_validation_artifact_is_frozen():
    from dataclasses import FrozenInstanceError
    a = ValidationArtifact("m", True, True, True, (), "ok")
    with pytest.raises(FrozenInstanceError):
        a.admissible = False


# --------------------------------------------------------------------------- #
# 2. held-out, null-calibrated transfer test
# --------------------------------------------------------------------------- #
def test_validate_recovery_refuses_non_train():
    # Pin validate_recovery's OWN message ("validation fits on TRAIN"), distinct
    # from fit_whitening's deeper guard ("may be fit on TRAIN only"). Removing the
    # outer guard would fall through to the inner one with a different message, so
    # matching the outer message makes removing it turn this red -- the outer
    # guard is proven to fire independently rather than being redundant.
    with pytest.raises(LeakageError, match="validation fits on TRAIN"):
        validate_recovery(_artifact(_cone(), "TEST"), _artifact(_cone(seed=1), "TEST"))


def test_validate_recovery_returns_a_verdict():
    rv = validate_recovery(_artifact(_cone(seed=0), "TRAIN"),
                           _artifact(_cone(seed=1), "TEST"), n_null=100)
    assert isinstance(rv, RecoveryValidation)
    assert rv.test_partition_role == "TEST"
    assert 0.0 <= rv.p_value <= 1.0


def test_cone_recovery_does_not_transfer_same_mean():
    """The recorded finding, regime 1: even when TRAIN and TEST share a
    population mean (two independent cones), whitening recovery does not exceed
    the null -- so the failure is not only about the mean, it is that whitening
    amplifies fit-sample noise that does not align out-of-sample."""
    rv = validate_recovery(_artifact(_cone(seed=0), "TRAIN"),
                           _artifact(_cone(seed=1), "TEST"), n_null=200)
    assert not rv.transfer_passed


def test_cone_recovery_does_not_transfer_different_mean():
    """The recorded finding, regime 2: the deployment-realistic different-mean
    case (single split) also fails transfer, and here the mean artifact is
    starkest."""
    tr, te = _cone_split()
    rv = validate_recovery(_artifact(tr, "TRAIN"), _artifact(te, "TEST"), n_null=200)
    assert not rv.transfer_passed


# --------------------------------------------------------------------------- #
# 3. the decomposition that explains the failure
# --------------------------------------------------------------------------- #
def _cone_split(n=4000, d=32, seed=0):
    """One cone draw split into TRAIN/TEST halves. The halves have DIFFERENT
    sample means (the realistic deployment case where TEST != TRAIN), which is
    where the per-partition mean artifact is exposed -- two INDEPENDENT cones
    from the same generator would share a population mean and hide it."""
    full = _cone(n=n, d=d, seed=seed)
    return full[: n // 2], full[n // 2:]


def test_decomposition_exposes_the_mean_artifact():
    tr, te = _cone_split()
    dec = decompose_recovery_source(_artifact(tr, "TRAIN"), _artifact(te, "TEST"))
    # Centering with TEST's OWN mean recovers far more than with the TRAIN mean:
    # the recovery is a per-partition mean the TRAIN fit cannot supply. This is
    # the decomposition that proved R3's transfer failure is a mean artifact.
    assert dec["test_own_mean_drop"] > dec["test_train_mean_drop"] + 0.5


# --------------------------------------------------------------------------- #
# 4. THE PERMANENT REGRESSION TEST
# --------------------------------------------------------------------------- #
def test_regression_common_direction_plus_noise_is_not_promoted():
    """PERMANENT (Monzia, 2026-07-21). A representation that is a common
    direction + independent noise + no transferable signal shows in-sample
    recovery but must FAIL held-out validation. If a future change to the
    metric or validator lets this be promoted, this test turns red -- the
    tripwire against this exact class of false discovery."""
    rng = np.random.default_rng(7)
    d, n = 48, 3000
    # common direction shared by all rows + independent per-row noise; the noise
    # carries NO structure that transfers between partitions.
    common = np.zeros(d); common[0] = 1.0
    def make(seed):
        r = np.random.default_rng(seed)
        return common + r.normal(scale=0.05, size=(n, d))
    train = _artifact(make(1), "TRAIN")
    test = _artifact(make(2), "TEST")

    # in-sample recovery APPEARS (centering removes the common direction)
    t = fit_whitening(train)
    xtr = np.asarray(train.embeddings)
    in_sample = (angular_concentration(xtr).value
                 - angular_concentration((xtr - t.mean) @ t.W).value)
    assert in_sample > 0.3, "fixture must show apparent in-sample recovery"

    # held-out validation must REFUSE it
    art, rv, dec = build_r3_validation_record(train, test, n_null=200)
    assert not rv.transfer_passed, "must fail held-out transfer"
    assert not art.admissible, "must not be admissible"
    assert art.negative_findings, "must record why it failed"


# --------------------------------------------------------------------------- #
# 5. the full record
# --------------------------------------------------------------------------- #
def test_full_record_on_cone_is_non_admissible_with_findings():
    art, rv, dec = build_r3_validation_record(
        _artifact(_cone(seed=0), "TRAIN"), _artifact(_cone(seed=1), "TEST"),
        n_null=200)
    assert art.metric_name == "R3.transfer_recovery"
    assert art.implementation_passed
    assert not art.held_out_transfer_passed
    assert not art.admissible
    assert len(art.negative_findings) >= 1
    assert "VALIDATED" in art.recommendation  # tells the reader not to promote


def test_record_manifest_serialises():
    art, rv, dec = build_r3_validation_record(
        _artifact(_cone(seed=0), "TRAIN"), _artifact(_cone(seed=1), "TEST"),
        n_null=50)
    m = art.to_manifest()
    assert m["admissible"] is False
    assert isinstance(m["negative_findings"], list)


# --------------------------------------------------------------------------- #
# 6. the null transform
# --------------------------------------------------------------------------- #
def test_random_orthogonal_is_orthogonal():
    rng = np.random.default_rng(0)
    q = random_orthogonal(10, rng)
    np.testing.assert_allclose(q @ q.T, np.eye(10), atol=1e-10)


def test_a_genuinely_transferable_signal_can_pass():
    """POSITIVE CONTROL. A genuinely transferable anisotropy -- one dominant axis,
    shared across partitions, rotated into general position -- produces recovery
    that DOES exceed the random-orthogonal null on held-out data, so
    transfer_passed is True. This is the counterpart to the negative finding: it
    proves the gate is not stuck at False, AND it makes the null real. If the null
    were replaced by the identity (a broken, too-strong null equal to pure
    centering), this genuine recovery would no longer beat it and the test would
    turn red -- so this positive control is what falsifies a broken null."""
    d = 12
    U = np.linalg.qr(np.random.default_rng(1).normal(size=(d, d)))[0]
    def make(seed):
        r = np.random.default_rng(seed)
        z = r.normal(size=(3000, d))
        z[:, 0] *= 6.0                 # one dominant axis, transferable
        return z @ U.T                 # rotated into general position
    train = _artifact(make(3), "TRAIN")   # seeds chosen so the control is stable
    test = _artifact(make(4), "TEST")
    rv = validate_recovery(train, test, n_null=200, seed=0)
    assert rv.transfer_passed, (
        "a genuinely transferable recovery must pass; if this fails, either the "
        "null is broken (e.g. identity instead of random-orthogonal) or the "
        "transfer criterion is mis-wired")
    assert rv.observed_recovery > rv.null_mean
