"""R3: the norm-angle probe decomposes the representation and recovers spread
label-free, and it refuses to fit on a held-out partition.

WHAT IS PINNED
==============
  1. decomposition -- norm and angular channels computed correctly.
  2. recovery      -- ZCA whitening, fit label-free, opens a collapsed cone's
                      angular spread (concentration drops toward 0).
  3. leakage guard -- fit_whitening REFUSES any artifact whose partition_role is
                      not TRAIN; the report applies the TRAIN transform unchanged
                      to held-out partitions and never refits.
  4. row-order     -- verify_row_order is called before any per-row work.
  5. MetricResult contract -- every value is a MetricResult with an honest status.

No torch. RepresentationArtifacts are built from numpy via the real extraction
boundary, so these tests exercise the true artifact type.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.capabilities import MetricStatus
from genomic_variant_classifier.evaluation.representation_artifact import (
    extract_focal_embeddings)
from genomic_variant_classifier.evaluation.norm_angle_probe import (
    fit_whitening, apply_whitening, norm_statistics, angular_concentration,
    norm_angle_report, WhiteningTransform, LeakageError)


class _MockOut:
    def __init__(self, e):
        self.focal_embeddings = e

    @property
    def has_embeddings(self):
        return self.focal_embeddings is not None


def _artifact(emb, role):
    keys = [f"v{i}" for i in range(len(emb))]
    return extract_focal_embeddings(
        _MockOut(emb), keys, representation_name="gnn.focal.pre_classifier",
        partition_role=role, model_class="VariantGAT", git_sha="test")


def _cone_collapse(n=400, d=64, spread=0.03, seed=0):
    rng = np.random.default_rng(seed)
    cone = rng.normal(size=(n, d)) * spread
    cone[:, 0] += 1.0
    cone /= np.linalg.norm(cone, axis=1, keepdims=True)
    scale = np.where(np.arange(n) < n // 2, 1.0, 3.0)
    return cone * scale[:, None]


def _spread(n=400, d=64, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


# --------------------------------------------------------------------------- #
# 1. decomposition
# --------------------------------------------------------------------------- #
def test_norm_statistics_are_metric_results():
    art = _artifact(_spread(), "TRAIN")
    stats = norm_statistics(art)
    for k in ("norm_mean", "norm_sd", "norm_cv"):
        assert stats[k].status is MetricStatus.OK
        assert np.isfinite(stats[k].value)


def test_angular_concentration_is_one_for_identical_directions():
    # all rows the same direction (different norms) -> concentration ~1
    v = np.tile([1.0, 0.0, 0.0], (10, 1)) * np.arange(1, 11)[:, None]
    r = angular_concentration(v)
    assert r.status is MetricStatus.OK
    assert r.value > 0.999


def test_angular_concentration_is_low_for_spread_directions():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(2000, 32))
    r = angular_concentration(v)
    assert r.value < 0.1, "isotropic directions should nearly cancel"


def test_angular_concentration_excludes_zero_norm_rows():
    v = np.vstack([np.eye(3), np.zeros((1, 3))])  # 3 unit + 1 zero
    r = angular_concentration(v)
    assert r.status is MetricStatus.OK  # 3 nonzero rows is enough


def test_angular_concentration_insufficient_when_all_zero():
    r = angular_concentration(np.zeros((5, 4)))
    assert r.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert np.isnan(r.value)


# --------------------------------------------------------------------------- #
# 2. recovery -- the scientific claim
# --------------------------------------------------------------------------- #
def test_whitening_recovers_angular_spread_on_a_cone_collapse():
    art = _artifact(_cone_collapse(), "TRAIN")
    raw = angular_concentration(art.embeddings)
    t = fit_whitening(art)
    whitened = t.transform(np.asarray(art.embeddings))
    wht = angular_concentration(whitened)
    assert raw.value > 0.5, "fixture should be a collapsed cone"
    assert wht.value < raw.value - 0.3, (
        "label-free whitening should open the collapsed cone's angular spread")


def test_whitening_makes_train_covariance_identity():
    art = _artifact(_spread(d=16), "TRAIN")
    t = fit_whitening(art)
    w = t.transform(np.asarray(art.embeddings))
    cov = np.cov(w, rowvar=False)
    np.testing.assert_allclose(cov, np.eye(16), atol=0.05)


# --------------------------------------------------------------------------- #
# 3. the leakage guard -- the hard contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("role", ["TEST", "STRUCTURE", "TUNE", "STRUCTURE_TEST",
                                  "UNPARTITIONED"])
def test_fit_whitening_refuses_non_train(role):
    art = _artifact(_spread(), role)
    with pytest.raises(LeakageError):
        fit_whitening(art)


def test_fit_whitening_accepts_train():
    fit_whitening(_artifact(_spread(), "TRAIN"))  # must not raise


def test_report_requires_train_anchor():
    # Pin the report-level guard's DISTINCT message ("anchor"). fit_whitening has
    # its own guard with a different message ("may be fit on TRAIN only"); pinning
    # "anchor" means removing the report guard -- and falling through to
    # fit_whitening's message -- turns this red, so the report guard is proven to
    # fire independently rather than being redundant decoration.
    with pytest.raises(LeakageError, match="anchor"):
        norm_angle_report(_artifact(_spread(), "TEST"))


def test_report_applies_train_transform_to_others_without_refitting():
    train = _artifact(_cone_collapse(seed=0), "TRAIN")
    test = _artifact(_cone_collapse(seed=2), "TEST")
    rep = norm_angle_report(train, others={"TEST": test})
    assert set(rep) == {"TRAIN", "TEST"}
    # the TEST block must exist and carry recovery evidence computed with the
    # TRAIN transform (we cannot fit on TEST, so this is the only legal path).
    assert rep["TEST"]["angular_recovery_delta"].status is MetricStatus.OK


def test_apply_whitening_uses_the_given_transform_only():
    train = _artifact(_spread(seed=0), "TRAIN")
    test = _artifact(_spread(seed=5), "TEST")
    t = fit_whitening(train)
    out = apply_whitening(t, test)
    # applying the TRAIN transform to TEST is just (test - train_mean) @ W
    expected = (np.asarray(test.embeddings) - t.mean) @ t.W
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_whitening_transform_records_its_fit_partition():
    t = fit_whitening(_artifact(_spread(), "TRAIN"))
    assert t.fit_partition_role == "TRAIN"
    assert t.n_fit_rows == 400


def test_whitening_transform_is_frozen():
    from dataclasses import FrozenInstanceError
    t = fit_whitening(_artifact(_spread(), "TRAIN"))
    with pytest.raises(FrozenInstanceError):
        t.mean = np.zeros(1)


# --------------------------------------------------------------------------- #
# 4. row-order verification precedes per-row work
# --------------------------------------------------------------------------- #
def test_norm_statistics_verifies_row_order():
    art = _artifact(_spread(), "TRAIN")
    # tamper: build an artifact whose recorded hash disagrees with its keys is
    # impossible (constructor checks), so instead assert the call path invokes
    # verify_row_order by monkeypatching it to raise.
    called = {}
    orig = type(art).verify_row_order

    def spy(self, keys):
        called["yes"] = True
        return orig(self, keys)

    type(art).verify_row_order = spy
    try:
        norm_statistics(art)
    finally:
        type(art).verify_row_order = orig
    assert called.get("yes"), "norm_statistics must verify row order first"


# --------------------------------------------------------------------------- #
# 5. transform shape safety
# --------------------------------------------------------------------------- #
def test_transform_rejects_wrong_width():
    # Pin the guard's specific message. Without the guard numpy still raises a
    # ValueError from broadcasting, but with a different message -- so matching
    # "cannot whiten shape" makes removing the guard turn this red rather than
    # passing for the wrong reason.
    t = fit_whitening(_artifact(_spread(d=16), "TRAIN"))
    with pytest.raises(ValueError, match="cannot whiten shape"):
        t.transform(np.zeros((5, 8)))  # 8 != 16


def test_fit_requires_two_rows():
    art = _artifact(_spread(n=1, d=4), "TRAIN")
    with pytest.raises(ValueError):
        fit_whitening(art)


def test_ridge_is_recorded():
    t = fit_whitening(_artifact(_spread(), "TRAIN"), ridge=1e-4)
    assert t.ridge == 1e-4


# --------------------------------------------------------------------------- #
# ORTHOGONAL INVARIANCE OF ANGULAR CONCENTRATION (Commit 2, 2026-07-22)
# --------------------------------------------------------------------------- #
# These tests codify WHY a Haar random-orthogonal null cannot be an inferential
# reference for angular concentration: the statistic is EXACTLY invariant under
# any orthogonal map. The mean resultant length ||mean(v/||v||)|| rotates with
# the data (rotating every v by Q rotates the mean direction by Q) and a norm is
# rotation-invariant, so the value does not change at all.
#
# This is the POSITIVE reframing of what earlier looked like a "failed sabotage"
# (substituting the identity for a rotation null did not move the verdict): it
# did not move the verdict because it could not -- the estimand is rotation-
# invariant. rotation_metric == identity_metric is the CORRECT, EXPECTED result,
# not a gap. The contrast test then shows a rescaling (non-orthogonal) map DOES
# change the concentration, which is exactly why a matched-spectrum rescaling
# null (built in a later commit) is informative where a rotation null is not.


def _haar_orthogonal(dim, rng):
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


def test_angular_concentration_is_exactly_rotation_invariant():
    """The core invariance: an orthogonal map leaves angular concentration
    unchanged to machine precision, across many rotations and several inputs.
    A rotation-only null is therefore an invariance CONTROL, not an inferential
    reference -- this test is why."""
    rng = np.random.default_rng(0)
    for seed in range(4):
        x = np.random.default_rng(seed).normal(size=(800, 24))
        x[:, 0] += 2.0  # give it nontrivial concentration
        base = angular_concentration(x)
        assert base.status is MetricStatus.OK
        for _ in range(5):
            q = _haar_orthogonal(24, rng)
            rotated = angular_concentration(x @ q)
            assert rotated.status is MetricStatus.OK
            assert abs(rotated.value - base.value) < 1e-10, (
                "angular concentration must be invariant under orthogonal maps")


def test_rotation_null_equals_identity_for_this_statistic():
    """The reclassified S7, stated positively: applying a rotation and applying
    nothing (identity) yield the SAME angular concentration. This is the reason a
    rotation null adds nothing over a do-nothing baseline for this metric, and
    the reason it is retained only as an invariance control."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 16))
    x[:, 0] += 3.0
    identity_value = angular_concentration(x @ np.eye(16)).value
    rotation_value = angular_concentration(x @ _haar_orthogonal(16, rng)).value
    assert abs(identity_value - rotation_value) < 1e-10


def test_a_rescaling_map_does_change_angular_concentration():
    """The contrast that motivates a matched-spectrum null: a non-orthogonal
    RESCALING map (stretching axes by different amounts) DOES move angular
    concentration. So a null that preserves the whitening rescaling spectrum
    while randomising its orientation -- unlike a pure rotation -- is a
    genuinely informative reference. This is why Commits 3-4 exist."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(800, 12))
    x[:, 0] += 3.0
    base = angular_concentration(x).value
    # an anisotropic diagonal rescaling: not orthogonal
    scales = np.linspace(0.2, 5.0, 12)
    rescaled = angular_concentration(x * scales).value
    assert abs(rescaled - base) > 1e-3, (
        "a rescaling map must change angular concentration; if it does not, a "
        "matched-spectrum null would be as uninformative as a rotation")
