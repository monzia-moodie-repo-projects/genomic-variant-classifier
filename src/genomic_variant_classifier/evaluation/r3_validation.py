"""R3 validation: three independent questions about the recovery metric.

WHY THREE QUESTIONS, NOT ONE
============================
A metric can be right in one sense and wrong in another, and collapsing that into
a single "validated?" bit hides exactly the distinction that matters. This module
keeps three questions separate, following the decomposition Monzia set out:

  1. IMPLEMENTATION VALIDITY -- does the code compute the intended quantity
     correctly? (Covered by test_norm_angle_probe.py; asserted here as a
     precondition, not re-derived.)

  2. STATISTICAL (TRANSFER) VALIDITY -- does the quantity survive on held-out
     data it was not fit on, beyond a label-free null? A transform fit on TRAIN
     that only "recovers" on TRAIN has overfit.

  3. SCIENTIFIC UTILITY / ADMISSIBILITY -- does the quantity measure the intended
     estimand (recoverability of representation structure) rather than an
     artifact of the fitting procedure? A metric can transfer trivially and still
     measure the wrong thing.

Only a metric that passes ALL THREE is admissible, and only an admissible metric
may satisfy the capability release gate. R3, as first operationalised (recovery =
angular-concentration drop after ZCA whitening), passes (1), FAILS (2), and is
therefore not (3). That negative result is recorded, not erased.

THE RECORDED FINDING (2026-07-21)
---------------------------------
On a synthetic cone collapse the in-sample recovery was ~0.98. Held-out, it
vanished (~0.016) and did not exceed a random-orthogonal null (p ~ 1.0). A
decomposition proved WHY: centering alone accounts for 99.6% of the in-sample
drop, and centering with the TRAIN mean does NOT transfer to TEST (gain +0.016)
while centering with TEST's OWN mean fully recovers it (gain +0.982). The
"recovery" was the cone's common direction being removed as a per-partition mean
-- information unavailable at deployment. This is a leakage artifact of the
operationalisation, not evidence about the representation.

WHAT THIS FALSIFIES, AND WHAT IT DOES NOT
-----------------------------------------
Falsified: recovery-as-concentration-drop-after-whitening, with per-partition
centering, transfers. NOT falsified: the broader hypothesis that representations
carry linearly recoverable structure after a LEAKAGE-SAFE conditioning fit on
TRAIN only. The next operationalisation (leakage-safe protocol, split into a
diagnostic in-sample gain R3A and a transferable recovery R3B) gets its own
chance. Deprecation would be premature: one operationalisation is disproved, not
the estimand.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .representation_artifact import RepresentationArtifact
from .norm_angle_probe import (
    fit_whitening, angular_concentration, WhiteningTransform, LeakageError)
from .null_family import (
    NullKind, eigenvalue_assignment_null, matched_spectrum_orientation_null,
    FlatGainSpectrumError)

logger = logging.getLogger(__name__)

__all__ = [
    "ValidationArtifact",
    "RecoveryValidation",
    "RecoveryValidationDecision",
    "random_orthogonal",
    "validate_recovery",
    "validate_recovery_matched",
    "decide_recovery_admissibility",
    "decompose_recovery_source",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# the three-concept record -- a first-class negative-result artifact
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidationArtifact:
    """The independent verdicts on a metric, so a negative result is recorded as
    evidence rather than erased. This is a first-class artifact: a metric that
    fails transfer validation produces one of these, with the finding preserved.

    admissible is True ONLY when implementation AND held-out transfer both pass.
    A metric that computes correctly but does not transfer is implementation-
    valid and NOT admissible -- exactly the state the capability contract needs.
    """

    metric_name: str
    implementation_passed: bool
    held_out_transfer_passed: bool
    admissible: bool
    negative_findings: tuple[str, ...]
    recommendation: str

    def __post_init__(self) -> None:
        # admissibility is not a free variable: it is implementation AND transfer.
        derived = bool(self.implementation_passed and self.held_out_transfer_passed)
        if self.admissible != derived:
            raise ValueError(
                f"admissible={self.admissible} contradicts implementation="
                f"{self.implementation_passed} and transfer="
                f"{self.held_out_transfer_passed}; admissible must be their "
                "conjunction, not an independent claim")
        if not self.admissible and not self.negative_findings:
            raise ValueError(
                "a non-admissible metric must record at least one negative "
                "finding; an unexplained failure is not a usable record")

    def to_manifest(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "implementation_passed": self.implementation_passed,
            "held_out_transfer_passed": self.held_out_transfer_passed,
            "admissible": self.admissible,
            "negative_findings": list(self.negative_findings),
            "recommendation": self.recommendation,
        }


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-uniform random orthogonal matrix via QR of a Gaussian. This is the
    null transform: it reshapes directions without seeing the data's structure,
    so recoveries it produces are the label-free baseline the real conditioning
    must beat."""
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    return q


@dataclass(frozen=True)
class RecoveryValidation:
    """The held-out, null-calibrated numeric verdict on the recovery claim."""

    observed_recovery: float
    null_mean: float
    null_std: float
    null_p95: float
    p_value: float
    n_null: int
    alpha: float
    transfer_passed: bool
    test_partition_role: str

    def to_manifest(self) -> dict:
        return {
            "observed_recovery": self.observed_recovery,
            "null_mean": self.null_mean, "null_std": self.null_std,
            "null_p95": self.null_p95, "p_value": self.p_value,
            "n_null": self.n_null, "alpha": self.alpha,
            "transfer_passed": self.transfer_passed,
            "test_partition_role": self.test_partition_role,
        }


def _concentration_drop(x_raw: np.ndarray, mapped: np.ndarray) -> float:
    raw = angular_concentration(x_raw)
    wht = angular_concentration(mapped)
    if raw.status.value != "ok" or wht.status.value != "ok":
        return float("nan")
    return float(raw.value - wht.value)


def validate_recovery(
    train: RepresentationArtifact,
    test: RepresentationArtifact,
    *,
    n_null: int = 200,
    alpha: float = 0.05,
    ridge: float = 1e-6,
    seed: int = 0,
) -> RecoveryValidation:
    """STATISTICAL VALIDITY (question 2). Fit ZCA whitening on TRAIN, measure its
    recovery on held-out TEST, compare against a random-orthogonal null. The
    transform is fit on TRAIN ONLY (refused otherwise) and applied unchanged to
    TEST -- so the ONLY leakage that can remain is intrinsic to the metric's
    definition, which is exactly what this exposes."""
    if train.partition_role != "TRAIN":
        raise LeakageError(
            f"validation fits on TRAIN; got {train.partition_role!r}")
    train.verify_row_order(train.row_keys)
    test.verify_row_order(test.row_keys)

    transform = fit_whitening(train, ridge=ridge)
    x_test = np.asarray(test.embeddings, dtype=np.float64)
    observed = _concentration_drop(x_test, (x_test - transform.mean) @ transform.W)

    dim = x_test.shape[1]
    rng = np.random.default_rng(seed)
    nulls = np.array([
        _concentration_drop(x_test, (x_test - transform.mean) @ random_orthogonal(dim, rng))
        for _ in range(n_null)
    ])
    nulls = nulls[np.isfinite(nulls)]
    if nulls.size < max(10, n_null // 2):
        return RecoveryValidation(
            observed, float("nan"), float("nan"), float("nan"), float("nan"),
            int(nulls.size), alpha, False, test.partition_role)

    p_value = (1.0 + float((nulls >= observed).sum())) / (1.0 + nulls.size)
    transfer_passed = bool(observed > 0.0 and p_value < alpha)
    return RecoveryValidation(
        observed, float(nulls.mean()), float(nulls.std(ddof=1)),
        float(np.percentile(nulls, 95)), p_value, int(nulls.size), alpha,
        transfer_passed, test.partition_role)


def validate_recovery_matched(
    train: RepresentationArtifact,
    test: RepresentationArtifact,
    *,
    null_kind: NullKind,
    n_null: int = 200,
    alpha: float = 0.05,
    ridge: float = 1e-6,
    seed: int = 0,
) -> RecoveryValidation:
    """STATISTICAL VALIDITY against a MATCHED-SPECTRUM null (the inferential
    reference, unlike the rotation-only control in validate_recovery).

    Identical protocol to validate_recovery -- fit ZCA whitening on TRAIN, apply
    to TEST, measure the held-out concentration drop -- but the null operators are
    drawn from the matched-spectrum family instead of a plain rotation. The
    matched null preserves the whitening gain spectrum exactly (singular values,
    condition number, norms) while randomising the gain-to-direction alignment,
    so beating it means the covariance-aligned assignment of gains, not the sheer
    strength of the transform, produced the recovery.

    null_kind selects the primary (EIGENVALUE_PERMUTATION) or secondary
    (MATCHED_SPECTRUM_ORIENTATION) null. The primary refuses a flat gain spectrum
    (FlatGainSpectrumError propagates); the caller decides whether to fall back to
    the orientation null. The reported p_value is a Monte Carlo matched-null tail
    probability, not an exact permutation p-value.
    """
    if train.partition_role != "TRAIN":
        raise LeakageError(
            f"validation fits on TRAIN; got {train.partition_role!r}")
    train.verify_row_order(train.row_keys)
    test.verify_row_order(test.row_keys)

    transform = fit_whitening(train, ridge=ridge)
    x_test = np.asarray(test.embeddings, dtype=np.float64)
    observed = _concentration_drop(x_test, (x_test - transform.mean) @ transform.W)

    rng = np.random.default_rng(seed)
    if null_kind is NullKind.EIGENVALUE_PERMUTATION:
        factory = eigenvalue_assignment_null
    elif null_kind is NullKind.MATCHED_SPECTRUM_ORIENTATION:
        factory = matched_spectrum_orientation_null
    else:  # defensive: an unknown kind must not silently pick a default
        raise ValueError(f"unknown null_kind {null_kind!r}")

    nulls = np.array([
        _concentration_drop(
            x_test, (x_test - transform.mean) @ factory(transform, rng).matrix)
        for _ in range(n_null)
    ])
    nulls = nulls[np.isfinite(nulls)]
    if nulls.size < max(10, n_null // 2):
        return RecoveryValidation(
            observed, float("nan"), float("nan"), float("nan"), float("nan"),
            int(nulls.size), alpha, False, test.partition_role)

    p_value = (1.0 + float((nulls >= observed).sum())) / (1.0 + nulls.size)
    transfer_passed = bool(observed > 0.0 and p_value < alpha)
    return RecoveryValidation(
        observed, float(nulls.mean()), float(nulls.std(ddof=1)),
        float(np.percentile(nulls, 95)), p_value, int(nulls.size), alpha,
        transfer_passed, test.partition_role)


@dataclass(frozen=True)
class RecoveryValidationDecision:
    """The composite admissibility verdict under the matched null FAMILY.

    A recovery claim is admissible only when it clears an INTERSECTION of
    conditions (Monzia Moodie's design, section 14): the implementation is valid,
    the recovery transfers held-out, it beats BOTH the primary (eigenvalue-
    permutation) and secondary (matched-orientation) nulls, it is not dominated by
    mean centering, and the gene-cluster uncertainty interval excludes no-recovery.
    No single null can compensate for failing the other -- this is a conjunction,
    not an averaged score.

    uncertainty_passed is Optional and defaults to None (PENDING): the gene-cluster
    bootstrap infrastructure does not yet exist, so the honest state is "not yet
    checked", which forces admissible=False with a PENDING finding rather than
    silently passing. It becomes True/False only when a real cluster interval is
    supplied.
    """

    implementation_passed: bool
    transfer_passed: bool
    primary_null_passed: bool
    secondary_null_passed: bool
    centering_not_dominant: bool
    uncertainty_passed: Optional[bool]   # None == PENDING, never default-True
    admissible: bool
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation_passed": self.implementation_passed,
            "transfer_passed": self.transfer_passed,
            "primary_null_passed": self.primary_null_passed,
            "secondary_null_passed": self.secondary_null_passed,
            "centering_not_dominant": self.centering_not_dominant,
            "uncertainty_passed": self.uncertainty_passed,
            "admissible": self.admissible,
            "findings": list(self.findings),
        }


def decide_recovery_admissibility(
    *,
    implementation_passed: bool,
    transfer_delta: float,
    minimum_transfer_delta: float,
    primary_null: RecoveryValidation,
    secondary_null: RecoveryValidation,
    alpha: float,
    centering_fraction: float,
    maximum_centering_fraction: float,
    cluster_ci_low: Optional[float] = None,
) -> RecoveryValidationDecision:
    """Apply the intersection admissibility rule. cluster_ci_low None means the
    gene-cluster uncertainty check is PENDING (infrastructure not yet built), so
    uncertainty_passed is None and admissibility CANNOT be granted -- fail-loud,
    never default-pass. Supply a real lower confidence bound to activate it."""
    transfer_passed = transfer_delta >= minimum_transfer_delta
    primary_null_passed = primary_null.p_value <= alpha and primary_null.observed_recovery > 0.0
    secondary_null_passed = secondary_null.p_value <= alpha and secondary_null.observed_recovery > 0.0
    centering_not_dominant = centering_fraction <= maximum_centering_fraction

    if cluster_ci_low is None:
        uncertainty_passed: Optional[bool] = None   # PENDING
    else:
        uncertainty_passed = cluster_ci_low > 0.0

    admissible = bool(
        implementation_passed
        and transfer_passed
        and primary_null_passed
        and secondary_null_passed
        and centering_not_dominant
        and uncertainty_passed is True   # None (PENDING) and False both block
    )

    findings: list[str] = []
    if not transfer_passed:
        findings.append("recovery_does_not_transfer")
    if not primary_null_passed:
        findings.append("does_not_beat_eigenvalue_assignment_null")
    if not secondary_null_passed:
        findings.append("does_not_beat_matched_orientation_null")
    if not centering_not_dominant:
        findings.append("recovery_dominated_by_centering")
    if uncertainty_passed is None:
        findings.append("gene_cluster_uncertainty_pending_infrastructure")
    elif uncertainty_passed is False:
        findings.append("gene_cluster_interval_includes_no_recovery")
    if not implementation_passed and not findings:
        findings.append("implementation_validation_failed")

    return RecoveryValidationDecision(
        implementation_passed=implementation_passed,
        transfer_passed=transfer_passed,
        primary_null_passed=primary_null_passed,
        secondary_null_passed=secondary_null_passed,
        centering_not_dominant=centering_not_dominant,
        uncertainty_passed=uncertainty_passed,
        admissible=admissible,
        findings=tuple(findings),
    )


def decompose_recovery_source(
    train: RepresentationArtifact, test: RepresentationArtifact,
) -> dict[str, float]:
    """SCIENTIFIC UTILITY (question 3). Attribute the in-sample recovery to its
    sources, so a transfer failure can be EXPLAINED, not just reported.

    Returns the concentration drops for: TRAIN centering-only, TEST centering
    with the TRAIN mean (leakage-safe), and TEST centering with its OWN mean
    (leaky). If the own-mean drop >> the train-mean drop, the recovery is a
    per-partition mean artifact -- the cone's shared direction removed as a mean,
    unavailable at deployment. This is the decomposition that proved R3's failure.
    """
    train.verify_row_order(train.row_keys)
    test.verify_row_order(test.row_keys)
    xtr = np.asarray(train.embeddings, dtype=np.float64)
    xte = np.asarray(test.embeddings, dtype=np.float64)
    mu_tr, mu_te = xtr.mean(0), xte.mean(0)
    return {
        "train_centering_drop": _concentration_drop(xtr, xtr - mu_tr),
        "test_train_mean_drop": _concentration_drop(xte, xte - mu_tr),
        "test_own_mean_drop": _concentration_drop(xte, xte - mu_te),
    }


def build_r3_validation_record(
    train: RepresentationArtifact, test: RepresentationArtifact,
    *, implementation_passed: bool = True, n_null: int = 200,
    alpha: float = 0.05, seed: int = 0,
) -> tuple[ValidationArtifact, RecoveryValidation, dict[str, float]]:
    """Assemble the full three-concept record for R3's recovery metric."""
    rv = validate_recovery(train, test, n_null=n_null, alpha=alpha, seed=seed)
    decomp = decompose_recovery_source(train, test)

    findings: list[str] = []
    if not rv.transfer_passed:
        findings.append(
            f"Held-out recovery {rv.observed_recovery:.3f} did not exceed the "
            f"current reference null on held-out data (Monte Carlo matched-null "
            f"tail probability {rv.p_value:.3f} >= alpha={alpha}).")
    safe = decomp["test_train_mean_drop"]
    leaky = decomp["test_own_mean_drop"]
    if np.isfinite(safe) and np.isfinite(leaky) and leaky > safe + 0.2:
        findings.append(
            f"In-sample recovery is a per-partition mean artifact: centering "
            f"with the TRAIN mean recovers {safe:.3f} on TEST, but centering "
            f"with TEST's OWN mean recovers {leaky:.3f}. The gain is the cone's "
            "common direction removed as a mean, unavailable at deployment.")
    # NULL-ADEQUACY finding (2026-07-21). The current reference null is a Haar
    # random-orthogonal transform. Angular concentration (mean resultant length)
    # is EXACTLY rotation-invariant, so a rotation-only null cannot separate the
    # observed whitening from chance on this statistic: it is retained only as an
    # invariance CONTROL, not an inferential reference. A matched-spectrum null
    # family (eigenvalue-permutation and random-orientation SPD, both preserving
    # the whitening gain spectrum while randomising covariance alignment) is the
    # correct inferential reference and is added in a separate methodological
    # commit. Until then the transfer verdict rests on the held-out sign of the
    # recovery, and this finding records that the null itself needs refinement.
    findings.append(
        "Null adequacy: the rotation-only reference null is rotation-invariant "
        "for angular concentration and is therefore retained only as an "
        "invariance control, not an inferential reference. A matched-spectrum "
        "null family is required and is deferred to a separate commit.")

    admissible = bool(implementation_passed and rv.transfer_passed)
    if admissible:
        rec = "Recovery transfers beyond the null; admissible as a diagnostic."
    else:
        rec = ("Do not promote to VALIDATED. Redefine recovery leakage-safe "
               "(fit centering+whitening on TRAIN only, freeze, apply unchanged) "
               "and split into diagnostic in-sample gain (R3A) and transferable "
               "recovery (R3B); only R3B may satisfy the gate.")

    artifact = ValidationArtifact(
        metric_name="R3.transfer_recovery",
        implementation_passed=implementation_passed,
        held_out_transfer_passed=rv.transfer_passed,
        admissible=admissible,
        negative_findings=tuple(findings),
        recommendation=rec,
    )
    return artifact, rv, decomp
