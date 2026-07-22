"""Alignment-sensitive whitening recovery -- the estimand a matched null can test.

WHY THIS EXISTS
===============
Panel R stage R3 asks whether a TRAIN-fitted transform recovers held-out
structure. An earlier operationalisation used ANGULAR concentration (directional
dispersion). A matched-spectrum null revealed that angular concentration is
ALIGNMENT-BLIND: it responds to the whitening gain-spectrum MAGNITUDE, which the
matched null preserves by construction, so it cannot beat the matched null and
cannot serve as evidence that a transform used the CORRECT covariance alignment
(see r3_validation.py and the compatibility contract below).

This module supplies the estimand the matched null CAN test: covariance-shape
recovery. It measures whether a TRAIN-fitted transform makes the held-out
covariance more identity-like in SHAPE -- which is exactly the property a gain-
to-direction permutation destroys. Two distinct geometric questions were
conflated under the single word "recovery"; this separates them.

SHAPE VERSUS SCALE
------------------
A raw covariance-to-identity Frobenius norm mixes two effects: covariance SHAPE
(anisotropy / alignment) and global variance SCALE. A transform that merely
shrinks all variances would reduce the raw norm without improving alignment. To
isolate alignment, the covariance is TRACE-NORMALISED before the shape distance
is taken, so a global rescaling leaves the shape error exactly unchanged. Scale
is reported separately as a log-average-variance term. This prevents a global
variance contraction from masquerading as alignment recovery.

For a held-out covariance C of dimension d:
    C_bar    = (d / tr(C)) * C                 trace-normalised (unit mean eigenvalue)
    E_shape  = || C_bar - I ||_F / sqrt(d)      alignment error, scale-invariant
    E_scale  = | log(tr(C) / d) |               global-scale departure, reported apart
    delta_shape = E_shape(C_raw) - E_shape(C_transformed)   > 0 == shape recovered

LEAKAGE
-------
Every covariance is centred with the TRAIN mean, never the held-out partition's
own mean; the transformed covariance is centred at zero because TRAIN centring is
already applied. The valid API accepts only a FittedWhiteningTransform proven to
be fitted on TRAIN -- the leaking path (using a held-out mean) is not expressible
through the public signature and raises LeakageError if forced.

Author: written for Monzia Moodie, 2026-07-22.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .norm_angle_probe import LeakageError

FloatMatrix = NDArray[np.float64]

__all__ = [
    "PartitionRole",
    "FittedWhiteningTransform",
    "CovarianceIdentityError",
    "WhiteningAlignmentRecovery",
    "CalibrationSummary",
    "StatisticSensitivity",
    "NullTarget",
    "RecoveryStatisticSpecification",
    "WHITENING_ALIGNMENT_SPEC",
    "ANGULAR_DISPERSION_SPEC",
    "validate_statistic_null_compatibility",
    "empirical_covariance",
    "covariance_identity_error",
    "whitening_alignment_recovery",
    "REASON_HELDOUT_PARTITION_MEAN",
    "REASON_NON_TRAIN_FIT",
]

_EPS = 1e-12

# Stable machine-readable leakage reasons (the contract; messages may evolve).
REASON_HELDOUT_PARTITION_MEAN = "heldout_partition_mean_used"
REASON_NON_TRAIN_FIT = "transform_not_fitted_on_train"


class PartitionRole(str, Enum):
    TRAIN = "TRAIN"
    TUNE = "TUNE"
    STRUCTURE = "STRUCTURE"
    TEST = "TEST"
    STRUCTURE_TEST = "STRUCTURE_TEST"
    UNPARTITIONED = "UNPARTITIONED"


@dataclass(frozen=True)
class FittedWhiteningTransform:
    """A whitening operator plus the provenance that makes leakage-safe use the
    ONLY expressible use. The public recovery API accepts this, not a bare mean +
    operator pair, so a held-out mean cannot be supplied by mistake."""

    train_mean: NDArray[np.float64]     # (1, dim) or (dim,), fitted on TRAIN
    operator: FloatMatrix               # (dim, dim)
    fitted_partition: PartitionRole

    def __post_init__(self) -> None:
        if not isinstance(self.fitted_partition, PartitionRole):
            raise TypeError("fitted_partition must be a PartitionRole")


class StatisticSensitivity(str, Enum):
    ROTATION_INVARIANT = "rotation_invariant"
    SPECTRUM_MAGNITUDE_SENSITIVE = "spectrum_magnitude_sensitive"
    ALIGNMENT_SENSITIVE = "alignment_sensitive"


class NullTarget(str, Enum):
    GAIN_ASSIGNMENT = "gain_assignment"
    OPERATOR_ORIENTATION = "operator_orientation"
    NO_TRANSFORMATION = "no_transformation"


@dataclass(frozen=True)
class RecoveryStatisticSpecification:
    """Types a recovery statistic by what it can SEE, and which null targets it
    may be validly compared against. This encodes the empirical finding -- angular
    dispersion is alignment-blind -- as a STRUCTURAL guarantee: a future caller
    cannot pair a gain-assignment null with an alignment-blind statistic and
    mistake structural-zero-power for a broken implementation."""

    name: str
    sensitivity: StatisticSensitivity
    compatible_null_targets: tuple[NullTarget, ...]
    larger_is_better: bool
    implementation_version: str


WHITENING_ALIGNMENT_SPEC = RecoveryStatisticSpecification(
    name="whitening_alignment_recovery",
    sensitivity=StatisticSensitivity.ALIGNMENT_SENSITIVE,
    compatible_null_targets=(
        NullTarget.GAIN_ASSIGNMENT,
        NullTarget.OPERATOR_ORIENTATION,
    ),
    larger_is_better=True,
    implementation_version="1",
)

ANGULAR_DISPERSION_SPEC = RecoveryStatisticSpecification(
    name="angular_dispersion_delta",
    sensitivity=StatisticSensitivity.SPECTRUM_MAGNITUDE_SENSITIVE,
    compatible_null_targets=(NullTarget.NO_TRANSFORMATION,),
    larger_is_better=True,
    implementation_version="2",
)


def validate_statistic_null_compatibility(
    *,
    statistic: RecoveryStatisticSpecification,
    null_target: NullTarget,
) -> None:
    """Refuse a statistic/null pairing the statistic cannot inform. Pairing a
    gain-assignment null with an alignment-blind statistic would report zero power
    that a reader might mistake for a failed implementation -- so it raises."""
    if null_target not in statistic.compatible_null_targets:
        raise ValueError(
            f"statistic {statistic.name!r} (sensitivity "
            f"{statistic.sensitivity.value!r}) is not compatible with null "
            f"target {null_target.value!r}; compatible targets are "
            f"{[t.value for t in statistic.compatible_null_targets]}")


@dataclass(frozen=True)
class CovarianceIdentityError:
    dimension: int
    shape_error: float          # scale-invariant alignment error
    scale_error: float          # global-scale departure, reported separately
    trace: float
    effective_sample_size: int


@dataclass(frozen=True)
class WhiteningAlignmentRecovery:
    raw: CovarianceIdentityError
    transformed: CovarianceIdentityError
    shape_recovery_delta: float     # > 0 == held-out covariance shape more identity-like
    scale_recovery_delta: float


@dataclass(frozen=True)
class CalibrationSummary:
    """A calibration outcome that CANNOT misreport its own rate: the numerator and
    denominator are stored, and the rate is checked against them at construction.
    This makes the '0.067 from 40 trials' class of arithmetic drift impossible to
    state -- the number cannot diverge from its counts."""

    n_simulations: int
    n_admitted: int
    observed_rate: float

    def __post_init__(self) -> None:
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if not 0 <= self.n_admitted <= self.n_simulations:
            raise ValueError("n_admitted must be between 0 and n_simulations")
        expected = self.n_admitted / self.n_simulations
        if abs(self.observed_rate - expected) > 1e-12:
            raise ValueError(
                f"observed_rate {self.observed_rate} does not match "
                f"n_admitted/n_simulations = {self.n_admitted}/"
                f"{self.n_simulations} = {expected}")


def _validate_embedding_matrix(x: FloatMatrix, *, name: str) -> FloatMatrix:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; received {array.shape}")
    if array.shape[0] < 3:
        raise ValueError(f"{name} requires at least three observations")
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one dimension")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def empirical_covariance(
    x: FloatMatrix, *, center: NDArray[np.float64],
) -> FloatMatrix:
    """Covariance using a FIXED, externally supplied center -- never the array's
    own mean. This is what keeps the held-out estimate from re-centering on
    held-out information."""
    array = _validate_embedding_matrix(x, name="x")
    center_array = np.asarray(center, dtype=np.float64)
    if center_array.shape == (array.shape[1],):
        center_array = center_array.reshape(1, -1)
    if center_array.shape != (1, array.shape[1]):
        raise ValueError("center must have shape [dimension] or [1, dimension]")
    if not np.isfinite(center_array).all():
        raise ValueError("center contains non-finite values")
    centered = array - center_array
    return centered.T @ centered / (centered.shape[0] - 1)


def covariance_identity_error(
    covariance: FloatMatrix, *,
    effective_sample_size: int,
    eigenvalue_floor: float = 1e-12,
) -> CovarianceIdentityError:
    """Trace-normalised covariance-to-identity distance. shape_error is EXACTLY
    invariant to global rescaling of the covariance; scale_error captures the
    global variance departure separately."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2:
        raise ValueError("covariance must be two-dimensional")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance contains non-finite values")
    if effective_sample_size < 3:
        raise ValueError("effective_sample_size must be at least three")
    if eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be positive")

    symmetric = 0.5 * (covariance + covariance.T)
    dimension = symmetric.shape[0]
    trace = float(np.trace(symmetric))
    if trace <= eigenvalue_floor:
        raise ValueError("covariance trace is too small to normalise")

    shape_normalised = (float(dimension) / trace) * symmetric
    identity = np.eye(dimension, dtype=np.float64)
    shape_error = float(
        np.linalg.norm(shape_normalised - identity, ord="fro") / np.sqrt(dimension))

    average_variance = trace / dimension
    scale_error = float(abs(np.log(average_variance)))

    return CovarianceIdentityError(
        dimension=dimension, shape_error=shape_error, scale_error=scale_error,
        trace=trace, effective_sample_size=effective_sample_size)


def whitening_alignment_recovery(
    *,
    x_heldout: FloatMatrix,
    transform: FittedWhiteningTransform,
) -> WhiteningAlignmentRecovery:
    """Evaluate a TRAIN-fitted transform on held-out embeddings. The transform's
    provenance is checked: a non-TRAIN fit raises LeakageError. The held-out
    partition's own mean is never used -- the TRAIN mean centres both the raw and
    transformed covariance."""
    if transform.fitted_partition is not PartitionRole.TRAIN:
        raise LeakageError(
            "whitening alignment recovery requires a TRAIN-fitted transform; "
            f"got a transform fitted on {transform.fitted_partition.value}. "
            "A transform fitted on a held-out partition leaks its geometry.",
            reason=REASON_NON_TRAIN_FIT)

    x = _validate_embedding_matrix(x_heldout, name="x_heldout")
    operator_array = np.asarray(transform.operator, dtype=np.float64)
    if operator_array.shape != (x.shape[1], x.shape[1]):
        raise ValueError("operator shape does not match embedding dimension")
    if not np.isfinite(operator_array).all():
        raise ValueError("operator contains non-finite values")

    train_mean = np.asarray(transform.train_mean, dtype=np.float64).reshape(1, -1)
    if train_mean.shape != (1, x.shape[1]):
        raise ValueError("train_mean shape does not match embedding dimension")

    raw_covariance = empirical_covariance(x, center=train_mean)
    centered = x - train_mean
    transformed = centered @ operator_array
    # Zero-centred: TRAIN centring is already applied, and the transformed
    # partition must NOT be re-centred with held-out information.
    transformed_covariance = empirical_covariance(
        transformed, center=np.zeros((1, x.shape[1]), dtype=np.float64))

    raw_error = covariance_identity_error(
        raw_covariance, effective_sample_size=x.shape[0])
    transformed_error = covariance_identity_error(
        transformed_covariance, effective_sample_size=x.shape[0])

    return WhiteningAlignmentRecovery(
        raw=raw_error,
        transformed=transformed_error,
        shape_recovery_delta=raw_error.shape_error - transformed_error.shape_error,
        scale_recovery_delta=raw_error.scale_error - transformed_error.scale_error)


# --------------------------------------------------------------------------- #
# matched-null calibration for the alignment-sensitive statistic
# --------------------------------------------------------------------------- #
from .norm_angle_probe import fit_whitening, WhiteningTransform          # noqa: E402
from .null_family import (                                               # noqa: E402
    NullKind, eigenvalue_assignment_null, matched_spectrum_orientation_null)
from .representation_artifact import RepresentationArtifact              # noqa: E402


@dataclass(frozen=True)
class AlignmentRecoveryValidation:
    """Held-out, matched-null-calibrated verdict for the alignment statistic."""

    observed_shape_recovery: float
    null_mean: float
    null_p95: float
    p_value: float                 # Monte Carlo matched-null tail probability
    n_null: int
    alpha: float
    beats_matched_null: bool
    null_kind: str


def _transform_from(whitening: WhiteningTransform) -> FittedWhiteningTransform:
    return FittedWhiteningTransform(
        train_mean=np.asarray(whitening.mean, dtype=np.float64).reshape(1, -1),
        operator=whitening.W,
        fitted_partition=PartitionRole.TRAIN)


def validate_alignment_recovery_matched(
    train: RepresentationArtifact,
    test: RepresentationArtifact,
    *,
    null_kind: NullKind = NullKind.EIGENVALUE_PERMUTATION,
    n_null: int = 40,
    alpha: float = 0.05,
    ridge: float = 1e-6,
    seed: int = 0,
) -> AlignmentRecoveryValidation:
    """Fit whitening on TRAIN, measure held-out covariance-SHAPE recovery on TEST,
    and compare against a matched-spectrum null. The statistic is alignment-
    sensitive, so the matched null (which scrambles gain-to-direction alignment
    while preserving the spectrum) is a valid inferential reference for it -- the
    compatibility contract permits GAIN_ASSIGNMENT and OPERATOR_ORIENTATION."""
    if train.partition_role != "TRAIN":
        raise LeakageError(
            f"alignment validation fits on TRAIN; got {train.partition_role!r}",
            reason=REASON_HELDOUT_PARTITION_MEAN)

    whitening = fit_whitening(train, ridge=ridge)
    transform = _transform_from(whitening)
    x_test = np.asarray(test.embeddings, dtype=np.float64)

    observed = whitening_alignment_recovery(
        x_heldout=x_test, transform=transform).shape_recovery_delta

    rng = np.random.default_rng(seed)
    if null_kind is NullKind.EIGENVALUE_PERMUTATION:
        factory = eigenvalue_assignment_null
    elif null_kind is NullKind.MATCHED_SPECTRUM_ORIENTATION:
        factory = matched_spectrum_orientation_null
    else:
        raise ValueError(f"unknown null_kind {null_kind!r}")

    nulls = []
    for _ in range(n_null):
        null_op = factory(whitening, rng).matrix
        null_transform = FittedWhiteningTransform(
            train_mean=transform.train_mean, operator=null_op,
            fitted_partition=PartitionRole.TRAIN)
        nulls.append(whitening_alignment_recovery(
            x_heldout=x_test, transform=null_transform).shape_recovery_delta)
    nulls = np.array(nulls, dtype=np.float64)
    nulls = nulls[np.isfinite(nulls)]

    if nulls.size < max(5, n_null // 2):
        return AlignmentRecoveryValidation(
            observed, float("nan"), float("nan"), float("nan"),
            int(nulls.size), alpha, False, null_kind.value)

    p_value = (1.0 + float((nulls >= observed).sum())) / (1.0 + nulls.size)
    beats = bool(observed > 0.0 and p_value <= alpha)
    return AlignmentRecoveryValidation(
        float(observed), float(nulls.mean()), float(np.percentile(nulls, 95)),
        float(p_value), int(nulls.size), alpha, beats, null_kind.value)
