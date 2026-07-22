"""Panel R stage R3: norm-angle decomposition and a label-free recoverability probe.

THE SCIENTIFIC QUESTION
=======================
Panel R stage one (R1/R2) MEASURED representation collapse: a cone-collapsed
128-dimensional representation, where every vector points in nearly the same
direction and the class is encoded only in the norm, was rated BETTER by three
Panel Q clustering metrics while a spherical-cosine Davies-Bouldin fired 142x.
The metrics disagreed; nothing DIAGNOSED.

R3 asks the next question: when the geometry deteriorates like that, can a
LABEL-FREE linear map recover usable structure? If a collapsed cone can be
whitened -- using only an unsupervised mean and covariance, no labels -- into a
representation whose directions spread back out, that is evidence the collapse
was a reparameterisation artifact a linear map can undo, not a destruction of
information. If it cannot, the collapse is real.

THE DECOMPOSITION
-----------------
Each embedding row v splits into two channels:
    norm  = ||v||_2         a scalar per row -- radial magnitude
    angle = v / ||v||_2      a unit vector per row -- direction on the sphere
A cone collapse concentrates the angles (they become nearly identical) while the
norms may still separate structure. Measuring the two channels separately is the
point: the angle channel can collapse while the norm channel does not.

THE LABEL-FREE LINEAR MAP
-------------------------
ZCA whitening: fit a mean mu and a covariance Sigma, transform
x -> W (x - mu) with W = Sigma^{-1/2}, so the fitted covariance becomes identity.
This is fully UNSUPERVISED -- no labels enter the fit. Angular concentration
(mean resultant length) measured before and after whitening quantifies how much
directional spread a linear map recovers.

THE LEAKAGE GUARD -- THE HARD CONTRACT
--------------------------------------
The whitening transform MUST be fit on the TRAIN partition only. Fitting mu and
Sigma on STRUCTURE or TEST leaks the evaluation distribution into the transform
EVEN WITH NO LABELS -- the second-moment structure of the held-out set is itself
information that must not touch the map. RepresentationArtifact carries a bound
partition_role; fit_whitening REFUSES any artifact whose role is not TRAIN. The
refusal is structural (an exception), not a comment. The fitted transform is then
applied UNCHANGED to TUNE/STRUCTURE/TEST via apply_whitening, which never refits.

This module runs, produces MetricResults, and returns a transform. It does not
persist anything and it does not decide admissibility: that keeps R3 at
OUTPUT_AVAILABLE, one rung above the extraction boundary, not at VALIDATED.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .capabilities import MetricStatus
from .clustering_metrics import MetricResult
from .representation_artifact import RepresentationArtifact

logger = logging.getLogger(__name__)

__all__ = [
    "WhiteningTransform",
    "fit_whitening",
    "apply_whitening",
    "norm_statistics",
    "angular_concentration",
    "norm_angle_report",
    "LeakageError",
]

_EPS = 1e-12


class LeakageError(ValueError):
    """Raised when a fit is attempted on a partition other than TRAIN. Fitting a
    transform on STRUCTURE or TEST leaks the evaluation distribution into the map
    even without labels, so the boundary refuses rather than warns."""


@dataclass(frozen=True)
class WhiteningTransform:
    """A ZCA whitening map fitted on TRAIN, applicable unchanged elsewhere.

    x -> (x - mean) @ W. Frozen so a fitted transform cannot be mutated after the
    fact and then believed to still describe the TRAIN partition it came from.
    """

    mean: np.ndarray             # (dim,)
    W: np.ndarray                # (dim, dim), symmetric Sigma^{-1/2}
    fit_partition_role: str      # always "TRAIN"; recorded for provenance
    n_fit_rows: int
    ridge: float                 # the ridge added to the covariance diagonal

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"cannot whiten shape {x.shape} with a transform fit on "
                f"dim {self.mean.shape[0]}")
        return (x - self.mean) @ self.W


def fit_whitening(artifact: RepresentationArtifact, *, ridge: float = 1e-6
                  ) -> WhiteningTransform:
    """Fit ZCA whitening on a TRAIN artifact. Refuses any other partition role.

    ridge is added to the covariance diagonal before inversion so a rank-deficient
    or near-singular covariance (a cone collapse produces exactly this) does not
    blow up Sigma^{-1/2}. The ridge is recorded on the transform.
    """
    if artifact.partition_role != "TRAIN":
        raise LeakageError(
            f"whitening may be fit on TRAIN only; this artifact is "
            f"{artifact.partition_role!r}. Fitting on a held-out partition leaks "
            "its distribution into the transform even with no labels.")
    x = np.asarray(artifact.embeddings, dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("need at least 2 rows to estimate a covariance")

    mean = x.mean(axis=0)
    centered = x - mean
    # sample covariance (n-1); symmetric by construction
    cov = (centered.T @ centered) / (x.shape[0] - 1)
    cov = cov + ridge * np.eye(cov.shape[0])
    # ZCA: W = Sigma^{-1/2} via symmetric eigendecomposition (cov is symmetric PSD)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, _EPS, None)          # guard tiny/negative from roundoff
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return WhiteningTransform(
        mean=mean, W=W, fit_partition_role="TRAIN",
        n_fit_rows=int(x.shape[0]), ridge=float(ridge))


def apply_whitening(transform: WhiteningTransform,
                    artifact: RepresentationArtifact) -> np.ndarray:
    """Apply a TRAIN-fitted transform to any artifact, refitting NOTHING. This is
    the only sanctioned way STRUCTURE/TEST embeddings are whitened -- with the
    TRAIN map, unchanged."""
    return transform.transform(np.asarray(artifact.embeddings, dtype=np.float64))


def _row_keys_or_raise(artifact: RepresentationArtifact) -> None:
    """Verify the artifact's own keys hash to its recorded order before any
    per-row work. A no-op on an honest artifact; a tripwire on a tampered one."""
    artifact.verify_row_order(artifact.row_keys)


def norm_statistics(artifact: RepresentationArtifact) -> dict[str, MetricResult]:
    """The radial channel: distribution of ||v|| across rows. Label-free."""
    _row_keys_or_raise(artifact)
    x = np.asarray(artifact.embeddings, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    n = norms.shape[0]
    if n < 2:
        bad = f"norm statistics need >=2 rows, got {n}"
        nan = MetricResult(float("nan"), MetricStatus.INSUFFICIENT_SUPPORT, bad)
        return {"norm_mean": nan, "norm_sd": nan, "norm_cv": nan}
    mean = float(norms.mean())
    sd = float(norms.std(ddof=1))
    out = {
        "norm_mean": MetricResult(mean, MetricStatus.OK),
        "norm_sd": MetricResult(sd, MetricStatus.OK),
    }
    out["norm_cv"] = (
        MetricResult(sd / mean, MetricStatus.OK) if mean > _EPS
        else MetricResult(float("nan"), MetricStatus.UNDEFINED,
                          "norm coefficient of variation undefined at zero mean"))
    return out


def angular_concentration(x: np.ndarray) -> MetricResult:
    """Mean resultant length of the row directions: ||mean(v/||v||)||.

    1.0 means every direction is identical (a fully collapsed cone); 0.0 means the
    directions cancel (spread over the sphere). This is the angle channel's
    collapse measure, and the before/after-whitening pair is R3's core evidence.
    Rows with zero norm have no direction and are excluded, reported via status.
    """
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    keep = norms > _EPS
    n_zero = int((~keep).sum())
    if keep.sum() < 2:
        return MetricResult(
            float("nan"), MetricStatus.INSUFFICIENT_SUPPORT,
            f"need >=2 nonzero-norm rows, got {int(keep.sum())} "
            f"({n_zero} zero-norm rows excluded)")
    directions = x[keep] / norms[keep][:, None]
    resultant = np.linalg.norm(directions.mean(axis=0))
    status = MetricStatus.OK
    reason = None
    return MetricResult(float(resultant), status, reason)


def norm_angle_report(
    train: RepresentationArtifact,
    *,
    others: Optional[dict[str, RepresentationArtifact]] = None,
    ridge: float = 1e-6,
) -> dict[str, dict[str, MetricResult]]:
    """R3's output: fit whitening on TRAIN, measure norm + angular concentration
    before and after, on TRAIN and on each provided held-out partition.

    `others` maps a partition label (e.g. "STRUCTURE", "TEST") to its artifact.
    Each is whitened with the TRAIN transform, never its own -- so the report
    cannot accidentally leak. Returns a nested dict: partition -> metric -> result.
    """
    if train.partition_role != "TRAIN":
        raise LeakageError(
            f"norm_angle_report requires the TRAIN artifact as its anchor; got "
            f"{train.partition_role!r}")

    transform = fit_whitening(train, ridge=ridge)

    def _one(art: RepresentationArtifact) -> dict[str, MetricResult]:
        _row_keys_or_raise(art)
        raw = np.asarray(art.embeddings, dtype=np.float64)
        whitened = transform.transform(raw)
        block = dict(norm_statistics(art))
        block["angular_concentration_raw"] = angular_concentration(raw)
        block["angular_concentration_whitened"] = angular_concentration(whitened)
        # recovery: how much the direction spread OPENED after whitening.
        # raw high (collapsed) -> whitened lower (spread) is recovery; report the
        # drop as a signed delta so a consumer sees the direction and size.
        r_raw = block["angular_concentration_raw"]
        r_wht = block["angular_concentration_whitened"]
        if r_raw.status is MetricStatus.OK and r_wht.status is MetricStatus.OK:
            block["angular_recovery_delta"] = MetricResult(
                float(r_raw.value - r_wht.value), MetricStatus.OK)
        else:
            block["angular_recovery_delta"] = MetricResult(
                float("nan"), MetricStatus.INSUFFICIENT_SUPPORT,
                "recovery delta needs both raw and whitened concentration")
        return block

    report = {"TRAIN": _one(train)}
    for label, art in (others or {}).items():
        report[label] = _one(art)
    return report
