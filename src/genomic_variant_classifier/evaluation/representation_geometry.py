"""Panel R -- representation geometry, collapse and conditioning (stage one).

WHY THIS PANEL EXISTS
=====================
Loss curves and cluster metrics can look healthy while the representation
underneath degenerates. The specific pathology: embeddings collapse into a
narrow cone, angular separation is lost, and the task signal migrates into the
VECTOR NORM. Downstream evaluation keeps improving because the information is
still accessible -- just through the wrong channel, in a poorly conditioned
space.

MEASURED AGAINST THIS REPOSITORY, 2026-07-21
---------------------------------------------
Two 128-dimensional two-class representations, n = 600. The first separates its
classes ANGULARLY and is near-isotropic. The second forces every vector into a
narrow cone and encodes the class only in the norm.

    metric                              healthy   collapsed   reads as
    Davies-Bouldin euclidean centroid     3.934      1.131     3.5x BETTER
    Calinski-Harabasz                    38.49     371.6       9.7x BETTER
    silhouette (euclidean, sampled)       0.057      0.404     7.1x BETTER
    Davies-Bouldin spherical cosine       1.324    187.98    142x WORSE
    ---
    mean resultant length                 0.238      0.870
    effective rank (of 128)             105.9       34.6

THREE Panel Q metrics rate the DEGENERATE representation as substantially
better, because radial separation is exactly what they reward.

AN IMPORTANT CORRECTION TO THE ORIGINAL SPECIFICATION. It listed Davies-Bouldin
among the metrics a cone collapse can pass. That holds for the Euclidean variant
and is FALSE for the spherical one, which already exists in this codebase at
clustering_metrics.davies_bouldin_spherical_cosine and moved 142-fold in the
wrong direction. The honest justification for Panel R is therefore NOT that
nothing detects the pathology. It is that ONE METRIC ALARMS AND NOTHING
DIAGNOSES: a spherical Davies-Bouldin of 188 says something is wrong, and says
nothing about whether the information was destroyed or merely badly arranged,
nor whether a label-free linear map recovers it.

    Panel Q asks whether the space forms clusters.
    Panel R asks whether the space is still fit for thought.

WHAT IS IN STAGE ONE, AND WHAT IS DELIBERATELY NOT
---------------------------------------------------
R1 (directional anisotropy) and R2 (rank and spectral utilisation) are pure
functions of a matrix. They need no probes, no partitions and no training loop,
so they are built here and are fail-closed.

R3 (norm-angle probe decomposition), R4 (conditioning and recoverability),
R5 (hubness and local geometry), R6 (training-time trajectories) and
R7 (downstream sensitivity) all require a representation this project does not
yet export. `models/gnn.py:357` computes `focal_embeddings` and returns only
`self.classifier(focal_embeddings)`; the embedding is discarded. Building probe
panels over an absent representation would produce exactly the vacuous pass the
capability contract exists to prevent, so those stages are registered through
`capabilities.CapabilityEvidence` with honest states instead of stubbed. See
`panel_r_capabilities()` below.

A NOTE ON PASSES
----------------
Everything except centred directional statistics is available from a SINGLE
streaming pass: n, the sum of x, the sum of outer products, the sum of unit
vectors, and the sum of norms and squared norms. Centred anisotropy genuinely
needs a second pass, because each unit vector must be recomputed as
(x - mu)/||x - mu||, and mu is not known until the first pass finishes. That is
stated rather than hidden, because a reader who assumes one pass will
mis-budget a 1.5-million-row cohort.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .capabilities import (
    REASON_NO_OUTPUT_ARTIFACT,
    CapabilityEvidence,
    CapabilityState,
    MetricStatus,
    TargetState,
)
from .clustering_metrics import MetricResult

logger = logging.getLogger(__name__)

__all__ = [
    "GeometrySummary",
    "REASON_EMPTY_COHORT",
    "REASON_NONFINITE_VALUES",
    "REASON_TOO_FEW_OBSERVATIONS",
    "REASON_ALL_ZERO_NORM",
    "REASON_DEGENERATE_SPECTRUM",
    "COLLAPSE_HEALTHY",
    "COLLAPSE_COMMON_DIRECTION",
    "COLLAPSE_NORM_COMPENSATED",
    "COLLAPSE_DESTRUCTIVE",
    "COLLAPSE_EXPLOSION",
    "COLLAPSE_UNDETERMINED",
    "COLLAPSE_CONE_UNRESOLVED",
    "CONE_MEAN_RESULTANT_LENGTH_THRESHOLD",
    "THIN_NORMALIZED_EFFECTIVE_RANK_THRESHOLD",
    "COMMON_DIRECTION_CENTERED_THRESHOLD",
    "mean_resultant_length",
    "norm_statistics",
    "spectral_geometry",
    "summarize_geometry",
    "classify_collapse",
    "panel_r_capabilities",
]

REASON_EMPTY_COHORT = "empty_cohort"
REASON_NONFINITE_VALUES = "nonfinite_values_present"
REASON_TOO_FEW_OBSERVATIONS = "too_few_observations"
REASON_ALL_ZERO_NORM = "all_observations_have_zero_norm"
REASON_DEGENERATE_SPECTRUM = "degenerate_spectrum"

# A representation needs more rows than dimensions before its covariance
# spectrum means anything. Below that the sample covariance is rank-deficient by
# construction and "effective rank 40" would describe the SAMPLE, not the model.
MINIMUM_ROWS_PER_DIMENSION: float = 2.0
MINIMUM_ROWS: int = 8

# ---------------------------------------------------------------------------
# COLLAPSE THRESHOLDS -- PLACEHOLDERS, NOT MEASURED ON THIS PROJECT'S DATA.
#
# These are round numbers chosen to be defensible in the absence of evidence,
# NOT values derived from any representation this project has produced. They are
# named, module-level and overridable precisely so that they can be found and
# replaced, rather than buried as literals inside a comparison.
#
# The right way to set them, once models/gnn.py exports focal_embeddings, is the
# one the specification describes: three references -- an initialization
# baseline, a healthy smaller-model baseline, and the previous accepted
# checkpoint -- with every threshold prespecified, layer-specific, measured
# across seeds, and stored in versioned configuration. Until such a baseline
# exists there is nothing to derive from, and inventing a precise-looking number
# would be worse than an obviously round one.
#
# Calibration note: on the synthetic 128-dimensional cone used in the tests, the
# raw mean resultant length is 0.870 and the normalized effective rank is 0.271,
# so both thresholds fire with margin. On the isotropic control they read 0.238
# and 0.828. That is a sanity check on the DIRECTION of the rules, not evidence
# that 0.5 is the right cut for a graph attention network's focal embeddings.
# ---------------------------------------------------------------------------
CONE_MEAN_RESULTANT_LENGTH_THRESHOLD: float = 0.5
THIN_NORMALIZED_EFFECTIVE_RANK_THRESHOLD: float = 0.5
COMMON_DIRECTION_CENTERED_THRESHOLD: float = 0.2
ZERO_NORM_FRACTION_THRESHOLD: float = 0.01
CONDITION_NUMBER_CEILING: float = 1e12

COLLAPSE_HEALTHY = "healthy"
COLLAPSE_COMMON_DIRECTION = "common_direction_growth"
COLLAPSE_NORM_COMPENSATED = "norm_compensated_partial_collapse"
COLLAPSE_DESTRUCTIVE = "destructive_collapse"
COLLAPSE_EXPLOSION = "representation_explosion"
COLLAPSE_UNDETERMINED = "undetermined"
# DISTINCT FROM UNDETERMINED, and the distinction is load-bearing. UNDETERMINED
# means the panel could not evaluate -- metrics missing, cohort unusable. This
# means a collapse WAS detected and stage one cannot say which kind, because
# separating norm-compensated from destructive needs the R3 probe decomposition.
# Collapsing the two would repeat the drift monitor's mistake, where UNKNOWN was
# recorded as `none` and the notification job never fired once.
COLLAPSE_CONE_UNRESOLVED = "cone_collapse_mechanism_unresolved"


def _validate(x: np.ndarray) -> Optional[str]:
    """Shared entry checks. Returns a reason token, or None when usable."""
    if x.ndim != 2:
        return f"expected a two-dimensional array, received shape {x.shape}"
    if x.shape[0] == 0 or x.shape[1] == 0:
        return REASON_EMPTY_COHORT
    if not np.isfinite(x).all():
        # NOT silently dropped. A non-finite embedding is itself a finding --
        # representation explosion produces them -- and quietly filtering would
        # hide the very pathology this panel exists to catch.
        return REASON_NONFINITE_VALUES
    if x.shape[0] < MINIMUM_ROWS:
        return REASON_TOO_FEW_OBSERVATIONS
    return None


def mean_resultant_length(x: np.ndarray, *, center: bool = False,
                          eps: float = 1e-12) -> MetricResult:
    """R1. Length of the mean unit vector: 0 dispersed, 1 a single direction.

    `center=True` subtracts the mean first, which is the diagnostically
    important variant: a large gap between the raw and centred values means a
    dominant COMMON direction, which is linearly removable, rather than genuine
    loss of angular diversity.

    Zero-norm rows are excluded from the direction average -- a zero vector has
    no direction -- but the count is reported in metadata rather than dropped
    silently, because a growing zero-norm fraction is itself a collapse symptom.
    """
    x = np.asarray(x, dtype=np.float64)
    bad = _validate(x)
    if bad:
        return MetricResult(float("nan"), MetricStatus.INSUFFICIENT_SUPPORT, bad)

    work = x - x.mean(axis=0, keepdims=True) if center else x
    norms = np.linalg.norm(work, axis=1)
    keep = norms > eps
    n_zero = int((~keep).sum())
    if int(keep.sum()) < 2:
        return MetricResult(float("nan"), MetricStatus.INSUFFICIENT_SUPPORT,
                            REASON_ALL_ZERO_NORM,
                            {"n_zero_norm": n_zero, "n_rows": int(x.shape[0])})

    unit = work[keep] / norms[keep, None]
    value = float(np.linalg.norm(unit.mean(axis=0)))
    return MetricResult(value, MetricStatus.OK, None,
                        {"centered": bool(center),
                         "n_used": int(keep.sum()),
                         "n_zero_norm": n_zero,
                         "dimension": int(x.shape[1])})


def norm_statistics(x: np.ndarray) -> dict:
    """R1/R3 support. The norm distribution, where migrated signal accumulates.

    A growing norm is not pathological on its own. It becomes suspicious when it
    rises WHILE angular diversity falls, which is why this is reported beside
    the directional statistics rather than alone.
    """
    x = np.asarray(x, dtype=np.float64)
    bad = _validate(x)
    if bad:
        nan = MetricResult(float("nan"), MetricStatus.INSUFFICIENT_SUPPORT, bad)
        return {k: nan for k in ("mean_norm", "norm_sd", "norm_cv",
                                 "max_to_median_norm", "zero_norm_fraction")}

    norms = np.linalg.norm(x, axis=1)
    mean = float(norms.mean())
    sd = float(norms.std(ddof=1)) if norms.size > 1 else 0.0
    median = float(np.median(norms))
    zero_fraction = float((norms <= 1e-12).mean())

    out = {
        "mean_norm": MetricResult(mean, MetricStatus.OK),
        "norm_sd": MetricResult(sd, MetricStatus.OK),
        "zero_norm_fraction": MetricResult(zero_fraction, MetricStatus.OK),
    }
    # A coefficient of variation is undefined at a mean of zero, and a
    # max-to-median ratio is undefined at a median of zero. Both are reported as
    # UNDEFINED rather than as a large number or a silent NaN.
    out["norm_cv"] = (MetricResult(sd / mean, MetricStatus.OK) if mean > 1e-12
                      else MetricResult(float("nan"), MetricStatus.UNDEFINED,
                                        "mean norm is zero"))
    out["max_to_median_norm"] = (
        MetricResult(float(norms.max() / median), MetricStatus.OK) if median > 1e-12
        else MetricResult(float("nan"), MetricStatus.UNDEFINED,
                          "median norm is zero"))
    return out


def spectral_geometry(x: np.ndarray, *, top_k=(1, 5, 10, 25, 50)) -> dict:
    """R2. How much of the available dimensionality is actually used.

    Computed from the singular values of the CENTRED matrix, so the spectrum
    describes variance about the mean rather than being dominated by a common
    offset -- the mean direction is R1's business and is reported there.

    effective rank      exp(spectral entropy); "how many dimensions are in use"
    participation ratio (sum lambda)^2 / sum lambda^2; more sensitive to a few
                        dominant directions
    condition number    lambda_1 / lambda_min over the retained spectrum
    """
    x = np.asarray(x, dtype=np.float64)
    bad = _validate(x)
    d = int(x.shape[1]) if x.ndim == 2 and x.shape[1] else 0
    if bad:
        nan = MetricResult(float("nan"), MetricStatus.INSUFFICIENT_SUPPORT, bad)
        return {k: nan for k in ("effective_rank", "normalized_effective_rank",
                                 "participation_ratio", "spectral_entropy",
                                 "condition_number")} | {
            f"top{k}_variance_fraction": nan for k in top_k}

    n = int(x.shape[0])
    if n < MINIMUM_ROWS_PER_DIMENSION * d:
        # Rank-deficient by construction. Reporting a number here would describe
        # the sample size, not the model.
        reason = (f"{n} rows for {d} dimensions is below the "
                  f"{MINIMUM_ROWS_PER_DIMENSION:g}x floor; the sample covariance "
                  "is rank-deficient by construction")
        nan = MetricResult(float("nan"), MetricStatus.INSUFFICIENT_SUPPORT, reason)
        return {k: nan for k in ("effective_rank", "normalized_effective_rank",
                                 "participation_ratio", "spectral_entropy",
                                 "condition_number")} | {
            f"top{k}_variance_fraction": nan for k in top_k}

    xc = x - x.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(xc, compute_uv=False)
    lam = sv ** 2
    total = float(lam.sum())
    if total <= 0:
        nan = MetricResult(float("nan"), MetricStatus.DEGENERATE
                           if hasattr(MetricStatus, "DEGENERATE")
                           else MetricStatus.UNDEFINED, REASON_DEGENERATE_SPECTRUM)
        return {k: nan for k in ("effective_rank", "normalized_effective_rank",
                                 "participation_ratio", "spectral_entropy",
                                 "condition_number")} | {
            f"top{k}_variance_fraction": nan for k in top_k}

    p = lam / total
    nz = p[p > 1e-15]
    entropy = float(-(nz * np.log(nz)).sum())
    eff = float(np.exp(entropy))
    pr = float(total ** 2 / float((lam ** 2).sum()))

    out = {
        "spectral_entropy": MetricResult(entropy, MetricStatus.OK),
        "effective_rank": MetricResult(eff, MetricStatus.OK),
        "normalized_effective_rank": MetricResult(eff / d, MetricStatus.OK,
                                                  None, {"dimension": d}),
        "participation_ratio": MetricResult(pr, MetricStatus.OK),
    }
    lam_min = float(lam[lam > 1e-15].min()) if (lam > 1e-15).any() else 0.0
    out["condition_number"] = (
        MetricResult(float(lam[0] / lam_min), MetricStatus.OK) if lam_min > 0
        else MetricResult(float("nan"), MetricStatus.UNDEFINED,
                          "smallest retained eigenvalue is zero"))
    for k in top_k:
        key = f"top{k}_variance_fraction"
        if k > lam.size:
            out[key] = MetricResult(
                float("nan"), MetricStatus.NOT_APPLICABLE,
                f"top-{k} requested but the spectrum has {lam.size} components")
        else:
            out[key] = MetricResult(float(lam[:k].sum() / total), MetricStatus.OK)
    return out


@dataclass(frozen=True)
class GeometrySummary:
    """One representation, one partition role, one checkpoint."""

    representation_name: str
    partition_role: str
    n_observations: int
    dimension: int
    metrics: dict = field(default_factory=dict)
    collapse_status: str = COLLAPSE_UNDETERMINED
    reasons: tuple = ()

    def to_dict(self) -> dict:
        return {
            "representation_name": self.representation_name,
            "partition_role": self.partition_role,
            "n_observations": self.n_observations,
            "dimension": self.dimension,
            "metrics": {k: v.to_dict() if hasattr(v, "to_dict") else v
                        for k, v in self.metrics.items()},
            "collapse_status": self.collapse_status,
            "reasons": list(self.reasons),
        }


def summarize_geometry(x: np.ndarray, *, representation_name: str,
                       partition_role: str, **thresholds) -> GeometrySummary:
    """R1 + R2 for one representation. Every metric carries its own status."""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.shape[0]) if x.ndim == 2 else 0
    d = int(x.shape[1]) if x.ndim == 2 and x.shape[1] else 0

    metrics = {
        "mean_resultant_length_raw": mean_resultant_length(x, center=False),
        "mean_resultant_length_centered": mean_resultant_length(x, center=True),
    }
    metrics.update(norm_statistics(x))
    metrics.update(spectral_geometry(x))

    raw = metrics["mean_resultant_length_raw"]
    cen = metrics["mean_resultant_length_centered"]
    if raw.status is MetricStatus.OK and cen.status is MetricStatus.OK:
        metrics["mean_resultant_length_delta"] = MetricResult(
            raw.value - cen.value, MetricStatus.OK, None,
            {"interpretation": "large positive means a dominant COMMON direction, "
                               "which is linearly removable"})
    else:
        metrics["mean_resultant_length_delta"] = MetricResult(
            float("nan"), MetricStatus.INSUFFICIENT_SUPPORT,
            "raw or centered mean resultant length unavailable")

    status, reasons = classify_collapse(metrics, **thresholds)
    return GeometrySummary(representation_name, partition_role, n, d,
                           metrics, status, tuple(reasons))


def classify_collapse(
    metrics: dict,
    *,
    cone_threshold: float = CONE_MEAN_RESULTANT_LENGTH_THRESHOLD,
    thin_threshold: float = THIN_NORMALIZED_EFFECTIVE_RANK_THRESHOLD,
    common_direction_threshold: float = COMMON_DIRECTION_CENTERED_THRESHOLD,
    zero_norm_threshold: float = ZERO_NORM_FRACTION_THRESHOLD,
    condition_ceiling: float = CONDITION_NUMBER_CEILING,
) -> tuple:
    """Encode the diagnostic patterns as RULES rather than leaving them to the eye.

    Deliberately conservative. Every branch that cannot be evaluated returns
    UNDETERMINED with a reason, because "we could not tell" must never be
    reported as "healthy" -- the mistake the drift monitor made when UNKNOWN was
    recorded as none and its notification never fired once.

    Stage one sees only R1 and R2, so it cannot distinguish NORM-COMPENSATED
    from DESTRUCTIVE collapse: that distinction needs the R3 probe decomposition
    against a representation this project does not yet export. Both surface as
    a cone-collapse finding with the ambiguity stated, never resolved by guess.
    """
    reasons = []

    def ok(key):
        m = metrics.get(key)
        return m is not None and m.status is MetricStatus.OK

    if not ok("mean_resultant_length_raw") or not ok("normalized_effective_rank"):
        return COLLAPSE_UNDETERMINED, (
            "mean resultant length or normalized effective rank unavailable; "
            "an unevaluable representation is not a healthy one",)

    raw_mrl = metrics["mean_resultant_length_raw"].value
    cen_mrl = (metrics["mean_resultant_length_centered"].value
               if ok("mean_resultant_length_centered") else float("nan"))
    nrank = metrics["normalized_effective_rank"].value
    top1 = metrics["top1_variance_fraction"].value if ok("top1_variance_fraction") else float("nan")

    zero_fraction = (metrics["zero_norm_fraction"].value
                     if ok("zero_norm_fraction") else 0.0)
    cond_bad = (metrics["condition_number"].status is MetricStatus.UNDEFINED
                or (ok("condition_number")
                    and metrics["condition_number"].value > condition_ceiling))

    if zero_fraction > zero_norm_threshold or cond_bad:
        reasons.append(
            f"zero-norm fraction {zero_fraction:.4f} or condition number beyond "
            "the numerical ceiling")
        return COLLAPSE_EXPLOSION, tuple(reasons)

    cone = raw_mrl > cone_threshold
    thin = nrank < thin_threshold

    if cone and np.isfinite(cen_mrl) and cen_mrl < common_direction_threshold:
        reasons.append(
            f"raw mean resultant length {raw_mrl:.3f} falls to {cen_mrl:.3f} after "
            "centering, so the anisotropy is a common direction and is linearly "
            "removable")
        if not thin:
            return COLLAPSE_COMMON_DIRECTION, tuple(reasons)

    if cone or thin:
        if cone:
            reasons.append(f"mean resultant length {raw_mrl:.3f} indicates a narrow cone")
        if thin:
            reasons.append(
                f"normalized effective rank {nrank:.3f} means most dimensions are idle")
        if np.isfinite(top1) and top1 > 0.5:
            reasons.append(f"top-1 component holds {top1:.1%} of the variance")
        reasons.append(
            "STAGE ONE CANNOT SEPARATE norm-compensated from destructive collapse: "
            "that needs the R3 norm-angle probe decomposition, which requires a "
            "representation this project does not yet export")
        # A DETECTED collapse whose mechanism is unresolved -- NOT the same as
        # "could not evaluate". A reader and a gate must be able to tell those
        # apart, and a single status cannot carry both.
        return COLLAPSE_CONE_UNRESOLVED, tuple(reasons)

    return COLLAPSE_HEALTHY, ()


def panel_r_capabilities() -> tuple:
    """The stages of Panel R that are NOT built, declared honestly.

    Registering these as evidence rather than stubbing them is the whole point
    of the capability contract: a panel evaluating an absent representation
    would pass vacuously, and a green Panel R would then be cited as proof the
    geometry was checked.
    """
    absent = ("R3 through R7 need a stored representation. models/gnn.py:357 "
              "computes focal_embeddings and returns only "
              "classifier(focal_embeddings); the embedding is discarded.")
    stages = (
        ("panel_r3_norm_angle_decomposition", absent),
        ("panel_r4_conditioning_recoverability", absent),
        ("panel_r5_hubness_local_geometry", absent),
        ("panel_r6_training_trajectory", absent),
        ("panel_r7_downstream_sensitivity", absent),
    )
    return tuple(
        CapabilityEvidence(
            capability_name=name,
            capability_state=CapabilityState.NOT_IMPLEMENTED,
            target_state=TargetState.ABSENT,
            output_artifact=None,
            target_manifest=None,
            status=MetricStatus.NOT_IMPLEMENTED,
            reason=REASON_NO_OUTPUT_ARTIFACT,
        )
        for name, _why in stages
    )
