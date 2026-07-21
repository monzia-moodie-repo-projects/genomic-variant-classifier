"""Clustering and representation-structure metrics (metric specification, Panel Q).

WHAT THIS MODULE IS FOR
=======================
Panel Q evaluates whether a representation or cohort exhibits coherent
structure. It does NOT evaluate whether predictions are correct. A strong result
here does not make a better classifier, and nothing produced by this module may
be reported as evidence of clinical superiority.

That separation is why this is a new module rather than an extension of
evaluation/metrics.py: the concerns differ in input contract, computational
characteristics, dependencies, failure modes, partition policy, and the kind of
scientific claim they support.

FIVE FAILURE MODES CLOSED BY CONSTRUCTION
==========================================

1. NO SILENT NOT-A-NUMBER. A naive implementation returns float("nan") when a
   metric is undefined. A consumer calling numpy.nanmean then skips it
   invisibly, and an absent measurement becomes indistinguishable from one that
   was legitimately excluded. Every value here is a MetricResult carrying a
   MetricStatus and a STABLE machine-readable reason token. The invariant is
   enforced in __post_init__, not only in the factory helpers, so the raw
   constructor cannot be used to build an unexplained failure.

2. REASONS ARE TOKENS; NUMBERS LIVE IN METADATA. An earlier draft built reasons
   by interpolation -- "sample_size_20001_exceeds_ceiling_20000_would_allocate_
   3.0_GiB". That couples every consumer and every test to a float format. The
   reason is now a fixed token from REASON_*; the quantities that produced it
   are in `metadata`, where they can be asserted numerically and formatted
   separately.

3. THREE EXCLUSION ROUTES, COUNTED SEPARATELY, IN A FIXED ORDER. Observations
   leave by non-finite features, by an algorithm's noise label, or by a
   minimum-cluster-size filter. A single "noise_fraction" covers only the second
   and silently absorbs the third. The order of exclusion is fixed and tested:
   an observation that is both non-finite and noise-labelled counts ONCE, as
   non-finite. A reconciliation check alone would not catch a reordering,
   because the total is the same either way.

4. GEOMETRY IS EXPLICIT, AND THE SPHERICAL CASE IS A DIFFERENT ESTIMAND. For
   unit-normalized vectors, squared Euclidean distance equals 2 - 2*cos, so
   pairwise Euclidean and cosine distances are monotone. That holds for PAIRWISE
   measures such as the silhouette coefficient. It does NOT make a
   Euclidean-centroid Davies-Bouldin index a cosine one, because the Euclidean
   mean of points on a sphere has norm below one and does not lie on the sphere.
   Measured on a directional fixture: 2.9188 against 1.3538, with cluster mean
   norms near 0.47. Two functions, two names, neither reportable as "the
   Davies-Bouldin index".

5. THE SILHOUETTE GUARD REFUSES BEFORE ALLOCATING. silhouette_score
   materializes an n-by-n distance matrix. The limit is expressed in MEMORY,
   not in a sample count, because memory is the actual constraint and a count
   silently assumes a dtype:

       maximum_distance_matrix_gib=3.0  ->  max n = floor(sqrt(3*2^30/8)) = 20,066

   Measured on 1280-dimensional embeddings: 0.32 s at n=2,000; 7.53 s at
   10,000; 42.44 s and 1.0 GiB at 30,000; ~27 GiB predicted at 60,000; 16.4
   TiB at 1.5 million. A request above the ceiling returns
   COMPUTATIONALLY_DEFERRED with the numbers in metadata before anything is
   allocated. It does not attempt the computation and fail partway, because
   discovering the allocation empirically costs a cloud rental to learn what
   arithmetic gives for free.

MEASUREMENT IS SEPARATED FROM DECISION
---------------------------------------
decide_confounder_gate() is pure, total, and free of randomness: it takes
AgreementEstimate values and a policy and returns a verdict. That makes the
SCIENTIFIC POLICY -- interval separation, technical versus nuisance, refusal
versus warning -- testable as arithmetic with hand-written intervals, and lets
the policy move to configuration later without touching the estimator.
evaluate_confounder_gate() measures, then calls it.

RESAMPLING IS SUBSAMPLING, NOT BOOTSTRAP
-----------------------------------------
The ordinary bootstrap is unsuitable for partition agreement: observations
repeat, some vanish, duplicates overweight, and the shared-observation
denominator moves between replicates. Fixed-rate subsampling without
replacement at the GENE level keeps the denominator fixed and prevents large
genes from dominating -- the same dependence concern that makes the project's
splits gene-disjoint.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        homogeneity_completeness_v_measure,
        normalized_mutual_info_score,
        silhouette_score,
    )
    _SKLEARN = True
except Exception:  # pragma: no cover
    _SKLEARN = False

BYTES_PER_DISTANCE = 8  # float64
DEFAULT_MAX_DISTANCE_MATRIX_GIB = 3.0

# Stable reason tokens. Consumers and tests assert on these; the numbers that
# produced them are in MetricResult.metadata / EstimatedMetric.metadata.
REASON_SKLEARN_MISSING = "scikit_learn_not_installed"
REASON_NO_OBSERVATIONS = "no_observations"
REASON_FEWER_THAN_TWO_CLUSTERS = "fewer_than_two_clusters"
REASON_CLUSTERS_NOT_FEWER_THAN_OBSERVATIONS = "clusters_not_fewer_than_observations"
REASON_MEMORY_LIMIT_EXCEEDED = "estimated_distance_matrix_exceeds_memory_limit"
REASON_NO_VALID_REPLICATES = "no_valid_replicates"
REASON_ZERO_NORM_OBSERVATIONS = "zero_norm_observations_have_no_direction"
REASON_NO_SPHERICAL_CENTROID = "cluster_has_no_defined_spherical_centroid"
REASON_COINCIDENT_CENTROIDS = "coincident_cluster_centroids"
REASON_NON_OK_RESULTS_PRESENT = "non_ok_metric_results_present"
REASON_NO_RESULTS = "no_results_supplied"
REASON_NO_OK_RESULTS = "no_ok_results"
REASON_INSUFFICIENT_OVERLAP = "overlap_below_minimum"
REASON_PARTITION_HAS_ONE_GROUP = "a_partition_has_fewer_than_two_groups"


class MetricStatus(str, Enum):
    """Why a metric does or does not carry a value.

    Inheriting from str keeps these JSON-serialisable without a custom encoder,
    which matters because Panel Q records end up in run manifests.
    """

    OK = "ok"
    UNDEFINED = "undefined"                                # mathematically undefined
    INSUFFICIENT_SUPPORT = "insufficient_support"          # too few observations/overlap
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"      # package not installed
    COMPUTATIONALLY_DEFERRED = "computationally_deferred"  # refused on cost, before allocating
    FAILED = "failed"                                      # raised during computation


class CovariateType(str, Enum):
    """How a covariate is treated by the confounder gate.

    TECHNICAL      provenance and instrumentation: laboratory, platform, batch,
                   build, submitter, missingness pattern. Exceeding the target
                   is a REFUSAL.
    DESIGN         cohort-construction artefacts: source database, release,
                   review status, label availability. Exceeding is a WARNING.
    BIOLOGICAL_NUISANCE
                   real biology that is not the target: ancestry, tissue,
                   variant type, coding status. Exceeding is a WARNING plus a
                   stratified-analysis requirement, because legitimate biology
                   can correlate with these.
    TARGET         the primary biological variable itself.

    There is deliberately no default. An unclassified covariate cannot be gated,
    and defaulting it either way is a guess that would silently decide whether a
    solution is refused.
    """

    TECHNICAL = "technical"
    DESIGN = "design"
    BIOLOGICAL_NUISANCE = "biological_nuisance"
    TARGET = "target"


@dataclass(frozen=True)
class MetricResult:
    """A metric value that always knows whether it is a value.

    Invariants, enforced in __post_init__ so the raw constructor cannot bypass
    them:
      - a non-OK status REQUIRES a nonempty reason;
      - a non-OK status carries value NaN;
      - an OK status carries a finite value and no reason.

    Read `status` before `value`. A caller that reads `value` without checking
    `status` is making the mistake this class exists to prevent.
    """

    value: float
    status: MetricStatus
    reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, MetricStatus):
            raise TypeError(f"status must be a MetricStatus, got {type(self.status).__name__}")
        if self.status is MetricStatus.OK:
            if self.reason:
                raise ValueError(
                    "an OK MetricResult must not carry a reason; a reason explains "
                    "why a value is absent")
            if not np.isfinite(self.value):
                raise ValueError(
                    f"an OK MetricResult must carry a finite value, got {self.value}")
        else:
            if not self.reason:
                raise ValueError(
                    f"status {self.status.value!r} requires a nonempty reason. A "
                    "failure without an explanation is exactly the silent NaN this "
                    "class exists to prevent.")
            if np.isfinite(self.value):
                raise ValueError(
                    f"status {self.status.value!r} must carry NaN, got {self.value}; "
                    "a non-OK result holding a finite number invites it being used")

    @property
    def is_ok(self) -> bool:
        return self.status is MetricStatus.OK

    @classmethod
    def ok(cls, value: float, **metadata) -> "MetricResult":
        return cls(float(value), MetricStatus.OK, None, dict(metadata))

    @classmethod
    def not_ok(cls, status: MetricStatus, reason: str, **metadata) -> "MetricResult":
        if status is MetricStatus.OK:
            raise ValueError("not_ok() cannot construct an OK result")
        return cls(float("nan"), status, reason, dict(metadata))

    def to_dict(self) -> dict:
        return {"value": self.value, "status": self.status.value,
                "reason": self.reason, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, d: dict) -> "MetricResult":
        """Round-trip from to_dict(). NaN does not survive strict JSON, so a
        null value is read back as NaN rather than rejected."""
        v = d.get("value")
        value = float("nan") if v is None else float(v)
        return cls(value, MetricStatus(d["status"]), d.get("reason"),
                   dict(d.get("metadata") or {}))


def aggregate(results: Sequence[MetricResult], *,
              allow_non_ok: bool = False) -> MetricResult:
    """Mean of several MetricResults, refusing to average across failures.

    numpy.nanmean over a list containing undefined metrics silently produces the
    mean of whichever subset happened to work, reported as though it were the
    requested quantity. This refuses instead, and reports the full status
    census -- including how many were OK -- so the caller can see the shape of
    the problem rather than only its size.

    The refusal status is INSUFFICIENT_SUPPORT rather than FAILED: nothing
    failed here, the inputs were unsuitable.
    """
    if not results:
        return MetricResult.not_ok(MetricStatus.INSUFFICIENT_SUPPORT, REASON_NO_RESULTS)
    census: dict = {}
    for r in results:
        census[r.status.value] = census.get(r.status.value, 0) + 1
    bad = [r for r in results if not r.is_ok]
    if bad and not allow_non_ok:
        return MetricResult.not_ok(
            MetricStatus.INSUFFICIENT_SUPPORT, REASON_NON_OK_RESULTS_PRESENT,
            n_total=len(results), n_not_ok=len(bad), status_counts=census)
    good = [r.value for r in results if r.is_ok]
    if not good:
        return MetricResult.not_ok(MetricStatus.UNDEFINED, REASON_NO_OK_RESULTS,
                                   status_counts=census)
    return MetricResult.ok(float(np.mean(good)), n_averaged=len(good),
                           n_skipped=len(results) - len(good), status_counts=census)


@dataclass(frozen=True)
class EstimatedMetric:
    """A metric estimated by subsampling rather than computed exactly.

    Same non-OK-requires-reason invariant as MetricResult, for the same reason.
    """

    estimate: float
    standard_deviation: float
    ci95: tuple
    sample_size: int
    sampling_fraction: float
    n_replicates: int
    seeds: tuple
    minimum_per_cluster: int
    distance_metric: str
    sampling_method: str
    status: MetricStatus
    reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is not MetricStatus.OK and not self.reason:
            raise ValueError(
                f"status {self.status.value!r} requires a nonempty reason")
        if self.status is MetricStatus.OK and not np.isfinite(self.estimate):
            raise ValueError("an OK EstimatedMetric must carry a finite estimate")

    @property
    def is_ok(self) -> bool:
        return self.status is MetricStatus.OK

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["status"] = self.status.value
        d["ci95"] = list(self.ci95)
        d["seeds"] = list(self.seeds)
        d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def not_ok(cls, status: MetricStatus, reason: str, *, sample_size: int = 0,
               distance_metric: str = "unknown", **metadata) -> "EstimatedMetric":
        nan = float("nan")
        return cls(nan, nan, (nan, nan), sample_size, 0.0, 0, (), 0,
                   distance_metric, "stratified_by_cluster_without_replacement",
                   status, reason, dict(metadata))


@dataclass(frozen=True)
class ClusteringPopulationAccounting:
    """Every route by which an observation leaves the analysis, counted separately.

    A single aggregate exclusion figure lets a density-based algorithm discard
    the difficult part of the cohort and score well on what remains.

    The four buckets must sum to the input count. __post_init__ enforces that,
    because an unaccounted observation is an unreported exclusion.
    """

    n_input: int
    n_nonfinite_excluded: int
    n_algorithm_noise: int
    n_small_cluster_excluded: int
    n_analyzed: int

    nonfinite_exclusion_fraction: float
    algorithm_noise_fraction: float
    small_cluster_exclusion_fraction: float
    analyzed_fraction: float

    minimum_cluster_size: int
    noise_label: Optional[Any]

    def __post_init__(self) -> None:
        total = (self.n_nonfinite_excluded + self.n_algorithm_noise
                 + self.n_small_cluster_excluded + self.n_analyzed)
        if total != self.n_input:
            raise ValueError(
                f"population accounting does not reconcile: "
                f"{self.n_nonfinite_excluded} non-finite + {self.n_algorithm_noise} "
                f"noise + {self.n_small_cluster_excluded} small-cluster + "
                f"{self.n_analyzed} analyzed = {total}, but input was "
                f"{self.n_input}. Every observation must be accounted for exactly "
                "once; an unaccounted observation is an unreported exclusion.")

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["noise_label"] = None if self.noise_label is None else str(self.noise_label)
        return d


def account_for_population(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    noise_label: Optional[Any] = -1,
    minimum_cluster_size: int = 2,
) -> tuple:
    """Partition observations into the four accounting buckets.

    Returns (accounting, x_analyzed, labels_analyzed).

    THE ORDER OF EXCLUSION IS FIXED AND LOAD-BEARING:
      1. non-finite features   -- those rows cannot be measured at all
      2. the algorithm's noise label, among rows that survived (1)
      3. the minimum-cluster-size filter, among rows that survived (1) and (2)

    An observation that qualifies for more than one route counts ONCE, in the
    earliest. Reordering would move counts between buckets while still
    reconciling to the same total, so a reconciliation check alone cannot detect
    it; the ordering has its own test.
    """
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    if x.ndim != 2:
        raise ValueError(f"x must have shape (n_samples, n_features); got {x.shape}")
    if len(x) != len(labels):
        raise ValueError(f"x has {len(x)} rows but labels has {len(labels)} entries")
    n_input = len(x)
    if n_input == 0:
        raise ValueError("x has zero rows")

    finite = np.isfinite(x).all(axis=1)
    n_nonfinite = int((~finite).sum())

    if noise_label is None:
        is_noise = np.zeros(n_input, dtype=bool)
    else:
        is_noise = labels == noise_label
    # Noise is counted only among rows that survived the non-finite filter, so a
    # row that is both counts once, as non-finite.
    n_noise = int((is_noise & finite).sum())

    keep = finite & ~is_noise
    kept_labels = labels[keep]
    if len(kept_labels):
        uniq, counts = np.unique(kept_labels, return_counts=True)
        too_small = uniq[counts < minimum_cluster_size]
        small_mask = (np.isin(kept_labels, too_small) if len(too_small)
                      else np.zeros(len(kept_labels), dtype=bool))
    else:
        small_mask = np.zeros(0, dtype=bool)
    n_small = int(small_mask.sum())

    kept_positions = np.flatnonzero(keep)
    analyzed_positions = (kept_positions[~small_mask] if len(kept_labels)
                          else np.array([], dtype=int))
    n_analyzed = int(len(analyzed_positions))

    acc = ClusteringPopulationAccounting(
        n_input=n_input,
        n_nonfinite_excluded=n_nonfinite,
        n_algorithm_noise=n_noise,
        n_small_cluster_excluded=n_small,
        n_analyzed=n_analyzed,
        nonfinite_exclusion_fraction=n_nonfinite / n_input,
        algorithm_noise_fraction=n_noise / n_input,
        small_cluster_exclusion_fraction=n_small / n_input,
        analyzed_fraction=n_analyzed / n_input,
        minimum_cluster_size=minimum_cluster_size,
        noise_label=noise_label,
    )
    return acc, x[analyzed_positions], labels[analyzed_positions]


def _guard(x: np.ndarray, labels: np.ndarray) -> Optional[MetricResult]:
    """Shared preconditions. Returns a non-OK MetricResult, or None to proceed."""
    if not _SKLEARN:
        return MetricResult.not_ok(MetricStatus.DEPENDENCY_UNAVAILABLE,
                                   REASON_SKLEARN_MISSING)
    n = len(x)
    k = int(len(np.unique(labels)))
    if n == 0:
        return MetricResult.not_ok(MetricStatus.UNDEFINED, REASON_NO_OBSERVATIONS)
    if k < 2:
        return MetricResult.not_ok(MetricStatus.UNDEFINED,
                                   REASON_FEWER_THAN_TWO_CLUSTERS, n_clusters=k)
    if k >= n:
        return MetricResult.not_ok(MetricStatus.UNDEFINED,
                                   REASON_CLUSTERS_NOT_FEWER_THAN_OBSERVATIONS,
                                   n_clusters=k, n_observations=n)
    return None


def davies_bouldin_euclidean_centroid(x: np.ndarray, labels: np.ndarray) -> MetricResult:
    """Davies-Bouldin index under Euclidean geometry with arithmetic centroids.

    scikit-learn's definition. Appropriate for standardized features and for
    representations whose intended geometry is Euclidean. It is NOT a cosine
    Davies-Bouldin index even if x was L2-normalized first; see
    davies_bouldin_spherical_cosine.

    Lower is better; zero is the unattainable optimum.
    """
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    bad = _guard(x, labels)
    if bad is not None:
        return bad
    try:
        v = davies_bouldin_score(x, labels)
    except Exception as e:
        return MetricResult.not_ok(MetricStatus.FAILED, "computation_raised",
                                   exception=f"{type(e).__name__}: {e}"[:200])
    return MetricResult.ok(v, geometry="euclidean_centroid",
                           n_observations=int(len(x)),
                           n_clusters=int(len(np.unique(labels))))


def davies_bouldin_spherical_cosine(x: np.ndarray, labels: np.ndarray) -> MetricResult:
    """Davies-Bouldin index under angular geometry with SPHERICAL centroids.

    Scatter is mean cosine distance from each member to its cluster's spherical
    centroid; separation is cosine distance between spherical centroids. The
    spherical centroid is the arithmetic mean RE-NORMALIZED to unit length:

        mu_k = sum_i(x_i) / || sum_i(x_i) ||

    Using the arithmetic mean directly -- which is what an ordinary Euclidean
    Davies-Bouldin index does after L2-normalizing its inputs -- places the
    centroid strictly inside the sphere and measures scatter in the ambient
    space. Measured on a directional fixture the two differ by more than a
    factor of two, with cluster mean norms near 0.47.

    A cluster whose members are near-antipodal sums to approximately zero and
    has no defined direction. That is reported, not silently repaired.
    """
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    bad = _guard(x, labels)
    if bad is not None:
        return bad

    norms = np.linalg.norm(x, axis=1)
    n_zero = int((norms == 0).sum())
    if n_zero:
        return MetricResult.not_ok(MetricStatus.UNDEFINED,
                                   REASON_ZERO_NORM_OBSERVATIONS,
                                   n_zero_norm=n_zero)

    unit = x / norms[:, None]
    clusters = np.unique(labels)
    centroids, scatters = [], []
    for c in clusters:
        members = unit[labels == c]
        mean = members.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean))
        if mean_norm < 1e-12:
            return MetricResult.not_ok(
                MetricStatus.UNDEFINED, REASON_NO_SPHERICAL_CENTROID,
                cluster=str(c), mean_norm=mean_norm,
                note="members are near-antipodal; their mean direction is undefined")
        centroid = mean / mean_norm
        centroids.append(centroid)
        scatters.append(float((1.0 - members @ centroid).mean()))

    C = np.vstack(centroids)
    S = np.asarray(scatters)
    separation = 1.0 - (C @ C.T)
    np.fill_diagonal(separation, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = (S[:, None] + S[None, :]) / separation
    worst = np.nanmax(ratios, axis=1)
    if not np.isfinite(worst).all():
        return MetricResult.not_ok(
            MetricStatus.UNDEFINED, REASON_COINCIDENT_CENTROIDS,
            note="two clusters share a direction; their separation is zero")

    return MetricResult.ok(float(worst.mean()), geometry="spherical_cosine",
                           n_observations=int(len(x)),
                           n_clusters=int(len(clusters)))


def calinski_harabasz(x: np.ndarray, labels: np.ndarray) -> MetricResult:
    """Ratio of between-cluster to within-cluster dispersion. Higher is better."""
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    bad = _guard(x, labels)
    if bad is not None:
        return bad
    try:
        v = calinski_harabasz_score(x, labels)
    except Exception as e:
        return MetricResult.not_ok(MetricStatus.FAILED, "computation_raised",
                                   exception=f"{type(e).__name__}: {e}"[:200])
    return MetricResult.ok(v, geometry="euclidean_centroid",
                           n_observations=int(len(x)))


def distance_matrix_gib(n: int) -> float:
    """Gibibytes an n-by-n float64 distance matrix would occupy."""
    return (int(n) * int(n) * BYTES_PER_DISTANCE) / 1024 ** 3


def maximum_n_for_memory(limit_gib: float) -> int:
    """Largest n whose distance matrix fits in `limit_gib`.

    The guard is expressed in MEMORY rather than in a sample count because
    memory is the actual constraint; a count silently assumes a dtype. At the
    3.0 GiB default this yields 20,066, which is where the previous draft's
    magic 20,000 came from implicitly.
    """
    if limit_gib <= 0:
        raise ValueError(f"limit_gib must be positive, got {limit_gib}")
    return int(math.floor(math.sqrt(limit_gib * 1024 ** 3 / BYTES_PER_DISTANCE)))


def stratified_cluster_sample(labels: np.ndarray, *, sample_size: int, seed: int,
                              minimum_per_cluster: int = 25) -> np.ndarray:
    """Sample positions stratified by cluster, protecting small clusters first.

    Uniform sampling underrepresents small clusters and can remove them
    entirely. That does not add noise to the estimate -- it changes the ESTIMAND,
    because the metric is then computed over a different cluster count than the
    clustering produced, and the change is invisible in the reported number.

    Each cluster is first allocated up to `minimum_per_cluster` (or its full
    size, whichever is smaller); any remaining budget is distributed in
    proportion to spare capacity. If the floors alone exceed the budget the
    allocation falls back to proportional, which is a documented behaviour
    rather than an error -- a caller asking for fewer observations than there
    are clusters-times-floor has to lose something.

    Deterministic in `seed`. Never returns a duplicate position.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) < 2:
        raise ValueError("stratified sampling requires at least two clusters")
    n = len(labels)
    target = int(min(sample_size, n))

    alloc = {c: int(min(cnt, minimum_per_cluster)) for c, cnt in zip(uniq, counts)}
    remaining = target - sum(alloc.values())

    if remaining < 0:
        prop = counts / counts.sum()
        raw = prop * target
        base = np.floor(raw).astype(int)
        while base.sum() < target:
            base[int(np.argmax(raw - base))] += 1
        alloc = {c: int(min(cnt, b)) for c, cnt, b in zip(uniq, counts, base)}
    elif remaining > 0:
        capacity = np.array([cnt - alloc[c] for c, cnt in zip(uniq, counts)],
                            dtype=float)
        if capacity.sum() > 0:
            raw = capacity / capacity.sum() * remaining
            extra = np.minimum(np.floor(raw).astype(int), capacity.astype(int))
            while extra.sum() < remaining and (capacity - extra).max() > 0:
                extra[int(np.argmax(capacity - extra))] += 1
            for c, e in zip(uniq, extra):
                alloc[c] += int(e)

    picked = []
    for c in uniq:
        idx = np.flatnonzero(labels == c)
        take = int(min(len(idx), alloc[c]))
        if take:
            picked.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(picked) if picked else np.array([], dtype=int)
    rng.shuffle(out)
    return out


def estimate_silhouette(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "euclidean",
    requested_sample_size: int = 10_000,
    n_replicates: int = 20,
    seed: int = 0,
    minimum_per_cluster: int = 25,
    maximum_distance_matrix_gib: float = DEFAULT_MAX_DISTANCE_MATRIX_GIB,
) -> EstimatedMetric:
    """Silhouette coefficient, ESTIMATED by stratified subsampling.

    The guard is the point of this function. The effective per-replicate size is
    min(requested_sample_size, n); if its distance matrix would exceed
    `maximum_distance_matrix_gib`, the call returns COMPUTATIONALLY_DEFERRED
    with the numbers in `metadata` BEFORE allocating anything. It does not
    attempt the computation and fail partway, because discovering the allocation
    empirically costs a cloud rental to learn what arithmetic gives for free.

    `metadata` on a deferral carries effective_sample_size, maximum_sample_size,
    estimated_gib and limit_gib, so a caller can report the numbers without
    parsing the reason token.
    """
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    if not _SKLEARN:
        return EstimatedMetric.not_ok(MetricStatus.DEPENDENCY_UNAVAILABLE,
                                      REASON_SKLEARN_MISSING, distance_metric=metric)
    if x.ndim != 2:
        raise ValueError(f"x must be rank 2; got {x.shape}")
    if len(x) != len(labels):
        raise ValueError(f"x has {len(x)} rows, labels has {len(labels)}")

    n = len(x)
    k = int(len(np.unique(labels)))
    if k < 2:
        return EstimatedMetric.not_ok(MetricStatus.UNDEFINED,
                                      REASON_FEWER_THAN_TWO_CLUSTERS,
                                      distance_metric=metric)
    if n <= k:
        return EstimatedMetric.not_ok(MetricStatus.UNDEFINED,
                                      REASON_CLUSTERS_NOT_FEWER_THAN_OBSERVATIONS,
                                      distance_metric=metric)

    effective = int(min(requested_sample_size, n))
    max_n = maximum_n_for_memory(maximum_distance_matrix_gib)
    if effective > max_n:
        return EstimatedMetric.not_ok(
            MetricStatus.COMPUTATIONALLY_DEFERRED, REASON_MEMORY_LIMIT_EXCEEDED,
            sample_size=effective, distance_metric=metric,
            effective_sample_size=effective, maximum_sample_size=max_n,
            estimated_gib=distance_matrix_gib(effective),
            limit_gib=float(maximum_distance_matrix_gib),
            requested_sample_size=int(requested_sample_size), n_observations=n)

    values, seeds_used = [], []
    for r in range(n_replicates):
        s = seed + r
        try:
            idx = stratified_cluster_sample(labels, sample_size=effective, seed=s,
                                            minimum_per_cluster=minimum_per_cluster)
        except ValueError:
            continue
        sub = labels[idx]
        if len(np.unique(sub)) < 2:
            continue
        try:
            values.append(float(silhouette_score(x[idx], sub, metric=metric)))
            seeds_used.append(s)
        except Exception as e:  # pragma: no cover
            logger.warning("silhouette replicate %d failed: %s", s, e)

    if not values:
        return EstimatedMetric.not_ok(MetricStatus.INSUFFICIENT_SUPPORT,
                                      REASON_NO_VALID_REPLICATES,
                                      sample_size=effective, distance_metric=metric)

    arr = np.asarray(values)
    lo, hi = (np.quantile(arr, [0.025, 0.975]) if len(arr) > 1 else (arr[0], arr[0]))
    return EstimatedMetric(
        estimate=float(arr.mean()),
        standard_deviation=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        ci95=(float(lo), float(hi)),
        sample_size=effective,
        sampling_fraction=effective / n,
        n_replicates=len(arr),
        seeds=tuple(seeds_used),
        minimum_per_cluster=minimum_per_cluster,
        distance_metric=metric,
        sampling_method="stratified_by_cluster_without_replacement",
        status=MetricStatus.OK,
        metadata={"effective_sample_size": effective,
                  "maximum_sample_size": max_n,
                  "estimated_gib": distance_matrix_gib(effective),
                  "limit_gib": float(maximum_distance_matrix_gib),
                  "n_observations": n},
    )


@dataclass(frozen=True)
class PartitionAgreementPanel:
    """Agreement between two partitions of the SAME observations.

    ARGUMENT ORDER MATTERS FOR TWO OF THESE. The adjusted Rand index, adjusted
    mutual information and normalized mutual information are symmetric.
    Homogeneity and completeness are NOT: homogeneity asks whether each
    discovered cluster contains mostly one reference class; completeness asks
    whether members of a reference class are kept together. Swapping the
    arguments exchanges the two values and produces a plausible-looking,
    wrong result. `reference` is always first, `clustering` second.
    """

    n_samples: int
    n_clusters_reference: int
    n_clusters_discovered: int
    adjusted_rand_index: MetricResult
    adjusted_mutual_information: MetricResult
    normalized_mutual_information: MetricResult
    homogeneity: MetricResult
    completeness: MetricResult
    v_measure: MetricResult

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_clusters_reference": self.n_clusters_reference,
            "n_clusters_discovered": self.n_clusters_discovered,
            **{k: getattr(self, k).to_dict() for k in
               ("adjusted_rand_index", "adjusted_mutual_information",
                "normalized_mutual_information", "homogeneity",
                "completeness", "v_measure")},
        }


def evaluate_partition_agreement(
    reference: np.ndarray,
    clustering: np.ndarray,
    *,
    ignore_labels: Optional[set] = None,
    minimum_overlap: int = 50,
) -> PartitionAgreementPanel:
    """Agreement between a reference partition and a discovered clustering.

    Normalized mutual information is computed but is SECONDARY: it is not
    adjusted for agreement expected by chance and inflates when clusters are
    numerous or samples small. It carries a metadata note saying so, and must
    never be reported without adjusted mutual information beside it.
    """
    ref = np.asarray(reference)
    clu = np.asarray(clustering)
    if ref.ndim != 1 or clu.ndim != 1:
        raise ValueError("partition labels must be one-dimensional")
    if len(ref) != len(clu):
        raise ValueError(
            f"partitions cover different observation counts: {len(ref)} vs {len(clu)}. "
            "Agreement indices are undefined across different observation sets; "
            "intersect them first.")

    keep = np.ones(len(ref), dtype=bool)
    if ignore_labels:
        keep &= ~np.isin(ref, list(ignore_labels))
        keep &= ~np.isin(clu, list(ignore_labels))
    ref, clu = ref[keep], clu[keep]

    n = len(ref)
    kr, kc = int(len(np.unique(ref))), int(len(np.unique(clu)))

    def refuse(status, reason, **md):
        r = MetricResult.not_ok(status, reason, **md)
        return PartitionAgreementPanel(n, kr, kc, r, r, r, r, r, r)

    if not _SKLEARN:
        return refuse(MetricStatus.DEPENDENCY_UNAVAILABLE, REASON_SKLEARN_MISSING)
    if n < minimum_overlap:
        return refuse(MetricStatus.INSUFFICIENT_SUPPORT, REASON_INSUFFICIENT_OVERLAP,
                      n_shared=n, minimum_overlap=int(minimum_overlap))
    if kr < 2 or kc < 2:
        return refuse(MetricStatus.UNDEFINED, REASON_PARTITION_HAS_ONE_GROUP,
                      n_reference_groups=kr, n_discovered_groups=kc)

    try:
        h, c, v = homogeneity_completeness_v_measure(ref, clu)
        return PartitionAgreementPanel(
            n_samples=n, n_clusters_reference=kr, n_clusters_discovered=kc,
            adjusted_rand_index=MetricResult.ok(adjusted_rand_score(ref, clu)),
            adjusted_mutual_information=MetricResult.ok(
                adjusted_mutual_info_score(ref, clu, average_method="arithmetic")),
            normalized_mutual_information=MetricResult.ok(
                normalized_mutual_info_score(ref, clu, average_method="arithmetic"),
                note="secondary; not chance-adjusted; never report alone"),
            homogeneity=MetricResult.ok(h, argument_order="reference_then_clustering"),
            completeness=MetricResult.ok(c, argument_order="reference_then_clustering"),
            v_measure=MetricResult.ok(v),
        )
    except Exception as e:  # pragma: no cover
        return refuse(MetricStatus.FAILED, "computation_raised",
                      exception=f"{type(e).__name__}: {e}"[:200])


def permute_covariate_by_gene_block(covariate: np.ndarray, groups: np.ndarray,
                                    rng: np.random.Generator, *,
                                    size_strata: bool = False) -> np.ndarray:
    """GENE-BLOCK permutation: whole genes exchange covariate values.

    Renamed from `permute_covariate_by_group` on 2026-07-21. "Gene-block
    permutation" is the standard name for this scheme, and the old name said
    only that groups were involved, not that whole blocks were exchanged as
    units. The reported `permutation_scheme` now names the scheme rather than
    the vague "group".

    Public because it is the mechanism the permutation null rests on, and
    testing the mechanism directly -- that every member of a block still shares
    one value, and that the multiset of block values is preserved -- is more
    fundamental and less brittle than testing a downstream quantile.

    THE REPRESENTATIVE RULE. Where a block carries several covariate values the
    FIRST is taken as its representative. This is recorded in the result as
    `representative_rule`, alongside how many blocks it applied to, because a
    reader who is told only that twelve blocks carried multiple values still
    cannot tell WHICH value was used.

    THE MARGINAL DRIFT, AND WHY IT IS INHERENT RATHER THAN A DEFECT.
    Gene-block permutation with UNEQUAL block sizes necessarily changes the
    row-level marginal distribution of the covariate. Swapping a 10-row gene's
    value with a 4-row gene's moves 10 rows out of one level and 4 into it.
    Measured 2026-07-21 on a cohort of 1,149 rows across 150 genes with a
    PURELY gene-level covariate -- zero genes carrying more than one value --
    the mean total variation distance between the observed and permuted
    marginals was 0.0145 (standard deviation 0.0076), and it was ZERO in 0 of
    200 permutations.

    Within-block heterogeneity only modulates this; it is not the cause. Across
    within-gene variation probabilities of 0.00, 0.05, 0.15, 0.35, 0.60 and
    1.00 the drift measured 0.0171, 0.0160, 0.0264, 0.0458, 0.0630 and 0.0172:
    it PEAKS near 0.60 and falls back, because at full heterogeneity the first
    value is itself a uniform draw.

    THE CONSEQUENCE IS NEGLIGIBLE FOR ADJUSTED MUTUAL INFORMATION, which is
    chance-corrected against exactly the marginals that drift. Measured
    p-values, unstratified against size-stratified, over six seeds each:

        association 0.0   0.6024 vs 0.6013   difference -0.0011
        association 0.2   0.0659 vs 0.0548   difference -0.0111
        association 0.4   0.0033 vs 0.0033   difference  0.0000
        association 0.6   0.0033 vs 0.0033   difference  0.0000

    So the unstratified scheme remains the DEFAULT. The drift is measured and
    reported per run rather than restructured away, because restructuring buys
    nothing in the quantity anyone reads and costs something real (below).

    SIZE-STRATIFIED MODE (opt-in, `size_strata=True`) restricts swaps to blocks
    of equal size, which preserves the row-level marginal EXACTLY -- measured
    0.000000 drift in 200 of 200 permutations. Its cost is that a block alone
    in its size stratum can only swap with itself, so its value is FROZEN across
    every permutation. That count is reported as `n_blocks_frozen_in_stratum`
    and MUST be read: a stratification that freezes much of the cohort is a
    weaker null, not a stronger one. Do not assume the cost is small -- a
    synthetic cohort with sizes drawn uniformly from 1 to 300 froze zero blocks,
    but real ClinVar per-gene counts are far more heavy-tailed and large genes
    are frequently unique in size.
    """
    cov = np.asarray(covariate)
    g = np.asarray(groups)
    if len(cov) != len(g):
        raise ValueError(f"covariate has {len(cov)} entries, groups has {len(g)}")
    uniq, inv = np.unique(g, return_inverse=True)
    first_pos = np.zeros(len(uniq), dtype=int)
    for j_ in range(len(uniq)):
        first_pos[j_] = int(np.flatnonzero(inv == j_)[0])
    block_values = cov[first_pos]

    if not size_strata:
        return rng.permutation(block_values)[inv]

    sizes = np.array([int((inv == j_).sum()) for j_ in range(len(uniq))])
    out = block_values.copy()
    for size in np.unique(sizes):
        idx = np.flatnonzero(sizes == size)
        out[idx] = rng.permutation(block_values[idx])
    return out[inv]


def count_blocks_frozen_in_size_strata(groups: np.ndarray) -> int:
    """Blocks that are the only member of their size stratum.

    Under `size_strata=True` such a block can only swap with itself, so its
    covariate value never moves. Reported so the cost of stratifying is visible
    rather than assumed.
    """
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    sizes = np.array([int((inv == j).sum()) for j in range(inv.max() + 1)])
    vals, counts = np.unique(sizes, return_counts=True)
    return int(counts[counts == 1].sum())


def _total_variation_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Half the L1 distance between two empirical distributions over the same
    support. Zero when the marginals match exactly."""
    ka, ca = np.unique(a, return_counts=True)
    kb, cb = np.unique(b, return_counts=True)
    da = dict(zip(ka.tolist(), (ca / max(len(a), 1)).tolist()))
    db = dict(zip(kb.tolist(), (cb / max(len(b), 1)).tolist()))
    return 0.5 * sum(abs(da.get(k, 0.0) - db.get(k, 0.0)) for k in set(da) | set(db))


# Above this fraction of blocks carrying more than one covariate value, the
# covariate is not a block-level property and collapsing it to one
# representative discards the variation being tested. The run WARNS and records
# the fraction; it does not refuse, because refusing what the previous
# implementation accepted is a regression rather than a stricter standard --
# the lesson learned on 2026-07-21 when an absolute calibration-fold floor broke
# a 427-row cohort that the code it replaced had always handled.
DEFAULT_BLOCK_HETEROGENEITY_ADVISORY: float = 0.20
REASON_BLOCK_COVARIATE_HETEROGENEOUS = "block_covariate_heterogeneous"


def permutation_null_ami(
    clustering: np.ndarray,
    covariate: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    n_permutations: int = 200,
    seed: int = 0,
    size_strata: bool = False,
    heterogeneity_advisory: float = DEFAULT_BLOCK_HETEROGENEITY_ADVISORY,
) -> dict:
    """Null distribution for adjusted mutual information, respecting dependence.

    When observations are grouped -- variants within a gene -- permuting
    individual rows destroys nothing, because the covariate is often constant
    within a group and the permutation largely reproduces the observed
    association. Measured on a gene-structured fixture, the gene-level null 95th
    percentile was 0.0516 against a row-level 0.0011: a 47-fold difference, so
    the row-level null would make a spurious association look far more
    significant than it is.

    This follows the logic of the n_pathogenic_in_gene permutation ablation,
    which established that feature's contribution against a permuted null rather
    than by inspection.

    groups=None permutes rows, which is correct only for genuinely independent
    observations. The unit used is always recorded in the result.
    """
    clu = np.asarray(clustering)
    cov = np.asarray(covariate)
    if len(clu) != len(cov):
        raise ValueError(f"clustering has {len(clu)} entries, covariate {len(cov)}")
    if not _SKLEARN:
        return {"status": MetricStatus.DEPENDENCY_UNAVAILABLE.value,
                "reason": REASON_SKLEARN_MISSING}

    observed = float(adjusted_mutual_info_score(cov, clu, average_method="arithmetic"))
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)

    if groups is None:
        for i in range(n_permutations):
            null[i] = adjusted_mutual_info_score(rng.permutation(cov), clu,
                                                 average_method="arithmetic")
        unit = "row"
        extra = {"representative_rule": None, "size_strata": False,
                 "marginal_tvd_mean": 0.0, "marginal_tvd_max": 0.0}
    else:
        g = np.asarray(groups)
        if len(g) != len(clu):
            raise ValueError("groups must align with clustering")
        n_blocks = int(len(np.unique(g)))
        _, inv = np.unique(g, return_inverse=True)
        multi = int(sum(1 for j in range(inv.max() + 1)
                        if len(np.unique(cov[inv == j])) > 1))
        frac_multi = multi / max(n_blocks, 1)

        # The marginal drift is MEASURED on the permutations actually drawn,
        # not assumed. Gene-block permutation with unequal block sizes always
        # moves the row-level marginal; recording it lets a reader judge the
        # null rather than trust it. See permute_covariate_by_gene_block.
        tvds = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            permuted = permute_covariate_by_gene_block(cov, g, rng,
                                                       size_strata=size_strata)
            tvds[i] = _total_variation_distance(cov, permuted)
            null[i] = adjusted_mutual_info_score(permuted, clu,
                                                 average_method="arithmetic")

        unit = "gene_block_size_stratified" if size_strata else "gene_block"
        extra = {
            "n_blocks": n_blocks,
            # Retained under the old key as well: run manifests already carry it.
            "n_groups": n_blocks,
            "n_blocks_with_multiple_covariate_values": multi,
            "n_groups_with_multiple_covariate_values": multi,
            "fraction_blocks_with_multiple_covariate_values": float(frac_multi),
            "representative_rule": "first_value_per_block",
            "marginal_tvd_mean": float(tvds.mean()),
            "marginal_tvd_max": float(tvds.max()),
            "size_strata": bool(size_strata),
            "n_blocks_frozen_in_stratum": (
                count_blocks_frozen_in_size_strata(g) if size_strata else 0),
        }

        if frac_multi > heterogeneity_advisory:
            logger.warning(
                "Gene-block permutation null: %d of %d blocks (%.1f%%) carry more "
                "than one covariate value, above the %.0f%% advisory. The "
                "representative rule keeps the FIRST value per block, so the "
                "within-block variation is discarded and the null tests a "
                "collapsed covariate. Above this level the covariate is arguably "
                "not a block-level property and gene-block permutation may be the "
                "wrong scheme. Proceeding; the fraction is recorded as "
                "fraction_blocks_with_multiple_covariate_values.",
                multi, n_blocks, 100.0 * frac_multi, 100.0 * heterogeneity_advisory)
            extra["advisory"] = REASON_BLOCK_COVARIATE_HETEROGENEOUS

    p = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return {"status": MetricStatus.OK.value, "observed_ami": observed,
            "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
            "permutation_p_value": p, "effect_over_null": float(observed - null.mean()),
            # One-sided upper: the question is whether the OBSERVED agreement
            # exceeds what the null produces, never whether it falls short.
            "p_value_sidedness": "one_sided_upper",
            "n_permutations": int(n_permutations), "permutation_unit": unit,
            "permutation_scheme": unit,
            "seed": int(seed), **extra}


@dataclass(frozen=True)
class AgreementEstimate:
    """An adjusted-mutual-information estimate with its interval and provenance.

    This is the unit the confounder gate decides on. Separating it from the
    estimator is what makes the POLICY testable as pure arithmetic with
    hand-written intervals, and lets the policy move to configuration later
    without touching the estimator.
    """

    name: str
    covariate_type: CovariateType
    point_estimate: float
    ci95: tuple
    n_replicates: int = 0
    permutation: Optional[dict] = None

    def __post_init__(self) -> None:
        if not isinstance(self.covariate_type, CovariateType):
            raise ValueError(
                f"covariate {self.name!r} has covariate_type "
                f"{self.covariate_type!r}, which is not a CovariateType. Every "
                "covariate must be explicitly classified: an unclassified "
                "covariate cannot be gated, and defaulting it either way would "
                "silently decide whether the solution is refused.")
        if not np.isfinite(self.point_estimate):
            raise ValueError(
                f"covariate {self.name!r} has a non-finite point estimate "
                f"{self.point_estimate}. An AgreementEstimate is a measurement; if "
                "adjusted mutual information could not be computed there is no "
                "estimate to gate on, and the caller must report that as a non-OK "
                "MetricResult rather than smuggle a NaN into the decision.")
        lo, hi = self.ci95
        # A NON-FINITE INTERVAL IS LEGITIMATE and is handled fail-closed by
        # decide_confounder_gate: it arises when too few subsample replicates were
        # usable, and "we could not bound this" must never read as separation.
        if np.isfinite(lo) and np.isfinite(hi) and lo > hi:
            raise ValueError(
                f"covariate {self.name!r} has an inverted interval [{lo}, {hi}]")

    def to_dict(self) -> dict:
        return {"name": self.name, "covariate_type": self.covariate_type.value,
                "point_estimate": self.point_estimate, "ci95": list(self.ci95),
                "n_replicates": self.n_replicates, "permutation": self.permutation}


@dataclass(frozen=True)
class ConfounderComparison:
    variable_name: str
    variable_type: CovariateType
    adjusted_mutual_information: MetricResult
    ci95: tuple
    permutation: Optional[dict]
    separated_from_target: bool
    verdict: str            # "pass" | "refuse" | "warn"

    def to_dict(self) -> dict:
        return {"variable_name": self.variable_name,
                "variable_type": self.variable_type.value,
                "adjusted_mutual_information": self.adjusted_mutual_information.to_dict(),
                "ci95": list(self.ci95), "permutation": self.permutation,
                "separated_from_target": self.separated_from_target,
                "verdict": self.verdict}


@dataclass(frozen=True)
class ConfounderGate:
    """The refusal condition, decided on INTERVALS rather than point estimates.

    A solution passes only when, for every TECHNICAL covariate T and the primary
    biological target B:

        upper95( AMI(C, T) )  <  lower95( AMI(C, B) )

    Comparing point estimates would let a margin narrower than the uncertainty
    count as a margin. A DESIGN or BIOLOGICAL_NUISANCE covariate in the same
    position raises a warning and a stratified-analysis requirement rather than
    a refusal, because legitimate biology can correlate with ancestry, tissue or
    variant class.

    Fail-closed: an interval that cannot be computed is not separated. An
    unknown is not a pass.
    """

    biological_target_name: str
    biological_target_ami: MetricResult
    biological_target_ci95: tuple
    comparisons: tuple
    passed: bool
    refusal_reasons: tuple
    warnings: tuple

    def to_dict(self) -> dict:
        return {"biological_target_name": self.biological_target_name,
                "biological_target_ami": self.biological_target_ami.to_dict(),
                "biological_target_ci95": list(self.biological_target_ci95),
                "comparisons": [c.to_dict() for c in self.comparisons],
                "passed": self.passed,
                "refusal_reasons": list(self.refusal_reasons),
                "warnings": list(self.warnings)}


def decide_confounder_gate(target: AgreementEstimate,
                           covariates: Sequence[AgreementEstimate]) -> ConfounderGate:
    """Pure, total, randomness-free gate decision.

    Takes estimates, returns a verdict. No sampling, no permutation, no
    scikit-learn call. That is the point: the scientific policy can be tested
    with hand-written intervals, and a future configuration-driven policy can
    replace this function without touching any estimator.
    """
    if target.covariate_type is not CovariateType.TARGET:
        raise ValueError(
            f"the biological target {target.name!r} is classified "
            f"{target.covariate_type.value!r}; it must be "
            f"{CovariateType.TARGET.value!r}")
    t_lo = target.ci95[0]
    comparisons, refusals, warns = [], [], []

    for est in covariates:
        if est.covariate_type is CovariateType.TARGET:
            raise ValueError(
                f"covariate {est.name!r} is classified as TARGET; only the "
                "biological target may carry that type")
        c_hi = est.ci95[1]
        # Fail-closed: a non-finite bound is not separation.
        separated = bool(np.isfinite(c_hi) and np.isfinite(t_lo) and c_hi < t_lo)
        if separated:
            verdict = "pass"
        elif est.covariate_type is CovariateType.TECHNICAL:
            verdict = "refuse"
            refusals.append(
                f"technical covariate {est.name!r} is not separated from target "
                f"{target.name!r}: adjusted mutual information upper 95% bound "
                f"{c_hi:.4f} is not below target lower 95% bound {t_lo:.4f}")
        else:
            verdict = "warn"
            warns.append(
                f"{est.covariate_type.value} covariate {est.name!r} is not "
                f"separated from target {target.name!r} (upper 95% {c_hi:.4f} vs "
                f"target lower 95% {t_lo:.4f}); stratified analysis required")
        comparisons.append(ConfounderComparison(
            variable_name=est.name, variable_type=est.covariate_type,
            adjusted_mutual_information=MetricResult.ok(
                est.point_estimate, n_subsample_replicates=est.n_replicates),
            ci95=est.ci95, permutation=est.permutation,
            separated_from_target=separated, verdict=verdict))

    return ConfounderGate(
        biological_target_name=target.name,
        biological_target_ami=MetricResult.ok(
            target.point_estimate, n_subsample_replicates=target.n_replicates),
        biological_target_ci95=target.ci95,
        comparisons=tuple(comparisons),
        passed=not refusals,
        refusal_reasons=tuple(refusals),
        warnings=tuple(warns))


def ami_interval(clustering: np.ndarray, covariate: np.ndarray, *,
                 groups: Optional[np.ndarray] = None, n_replicates: int = 50,
                 subsample_fraction: float = 0.8, seed: int = 0) -> tuple:
    """Point estimate, 95% interval and usable-replicate count for adjusted
    mutual information.

    The interval comes from fixed-rate SUBSAMPLING WITHOUT REPLACEMENT, at the
    group level when groups are supplied. The ordinary bootstrap is unsuitable:
    observations repeat, duplicates overweight, and the denominator moves
    between replicates, so the resulting spread is not the sampling variability
    of the statistic on a fixed-size sample.
    """
    clu = np.asarray(clustering)
    cov = np.asarray(covariate)
    point = float(adjusted_mutual_info_score(cov, clu, average_method="arithmetic"))
    rng = np.random.default_rng(seed)
    vals = []
    if groups is None:
        n = len(clu)
        take = max(2, int(round(subsample_fraction * n)))
        for _ in range(n_replicates):
            idx = rng.choice(n, size=take, replace=False)
            if len(np.unique(clu[idx])) < 2 or len(np.unique(cov[idx])) < 2:
                continue
            vals.append(adjusted_mutual_info_score(cov[idx], clu[idx],
                                                   average_method="arithmetic"))
    else:
        g = np.asarray(groups)
        uniq = np.unique(g)
        take = max(2, int(round(subsample_fraction * len(uniq))))
        for _ in range(n_replicates):
            keep = rng.choice(uniq, size=take, replace=False)
            idx = np.flatnonzero(np.isin(g, keep))
            if len(np.unique(clu[idx])) < 2 or len(np.unique(cov[idx])) < 2:
                continue
            vals.append(adjusted_mutual_info_score(cov[idx], clu[idx],
                                                   average_method="arithmetic"))
    if len(vals) < 2:
        return point, (float("nan"), float("nan")), len(vals)
    lo, hi = np.quantile(np.asarray(vals), [0.025, 0.975])
    return point, (float(lo), float(hi)), len(vals)


def evaluate_confounder_gate(
    clustering: np.ndarray,
    *,
    biological_target: np.ndarray,
    biological_target_name: str,
    covariates: dict,
    covariate_types: dict,
    groups: Optional[np.ndarray] = None,
    n_permutations: int = 200,
    n_subsamples: int = 50,
    subsample_fraction: float = 0.8,
    seed: int = 0,
) -> ConfounderGate:
    """Measure the agreement estimates, then call decide_confounder_gate().

    This is the convenience layer. All policy lives in decide_confounder_gate();
    everything here is measurement, so a policy change cannot be made by
    accident while editing an estimator, and vice versa.

    `covariate_types` maps each covariate name to a CovariateType (or its string
    value). Every covariate must be classified; there is no default.
    """
    missing = set(covariates) - set(covariate_types)
    if missing:
        raise ValueError(
            f"no covariate type declared for {sorted(missing)}. Every covariate "
            "must be explicitly classified before the gate can decide whether "
            "exceeding the target is a refusal or a warning.")
    if not _SKLEARN:
        r = MetricResult.not_ok(MetricStatus.DEPENDENCY_UNAVAILABLE,
                                REASON_SKLEARN_MISSING)
        return ConfounderGate(biological_target_name, r, (float("nan"),) * 2,
                              (), False, (REASON_SKLEARN_MISSING,), ())

    clu = np.asarray(clustering)
    t_point, t_ci, t_n = ami_interval(
        clu, np.asarray(biological_target), groups=groups,
        n_replicates=n_subsamples, subsample_fraction=subsample_fraction, seed=seed)
    target = AgreementEstimate(biological_target_name, CovariateType.TARGET,
                               t_point, t_ci, t_n, None)

    estimates = []
    for i, (name, values) in enumerate(sorted(covariates.items())):
        v = np.asarray(values)
        if len(v) != len(clu):
            raise ValueError(
                f"covariate {name!r} has {len(v)} entries, clustering has {len(clu)}")
        ctype = covariate_types[name]
        ctype = ctype if isinstance(ctype, CovariateType) else CovariateType(ctype)
        point, ci, n_rep = ami_interval(
            clu, v, groups=groups, n_replicates=n_subsamples,
            subsample_fraction=subsample_fraction, seed=seed + 1 + i)
        perm = permutation_null_ami(clu, v, groups=groups,
                                    n_permutations=n_permutations, seed=seed + 1 + i)
        estimates.append(AgreementEstimate(name, ctype, point, ci, n_rep, perm))

    return decide_confounder_gate(target, estimates)
