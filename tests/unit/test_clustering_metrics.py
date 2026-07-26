"""Tests for Panel Q clustering and representation-structure metrics.

WHAT THESE PIN, AND WHY
=======================
The module's value is entirely in its refusals. Any of these metrics is three
lines of scikit-learn; what makes the module worth having is that it declines to
produce a number it cannot justify. So most of these tests assert that something
does NOT happen, and several assert against an independently computed answer
rather than against a golden constant.

Five properties, in order of what their absence would cost:

1. THE SILHOUETTE GUARD REFUSES BEFORE ALLOCATING. It is expressed in memory,
   not in a sample count, because memory is the constraint and a count silently
   assumes a dtype. A request above the ceiling must return in milliseconds
   having allocated nothing.

2. NO SILENT NOT-A-NUMBER. Enforced in __post_init__, so the raw constructor
   cannot build an unexplained failure -- an earlier draft enforced it only in
   the factory, leaving one of two construction paths open.

3. EXCLUSION ORDER, not merely exclusion arithmetic. An observation qualifying
   for two exclusion routes counts once, in the earliest. A reconciliation test
   passes whether or not the order is right, because the total is unchanged;
   only a membership test detects a reordering.

4. THE TWO DAVIES-BOULDIN GEOMETRIES ARE DIFFERENT ESTIMANDS. Checked against
   independent implementations written in this file rather than against golden
   numbers: a constant catches drift, an independent implementation catches
   wrongness. This is the pattern that caught the vectorised-expansion defect in
   conformal/ordinal.py.

5. THE GATE IS PURE AND FAIL-CLOSED. decide_confounder_gate takes estimates and
   returns a verdict with no randomness, so the scientific policy is testable as
   arithmetic with hand-written intervals. An interval that cannot be computed
   is not separation.

Placement: tests/unit/test_clustering_metrics.py
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.clustering_metrics import (
    AgreementEstimate,
    ClusteringPopulationAccounting,
    CovariateType,
    EstimatedMetric,
    MetricResult,
    MetricStatus,
    REASON_FEWER_THAN_TWO_CLUSTERS,
    REASON_INSUFFICIENT_OVERLAP,
    REASON_MEMORY_LIMIT_EXCEEDED,
    REASON_NON_OK_RESULTS_PRESENT,
    REASON_NO_RESULTS,
    account_for_population,
    aggregate,
    ami_interval,
    calinski_harabasz,
    davies_bouldin_euclidean_centroid,
    davies_bouldin_spherical_cosine,
    decide_confounder_gate,
    distance_matrix_gib,
    estimate_silhouette,
    evaluate_confounder_gate,
    evaluate_partition_agreement,
    maximum_n_for_memory,
    permutation_null_ami,
    permute_covariate_by_gene_block,
    stratified_cluster_sample,
)


# --------------------------------------------------------------------------- #
# fixtures and independent reference implementations
# --------------------------------------------------------------------------- #
def directional_clusters(n_per=200, k=3, d=16, spread=0.45, seed=3):
    """Clusters defined by DIRECTION on the unit sphere, which is the regime
    where the Euclidean and spherical Davies-Bouldin indices must disagree."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    x, y = [], []
    for i, c in enumerate(centres):
        x.append(c + spread * rng.normal(size=(n_per, d)))
        y += [i] * n_per
    return np.vstack(x), np.asarray(y)


def reference_davies_bouldin_euclidean(x, labels):
    """Independent implementation, deliberately naive.

    Its only purpose is to disagree with the module if the module is wrong. It
    shares no code with scikit-learn's implementation beyond numpy.
    """
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    centroids = np.vstack([x[labels == c].mean(axis=0) for c in uniq])
    scatter = np.array([
        np.linalg.norm(x[labels == c] - centroids[i], axis=1).mean()
        for i, c in enumerate(uniq)])
    worst = []
    for i in range(len(uniq)):
        ratios = []
        for j in range(len(uniq)):
            if i == j:
                continue
            sep = np.linalg.norm(centroids[i] - centroids[j])
            ratios.append((scatter[i] + scatter[j]) / sep)
        worst.append(max(ratios))
    return float(np.mean(worst))


def reference_davies_bouldin_spherical(x, labels):
    """Independent implementation with SPHERICAL centroids -- the mean
    re-normalized to unit length, which keeps the computation on the sphere."""
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    uniq = np.unique(labels)
    cents, scat = [], []
    for c in uniq:
        m = unit[labels == c].mean(axis=0)
        m = m / np.linalg.norm(m)
        cents.append(m)
        scat.append(float((1.0 - unit[labels == c] @ m).mean()))
    worst = []
    for i in range(len(uniq)):
        ratios = []
        for j in range(len(uniq)):
            if i == j:
                continue
            sep = 1.0 - float(cents[i] @ cents[j])
            ratios.append((scat[i] + scat[j]) / sep)
        worst.append(max(ratios))
    return float(np.mean(worst))


def gene_structured_cohort(n_genes=120, per_gene=25, seed=7):
    rng = np.random.default_rng(seed)
    genes = np.repeat([f"GENE{i:03d}" for i in range(n_genes)], per_gene)
    laboratory = np.repeat(rng.integers(0, 4, size=n_genes), per_gene)
    mechanism = np.repeat(rng.integers(0, 3, size=n_genes), per_gene)
    return genes, laboratory, mechanism


# --------------------------------------------------------------------------- #
# 1. MetricResult -- no silent not-a-number
# --------------------------------------------------------------------------- #
def test_non_ok_without_a_reason_is_rejected_by_the_raw_constructor():
    """The invariant must hold on BOTH construction paths. An earlier draft
    enforced it only in not_ok(), leaving the dataclass constructor open."""
    with pytest.raises(ValueError, match="requires a nonempty reason"):
        MetricResult(float("nan"), MetricStatus.UNDEFINED, None)
    with pytest.raises(ValueError, match="requires a nonempty reason"):
        MetricResult(float("nan"), MetricStatus.UNDEFINED, "")


def test_non_ok_carrying_a_finite_value_is_rejected():
    """A failed result holding a real number invites it being used."""
    with pytest.raises(ValueError, match="must carry NaN"):
        MetricResult(1.0, MetricStatus.UNDEFINED, "some_reason")


def test_ok_must_be_finite_and_reasonless():
    with pytest.raises(ValueError, match="must not carry a reason"):
        MetricResult(1.0, MetricStatus.OK, "why")
    with pytest.raises(ValueError, match="must carry a finite value"):
        MetricResult(float("nan"), MetricStatus.OK, None)


def test_status_must_be_a_metric_status():
    with pytest.raises(TypeError, match="must be a MetricStatus"):
        MetricResult(1.0, "ok", None)


def test_not_ok_cannot_construct_an_ok_result():
    with pytest.raises(ValueError, match="cannot construct an OK result"):
        MetricResult.not_ok(MetricStatus.OK, "reason")


def test_metric_result_round_trips_through_a_dict():
    a = MetricResult.ok(0.75, geometry="euclidean_centroid")
    assert MetricResult.from_dict(a.to_dict()) == a
    b = MetricResult.not_ok(MetricStatus.UNDEFINED, REASON_FEWER_THAN_TWO_CLUSTERS,
                            n_clusters=1)
    c = MetricResult.from_dict(b.to_dict())
    assert c.status is MetricStatus.UNDEFINED
    assert c.reason == REASON_FEWER_THAN_TWO_CLUSTERS
    assert math.isnan(c.value)
    assert c.metadata == {"n_clusters": 1}


def test_round_trip_survives_a_null_value():
    """Not-a-number does not survive strict JSON. A null must read back as NaN
    rather than raise, or a manifest written by one process cannot be read by
    another."""
    d = MetricResult.not_ok(MetricStatus.FAILED, "boom").to_dict()
    d["value"] = None
    assert math.isnan(MetricResult.from_dict(d).value)


# --------------------------------------------------------------------------- #
# 2. aggregate -- refuses to average across failures
# --------------------------------------------------------------------------- #
def test_aggregate_refuses_when_any_result_is_not_ok():
    """numpy.nanmean would silently return the mean of whatever worked."""
    out = aggregate([MetricResult.ok(1.0),
                     MetricResult.not_ok(MetricStatus.UNDEFINED, "no_clusters")])
    assert out.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert out.reason == REASON_NON_OK_RESULTS_PRESENT
    assert out.metadata["n_not_ok"] == 1
    assert out.metadata["n_total"] == 2


def test_aggregate_reports_the_full_status_census_including_ok():
    """Counting only the failures tells the caller the size of the problem but
    not its shape."""
    out = aggregate([MetricResult.ok(1.0), MetricResult.ok(2.0),
                     MetricResult.not_ok(MetricStatus.UNDEFINED, "a"),
                     MetricResult.not_ok(MetricStatus.FAILED, "b")])
    assert out.metadata["status_counts"] == {"ok": 2, "undefined": 1, "failed": 1}


def test_aggregate_averages_when_all_are_ok():
    out = aggregate([MetricResult.ok(1.0), MetricResult.ok(3.0)])
    assert out.is_ok and out.value == pytest.approx(2.0)
    assert out.metadata["n_averaged"] == 2


def test_aggregate_with_override_skips_the_bad_ones():
    out = aggregate([MetricResult.ok(2.0), MetricResult.ok(4.0),
                     MetricResult.not_ok(MetricStatus.UNDEFINED, "x")],
                    allow_non_ok=True)
    assert out.is_ok and out.value == pytest.approx(3.0)
    assert out.metadata["n_skipped"] == 1


def test_aggregate_of_nothing_is_a_refusal_not_a_zero():
    out = aggregate([])
    assert out.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert out.reason == REASON_NO_RESULTS


# --------------------------------------------------------------------------- #
# 3. the silhouette guard -- memory, not sample count
# --------------------------------------------------------------------------- #
def test_maximum_n_is_derived_from_the_memory_limit():
    """floor(sqrt(limit_bytes / 8)). At 3.0 GiB this is 20,066, which is where
    the previous draft's magic 20,000 came from implicitly."""
    assert maximum_n_for_memory(3.0) == 20_066
    assert maximum_n_for_memory(8.0) == 32_768
    assert maximum_n_for_memory(1.0) == 11_585
    assert distance_matrix_gib(20_066) == pytest.approx(3.0, rel=1e-3)


def test_a_nonpositive_memory_limit_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        maximum_n_for_memory(0.0)


def test_silhouette_refuses_above_the_ceiling_without_allocating():
    """The whole point. Must return in milliseconds, not after a long
    allocation that fails -- a job dying twenty minutes into a cloud rental has
    paid real money to learn what arithmetic gives for free."""
    rng = np.random.default_rng(0)
    n = 50_000
    x = rng.normal(size=(n, 4))
    y = rng.integers(0, 6, size=n)
    t0 = time.perf_counter()
    out = estimate_silhouette(x, y, requested_sample_size=100_000,
                              maximum_distance_matrix_gib=3.0)
    elapsed = time.perf_counter() - t0
    assert out.status is MetricStatus.COMPUTATIONALLY_DEFERRED
    assert out.reason == REASON_MEMORY_LIMIT_EXCEEDED
    assert elapsed < 1.0, f"refusal took {elapsed:.2f}s; it should not have computed"


def test_the_deferral_reports_the_effective_size_not_the_requested_one():
    """effective = min(requested, n). Reporting the request would overstate the
    allocation and mislead a caller trying to choose a workable budget."""
    rng = np.random.default_rng(0)
    n = 50_000
    x = rng.normal(size=(n, 4))
    y = rng.integers(0, 6, size=n)
    out = estimate_silhouette(x, y, requested_sample_size=1_000_000,
                              maximum_distance_matrix_gib=3.0)
    assert out.metadata["effective_sample_size"] == n
    assert out.metadata["requested_sample_size"] == 1_000_000
    assert out.metadata["maximum_sample_size"] == 20_066
    assert out.metadata["estimated_gib"] == pytest.approx(distance_matrix_gib(n))
    assert out.metadata["limit_gib"] == 3.0


def test_the_reason_is_a_stable_token_and_the_numbers_live_in_metadata():
    """An earlier draft interpolated the numbers into the reason, coupling every
    consumer and every test to a float format."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30_000, 3))
    y = rng.integers(0, 4, size=30_000)
    out = estimate_silhouette(x, y, requested_sample_size=30_000,
                              maximum_distance_matrix_gib=0.5)
    assert out.reason == REASON_MEMORY_LIMIT_EXCEEDED
    assert not any(ch.isdigit() for ch in out.reason)
    assert out.metadata["estimated_gib"] > 0.5


def test_the_ceiling_boundary_is_exact():
    """max_n is accepted, max_n + 1 is refused. A guard whose boundary is
    untested is a guard whose boundary is unknown."""
    limit = 0.01
    max_n = maximum_n_for_memory(limit)
    rng = np.random.default_rng(2)
    n = max_n + 500
    x = rng.normal(size=(n, 3))
    y = rng.integers(0, 3, size=n)
    ok = estimate_silhouette(x, y, requested_sample_size=max_n, n_replicates=2,
                             minimum_per_cluster=5, maximum_distance_matrix_gib=limit)
    assert ok.status is MetricStatus.OK
    over = estimate_silhouette(x, y, requested_sample_size=max_n + 1, n_replicates=2,
                               minimum_per_cluster=5,
                               maximum_distance_matrix_gib=limit)
    assert over.status is MetricStatus.COMPUTATIONALLY_DEFERRED


def test_silhouette_within_the_ceiling_reports_its_estimation_provenance():
    """An estimate reported without its sample size, seeds and spread is
    indistinguishable from an exact computation."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4_000, 6))
    y = rng.integers(0, 5, size=4_000)
    out = estimate_silhouette(x, y, requested_sample_size=800, n_replicates=6,
                              seed=11, minimum_per_cluster=10)
    assert out.status is MetricStatus.OK
    assert out.sample_size == 800
    assert out.n_replicates == 6
    assert out.seeds == tuple(range(11, 17))
    assert out.sampling_fraction == pytest.approx(800 / 4_000)
    assert out.sampling_method == "stratified_by_cluster_without_replacement"
    assert out.standard_deviation >= 0.0
    assert out.ci95[0] <= out.estimate <= out.ci95[1]


def test_silhouette_is_deterministic_in_its_seed():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2_000, 5))
    y = rng.integers(0, 4, size=2_000)
    a = estimate_silhouette(x, y, requested_sample_size=500, n_replicates=4, seed=9)
    b = estimate_silhouette(x, y, requested_sample_size=500, n_replicates=4, seed=9)
    assert a.estimate == b.estimate and a.seeds == b.seeds


def test_silhouette_refuses_a_single_cluster():
    x = np.random.default_rng(5).normal(size=(100, 3))
    out = estimate_silhouette(x, np.zeros(100, dtype=int))
    assert out.status is MetricStatus.UNDEFINED
    assert out.reason == REASON_FEWER_THAN_TWO_CLUSTERS


def test_estimated_metric_also_requires_a_reason_when_not_ok():
    with pytest.raises(ValueError, match="requires a nonempty reason"):
        EstimatedMetric(float("nan"), float("nan"), (0.0, 0.0), 0, 0.0, 0, (), 0,
                        "euclidean", "stratified", MetricStatus.FAILED, None)


# --------------------------------------------------------------------------- #
# 4. the stratified sampler -- protects the estimand, not just the estimate
# --------------------------------------------------------------------------- #
def test_stratified_sample_keeps_every_cluster_present():
    """Uniform sampling can remove a small cluster entirely. That does not add
    noise to the estimate; it changes the ESTIMAND, because the metric is then
    computed over a different cluster count than the clustering produced, and
    the change is invisible in the reported number."""
    labels = np.array([0] * 5_000 + [1] * 5_000 + [2] * 30)
    idx = stratified_cluster_sample(labels, sample_size=500, seed=0,
                                    minimum_per_cluster=25)
    assert set(np.unique(labels[idx])) == {0, 1, 2}
    assert (labels[idx] == 2).sum() >= 25


def test_stratified_sample_returns_no_duplicates():
    labels = np.repeat([0, 1, 2, 3], 300)
    idx = stratified_cluster_sample(labels, sample_size=400, seed=1)
    assert len(idx) == len(np.unique(idx))


def test_stratified_sample_respects_the_budget():
    labels = np.repeat([0, 1, 2], 400)
    idx = stratified_cluster_sample(labels, sample_size=300, seed=2)
    assert len(idx) <= 300


def test_stratified_sample_is_deterministic_and_seed_sensitive():
    labels = np.repeat([0, 1, 2], 400)
    a = stratified_cluster_sample(labels, sample_size=200, seed=3)
    b = stratified_cluster_sample(labels, sample_size=200, seed=3)
    c = stratified_cluster_sample(labels, sample_size=200, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(np.sort(a), np.sort(c))


def test_stratified_sample_falls_back_when_the_floor_exceeds_the_budget():
    """Documented behaviour, not an error: a caller asking for fewer
    observations than clusters-times-floor has to lose something."""
    labels = np.repeat(np.arange(20), 50)
    idx = stratified_cluster_sample(labels, sample_size=40, seed=5,
                                    minimum_per_cluster=25)
    assert len(idx) <= 40
    assert len(np.unique(idx)) == len(idx)


def test_stratified_sample_needs_two_clusters():
    with pytest.raises(ValueError, match="at least two clusters"):
        stratified_cluster_sample(np.zeros(50, dtype=int), sample_size=10, seed=0)


# --------------------------------------------------------------------------- #
# 5. population accounting -- order, not just arithmetic
# --------------------------------------------------------------------------- #
def test_the_four_buckets_reconcile_to_the_input():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(100, 5))
    x[0:3] = np.nan
    labels = np.array([0] * 40 + [1] * 40 + [-1] * 15 + [7] * 5)
    acc, _, _ = account_for_population(x, labels, noise_label=-1,
                                       minimum_cluster_size=6)
    assert acc.n_input == 100
    assert (acc.n_nonfinite_excluded + acc.n_algorithm_noise
            + acc.n_small_cluster_excluded + acc.n_analyzed) == 100


def test_an_unreconciled_accounting_is_rejected_at_construction():
    with pytest.raises(ValueError, match="does not reconcile"):
        ClusteringPopulationAccounting(
            n_input=100, n_nonfinite_excluded=1, n_algorithm_noise=1,
            n_small_cluster_excluded=1, n_analyzed=1,
            nonfinite_exclusion_fraction=0.01, algorithm_noise_fraction=0.01,
            small_cluster_exclusion_fraction=0.01, analyzed_fraction=0.01,
            minimum_cluster_size=2, noise_label=-1)


def test_exclusion_order_is_fixed_so_a_row_is_never_counted_twice():
    """THE TEST THAT RECONCILIATION CANNOT REPLACE.

    Rows 0 and 1 are BOTH non-finite AND noise-labelled. They must appear once,
    in the non-finite bucket. If the order were reversed they would count as
    noise instead -- and the total would be identical, so the reconciliation
    test above would still pass.
    """
    x = np.zeros((20, 3))
    x[0:2] = np.nan
    labels = np.array([-1, -1] + [0] * 9 + [1] * 9)
    acc, _, _ = account_for_population(x, labels, noise_label=-1,
                                       minimum_cluster_size=2)
    assert acc.n_nonfinite_excluded == 2
    assert acc.n_algorithm_noise == 0, (
        "rows that are both non-finite and noise were counted twice, or counted "
        "in the wrong bucket; non-finite must take precedence")
    assert acc.n_analyzed == 18


def test_small_cluster_exclusions_are_not_hidden_inside_the_noise_fraction():
    """A single aggregate figure would report 0.15 and conceal the rest."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100, 4))
    x[0:3] = np.nan
    labels = np.array([0] * 40 + [1] * 40 + [-1] * 15 + [7] * 5)
    acc, _, _ = account_for_population(x, labels, noise_label=-1,
                                       minimum_cluster_size=6)
    assert acc.algorithm_noise_fraction == pytest.approx(0.15)
    assert acc.small_cluster_exclusion_fraction > 0
    assert acc.nonfinite_exclusion_fraction > 0
    assert acc.analyzed_fraction < 1.0


def test_accounting_returns_only_the_analyzed_rows():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(60, 3))
    labels = np.array([0] * 25 + [1] * 25 + [-1] * 10)
    acc, xa, la = account_for_population(x, labels, noise_label=-1)
    assert len(xa) == len(la) == acc.n_analyzed == 50
    assert -1 not in set(np.unique(la))


def test_noise_label_none_means_no_noise_bucket():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(30, 3))
    labels = np.array([0] * 15 + [1] * 15)
    acc, _, _ = account_for_population(x, labels, noise_label=None)
    assert acc.n_algorithm_noise == 0 and acc.n_analyzed == 30


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="zero rows"):
        account_for_population(np.zeros((0, 3)), np.array([], dtype=int))


# --------------------------------------------------------------------------- #
# 6. geometry -- checked against independent implementations
# --------------------------------------------------------------------------- #
def test_euclidean_davies_bouldin_matches_an_independent_implementation():
    """A golden constant catches drift; an independent implementation catches
    wrongness. This is the pattern that caught the vectorised-expansion defect
    in conformal/ordinal.py."""
    x, y = directional_clusters()
    got = davies_bouldin_euclidean_centroid(x, y)
    assert got.is_ok
    assert got.value == pytest.approx(reference_davies_bouldin_euclidean(x, y),
                                      rel=1e-9)


def test_spherical_davies_bouldin_matches_an_independent_implementation():
    x, y = directional_clusters()
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    got = davies_bouldin_spherical_cosine(xn, y)
    assert got.is_ok
    assert got.value == pytest.approx(reference_davies_bouldin_spherical(xn, y),
                                      rel=1e-9)


def test_normalising_then_using_euclidean_is_not_the_spherical_index():
    """The claim the two functions exist to make. The Euclidean mean of unit
    vectors has norm below one and leaves the sphere, so scatter and separation
    are measured in the ambient space."""
    x, y = directional_clusters()
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    euclid = davies_bouldin_euclidean_centroid(xn, y)
    spher = davies_bouldin_spherical_cosine(xn, y)
    assert euclid.is_ok and spher.is_ok
    assert abs(euclid.value - spher.value) > 0.5, (
        "the two geometries agreed; either the fixture is not directional or the "
        "spherical implementation has silently become the Euclidean one")
    for c in np.unique(y):
        assert np.linalg.norm(xn[y == c].mean(axis=0)) < 0.95, (
            "cluster mean is nearly unit-norm, so this fixture cannot demonstrate "
            "the distinction")


def test_each_geometry_records_which_one_it_used():
    """A value reported without its geometry is unreportable."""
    x, y = directional_clusters()
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    assert davies_bouldin_euclidean_centroid(xn, y).metadata["geometry"] == \
        "euclidean_centroid"
    assert davies_bouldin_spherical_cosine(xn, y).metadata["geometry"] == \
        "spherical_cosine"


def test_spherical_index_refuses_a_zero_norm_observation():
    x = np.vstack([np.eye(4)[0], np.eye(4)[1], np.zeros(4)])
    out = davies_bouldin_spherical_cosine(x, np.array([0, 1, 1]))
    assert out.status is MetricStatus.UNDEFINED
    assert out.metadata["n_zero_norm"] == 1


def test_spherical_index_refuses_an_antipodal_cluster():
    """Members pointing in opposite directions sum to zero and have no mean
    direction. Reported, not silently repaired."""
    x = np.vstack([np.eye(4)[0], -np.eye(4)[0], np.eye(4)[1], np.eye(4)[2]])
    out = davies_bouldin_spherical_cosine(x, np.array([0, 0, 1, 2]))
    assert out.status is MetricStatus.UNDEFINED
    assert "antipodal" in out.metadata["note"]


@pytest.mark.parametrize("fn", [davies_bouldin_euclidean_centroid,
                                davies_bouldin_spherical_cosine,
                                calinski_harabasz])
def test_every_internal_metric_refuses_a_single_cluster(fn):
    x, _ = directional_clusters(n_per=30, k=2)
    out = fn(x, np.zeros(len(x), dtype=int))
    assert out.status is MetricStatus.UNDEFINED
    assert out.reason == REASON_FEWER_THAN_TWO_CLUSTERS
    assert math.isnan(out.value)


def test_calinski_harabasz_prefers_the_better_separated_clustering():
    """Higher is better. A metric whose direction is untested could be inverted
    and nothing would notice."""
    rng = np.random.default_rng(11)
    tight = np.vstack([rng.normal(0, 0.1, size=(200, 4)),
                       rng.normal(8, 0.1, size=(200, 4))])
    loose = np.vstack([rng.normal(0, 3.0, size=(200, 4)),
                       rng.normal(1, 3.0, size=(200, 4))])
    y = np.array([0] * 200 + [1] * 200)
    assert calinski_harabasz(tight, y).value > calinski_harabasz(loose, y).value


def test_davies_bouldin_prefers_the_better_separated_clustering():
    """Lower is better -- the opposite direction from Calinski-Harabasz, which
    is exactly the kind of thing worth pinning."""
    rng = np.random.default_rng(12)
    tight = np.vstack([rng.normal(0, 0.1, size=(200, 4)),
                       rng.normal(8, 0.1, size=(200, 4))])
    loose = np.vstack([rng.normal(0, 3.0, size=(200, 4)),
                       rng.normal(1, 3.0, size=(200, 4))])
    y = np.array([0] * 200 + [1] * 200)
    assert (davies_bouldin_euclidean_centroid(tight, y).value
            < davies_bouldin_euclidean_centroid(loose, y).value)


# --------------------------------------------------------------------------- #
# 7. partition agreement -- including the asymmetry that is easy to get wrong
# --------------------------------------------------------------------------- #
def test_identical_partitions_agree_perfectly():
    labels = np.repeat([0, 1, 2], 40)
    p = evaluate_partition_agreement(labels, labels.copy())
    assert p.adjusted_rand_index.value == pytest.approx(1.0)
    assert p.adjusted_mutual_information.value == pytest.approx(1.0)
    assert p.homogeneity.value == pytest.approx(1.0)
    assert p.completeness.value == pytest.approx(1.0)
    assert p.v_measure.value == pytest.approx(1.0)


def test_homogeneity_and_completeness_are_not_interchangeable():
    """THE DOCUMENTED FOOTGUN, NOW GUARDED.

    A clustering that SPLITS each reference class into two is perfectly
    homogeneous (every cluster is pure) but incomplete (each class is scattered
    across two clusters). Swapping the arguments exchanges the two values and
    produces a plausible-looking, wrong answer -- which is exactly why this
    needs a test rather than a docstring.
    """
    reference = np.repeat([0, 1, 2], 40)
    refined = np.concatenate([np.repeat([0, 1], 20), np.repeat([2, 3], 20),
                              np.repeat([4, 5], 20)])
    right = evaluate_partition_agreement(reference, refined)
    assert right.homogeneity.value == pytest.approx(1.0)
    assert right.completeness.value < 0.9
    swapped = evaluate_partition_agreement(refined, reference)
    assert swapped.homogeneity.value == pytest.approx(right.completeness.value)
    assert swapped.completeness.value == pytest.approx(right.homogeneity.value)


def test_adjusted_mutual_information_tolerates_refinement_better_than_the_rand_index():
    """Why adjusted mutual information is primary for external agreement: a
    discovered partition may legitimately refine a reference one."""
    reference = np.repeat([0, 1, 2], 60)
    refined = np.concatenate([np.repeat([0, 1, 2], 20), np.repeat([3, 4, 5], 20),
                              np.repeat([6, 7, 8], 20)])
    p = evaluate_partition_agreement(reference, refined)
    assert p.adjusted_mutual_information.value > p.adjusted_rand_index.value


def test_normalized_mutual_information_is_flagged_as_secondary():
    labels = np.repeat([0, 1, 2], 40)
    p = evaluate_partition_agreement(labels, labels.copy())
    assert "never report alone" in p.normalized_mutual_information.metadata["note"]


def test_agreement_refuses_insufficient_overlap():
    labels = np.repeat([0, 1], 10)
    p = evaluate_partition_agreement(labels, labels.copy(), minimum_overlap=50)
    assert p.adjusted_rand_index.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert p.adjusted_rand_index.reason == REASON_INSUFFICIENT_OVERLAP
    assert p.adjusted_rand_index.metadata["n_shared"] == 20


def test_agreement_refuses_a_single_group_partition():
    ref = np.repeat([0, 1], 40)
    p = evaluate_partition_agreement(ref, np.zeros(80, dtype=int))
    assert p.adjusted_rand_index.status is MetricStatus.UNDEFINED


def test_agreement_rejects_mismatched_observation_counts():
    with pytest.raises(ValueError, match="different observation counts"):
        evaluate_partition_agreement(np.zeros(10, dtype=int), np.zeros(11, dtype=int))


# --------------------------------------------------------------------------- #
# 8. the permutation mechanism -- tested directly, not through a quantile
# --------------------------------------------------------------------------- #
def test_group_permutation_moves_whole_blocks():
    """THE FUNDAMENTAL MECHANISM. More reliable than asserting a null quantile,
    which depends on seeds, numpy and scikit-learn."""
    genes = np.repeat([f"G{i}" for i in range(8)], 5)
    covariate = np.repeat(np.arange(10, 90, 10), 5)
    out = permute_covariate_by_gene_block(covariate, genes, np.random.default_rng(1))
    for g in np.unique(genes):
        assert len(set(out[genes == g])) == 1, (
            f"gene {g} holds several covariate values after permutation; the "
            "permutation broke the block structure it exists to preserve")
    assert sorted(np.unique(out).tolist()) == sorted(np.unique(covariate).tolist())


def test_group_permutation_actually_permutes():
    genes = np.repeat([f"G{i}" for i in range(20)], 3)
    covariate = np.repeat(np.arange(20), 3)
    out = permute_covariate_by_gene_block(covariate, genes, np.random.default_rng(2))
    assert not np.array_equal(out, covariate)


def test_group_permutation_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="covariate has"):
        permute_covariate_by_gene_block(np.arange(10), np.arange(11),
                                   np.random.default_rng(0))


def test_the_gene_level_null_is_wider_than_the_row_level_null():
    """Variants within a gene share the covariate, so permuting rows barely
    disturbs the association and the null collapses toward zero. Measured on
    this fixture the gene-level 95th percentile was roughly forty times the
    row-level one; the assertion uses a wide margin so it pins the direction and
    order of magnitude rather than a fragile ratio."""
    genes, laboratory, _ = gene_structured_cohort()
    clustering = laboratory.copy()
    by_gene = permutation_null_ami(clustering, laboratory, groups=genes,
                                   n_permutations=120, seed=1)
    by_row = permutation_null_ami(clustering, laboratory, groups=None,
                                  n_permutations=120, seed=1)
    # "gene_block" since 2026-07-21. The previous value, "group", named only
    # that groups were involved -- not that whole blocks were exchanged as
    # units, which is the property the null rests on.
    assert by_gene["permutation_unit"] == "gene_block"
    assert by_gene["permutation_scheme"] == "gene_block"
    assert by_row["permutation_unit"] == "row"
    assert by_gene["null_p95"] > 5 * by_row["null_p95"]


def test_the_permutation_records_its_unit_and_group_count():
    genes, laboratory, _ = gene_structured_cohort(n_genes=40)
    out = permutation_null_ami(laboratory, laboratory, groups=genes,
                               n_permutations=30, seed=0)
    assert out["n_groups"] == 40
    assert out["n_groups_with_multiple_covariate_values"] == 0
    assert 0.0 < out["permutation_p_value"] <= 1.0


def test_the_permutation_is_deterministic_in_its_seed():
    genes, laboratory, _ = gene_structured_cohort(n_genes=30)
    a = permutation_null_ami(laboratory, laboratory, groups=genes,
                             n_permutations=25, seed=5)
    b = permutation_null_ami(laboratory, laboratory, groups=genes,
                             n_permutations=25, seed=5)
    assert a["null_p95"] == b["null_p95"]


# --------------------------------------------------------------------------- #
# 9. the gate -- pure policy, tested as arithmetic
# --------------------------------------------------------------------------- #
def _estimate(name, covariate_type, lo, hi, point=None):
    """Hand-written interval. No sampling, no seeds, no scikit-learn."""
    return AgreementEstimate(name, covariate_type,
                             (lo + hi) / 2 if point is None else point, (lo, hi))


def test_a_clean_separation_passes():
    gate = decide_confounder_gate(
        _estimate("mechanism", CovariateType.TARGET, 0.60, 0.70),
        [_estimate("laboratory", CovariateType.TECHNICAL, 0.05, 0.10)])
    assert gate.passed
    assert gate.comparisons[0].verdict == "pass"
    assert gate.refusal_reasons == ()


def test_a_technical_covariate_overlapping_the_target_refuses():
    gate = decide_confounder_gate(
        _estimate("mechanism", CovariateType.TARGET, 0.30, 0.50),
        [_estimate("laboratory", CovariateType.TECHNICAL, 0.40, 0.60)])
    assert not gate.passed
    assert gate.comparisons[0].verdict == "refuse"
    assert "laboratory" in gate.refusal_reasons[0]


def test_touching_intervals_are_not_separated():
    """Target lower bound 0.40, covariate upper bound 0.40. Strict inequality:
    a margin equal to zero is not a margin."""
    gate = decide_confounder_gate(
        _estimate("mechanism", CovariateType.TARGET, 0.40, 0.60),
        [_estimate("laboratory", CovariateType.TECHNICAL, 0.20, 0.40)])
    assert not gate.passed


def test_a_hair_of_separation_passes():
    """The other side of the boundary, so the test above pins strictness rather
    than an off-by-one."""
    gate = decide_confounder_gate(
        _estimate("mechanism", CovariateType.TARGET, 0.40, 0.60),
        [_estimate("laboratory", CovariateType.TECHNICAL, 0.20, 0.3999)])
    assert gate.passed


@pytest.mark.parametrize("ctype", [CovariateType.BIOLOGICAL_NUISANCE,
                                   CovariateType.DESIGN])
def test_non_technical_covariates_warn_rather_than_refuse(ctype):
    """Legitimate biology can correlate with ancestry, tissue or variant class,
    so these require stratified analysis rather than blocking the claim."""
    gate = decide_confounder_gate(
        _estimate("mechanism", CovariateType.TARGET, 0.30, 0.50),
        [_estimate("ancestry", ctype, 0.40, 0.60)])
    assert gate.passed
    assert gate.comparisons[0].verdict == "warn"
    assert len(gate.warnings) == 1


def test_an_uncomputable_interval_fails_closed():
    """An unknown is not a pass. This is the difference between a gate and a
    suggestion."""
    gate = decide_confounder_gate(
        _estimate("mechanism", CovariateType.TARGET, 0.30, 0.50),
        [_estimate("laboratory", CovariateType.TECHNICAL,
                   float("nan"), float("nan"), point=0.11)])
    assert not gate.passed
    assert gate.comparisons[0].separated_from_target is False


def test_an_uncomputable_target_interval_also_fails_closed():
    gate = decide_confounder_gate(
        _estimate("mechanism", CovariateType.TARGET,
                  float("nan"), float("nan"), point=0.40),
        [_estimate("laboratory", CovariateType.TECHNICAL, 0.01, 0.02)])
    assert not gate.passed


def test_a_non_finite_point_estimate_is_rejected_at_the_boundary():
    """An AgreementEstimate is a measurement. If adjusted mutual information
    could not be computed there is nothing to gate on, and the caller must
    report that as a non-OK MetricResult rather than smuggle a NaN into the
    decision -- where it would raise deep inside the gate instead."""
    with pytest.raises(ValueError, match="non-finite point estimate"):
        AgreementEstimate("x", CovariateType.TECHNICAL, float("nan"), (0.4, 0.6))


def test_an_unclassified_covariate_is_rejected():
    with pytest.raises(ValueError, match="not a CovariateType"):
        AgreementEstimate("x", None, 0.5, (0.4, 0.6))
    with pytest.raises(ValueError, match="not a CovariateType"):
        AgreementEstimate("x", "technical", 0.5, (0.4, 0.6))


def test_an_inverted_interval_is_rejected():
    with pytest.raises(ValueError, match="inverted interval"):
        AgreementEstimate("x", CovariateType.TECHNICAL, 0.5, (0.6, 0.4))


def test_the_target_must_be_typed_as_the_target():
    with pytest.raises(ValueError, match="it must be 'target'"):
        decide_confounder_gate(
            _estimate("mechanism", CovariateType.TECHNICAL, 0.6, 0.7), [])


def test_a_covariate_may_not_be_typed_as_the_target():
    with pytest.raises(ValueError, match="only the biological target"):
        decide_confounder_gate(
            _estimate("mechanism", CovariateType.TARGET, 0.6, 0.7),
            [_estimate("other", CovariateType.TARGET, 0.1, 0.2)])


def test_the_gate_serialises_without_a_custom_encoder():
    import json
    gate = decide_confounder_gate(
        _estimate("mechanism", CovariateType.TARGET, 0.60, 0.70),
        [_estimate("laboratory", CovariateType.TECHNICAL, 0.05, 0.10)])
    assert json.loads(json.dumps(gate.to_dict()))["passed"] is True


# --------------------------------------------------------------------------- #
# 10. the measuring layer -- end to end on gene-structured data
# --------------------------------------------------------------------------- #
def test_clusters_tracking_the_laboratory_are_refused():
    """The scenario the gate exists for: a solution that looks elegant and is
    entirely provenance."""
    genes, laboratory, mechanism = gene_structured_cohort()
    gate = evaluate_confounder_gate(
        laboratory.copy(), biological_target=mechanism,
        biological_target_name="mechanism",
        covariates={"laboratory": laboratory, "gene_identity": genes},
        covariate_types={"laboratory": CovariateType.TECHNICAL,
                         "gene_identity": CovariateType.TECHNICAL},
        groups=genes, n_permutations=40, n_subsamples=20, seed=0)
    assert not gate.passed
    named = " ".join(gate.refusal_reasons)
    assert "laboratory" in named and "gene_identity" in named


def test_clusters_tracking_biology_pass():
    genes, laboratory, mechanism = gene_structured_cohort()
    gate = evaluate_confounder_gate(
        mechanism.copy(), biological_target=mechanism,
        biological_target_name="mechanism",
        covariates={"laboratory": laboratory},
        covariate_types={"laboratory": CovariateType.TECHNICAL},
        groups=genes, n_permutations=40, n_subsamples=20, seed=0)
    assert gate.passed


def test_the_measuring_layer_demands_a_type_for_every_covariate():
    genes, laboratory, mechanism = gene_structured_cohort(n_genes=30)
    with pytest.raises(ValueError, match="no covariate type declared"):
        evaluate_confounder_gate(
            laboratory.copy(), biological_target=mechanism,
            biological_target_name="mechanism",
            covariates={"batch": laboratory}, covariate_types={}, groups=genes)


def test_a_misaligned_covariate_is_rejected():
    genes, laboratory, mechanism = gene_structured_cohort(n_genes=30)
    with pytest.raises(ValueError, match="entries, clustering has"):
        evaluate_confounder_gate(
            laboratory.copy(), biological_target=mechanism,
            biological_target_name="mechanism",
            covariates={"short": laboratory[:10]},
            covariate_types={"short": CovariateType.TECHNICAL}, groups=genes)


def test_ami_interval_returns_a_point_an_interval_and_a_replicate_count():
    genes, laboratory, _ = gene_structured_cohort(n_genes=60)
    point, ci, n_rep = ami_interval(laboratory, laboratory, groups=genes,
                                    n_replicates=15, seed=0)
    assert point == pytest.approx(1.0)
    assert ci[0] <= point <= ci[1] or math.isnan(ci[0])
    assert n_rep > 0
