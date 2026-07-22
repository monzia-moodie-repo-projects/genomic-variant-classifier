"""Panel R stage one: the collapse a healthy-looking cluster panel would miss.

WHY THIS FILE EXISTS
====================
Cluster metrics can rate a degenerate representation as EXCELLENT. Measured on
2026-07-21 against this repository's own Panel Q, using two 128-dimensional
two-class representations of 600 rows each -- one separating classes angularly,
one forcing every vector into a narrow cone and encoding the class only in the
vector norm:

    Davies-Bouldin euclidean centroid     3.934 -> 1.131     3.5x BETTER
    Calinski-Harabasz                    38.49  -> 371.6     9.7x BETTER
    silhouette (euclidean, sampled)       0.057 -> 0.404     7.1x BETTER
    Davies-Bouldin spherical cosine       1.324 -> 187.98  142x WORSE

Three of the four rate the DEGENERATE representation as substantially better,
because radial separation is precisely what they reward.

test_panel_q_rates_the_collapsed_representation_as_better below RE-MEASURES
this every run. It is the load-bearing test in the file: if it ever stops
holding, the premise for Panel R has changed and that must surface as a failure
rather than as a quietly obsolete comment.

THE CORRECTION IT ENCODES
-------------------------
The specification listed Davies-Bouldin among the metrics a cone collapse can
pass. That is true of the Euclidean variant and FALSE of the spherical one,
which already exists at clustering_metrics.py:487 and screams. So Panel R's
warrant is not "nothing detects this" -- it is that ONE METRIC ALARMS AND
NOTHING DIAGNOSES. A spherical Davies-Bouldin of 188 cannot say whether the
information was destroyed or merely badly arranged.

WHAT STAGE ONE DELIBERATELY DOES NOT CLAIM
------------------------------------------
R1 and R2 are pure functions of a matrix. R3 through R7 need a stored
representation, and models/gnn.py:357 computes focal_embeddings then returns
only classifier(focal_embeddings). Those stages are registered as capability
evidence with NOT_IMPLEMENTED / ABSENT rather than stubbed, and the tests here
assert that NONE of them can satisfy a release gate.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.capabilities import (
    CapabilityState,
    MetricStatus,
    TargetState,
    release_gate_satisfied,
)
from genomic_variant_classifier.evaluation.clustering_metrics import (
    calinski_harabasz,
    davies_bouldin_euclidean_centroid,
    davies_bouldin_spherical_cosine,
    estimate_silhouette,
)
from genomic_variant_classifier.evaluation.representation_geometry import (
    COLLAPSE_COMMON_DIRECTION,
    COLLAPSE_CONE_UNRESOLVED,
    COLLAPSE_EXPLOSION,
    COLLAPSE_HEALTHY,
    COLLAPSE_UNDETERMINED,
    CONE_MEAN_RESULTANT_LENGTH_THRESHOLD,
    GeometrySummary,
    THIN_NORMALIZED_EFFECTIVE_RANK_THRESHOLD,
    classify_collapse,
    mean_resultant_length,
    norm_statistics,
    panel_r_capabilities,
    spectral_geometry,
    summarize_geometry,
)

D = 128
N = 600
SEED = 0


def _labels():
    return np.r_[np.zeros(N // 2), np.ones(N // 2)].astype(int)


@pytest.fixture(scope="module")
def healthy():
    """Isotropic, classes separated ANGULARLY along two different axes."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=(N // 2, D)); a[:, 0] += 4.0
    b = rng.normal(size=(N // 2, D)); b[:, 1] += 4.0
    return np.vstack([a, b])


@pytest.fixture(scope="module")
def collapsed():
    """A narrow cone. Every vector points near +e0; the class lives in the NORM."""
    rng = np.random.default_rng(SEED)
    cone = rng.normal(size=(N, D)) * 0.05
    cone[:, 0] += 1.0
    cone /= np.linalg.norm(cone, axis=1, keepdims=True)
    return cone * np.where(_labels() == 0, 1.0, 3.0)[:, None]


@pytest.fixture(scope="module")
def common_offset():
    """A large shared offset with the spectrum intact: linearly removable."""
    rng = np.random.default_rng(SEED)
    return rng.normal(size=(N, D)) + 30.0


# --------------------------------------------------------------------------- #
# 1. the premise, re-measured every run
# --------------------------------------------------------------------------- #
def test_panel_q_rates_the_collapsed_representation_as_better(healthy, collapsed):
    """THE LOAD-BEARING TEST. If this stops holding, Panel R's premise changed."""
    y = _labels()
    db_h = davies_bouldin_euclidean_centroid(healthy, y)
    db_c = davies_bouldin_euclidean_centroid(collapsed, y)
    ch_h = calinski_harabasz(healthy, y)
    ch_c = calinski_harabasz(collapsed, y)
    assert db_c.value < db_h.value, "Davies-Bouldin euclidean should look BETTER"
    assert ch_c.value > ch_h.value, "Calinski-Harabasz should look BETTER"

    s_h = estimate_silhouette(healthy, y, seed=SEED)
    s_c = estimate_silhouette(collapsed, y, seed=SEED)
    assert s_c.estimate > s_h.estimate, "euclidean silhouette should look BETTER"


def test_the_spherical_davies_bouldin_does_fire(healthy, collapsed):
    """The correction to the specification: this one is NOT fooled."""
    y = _labels()
    sph_h = davies_bouldin_spherical_cosine(healthy, y)
    sph_c = davies_bouldin_spherical_cosine(collapsed, y)
    assert sph_c.value > 10 * sph_h.value, (
        "the spherical variant already alarms; Panel R's warrant is that nothing "
        "DIAGNOSES, not that nothing detects")


def test_panel_r_separates_what_panel_q_confuses(healthy, collapsed):
    h = summarize_geometry(healthy, representation_name="r", partition_role="STRUCTURE")
    c = summarize_geometry(collapsed, representation_name="r", partition_role="STRUCTURE")
    assert h.collapse_status == COLLAPSE_HEALTHY
    assert c.collapse_status == COLLAPSE_CONE_UNRESOLVED


# --------------------------------------------------------------------------- #
# 2. R1 -- directional anisotropy
# --------------------------------------------------------------------------- #
def test_mean_resultant_length_is_near_one_for_a_single_direction():
    x = np.tile(np.r_[1.0, np.zeros(D - 1)], (32, 1))
    r = mean_resultant_length(x)
    assert r.status is MetricStatus.OK
    assert r.value == pytest.approx(1.0, abs=1e-9)


def test_mean_resultant_length_is_near_zero_for_dispersed_directions():
    rng = np.random.default_rng(SEED)
    r = mean_resultant_length(rng.normal(size=(4000, 8)))
    assert r.status is MetricStatus.OK
    assert r.value < 0.1


def test_mean_resultant_length_is_scale_invariant():
    """It is a statistic of DIRECTION. Rescaling every row must not move it."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(200, 16)) + 1.0
    scaled = x * rng.uniform(0.1, 10.0, size=(200, 1))
    assert mean_resultant_length(x).value == pytest.approx(
        mean_resultant_length(scaled).value, abs=1e-9)


def test_centering_reveals_a_common_direction(common_offset):
    raw = mean_resultant_length(common_offset, center=False)
    cen = mean_resultant_length(common_offset, center=True)
    assert raw.value > 0.9, "a large shared offset should look anisotropic raw"
    assert cen.value < 0.2, "and isotropic once centred"


def test_zero_norm_rows_are_counted_not_silently_dropped():
    x = np.zeros((32, 8)); x[:16, 0] = 1.0
    r = mean_resultant_length(x)
    assert r.status is MetricStatus.OK
    assert r.metadata["n_zero_norm"] == 16
    assert r.metadata["n_used"] == 16


def test_all_zero_norm_is_insufficient_support_not_zero():
    r = mean_resultant_length(np.zeros((32, 8)))
    assert r.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert r.reason


def test_the_refusal_branch_still_reports_how_many_rows_were_zero():
    """FOUND BY A SABOTAGE THAT FAILED TO FAIL, 2026-07-21.

    Mutating `n_zero_norm` in the INSUFFICIENT_SUPPORT branch left the suite
    green, because every existing assertion exercised the OK branch instead. A
    refusal still has to say HOW BAD it was: "31 of 32 rows had zero norm" and
    "1 of 32" are the same status and very different findings, and a caller
    triaging a failed run needs the number."""
    x = np.zeros((32, 8))
    x[0, 0] = 1.0                      # exactly one usable row: still < 2
    r = mean_resultant_length(x)
    assert r.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert r.metadata["n_zero_norm"] == 31
    assert r.metadata["n_rows"] == 32


def test_non_finite_values_are_reported_not_filtered():
    """A representation explosion PRODUCES non-finite values. Dropping them
    would hide the pathology the panel exists to catch."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(64, 8)); x[3, 2] = np.inf
    r = mean_resultant_length(x)
    assert r.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert "nonfinite" in r.reason


# --------------------------------------------------------------------------- #
# 3. R2 -- rank and spectral utilisation
# --------------------------------------------------------------------------- #
def test_effective_rank_is_near_the_dimension_for_isotropic_data():
    rng = np.random.default_rng(SEED)
    s = spectral_geometry(rng.normal(size=(4000, 16)))
    assert s["effective_rank"].status is MetricStatus.OK
    assert s["normalized_effective_rank"].value > 0.9


def test_effective_rank_collapses_for_a_rank_one_representation():
    rng = np.random.default_rng(SEED)
    direction = rng.normal(size=(1, 32))
    x = rng.normal(size=(500, 1)) * direction
    s = spectral_geometry(x)
    assert s["effective_rank"].value < 1.5
    assert s["top1_variance_fraction"].value > 0.99


def test_participation_ratio_also_collapses():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(500, 1)) * rng.normal(size=(1, 32))
    assert spectral_geometry(x)["participation_ratio"].value < 1.5


def test_below_the_absolute_row_floor_is_refused():
    """Fewer than MINIMUM_ROWS observations: nothing is estimable."""
    rng = np.random.default_rng(SEED)
    s = spectral_geometry(rng.normal(size=(4, D)))
    assert s["effective_rank"].status is MetricStatus.INSUFFICIENT_SUPPORT
    assert s["effective_rank"].reason == "too_few_observations"


def test_too_few_rows_for_the_dimension_is_refused():
    """100 rows clears the absolute floor of 8 but not the 2x-per-dimension one
    for 128 dimensions, so the sample covariance is rank-deficient BY
    CONSTRUCTION. Reporting "effective rank 40" here would describe the SAMPLE
    SIZE, not the model -- which is the kind of number that looks like evidence
    and is not.

    This test previously used 4 rows and therefore never reached this branch at
    all: the absolute floor fired first and the assertion passed on the wrong
    reason. Caught 2026-07-21 by the reason string not matching."""
    rng = np.random.default_rng(SEED)
    n = 100
    assert 8 <= n < 2 * D, "the fixture must clear one floor and fail the other"
    s = spectral_geometry(rng.normal(size=(n, D)))
    assert s["effective_rank"].status is MetricStatus.INSUFFICIENT_SUPPORT
    assert "rank-deficient" in s["effective_rank"].reason
    assert s["normalized_effective_rank"].status is MetricStatus.INSUFFICIENT_SUPPORT
    assert s["condition_number"].status is MetricStatus.INSUFFICIENT_SUPPORT


def test_just_above_the_per_dimension_floor_is_computed():
    """The complement: the guard must not refuse everything. 2x + margin works."""
    rng = np.random.default_rng(SEED)
    s = spectral_geometry(rng.normal(size=(2 * 16 + 4, 16)))
    assert s["effective_rank"].status is MetricStatus.OK


def test_top_k_beyond_the_spectrum_is_not_applicable_not_one():
    rng = np.random.default_rng(SEED)
    s = spectral_geometry(rng.normal(size=(200, 4)))
    assert s["top50_variance_fraction"].status is MetricStatus.NOT_APPLICABLE


def test_spectral_entropy_and_effective_rank_agree():
    rng = np.random.default_rng(SEED)
    s = spectral_geometry(rng.normal(size=(600, 12)))
    assert s["effective_rank"].value == pytest.approx(
        float(np.exp(s["spectral_entropy"].value)), rel=1e-9)


# --------------------------------------------------------------------------- #
# 4. norm statistics
# --------------------------------------------------------------------------- #
def test_norm_cv_is_undefined_at_zero_mean_norm_not_nan():
    stats = norm_statistics(np.zeros((32, 8)))
    assert stats["norm_cv"].status is MetricStatus.UNDEFINED
    assert stats["norm_cv"].reason


def test_norm_statistics_track_the_migrated_signal(collapsed):
    stats = norm_statistics(collapsed)
    assert stats["mean_norm"].status is MetricStatus.OK
    assert stats["norm_cv"].value > 0.3, (
        "the class signal lives in the norm here, so the norm must be dispersed")


# --------------------------------------------------------------------------- #
# 5. the five outcomes are distinct
# --------------------------------------------------------------------------- #
def test_healthy_is_healthy(healthy):
    s = summarize_geometry(healthy, representation_name="r", partition_role="S")
    assert s.collapse_status == COLLAPSE_HEALTHY
    assert s.reasons == ()


def test_a_common_offset_is_named_as_linearly_removable(common_offset):
    s = summarize_geometry(common_offset, representation_name="r", partition_role="S")
    assert s.collapse_status == COLLAPSE_COMMON_DIRECTION
    assert any("linearly removable" in r for r in s.reasons)


def test_a_detected_collapse_is_not_the_same_word_as_could_not_evaluate(collapsed):
    """UNDETERMINED means the panel could not look. CONE_UNRESOLVED means it
    looked and found something it cannot fully classify. Collapsing the two
    repeats the drift monitor's UNKNOWN-reported-as-none defect."""
    found = summarize_geometry(collapsed, representation_name="r", partition_role="S")
    rng = np.random.default_rng(SEED)
    blind = summarize_geometry(rng.normal(size=(4, D)),
                               representation_name="r", partition_role="S")
    assert found.collapse_status == COLLAPSE_CONE_UNRESOLVED
    assert blind.collapse_status == COLLAPSE_UNDETERMINED
    assert found.collapse_status != blind.collapse_status


def test_non_finite_input_is_undetermined_never_healthy():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N, D)); x[0, 0] = np.nan
    s = summarize_geometry(x, representation_name="r", partition_role="S")
    assert s.collapse_status == COLLAPSE_UNDETERMINED
    assert s.collapse_status != COLLAPSE_HEALTHY


def test_an_unevaluable_representation_never_reports_healthy():
    for bad in (np.zeros((0, D)), np.zeros((3, D)), np.zeros((64, 0))):
        s = summarize_geometry(bad, representation_name="r", partition_role="S")
        assert s.collapse_status != COLLAPSE_HEALTHY


def test_the_collapse_reason_names_the_missing_stage(collapsed):
    s = summarize_geometry(collapsed, representation_name="r", partition_role="S")
    assert any("R3" in r for r in s.reasons), (
        "the report must say WHY it cannot classify the mechanism")


# --------------------------------------------------------------------------- #
# 6. thresholds are live, not decorative
# --------------------------------------------------------------------------- #
def test_thresholds_are_overridable_and_actually_used(collapsed):
    strict = summarize_geometry(collapsed, representation_name="r",
                                partition_role="S")
    lenient = summarize_geometry(collapsed, representation_name="r",
                                 partition_role="S",
                                 cone_threshold=0.99, thin_threshold=0.05)
    assert strict.collapse_status == COLLAPSE_CONE_UNRESOLVED
    assert lenient.collapse_status == COLLAPSE_HEALTHY


def test_the_documented_thresholds_are_the_defaults():
    assert CONE_MEAN_RESULTANT_LENGTH_THRESHOLD == 0.5
    assert THIN_NORMALIZED_EFFECTIVE_RANK_THRESHOLD == 0.5


def test_classify_collapse_refuses_without_the_metrics_it_needs():
    assert classify_collapse({})[0] == COLLAPSE_UNDETERMINED
    assert classify_collapse({})[1], "an unevaluable panel must say why"


# --------------------------------------------------------------------------- #
# 7. the unbuilt stages are declared, not stubbed
# --------------------------------------------------------------------------- #
def test_all_five_stages_are_registered():
    caps = panel_r_capabilities()
    assert len(caps) == 5
    assert {c.capability_name for c in caps} == {
        "panel_r3_norm_angle_decomposition",
        "panel_r4_conditioning_recoverability",
        "panel_r5_hubness_local_geometry",
        "panel_r6_training_trajectory",
        "panel_r7_downstream_sensitivity",
    }


def test_no_unbuilt_stage_can_satisfy_a_release_gate():
    assert not any(release_gate_satisfied(c) for c in panel_r_capabilities())


# The Panel R stages sit at different rungs, and the tests assert each one where
# it actually is -- updated as probes land.
#   R3: OUTPUT_AVAILABLE  -- norm_angle_probe.py produces its decomposition and
#       recovery delta (this commit).
#   R4, R5: IMPLEMENTED_NO_OUTPUT -- the extraction boundary gave them a stored
#       representation, but their own probes are not built yet.
#   R6, R7: NOT_IMPLEMENTED -- R6 needs a checkpoint series, R7 longitudinal
#       outcomes; neither exists.
# Updated 2026-07-21 in the same commit as the R3 advance.
_OUTPUT_AVAILABLE = {"panel_r3_norm_angle_decomposition"}
_IMPLEMENTED_NO_OUTPUT = {
    "panel_r4_conditioning_recoverability",
    "panel_r5_hubness_local_geometry",
}
_STILL_BLOCKED = {
    "panel_r6_training_trajectory",
    "panel_r7_downstream_sensitivity",
}


def test_r3_reached_output_available():
    caps = {c.capability_name: c for c in panel_r_capabilities()}
    c = caps["panel_r3_norm_angle_decomposition"]
    assert c.capability_state is CapabilityState.OUTPUT_AVAILABLE, (
        "R3's probe protocol (norm_angle_probe.py) exists, so R3 advanced")
    # OUTPUT_AVAILABLE, not VALIDATED: output exists, unverified. The contract
    # forbids MetricStatus.OK outside VALIDATED, so the status is honestly
    # INSUFFICIENT_SUPPORT and there is no named artifact yet.
    assert c.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert c.output_artifact is None
    assert c.reason


def test_r4_r5_remain_implemented_no_output():
    caps = {c.capability_name: c for c in panel_r_capabilities()}
    for name in _IMPLEMENTED_NO_OUTPUT:
        assert caps[name].capability_state is CapabilityState.IMPLEMENTED_NO_OUTPUT, (
            f"{name} has a stored representation but no probe of its own yet")
        assert caps[name].output_artifact is None
        assert caps[name].status is MetricStatus.INSUFFICIENT_SUPPORT
        assert caps[name].reason


def test_r6_r7_did_not_advance():
    caps = {c.capability_name: c for c in panel_r_capabilities()}
    for name in _STILL_BLOCKED:
        assert caps[name].capability_state is CapabilityState.NOT_IMPLEMENTED, (
            f"{name} needs inputs the extraction boundary does not provide "
            "(R6: checkpoint series; R7: longitudinal outcomes)")
        assert caps[name].status is MetricStatus.NOT_IMPLEMENTED


def test_every_stage_is_honest_about_its_state():
    for c in panel_r_capabilities():
        assert c.target_state is TargetState.ABSENT
        assert c.reason
        assert c.output_artifact is None


def test_advancing_a_rung_did_not_make_any_stage_releasable():
    # The load-bearing invariant across the advance: moving three stages up one
    # rung must NOT make them citable as done. IMPLEMENTED_NO_OUTPUT is still
    # a long way below VALIDATED.
    for c in panel_r_capabilities():
        assert not release_gate_satisfied(c)


def test_the_summary_serialises(healthy):
    s = summarize_geometry(healthy, representation_name="gnn_focal",
                           partition_role="STRUCTURE")
    d = s.to_dict()
    assert d["representation_name"] == "gnn_focal"
    assert d["partition_role"] == "STRUCTURE"
    assert d["dimension"] == D
    assert d["n_observations"] == N
    assert isinstance(d["reasons"], list)
    assert isinstance(s, GeometrySummary)
