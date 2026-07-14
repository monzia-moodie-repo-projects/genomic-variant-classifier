"""The aggregate-only drift reference profile must be EXACT, and must not lie when it can't be.

Created 2026-07-13 (roadmap 6.20).

WHAT THESE TESTS PROTECT
------------------------
The scheduled drift monitor runs from an AGGREGATE PROFILE -- histogram counts and quantile
grids, no variant rows -- rather than the raw 23.8 MB Run-15 feature matrix, which would have
to be fetched from cloud storage with credentials on every hosted run. The profile is 1.4 MB
and lives in git.

(An earlier draft of this docstring justified the profile by claiming the matrix holds
dbNSFP-derived columns that data_manifest.yaml marks `tier: controlled` / "LICENSED (paid)".
That was false -- dbNSFP is `tier: academic`; the "LICENSED (paid)" note belongs to hgmd, and
all four controlled-tier columns in the matrix are constant zero. The profile is still the
right design; the reason recorded here is now the true one. See drift_reference_profile.py.)

That substitution is only safe if two things are true, and both are asserted here:

  1. **The Population Stability Index is EXACT.** Not close. Not within tolerance. IDENTICAL,
     bit-for-bit, to what the raw-data detector computes -- because PSI is the number every
     drift decision is made from, and a profile that shifted it by 1e-6 would be a monitor
     that quietly disagrees with the science it is supposed to be guarding. Tested with `==`
     on the raw float, not `pytest.approx`.

  2. **What the profile CANNOT compute, it reports as NOT COMPUTED.** The Maximum Mean
     Discrepancy and Székely-Rizzo energy tests need reference samples and cannot run from
     aggregates. The temptation is to default `mmd_pvalue` to something benign. That would be
     invisible, would look exactly like a clean run, and would permanently disarm the
     `mmd_pvalue < 0.001` escalation -- while the report said everything was fine.

     That is not a hypothetical failure mode. It is THIS EXACT SUBSYSTEM'S HISTORY: for its
     whole life `drift_monitor.yml` reported "no drift" every month having never checked
     anything (roadmap 6.20). These tests exist so that cannot happen a second time by a
     different route.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.monitoring.drift_detector import (
    PSI_MONITOR,
    PSI_RETRAIN,
    DriftDetector,
)
from genomic_variant_classifier.monitoring.drift_reference_profile import (
    PROFILE_FORMAT_VERSION,
    DriftReferenceProfile,
)


# ---------------------------------------------------------------------------
# Fixtures -- deliberately nasty. Exactness that only holds on tidy Gaussians
# is exactness that will break on real genomic features.
# ---------------------------------------------------------------------------

@pytest.fixture
def X_ref() -> pd.DataFrame:
    rng = np.random.default_rng(20260713)
    n = 4000
    return pd.DataFrame({
        "gaussian":     rng.normal(0, 1, n),
        "skewed":       rng.exponential(2.0, n),          # long right tail
        "bimodal":      np.concatenate([rng.normal(-3, 0.5, n // 2),
                                        rng.normal(+3, 0.5, n - n // 2)]),
        "bounded_01":   rng.beta(2, 5, n),                # like a pathogenicity score
        "discrete":     rng.integers(0, 5, n).astype(float),
        "binary_flag":  (rng.random(n) > 0.7).astype(float),
        "constant":     np.full(n, 1.5),                  # degenerate: p01 == p99
        "with_nans":    np.where(rng.random(n) < 0.1, np.nan, rng.normal(5, 2, n)),
        "heavy_tail":   rng.standard_cauchy(n),           # percentile clipping really matters
    })


@pytest.fixture
def X_new(X_ref: pd.DataFrame) -> pd.DataFrame:
    """A genuinely drifted copy -- so the PSI values compared are non-trivial.

    Comparing a reference against ITSELF gives PSI ~= 0 everywhere, and an equality assertion
    on a column of zeros proves nothing at all.
    """
    rng = np.random.default_rng(99)
    X = X_ref.sample(n=2500, random_state=7).copy()
    X["gaussian"]   = X["gaussian"] + 1.2          # mean shift
    X["skewed"]     = X["skewed"] * 1.8            # scale shift
    X["bounded_01"] = np.clip(X["bounded_01"] + 0.25, 0, 1)
    X["discrete"]   = rng.integers(0, 5, len(X)).astype(float)
    return X


@pytest.fixture
def profile(X_ref: pd.DataFrame) -> DriftReferenceProfile:
    return DriftReferenceProfile.from_reference(X_ref, source="test-fixture")


# ---------------------------------------------------------------------------
# 1. THE CENTRAL CLAIM: PSI is EXACT, not approximate
# ---------------------------------------------------------------------------

def test_psi_from_profile_is_bit_identical_to_raw(X_ref, X_new, profile):
    """The whole design rests on this. `==`, not approx."""
    detector = DriftDetector.from_reference(X_ref=X_ref)

    checked_nonzero = 0
    for i, feat in enumerate(detector.feature_names):
        ref_col = detector.ref_data[:, i]
        ref_col = ref_col[np.isfinite(ref_col)]
        new_col = X_new[feat].to_numpy(dtype=np.float64)
        new_col = new_col[np.isfinite(new_col)]

        psi_raw  = detector._psi(ref_col, new_col)
        psi_prof = profile.psi(feat, new_col)

        assert psi_raw == psi_prof, (
            f"PSI DIVERGED on {feat!r}: raw={psi_raw!r} profile={psi_prof!r} "
            f"(delta {abs(psi_raw - psi_prof):.3e}). The profile is not a faithful stand-in "
            f"for the reference matrix, and every drift decision made from it is suspect."
        )
        if psi_raw > 0:
            checked_nonzero += 1

    # Guard the guard: if every PSI were 0.0 this test would pass while proving nothing.
    assert checked_nonzero >= 4, (
        f"only {checked_nonzero} features had non-zero PSI -- the fixtures are not actually "
        f"drifted, so the equality assertion above is vacuous."
    )


def test_psi_identical_after_save_and_reload(X_ref, X_new, profile, tmp_path):
    """JSON round-trip must not cost a single bit of float precision."""
    path = profile.save(tmp_path / "p.json")
    reloaded = DriftReferenceProfile.load(path)

    for feat in profile.feature_names:
        new_col = X_new[feat].to_numpy(dtype=np.float64)
        new_col = new_col[np.isfinite(new_col)]
        assert profile.psi(feat, new_col) == reloaded.psi(feat, new_col), (
            f"{feat}: PSI changed across a JSON round-trip."
        )


def test_degenerate_constant_feature_matches_raw(X_ref, profile):
    """p01 == p99 -> _psi returns 0.0. The profile must agree, not divide by zero."""
    detector = DriftDetector.from_reference(X_ref=X_ref)
    i = detector.feature_names.index("constant")
    ref_col = detector.ref_data[:, i]

    new_col = np.full(500, 1.5)
    assert detector._psi(ref_col, new_col) == profile.psi("constant", new_col) == 0.0


def test_per_feature_action_is_identical(X_ref, X_new, profile):
    """PSI drives the action. Identical PSI must therefore mean identical actions."""
    raw_report  = DriftDetector.from_reference(X_ref=X_ref).check(X_new)
    prof_report = DriftDetector.from_profile(profile).check(X_new)

    raw_actions  = {r.feature: (r.psi, r.action) for r in raw_report.feature_results}
    prof_actions = {r.feature: (r.psi, r.action) for r in prof_report.feature_results}

    assert raw_actions == prof_actions
    assert raw_report.features_drifted   == prof_report.features_drifted
    assert raw_report.features_monitored == prof_report.features_monitored


# ---------------------------------------------------------------------------
# 2. WHAT IT CANNOT DO, IT MUST SAY -- never a benign default
# ---------------------------------------------------------------------------

def test_joint_tests_are_reported_as_not_run_never_as_passing(X_new, profile):
    report = DriftDetector.from_profile(profile).check(X_new)

    assert report.joint_tests_run is False
    assert report.mmd_score is None
    assert report.mmd_pvalue is None
    assert report.energy_statistic is None
    assert report.energy_pvalue is None

    # A benign default is the specific catastrophe: `mmd_pvalue = 1.0` would render fine on
    # every dashboard, satisfy every downstream `< 0.001` comparison, and mean NOTHING.
    # Spell it out so nobody "fixes" the None away in six months to make a chart draw.
    assert report.mmd_pvalue != 1.0
    assert report.joint_tests_reason and "require samples" in report.joint_tests_reason
    assert "NOT COMPUTED" in report.summary


def test_missing_mmd_does_not_suppress_a_psi_driven_escalation(profile):
    """`mmd_pvalue is None` must not short-circuit the PSI triggers: drift must still escalate."""
    rng = np.random.default_rng(3)
    # Violent drift on many features -> n_retrain > 3 -> urgent, with no MMD at all.
    X = pd.DataFrame({f: rng.normal(50, 5, 800) for f in profile.feature_names})

    report = DriftDetector.from_profile(profile).check(X)

    assert report.joint_tests_run is False
    assert report.features_drifted > 3
    assert report.recommended_action == "urgent_retrain", (
        "PSI found severe drift but the action was not escalated -- a missing MMD p-value has "
        "silently disarmed the escalation path. This is the bug the profile exists to avoid."
    )
    assert report.action_required is True


def test_missing_mmd_does_not_fabricate_an_escalation(X_ref, profile):
    """The converse: no drift + no joint test must not INVENT a retrain.

    Asserted as `action_required is False` rather than `action == "none"` on purpose. The
    fixtures include a Cauchy column, and a resample of a heavy-tailed feature can cross
    PSI_MONITOR by luck alone -- a "monitor" here would be honest sampling noise, not a
    failure. What must NEVER happen is a *retrain* conjured out of an absent measurement, and
    that is exactly what this asserts.
    """
    report = DriftDetector.from_profile(profile).check(X_ref.sample(1500, random_state=1))
    assert report.joint_tests_run is False
    assert report.action_required is False
    assert report.recommended_action in ("none", "monitor")


def test_print_summary_survives_absent_joint_tests(X_new, profile, capsys):
    """print_summary formatted mmd_score with `:.6f`. On None that is a TypeError."""
    DriftDetector.from_profile(profile).check(X_new).print_summary()
    out = capsys.readouterr().out
    assert "NOT COMPUTED" in out
    assert "JOINT TESTS DID NOT RUN" in out


def test_ks_and_wasserstein_are_flagged_approximate_only_in_profile_mode(X_ref, X_new, profile):
    raw  = DriftDetector.from_reference(X_ref=X_ref).check(X_new)
    prof = DriftDetector.from_profile(profile).check(X_new)

    assert all(r.ks_wasserstein_approximate is False for r in raw.feature_results)
    assert all(r.ks_wasserstein_approximate is True for r in prof.feature_results)


def test_report_serialises_with_none_joint_tests(X_new, profile, tmp_path):
    """to_json must not choke on the Nones -- the workflow writes this file."""
    report = DriftDetector.from_profile(profile).check(X_new)
    out = tmp_path / "r.json"
    report.to_json(out)

    payload = json.loads(out.read_text())
    assert payload["mmd_pvalue"] is None
    assert payload["joint_tests_run"] is False
    assert payload["joint_tests_reason"]


# ---------------------------------------------------------------------------
# 3. THE PROFILE MUST NOT CARRY WHAT IT PROMISED NOT TO CARRY
# ---------------------------------------------------------------------------

def test_profile_contains_no_variant_level_rows(X_ref, profile, tmp_path):
    """The entire safety argument is 'it is a histogram'. Assert that structurally.

    Every stored array must be tiny and fixed-size -- bins and quantiles -- and nothing may
    scale with the number of reference rows. If a future edit ever tucked the raw column in
    "just for the Kolmogorov-Smirnov test", this fails.
    """
    payload = json.loads(profile.save(tmp_path / "p.json").read_text())
    n_rows = len(X_ref)

    for name, spec in payload["features"].items():
        assert len(spec["hist"]) == 10, f"{name}: histogram is not 10 bins"
        assert len(spec["quantiles"]) < 2000, (
            f"{name}: quantile grid has {len(spec['quantiles'])} points -- that is not a "
            f"summary, that is starting to look like the data."
        )
        for key, value in spec.items():
            if isinstance(value, list):
                assert len(value) < n_rows / 2, (
                    f"{name}.{key} has {len(value)} entries against {n_rows} reference rows. "
                    f"An aggregate profile MUST NOT scale with the cohort size. Something is "
                    f"smuggling raw data into a file that is committed to a PUBLIC repo."
                )


def test_profile_is_small_enough_to_commit(profile, tmp_path):
    """Small in BYTES, and small in LINES. Both matter, for different reasons.

    Bytes: it lives in a PUBLIC git repository and is cloned by everyone.

    Lines: the first version used `json.dumps(indent=1)`, which puts every element of every
    array on its own line -- **79,805 lines** for the real 78-feature profile (78 x 1,001
    quantiles). Not wrong, but bad hygiene: nobody reads 78,000 lines of floats, so the
    readability that indent was bought for does not exist; and every regeneration produces a
    whole-file diff, so `git log` on the artifact tells you only THAT it changed, never WHAT.
    A reference DISTRIBUTION is exactly the thing you want to be able to diff meaningfully.

    `_dumps_compact_arrays` keeps the header pretty-printed (source, build time, row count,
    feature list -- the provenance, readable at a glance) and puts each histogram and quantile
    grid on ONE line. Contents are unchanged: json emits floats via `repr`, which is
    round-trip exact, and `test_psi_identical_after_save_and_reload` proves the Population
    Stability Index does not move by a single bit.

    The bound below is per-feature, not absolute, so it stays meaningful as features are added.
    """
    path = profile.save(tmp_path / "p.json")
    kb = path.stat().st_size / 1024
    assert kb < 5000, f"profile is {kb:.0f} KB -- too big to live in git"

    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    n_feat = len(profile.feature_names)
    # ~10 lines per feature (open brace, 7 keys, close) + a short header. `indent=1` would
    # give >1,000 lines PER FEATURE, so this fails loudly if anyone reverts the serialiser.
    assert n_lines < 20 * n_feat + 50, (
        f"the profile serialised to {n_lines} lines for {n_feat} features "
        f"(~{n_lines / max(n_feat, 1):.0f} lines each).\n"
        f"\n"
        f"That is the `json.dumps(indent=1)` shape, which put all 1,001 quantiles of every "
        f"feature on their own line -- 79,805 lines in the real profile. Use "
        f"_dumps_compact_arrays: the arrays belong on ONE line each. The CONTENTS are "
        f"identical either way; this is purely about not committing 80,000 lines of floats "
        f"to a public repository and destroying the diff."
    )


# ---------------------------------------------------------------------------
# 4. FAIL-LOUD: every way of getting this wrong must raise, not degrade
# ---------------------------------------------------------------------------

def test_bin_count_mismatch_raises(X_ref):
    """The histogram is already binned. Re-binning it silently would corrupt every PSI."""
    profile = DriftReferenceProfile.from_reference(X_ref, source="t", n_bins=10)
    with pytest.raises(ValueError, match="cannot be re-binned|n_bins"):
        DriftDetector.from_profile(profile, n_bins=20)


def test_passing_both_reference_and_profile_raises(X_ref, profile):
    with pytest.raises(ValueError, match="EITHER reference_data OR profile"):
        DriftDetector(
            reference_data=X_ref.to_numpy(dtype=np.float64),
            feature_names=list(X_ref.columns),
            profile=profile,
        )


def test_passing_neither_raises(X_ref):
    with pytest.raises(ValueError, match="either reference_data or a profile"):
        DriftDetector(reference_data=None, feature_names=list(X_ref.columns))


def test_missing_profile_file_raises_with_instructions(tmp_path):
    """Absent reference is NOT 'no drift'. It is 'cannot check'."""
    with pytest.raises(FileNotFoundError, match="NOT 'no drift'"):
        DriftReferenceProfile.load(tmp_path / "does_not_exist.json")


def test_unknown_format_version_raises_rather_than_guessing(profile, tmp_path):
    path = profile.save(tmp_path / "p.json")
    payload = json.loads(path.read_text())
    payload["format_version"] = PROFILE_FORMAT_VERSION + 99
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Refusing to guess"):
        DriftReferenceProfile.load(path)


def test_new_data_missing_a_reference_feature_raises(X_new, profile):
    with pytest.raises(KeyError, match="Refusing to report partial coverage"):
        DriftDetector.from_profile(profile).check(X_new.drop(columns=["gaussian"]))


def test_unchecked_extra_features_are_warned_not_swallowed(X_new, profile, caplog):
    """78-feature reference vs 97-feature contract: the 19 must not vanish quietly."""
    X = X_new.copy()
    X["brand_new_feature_the_reference_never_saw"] = 1.0

    with caplog.at_level("WARNING"):
        DriftDetector.from_profile(profile).check(X)

    assert "NOT DRIFT-CHECKED" in caplog.text
    assert "brand_new_feature_the_reference_never_saw" in caplog.text


def test_bare_ndarray_with_wrong_width_raises(X_ref, profile):
    """No names to align on -> would pair up the WRONG features. Must refuse."""
    with pytest.raises(ValueError, match="Pass a DataFrame"):
        DriftDetector.from_profile(profile).check(np.zeros((100, 3)))


# ---------------------------------------------------------------------------
# 5. THE REAL MATRIX -- skipped where the cohort is absent (i.e. in CI)
# ---------------------------------------------------------------------------

REAL_REF = Path("outputs/run15_rerun_report/full/splits/X_train.parquet")


@pytest.mark.skipif(not REAL_REF.is_file(),
                    reason="Run-15 reference matrix absent (gitignored cohort data; not in CI)")
def test_psi_exact_on_the_real_run15_matrix():
    """Synthetic exactness is necessary but not sufficient. Prove it on the real thing."""
    X = pd.read_parquet(REAL_REF).select_dtypes(include=[np.number])
    profile = DriftReferenceProfile.from_reference(X, source=str(REAL_REF))
    detector = DriftDetector.from_reference(X_ref=X)

    rng = np.random.default_rng(0)
    X_new = X.iloc[rng.choice(len(X), 40_000, replace=False)].copy()
    for col in list(X_new.columns)[:10]:
        X_new[col] = X_new[col] * 1.15 + 0.05

    for i, feat in enumerate(detector.feature_names):
        ref_col = detector.ref_data[:, i]
        ref_col = ref_col[np.isfinite(ref_col)]
        new_col = X_new[feat].to_numpy(dtype=np.float64)
        new_col = new_col[np.isfinite(new_col)]

        assert detector._psi(ref_col, new_col) == profile.psi(feat, new_col), (
            f"PSI diverged on real feature {feat!r}"
        )
