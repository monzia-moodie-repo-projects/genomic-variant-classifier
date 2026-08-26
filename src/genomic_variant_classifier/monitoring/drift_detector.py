"""
src/genomic_variant_classifier/monitoring/drift_detector.py
=================================
Comprehensive data drift detection for the Genomic Variant Classifier.

Implements four complementary drift detection strategies, each targeting a
different type of distributional change:

    PSI  — Population Stability Index (industry standard, fast, interpretable)
    KS   — Kolmogorov-Smirnov test (nonparametric, per-feature continuous)
    MMD  — Maximum Mean Discrepancy (kernel-based, catches subtle shifts)
    ADWIN — Adaptive Windowing (streaming detector for online use)

Genomic drift taxonomy addressed:
    Feature/covariate drift : P(X) changes — gnomAD cohort expansion,
                              AlphaMissense model updates, score recalibration
    Label drift             : P(Y) changes — ClinVar reclassifications
    Concept drift           : P(Y|X) changes — new biology, e.g. SpliceAI
                              dramatically altering splice variant interpretation
    Score drift             : a specific sub-type of feature drift where a
                              precomputed tool is retrained upstream

State-of-the-art additions beyond the Kirkpatrick/EWC literature:
    - Least-Squares Density Ratio Estimation (LSIF) for importance weighting:
      estimates the density ratio p_new(x) / p_old(x) without fitting two
      separate density models, which is numerically more stable than direct KL
    - Wasserstein-1 distance (Earth Mover's Distance) as a geometrically
      meaningful distance between score distributions — more sensitive than
      PSI for bimodal distributions common in pathogenicity scores
    - Two-sample energy statistic (Székely-Rizzo) — a distribution-free test
      that works well for multivariate drift in the joint feature space

Usage:
    from genomic_variant_classifier.monitoring.drift_detector import DriftDetector, DriftReport

    detector = DriftDetector.from_reference(
        X_ref=X_train,
        feature_names=list(X_train.columns),
        save_path="models/drift_reference.pkl",
    )
    report = detector.check(X_new)
    if report.action_required:
        trigger_retraining()
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

if TYPE_CHECKING:      # import-cycle-free; resolved only by type checkers
    from genomic_variant_classifier.monitoring.drift_reference_profile import (
        DriftReferenceProfile,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (conventional clinical / finance standards)
# ---------------------------------------------------------------------------
PSI_NEGLIGIBLE  = 0.10   # PSI < 0.10 → no action
PSI_MONITOR     = 0.20   # 0.10–0.20 → increase monitoring frequency
PSI_RETRAIN     = 0.25   # > 0.25 → trigger retraining

KS_ALPHA        = 0.01   # significance level for KS test (Bonferroni corrected later)
MMD_SIGMA       = 1.0    # RBF kernel bandwidth (median heuristic used if None)
WASSERSTEIN_WARN = 0.05  # Wasserstein distance threshold for score features


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FeatureDriftResult:
    """Drift statistics for a single feature."""
    feature:          str
    psi:              float
    ks_statistic:     float
    ks_pvalue:        float
    wasserstein:      float
    ref_mean:         float
    ref_std:          float
    new_mean:         float
    new_std:          float
    mean_shift_sigmas: float     # (new_mean - ref_mean) / ref_std
    action:           str        # "none" | "monitor" | "retrain"

    #: True when `ks_statistic` / `ks_pvalue` / `wasserstein` were computed against a
    #: quantile-reconstructed reference rather than the real reference column -- i.e. when the
    #: detector was driven from an aggregate-only profile (roadmap 6.20).
    #:
    #: `psi` and `action` are EXACT either way; they are the fields anything acts on. These
    #: three are informational, and when this flag is set they carry the resolution of the
    #: stored quantile grid rather than of the data. Label it, so nobody later reads an
    #: approximate Kolmogorov-Smirnov p-value as a measured one.
    ks_wasserstein_approximate: bool = False


@dataclass
class DriftReport:
    """Complete drift analysis report across all features."""
    timestamp:         str
    n_ref_samples:     int
    n_new_samples:     int
    features_checked:  int
    features_drifted:  int          # PSI > PSI_RETRAIN
    features_monitored: int         # PSI in [PSI_MONITOR, PSI_RETRAIN]

    # The joint (multivariate) tests. These are Optional[float] and may legitimately be None
    # -- see `joint_tests_run` below. NONE MEANS "NOT COMPUTED". IT NEVER MEANS "NO DRIFT".
    mmd_score:         Optional[float]   # joint Maximum Mean Discrepancy across all features
    mmd_pvalue:        Optional[float]
    energy_statistic:  Optional[float]   # Székely-Rizzo two-sample energy test
    energy_pvalue:     Optional[float]

    feature_results:   list[FeatureDriftResult] = field(default_factory=list)
    top_drifted:       list[str]    = field(default_factory=list)
    action_required:   bool         = False
    recommended_action: str        = "none"  # "none"|"monitor"|"retrain"|"urgent_retrain"
    summary:           str         = ""

    # ── Were the joint tests actually run? (roadmap 6.20) ──────────────────────────────
    #
    # The Maximum Mean Discrepancy and energy tests are MULTIVARIATE permutation tests: they
    # need real reference SAMPLES. When the detector is driven from an aggregate-only
    # reference profile (`DriftDetector.from_profile`, used by the monthly hosted drift
    # monitor because a 1.4 MB committed histogram beats fetching a 23.8 MB cohort matrix
    # from cloud storage with credentials), there are no reference rows and the joint tests
    # CANNOT run.
    #
    # They are then reported as NOT RUN -- explicitly, in the report, and in every export.
    # They are NEVER reported as passing.
    #
    # THIS FLAG IS LOAD-BEARING. `check()` escalates to urgent_retrain on
    # `mmd_pvalue < 0.001`. If a profile-driven run quietly substituted a benign p-value
    # there, that escalation would be permanently disarmed WHILE APPEARING TO WORK -- which
    # is exactly root pattern (c): a gate that checks a proxy instead of the thing it
    # protects is not a gate. A missing measurement must look missing.
    joint_tests_run:    bool           = True
    joint_tests_reason: Optional[str]  = None

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  DRIFT REPORT -- {self.timestamp}")
        print(f"{'='*60}")
        print(f"  Reference: {self.n_ref_samples:,} samples")
        print(f"  New data:  {self.n_new_samples:,} samples")
        print(f"  Features checked:    {self.features_checked}")
        print(f"  Features drifted:    {self.features_drifted}  (PSI > {PSI_RETRAIN})")
        print(f"  Features monitored:  {self.features_monitored}  (PSI > {PSI_MONITOR})")
        if self.joint_tests_run:
            print(f"  MMD score:           {self.mmd_score:.6f}  (p={self.mmd_pvalue:.4f})")
            print(f"  Energy statistic:    {self.energy_statistic:.4f}  (p={self.energy_pvalue:.4f})")
        else:
            # Say it loudly. A joint test that did not run must never be mistaken for a joint
            # test that found nothing.
            print(f"  MMD score:           NOT COMPUTED")
            print(f"  Energy statistic:    NOT COMPUTED")
            print(f"  ^^ JOINT TESTS DID NOT RUN: {self.joint_tests_reason}")
            print(f"     The per-feature Population Stability Index checks below DID run and")
            print(f"     are exact. The joint multivariate escalation did NOT. This report is")
            print(f"     NOT evidence that the joint distribution is unchanged.")
        print(f"  ACTION: {self.recommended_action.upper()}")
        if self.top_drifted:
            print(f"  Top drifted features: {', '.join(self.top_drifted[:5])}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Stateful drift detector that holds a reference distribution snapshot
    and exposes a check() method for periodic evaluation.

    The reference snapshot should be set once from the training data and
    persisted. It is reloaded at each monitoring run without re-fitting.
    """

    def __init__(
        self,
        reference_data:  Optional[np.ndarray],
        feature_names:   list[str],
        n_bins:          int  = 10,
        mmd_n_permute:   int  = 200,
        energy_n_permute: int = 200,
        random_state:    int  = 42,
        profile:         Optional["DriftReferenceProfile"] = None,
    ) -> None:
        self.feature_names    = list(feature_names)
        self.n_features       = len(feature_names)
        self.n_bins           = n_bins
        self.mmd_n_permute    = mmd_n_permute
        self.energy_n_permute = energy_n_permute
        self.rng              = np.random.default_rng(random_state)

        #: The aggregate-only reference, when there are no raw rows (roadmap 6.20).
        #: EXACTLY ONE of `ref_data` / `profile` is populated. When `profile` is set,
        #: `ref_data` is None and the multivariate tests cannot run -- see `check()`.
        self.profile = profile

        if profile is not None:
            if reference_data is not None:
                raise ValueError(
                    "Pass EITHER reference_data OR profile, not both. Two references is two "
                    "answers, and nothing would tell you which one a report came from."
                )
            self.ref_data  = None
            self.ref_stats = None
            self.ref_bins  = None
            self.mmd_sigma = None      # NOT MMD_SIGMA. There is no bandwidth without samples,
                                       # and a plausible-looking default here would let the
                                       # joint test appear to run on data it never saw.
            logger.info(
                "DriftDetector initialised FROM AN AGGREGATE PROFILE: %d features, %d "
                "reference rows summarised (source=%s, built %s). Population Stability Index "
                "is EXACT. The joint Maximum Mean Discrepancy and energy tests CANNOT run "
                "without reference samples and will be reported as NOT COMPUTED.",
                self.n_features, profile.n_ref_samples, profile.source, profile.built_at_utc,
            )
            return

        if reference_data is None:
            raise ValueError("DriftDetector needs either reference_data or a profile.")

        self.ref_data   = reference_data.astype(np.float64)
        self.ref_stats  = self._compute_stats(self.ref_data)
        self.ref_bins   = self._compute_bins(self.ref_data)

        # Median heuristic for MMD bandwidth
        pairwise = cdist(
            self.ref_data[:min(2000, len(self.ref_data))],
            self.ref_data[:min(2000, len(self.ref_data))],
        )
        self.mmd_sigma = float(np.median(pairwise[pairwise > 0])) or MMD_SIGMA

        logger.info(
            "DriftDetector initialised: %d features, %d reference samples, sigma_MMD=%.3f",
            self.n_features, len(self.ref_data), self.mmd_sigma,
        )

    # ── Class-method constructors ──────────────────────────────────────────

    @classmethod
    def from_reference(
        cls,
        X_ref:        pd.DataFrame | np.ndarray,
        feature_names: Optional[list[str]] = None,
        save_path:    Optional[str | Path]  = None,
        **kwargs,
    ) -> DriftDetector:
        if isinstance(X_ref, pd.DataFrame):
            feature_names = feature_names or list(X_ref.columns)
            arr = X_ref.to_numpy(dtype=np.float64)
        else:
            arr = X_ref.astype(np.float64)
            feature_names = feature_names or [f"feat_{i}" for i in range(arr.shape[1])]

        detector = cls(arr, feature_names, **kwargs)
        if save_path:
            detector.save(save_path)
        return detector

    @classmethod
    def from_profile(
        cls,
        profile: "str | Path | DriftReferenceProfile",
        **kwargs,
    ) -> DriftDetector:
        """Build a detector from an AGGREGATE-ONLY reference profile (roadmap 6.20).

        This is how the scheduled monthly drift monitor runs on a hosted runner: the raw
        reference matrix (`X_train.parquet`) is 23.8 MB of variant rows that would have to be
        fetched from cloud storage with credentials on every run. The profile is 1.4 MB of
        histograms that lives in git. That is the whole reason it exists -- and it also means
        no per-variant annotation from any source, academic or licensed, is ever redistributed.

        (An earlier version of this docstring said the matrix could not travel because dbNSFP
        is `tier: controlled` / "LICENSED (paid)". That was wrong: data_manifest.yaml marks
        dbNSFP `tier: academic`; the "LICENSED (paid)" note belongs to hgmd. See
        drift_reference_profile.py for the full correction.)

        WHAT YOU GET
            * Population Stability Index, per feature -- **EXACT**. Bit-for-bit identical to
              the raw-data detector. Proven by tests/unit/test_drift_reference_profile.py.
            * The per-feature action (none / monitor / retrain) -- **EXACT**, because it is a
              function of PSI alone.
            * Kolmogorov-Smirnov and Wasserstein -- APPROXIMATE, reconstructed from the stored
              quantile grid, and flagged as such. Nothing depends on them.

        WHAT YOU DO NOT GET
            * The joint Maximum Mean Discrepancy and Székely-Rizzo energy tests. They are
              multivariate permutation tests over the JOINT distribution and cannot be
              recovered from marginal aggregates. They are reported as NOT COMPUTED, with
              `DriftReport.joint_tests_run = False`. They are never reported as passing.
        """
        from genomic_variant_classifier.monitoring.drift_reference_profile import (
            DriftReferenceProfile,
        )

        if isinstance(profile, (str, Path)):
            profile = DriftReferenceProfile.load(profile)

        if profile.n_bins != kwargs.get("n_bins", 10):
            # Root pattern (a): a number written down in two places WILL disagree. The bin
            # count is baked into the stored histogram; a detector using a different one would
            # silently produce wrong PSI for every feature, with no error anywhere.
            raise ValueError(
                f"Profile was built with n_bins={profile.n_bins}, but this detector was asked "
                f"for n_bins={kwargs.get('n_bins', 10)}. The reference histogram is already "
                f"binned -- it cannot be re-binned, and using it anyway would make EVERY "
                f"Population Stability Index wrong. Rebuild the profile."
            )

        return cls(
            reference_data=None,
            feature_names=profile.feature_names,
            profile=profile,
            **kwargs,
        )

    @classmethod
    def load(cls, path: str | Path) -> DriftDetector:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected DriftDetector, got {type(obj).__name__}")
        return obj

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("DriftDetector saved -> %s", path)

    # ── Main public interface ──────────────────────────────────────────────

    def check(
        self,
        X_new: pd.DataFrame | np.ndarray,
        timestamp: Optional[str] = None,
    ) -> DriftReport:
        """
        Run full drift check against the reference distribution.

        Parameters
        ----------
        X_new : DataFrame or array of shape (n_samples, n_features)
        timestamp : optional ISO timestamp string; defaults to now

        Returns
        -------
        DriftReport with per-feature statistics and a recommended action
        """
        from datetime import datetime, timezone
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()

        if isinstance(X_new, pd.DataFrame):
            # ── The reference must COVER the new data, or the check is a subset lie ────────
            #
            # `X_new[self.feature_names]` silently SELECTS the reference's columns and drops
            # everything else. If the new matrix has features the reference has never seen,
            # those features are never drift-checked -- and the report still says "checked",
            # with a feature count that looks healthy.
            #
            # This is not hypothetical. The Run-15 reference matrix carries 78 features; the
            # current tabular contract (EXPECTED_TABULAR_FEATURE_COUNT) is 95 -- MEASURED
            # 2026-08-25 from its sole definition in models/variant_ensemble.py:193, where
            # TABULAR_FEATURES holds exactly 95 entries.
            #
            # DETECTOR-CONTRACT-COMMENT-STALE-1: this comment said 97 until 2026-08-25, and
            # 97 was itself a figure a preflight gate had ALREADY corrected once --
            # docs/sessions/SESSION_2026-08-02_pre1-preflight-contract-gate.md records
            # "97-feature contract (88 + 3 + 6)" being replaced by "95, 86 + 3 + 6". The
            # superseded number survived here because nothing binds a COMMENT to a constant.
            #
            # The subtraction moved with it: 97 - 78 = 19, but 95 - 78 = 17. Correcting the
            # contract and leaving "ignore 19" would have replaced one stale number with an
            # inconsistent pair.
            #
            # Pointed at today's data, an un-guarded detector would check 78, ignore 17, and
            # report no drift on features it never looked at. That is root pattern (c) -- a
            # gate that checks a proxy for the thing it protects -- and it is precisely how
            # this subsystem died the first time.
            missing = [f for f in self.feature_names if f not in X_new.columns]
            if missing:
                raise KeyError(
                    f"The new data is missing {len(missing)} feature(s) the reference "
                    f"expects, so they cannot be compared: {missing[:10]}"
                    f"{' ...' if len(missing) > 10 else ''}. Refusing to report partial "
                    f"coverage as a completed drift check."
                )

            unchecked = [c for c in X_new.columns if c not in self.feature_names]
            if unchecked:
                logger.warning(
                    "%d feature(s) in the new data are ABSENT FROM THE REFERENCE and are "
                    "therefore NOT DRIFT-CHECKED: %s%s. The reference is stale relative to "
                    "the current feature contract -- rebuild it. This report covers %d of %d "
                    "features and is NOT evidence about the rest.",
                    len(unchecked), unchecked[:10], " ..." if len(unchecked) > 10 else "",
                    len(self.feature_names), X_new.shape[1],
                )

            new_arr = X_new[self.feature_names].to_numpy(dtype=np.float64)
        else:
            new_arr = X_new.astype(np.float64)
            if new_arr.shape[1] != self.n_features:
                raise ValueError(
                    f"New data has {new_arr.shape[1]} columns but the reference has "
                    f"{self.n_features}. With a bare ndarray there are no names to align on, "
                    f"so this comparison would silently pair up the WRONG features. "
                    f"Pass a DataFrame."
                )

        from_profile = self.profile is not None

        feature_results = []
        for i, feat in enumerate(self.feature_names):
            new_col = new_arr[:, i]
            if from_profile:
                result = self._check_feature_from_profile(feat, new_col)
            else:
                result = self._check_feature(feat, self.ref_data[:, i], new_col)
            feature_results.append(result)

        # Sort by PSI descending
        feature_results.sort(key=lambda r: r.psi, reverse=True)

        n_retrain  = sum(1 for r in feature_results if r.action == "retrain")
        n_monitor  = sum(1 for r in feature_results if r.action == "monitor")
        top_drifted = [r.feature for r in feature_results if r.action == "retrain"][:5]

        # ── Joint (multivariate) tests ────────────────────────────────────────────────
        #
        # These need real reference SAMPLES. From an aggregate profile there are none, and no
        # amount of cleverness recovers a joint distribution from marginal histograms. So they
        # do not run -- and they SAY they did not run.
        #
        # The alternative -- quietly setting mmd_pvalue = 1.0 -- would have been invisible,
        # would have looked exactly like a healthy run, and would have permanently disarmed the
        # urgent_retrain escalation below. That is the defect this subsystem is being rescued
        # FROM (roadmap 6.20: drift_monitor.yml reported "no drift" every month having never
        # checked anything). A measurement that did not happen must never wear the costume of a
        # measurement that came back clean.
        if from_profile:
            mmd_score = mmd_pval = energy_stat = energy_p = None
            joint_tests_run = False
            joint_tests_reason = (
                "the detector was built from an aggregate-only reference profile, which "
                "contains histograms and quantile grids but no reference rows; the Maximum "
                "Mean Discrepancy and Szekely-Rizzo energy tests are multivariate permutation "
                "tests and require samples of the joint distribution. Run "
                "scripts/run_drift_monitor.py with --reference-splits, on a machine that holds "
                "the cohort matrix, to obtain them."
            )
            logger.warning(
                "Joint MMD/energy tests NOT RUN: %s The per-feature Population Stability "
                "Index checks are exact and DID run.", joint_tests_reason,
            )
        else:
            n_sub = min(3000, len(self.ref_data), len(new_arr))
            ref_sub = self.ref_data[self.rng.choice(len(self.ref_data), n_sub, replace=False)]
            new_sub = new_arr[self.rng.choice(len(new_arr), n_sub, replace=False)]

            mmd_score, mmd_pval   = self._mmd_test(ref_sub, new_sub)
            energy_stat, energy_p = self._energy_test(ref_sub, new_sub)
            mmd_score, mmd_pval   = float(mmd_score), float(mmd_pval)
            energy_stat, energy_p = float(energy_stat), float(energy_p)
            joint_tests_run = True
            joint_tests_reason = None

        # ── Overall action ────────────────────────────────────────────────────────────
        #
        # Written so that a MISSING mmd_pvalue can never be read as a PASSING one. `None` does
        # not participate in the escalation; it does not suppress it either. The Population
        # Stability Index triggers stand on their own -- which is the whole reason the profile
        # is useful: `n_retrain` and `n_monitor` are EXACT even with no reference rows.
        mmd_urgent = mmd_pval is not None and mmd_pval < 0.001
        mmd_retrain = mmd_pval is not None and mmd_pval < 0.01

        if n_retrain > 3 or mmd_urgent:
            action = "urgent_retrain"
        elif n_retrain > 0 or mmd_retrain:
            action = "retrain"
        elif n_monitor > 0:
            action = "monitor"
        else:
            action = "none"

        joint_txt = (
            f"Joint MMD p={mmd_pval:.4f}." if joint_tests_run
            else "Joint MMD/energy NOT COMPUTED (aggregate-only reference; PSI checks are exact)."
        )
        summary = (
            f"{n_retrain} features with significant drift (PSI>{PSI_RETRAIN}), "
            f"{n_monitor} under monitoring. "
            f"{joint_txt} "
            f"Recommended: {action}."
        )

        report = DriftReport(
            timestamp          = timestamp,
            n_ref_samples      = self.profile.n_ref_samples if from_profile else len(self.ref_data),
            n_new_samples      = len(new_arr),
            features_checked   = self.n_features,
            features_drifted   = n_retrain,
            features_monitored = n_monitor,
            mmd_score          = mmd_score,
            mmd_pvalue         = mmd_pval,
            energy_statistic   = energy_stat,
            energy_pvalue      = energy_p,
            feature_results    = feature_results,
            top_drifted        = top_drifted,
            action_required    = action in ("retrain", "urgent_retrain"),
            recommended_action = action,
            summary            = summary,
            joint_tests_run    = joint_tests_run,
            joint_tests_reason = joint_tests_reason,
        )

        logger.info("Drift check complete. %s", summary)
        return report

    # ── Per-feature analysis ───────────────────────────────────────────────

    def _check_feature(
        self, feature: str, ref_col: np.ndarray, new_col: np.ndarray
    ) -> FeatureDriftResult:
        ref_col = ref_col[np.isfinite(ref_col)]
        new_col = new_col[np.isfinite(new_col)]

        psi          = self._psi(ref_col, new_col)
        ks_stat, ks_p = stats.ks_2samp(ref_col, new_col)
        wasserstein  = float(stats.wasserstein_distance(ref_col, new_col))

        ref_mean, ref_std = float(np.mean(ref_col)), float(np.std(ref_col)) + 1e-9
        new_mean, new_std = float(np.mean(new_col)), float(np.std(new_col))
        shift_sigmas = (new_mean - ref_mean) / ref_std

        if psi > PSI_RETRAIN:
            action = "retrain"
        elif psi > PSI_MONITOR:
            action = "monitor"
        else:
            action = "none"

        return FeatureDriftResult(
            feature           = feature,
            psi               = round(psi, 5),
            ks_statistic      = round(float(ks_stat), 5),
            ks_pvalue         = round(float(ks_p), 6),
            wasserstein       = round(wasserstein, 5),
            ref_mean          = round(ref_mean, 5),
            ref_std           = round(ref_std, 5),
            new_mean          = round(new_mean, 5),
            new_std           = round(new_std, 5),
            mean_shift_sigmas = round(shift_sigmas, 3),
            action            = action,
        )

    def _check_feature_from_profile(
        self, feature: str, new_col: np.ndarray
    ) -> FeatureDriftResult:
        """Per-feature drift with NO reference rows -- only the aggregate profile (6.20).

        Mirrors `_check_feature` exactly, field for field. The Population Stability Index and
        therefore the ACTION are EXACT -- identical to what the raw-data path would return.
        The Kolmogorov-Smirnov and Wasserstein figures are reconstructed from the stored
        quantile grid and are flagged approximate.
        """
        prof    = self.profile.features[feature]
        new_col = new_col[np.isfinite(new_col)]

        # EXACT. Same percentiles, same edges, same denominator, same clipping.
        psi = self.profile.psi(feature, new_col)

        # APPROXIMATE. The quantile grid is the reference empirical cumulative distribution
        # function, compressed; interpolating it back gives a sample with the same
        # distribution to grid resolution -- and nothing else. No rows, no identities, no
        # joint structure.
        if prof.n_finite == 0 or len(new_col) == 0:
            ks_stat, ks_p, wasserstein = 0.0, 1.0, 0.0
        else:
            ref_recon = prof.reference_sample(n=min(10_000, max(prof.n_finite, 2)))
            ks_stat, ks_p = stats.ks_2samp(ref_recon, new_col)
            wasserstein   = float(stats.wasserstein_distance(ref_recon, new_col))

        # `+ 1e-9` reproduces _check_feature line-for-line: the epsilon is applied at use, not
        # stored. Drop it and every mean_shift_sigmas would differ in the last places -- a
        # small, silent divergence between the two code paths, which is precisely the shape of
        # bug this project keeps finding.
        ref_mean, ref_std = prof.mean, prof.std + 1e-9
        new_mean = float(np.mean(new_col)) if len(new_col) else 0.0
        new_std  = float(np.std(new_col)) if len(new_col) else 0.0
        shift_sigmas = (new_mean - ref_mean) / ref_std

        if psi > PSI_RETRAIN:
            action = "retrain"
        elif psi > PSI_MONITOR:
            action = "monitor"
        else:
            action = "none"

        return FeatureDriftResult(
            feature           = feature,
            psi               = round(psi, 5),
            ks_statistic      = round(float(ks_stat), 5),
            ks_pvalue         = round(float(ks_p), 6),
            wasserstein       = round(wasserstein, 5),
            ref_mean          = round(ref_mean, 5),
            ref_std           = round(ref_std, 5),
            new_mean          = round(new_mean, 5),
            new_std           = round(new_std, 5),
            mean_shift_sigmas = round(shift_sigmas, 3),
            action            = action,
            ks_wasserstein_approximate = True,
        )

    # ── Statistical methods ────────────────────────────────────────────────

    def _psi(self, ref: np.ndarray, new: np.ndarray) -> float:
        """Population Stability Index (10 equal-width bins over reference range)."""
        lo, hi = np.percentile(ref, 1), np.percentile(ref, 99)
        if lo == hi:
            return 0.0
        edges   = np.linspace(lo, hi, self.n_bins + 1)
        ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
        new_pct = np.histogram(new, bins=edges)[0] / max(len(new), 1)
        # Smooth zeros to avoid log(0)
        ref_pct = np.clip(ref_pct, 1e-4, None)
        new_pct = np.clip(new_pct, 1e-4, None)
        return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """RBF kernel matrix K(X, Y) using stored sigma."""
        sq_dists = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-sq_dists / (2 * self.mmd_sigma ** 2))

    def _mmd_score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Unbiased MMD^2 estimator."""
        n, m = len(X), len(Y)
        Kxx = self._rbf_kernel(X, X)
        Kyy = self._rbf_kernel(Y, Y)
        Kxy = self._rbf_kernel(X, Y)
        np.fill_diagonal(Kxx, 0)
        np.fill_diagonal(Kyy, 0)
        return (Kxx.sum() / (n * (n - 1)) +
                Kyy.sum() / (m * (m - 1)) -
                2 * Kxy.mean())

    def _mmd_test(
        self, ref: np.ndarray, new: np.ndarray
    ) -> tuple[float, float]:
        """Permutation test for MMD^2."""
        observed = self._mmd_score(ref, new)
        combined = np.vstack([ref, new])
        n = len(ref)
        perm_scores: list[float] = []
        for _ in range(self.mmd_n_permute):
            idx = self.rng.permutation(len(combined))
            perm_scores.append(
                self._mmd_score(combined[idx[:n]], combined[idx[n:]])
            )
        pval = float(np.mean(np.array(perm_scores) >= observed))
        return float(observed), pval

    def _energy_test(
        self, ref: np.ndarray, new: np.ndarray
    ) -> tuple[float, float]:
        """
        Székely-Rizzo two-sample energy statistic.
        E = (2nm)/(n+m) * [E|X-Y| - 0.5*E|X-X'| - 0.5*E|Y-Y'|]
        Sensitive to differences in shape, not just mean/variance.
        """
        n, m = len(ref), len(new)
        Exy  = cdist(ref, new).mean()
        Exx  = cdist(ref, ref).mean()
        Eyy  = cdist(new, new).mean()
        stat = (2 * n * m) / (n + m) * (Exy - 0.5 * Exx - 0.5 * Eyy)

        combined = np.vstack([ref, new])
        perm_stats: list[float] = []
        for _ in range(self.energy_n_permute):
            idx = self.rng.permutation(len(combined))
            r, s = combined[idx[:n]], combined[idx[n:]]
            perm_stats.append(
                (2 * n * m) / (n + m) * (
                    cdist(r, s).mean() - 0.5 * cdist(r, r).mean() - 0.5 * cdist(s, s).mean()
                )
            )
        pval = float(np.mean(np.array(perm_stats) >= stat))
        return float(stat), pval

    # ── Internal helpers ───────────────────────────────────────────────────

    def _compute_stats(self, arr: np.ndarray) -> dict:
        return {
            "mean":   arr.mean(axis=0),
            "std":    arr.std(axis=0) + 1e-9,
            "p1":     np.percentile(arr, 1,  axis=0),
            "p99":    np.percentile(arr, 99, axis=0),
        }

    def _compute_bins(self, arr: np.ndarray) -> list:
        bins = []
        for i in range(arr.shape[1]):
            col = arr[:, i]
            lo, hi = np.percentile(col, 1), np.percentile(col, 99)
            bins.append(np.linspace(lo if lo < hi else lo - 1, hi if lo < hi else hi + 1, self.n_bins + 1))
        return bins


# ---------------------------------------------------------------------------
# Streaming ADWIN detector (for continuous / online ingestion)
# ---------------------------------------------------------------------------

class ADWINDriftDetector:
    """
    Adaptive Windowing (ADWIN) detector for streaming variant ingestion.

    Maintains a sliding window of a scalar statistic (e.g. mean pathogenicity
    score or mean allele frequency of incoming variants). Flags drift when the
    mean in the most recent sub-window differs significantly from the full window.

    Reference: Bifet & Gavalda (2007), "Learning from Time-Changing Data with
    Adaptive Windowing". SIAM SDM 2007.

    Usage:
        adwin = ADWINDriftDetector(delta=0.002)
        for score in streaming_pathogenicity_scores:
            drifted = adwin.update(score)
            if drifted:
                trigger_retraining()
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta   = delta
        self.window: list[float] = []
        self._drift_detected = False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    def update(self, value: float) -> bool:
        """
        Add a new observation. Returns True if drift is detected.
        ADWIN shrinks the window from the left when it detects a
        statistically significant shift in the mean.
        """
        self._drift_detected = False
        self.window.append(float(value))

        if len(self.window) < 32:
            return False

        n   = len(self.window)
        arr = np.array(self.window)
        mu  = arr.mean()

        # Test all possible split points
        for cut in range(1, n - 1):
            n0, n1   = cut, n - cut
            mu0      = arr[:cut].mean()
            mu1      = arr[cut:].mean()
            diff     = abs(mu0 - mu1)
            epsilon  = np.sqrt(
                (1 / (2 * n0) + 1 / (2 * n1)) *
                np.log(4 * n / self.delta)
            )
            if diff >= epsilon:
                # Shrink window to the more recent sub-window
                self.window = self.window[cut:]
                self._drift_detected = True
                logger.info(
                    "ADWIN: drift detected at split %d/%d, |deltamu|=%.4f >= epsilon=%.4f",
                    cut, n, diff, epsilon,
                )
                break

        return self._drift_detected

    def reset(self) -> None:
        self.window = []
        self._drift_detected = False

    @property
    def mean(self) -> float:
        return float(np.mean(self.window)) if self.window else 0.0

    @property
    def window_size(self) -> int:
        return len(self.window)


# ---------------------------------------------------------------------------
# LSIF density ratio estimator for importance weighting
# ---------------------------------------------------------------------------

class LSIFImportanceWeighter:
    """
    Least-Squares Importance Fitting (LSIF) — estimates the density ratio
    w(x) = p_new(x) / p_ref(x) without fitting two separate density models.

    These weights can be used for:
      1. Sample re-weighting in retraining: give higher weight to variants
         whose feature distribution resembles the new data
      2. Identifying which training variants are most stale / unrepresentative
         of current data

    Reference: Kanamori et al. (2009), "A Least-Squares Approach to Direct
    Importance Estimation". JMLR 10.

    Usage:
        weighter = LSIFImportanceWeighter()
        weighter.fit(X_ref, X_new)
        weights = weighter.transform(X_train)  # per-sample importance weights
    """

    def __init__(self, sigma: float = 1.0, lambda_: float = 0.01, n_basis: int = 200) -> None:
        self.sigma    = sigma
        self.lambda_  = lambda_
        self.n_basis  = n_basis
        self._centers: Optional[np.ndarray] = None
        self._alpha:   Optional[np.ndarray] = None

    def fit(
        self,
        X_ref: np.ndarray | pd.DataFrame,
        X_new: np.ndarray | pd.DataFrame,
    ) -> LSIFImportanceWeighter:
        if isinstance(X_ref, pd.DataFrame):
            X_ref = X_ref.to_numpy(dtype=np.float64)
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.to_numpy(dtype=np.float64)

        X_ref = X_ref.astype(np.float64)
        X_new = X_new.astype(np.float64)

        # Select basis centres from X_new (or combined) using random subsampling
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_new), min(self.n_basis, len(X_new)), replace=False)
        self._centers = X_new[idx]

        # Compute kernel matrices
        K_ref = self._kernel(X_ref, self._centers)   # (n_ref, n_basis)
        K_new = self._kernel(X_new, self._centers)   # (n_new, n_basis)

        # LSIF objective: min_alpha ||Hα - h||^2 + λ||α||^2
        H = K_ref.T @ K_ref / len(X_ref)
        h = K_new.mean(axis=0)

        # Closed-form solution: α = (H + λI)^{-1} h
        self._alpha = np.linalg.solve(
            H + self.lambda_ * np.eye(self.n_basis),
            h,
        )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return importance weights w(x) = p_new(x) / p_ref(x) for each row."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)
        K = self._kernel(X.astype(np.float64), self._centers)
        w = K @ self._alpha
        return np.clip(w, 0.0, None)  # clip to non-negative (density ratios ≥ 0)

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq_dists = cdist(A, B, metric="sqeuclidean")
        return np.exp(-sq_dists / (2 * self.sigma ** 2))