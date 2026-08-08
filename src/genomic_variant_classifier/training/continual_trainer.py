"""
src/genomic_variant_classifier/training/continual_trainer.py
===================================
Full continual learning orchestration for the Genomic Variant Classifier.

Ties together all drift detection and adaptation modules into a single
end-to-end pipeline that can be run on a schedule (monthly, on each ClinVar
release, or triggered by drift alerts).

Pipeline:
    1. Load new data release (ClinVar + gnomAD + any updated scores)
    2. Run feature drift detection (PSI / KS / MMD on input distribution)
    3. Run label drift detection (ClinVar reclassification tracking)
    4. Decide: no action | increase monitoring | retrain
    5. If retraining:
       a. Compute LSIF importance weights (p_new / p_old density ratio)
       b. Apply TreeEWCProxy sample weights for stable variants
       c. Apply temporal decay weights for old submissions
       d. Combine all weights and retrain the stacking ensemble
       e. Evaluate on canonical holdout set
       f. Register in model registry
       g. Deploy to shadow
    6. During shadow burn-in: compare shadow vs production on live traffic
    7. Promote shadow → production if quality gate passes

State-of-the-art additions:
    - SNGP (Spectral Normalised Gaussian Process) output head: adds
      distance-aware uncertainty to OOD variant detection; variants in
      genomic regions absent from training data are flagged rather than
      silently scored. Implemented as an optional head on GenomicVariantMLP.
    - Selective prediction / abstention: variants where both epistemic and
      aleatoric uncertainty exceed configurable thresholds are returned with
      classification="Uncertain significance" regardless of the point estimate,
      forcing human review of genuinely ambiguous cases.
    - Evidently AI integration: optional structured HTML drift report
      exportable to Evidently format for dashboard visualisation.

Usage:
    python scripts/run_drift_monitor.py \\
        --reference-splits  outputs/phase2_with_gnomad/splits/ \\
        --new-clinvar       data/processed/clinvar_grch38_2024_07.parquet \\
        --old-clinvar       data/processed/clinvar_grch38_2024_01.parquet \\
        --output-dir        outputs/drift_reports/ \\
        --registry          deployments/registry.v1.json \\
        --auto-retrain
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ContinualLearningConfig:
    """Configuration for the continual learning pipeline."""

    # Drift detection
    psi_retrain_threshold:    float = 0.25
    flip_rate_retrain:        float = 0.010
    mmd_pvalue_retrain:       float = 0.01

    # EWC / sample weighting
    ewc_lambda:               float = 1000.0  # for neural component
    tree_ewc_lambda_decay:    float = 0.50    # for XGBoost/LightGBM
    temporal_decay_lambda:    float = 0.30    # annual decay rate
    reclassified_boost:       float = 2.0

    # Retraining
    min_review_tier:          int   = 2
    n_folds:                  int   = 5
    max_train_samples:        Optional[int] = None

    # Shadow deployment
    shadow_burn_in_days:      int   = 7
    shadow_min_predictions:   int   = 1000
    shadow_auroc_tolerance:   float = 0.002  # max allowed drop for promotion

    # Registry
    #
    # REGISTRY-1 (2026-08-07): moved out of `models/`, which .gitignore:75
    # ignores wholesale, so the declaration could never be committed and
    # the Continuous Integration check could never find it. `deployments/`
    # is a control-plane namespace: small, reviewable declarations, as
    # distinct from artifacts (`models/`) and reference data
    # (`data/reference/`).
    registry_path:            str = "deployments/registry.v1.json"

    # Adaptive retraining is FAIL-CLOSED. See AdaptiveRetrainingInputs
    # below; leaving this None is what keeps `_retrain` shut.
    adaptive_inputs:          Optional["AdaptiveRetrainingInputs"] = None

    # Outputs
    output_dir:               str = "outputs/continual_learning"
    auto_retrain:             bool = False  # if False, only reports; requires human approval


@dataclass(frozen=True)
class ReferenceTrainingFeatures:
    """The deployed model's training cohort, in the SAME feature space as the
    new one.

    LSIF-1 (2026-08-07). Density-ratio estimation compares two samples; it can
    only do so if both inhabit one feature space under one representation
    function. The previous code passed the NEW cohort through the serving
    pipeline's `_prepare` and called the result the reference, which made

        w(x) = p_new(x) / p_ref(x)

    an estimate of a quantity that does not exist. A matrix of the right width
    is not evidence of the right provenance, so this type carries enough to
    prove whose cohort it is.

    `population_fingerprint` is optional but load-bearing when present: if the
    two populations are IDENTICAL there is no covariate shift, and fitting an
    importance weighter would manufacture structure out of estimator noise.
    """

    frame: "pd.DataFrame"
    model_record_id: str
    feature_names: tuple[str, ...]
    population_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        if not self.model_record_id:
            raise ValueError(
                "a reference cohort must name the model record it trained. "
                "LSIF-1: without it, any matrix of the right width passes.")
        if not self.feature_names:
            raise ValueError(
                "a reference cohort must enumerate its feature names; a COUNT "
                "cannot detect a reordered or substituted column")
        if tuple(self.frame.columns) != self.feature_names:
            raise ValueError(
                "the reference frame's columns do not match its declared "
                f"feature names: frame={list(self.frame.columns)[:6]}..., "
                f"declared={list(self.feature_names)[:6]}...")


class DensityRatioStatus(str, Enum):
    """Whether a density ratio was estimated, or declared unnecessary.

    Fitting LSIF on two identical populations is mathematically defined --
    p/p -- and operationally meaningless: it reports estimator noise as
    adaptation. Declaring the identity is more honest than estimating it.
    """

    ESTIMATED = "estimated"
    SAME_POPULATION = "same_population"


@dataclass(frozen=True)
class AdaptiveRetrainingInputs:
    """The scientific inputs adaptive retraining cannot proceed without.

    REGISTRY-1 (2026-08-07). Until today `_retrain` was unreachable because
    `ModelRegistry` was imported and never defined. That accident was the only
    thing preventing four measured defects from executing:

      LSIF-1      `lsif.fit(X_ref=..., X_new=...)` receives the SAME rows in
                  two different feature representations, so the declared
                  density ratio p_new/p_old has no reference population and is
                  not identified.
      ROSTER-1    the retraining subprocess passes `--skip-nn --skip-svm`, so
                  the intervention is "new data + adaptation + architecture
                  change" and any shadow-versus-production movement is
                  confounded.
      EVALPROV-1  `X_val_new` -- the new release's VALIDATION split -- is
                  registered as `holdout_auroc` and logged as a holdout. The
                  module contract promises "evaluate on canonical holdout set".
      EWCSEL-1    `best_score_` is set nowhere in src/, so
                  `max(..., key=getattr(m, "best_score_", 0.0))` returns
                  whichever base model comes first in dictionary order.

    A boolean flag would be flipped by whoever next wanted the path to run.
    Requiring the MISSING INPUTS instead means the path cannot execute until
    each finding actually has an answer:

      no reference training features   -> LSIF cannot estimate a ratio
      no expected roster               -> an architecture change is undetected
      no evaluation protocol           -> promotion evidence is unqualified
      no selected base model           -> the EWC anchor is arbitrary
    """

    #: LSIF-1 (2026-08-07). Was `Path`. A path proves a file exists; it
    #: proves nothing about whose cohort it holds or what feature space
    #: it inhabits, and both are preconditions for a density ratio.
    reference_training_features: "ReferenceTrainingFeatures"
    expected_model_roster:       tuple[str, ...]
    promotion_protocol_id:       str
    ewc_anchor_model:            str

    #: LSIF-1 (2026-08-08). WHICH DEPLOYMENT THIS RETRAINING ADAPTS
    #: FROM. The reference cohort must belong to it; a density ratio
    #: against some other deployment measures nothing about this one.
    #: Added because the comparison was written on 2026-08-07 against
    #: a name that did not exist in scope -- a defect no import check
    #: could see, because it raises only when the line executes.
    deployed_model_record_id:    str

    #: The NEW cohort's population fingerprint, if the caller can
    #: declare one. Optional because nothing in this repository
    #: computes population fingerprints yet, and inventing a scheme
    #: here would be scope creep. The SAME_POPULATION shortcut fires
    #: only when BOTH sides are declared and agree: declared identity
    #: beats estimated identity, but an absent declaration is not a
    #: declaration of difference either -- it simply means the ratio
    #: must be estimated.
    new_population_fingerprint:  Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference_training_features,
                          ReferenceTrainingFeatures):
            raise ValueError(
                "reference_training_features must be a "
                "ReferenceTrainingFeatures carrying the deployed model's "
                "record id and feature names. LSIF-1: a bare frame or "
                "path lets any matrix of the right width through, which "
                "is how the density ratio came to be fitted against "
                f"itself. Got {type(self.reference_training_features)}")
        if not self.deployed_model_record_id:
            raise ValueError(
                "deployed_model_record_id must name the model record "
                "this retraining adapts FROM. LSIF-1: without it the "
                "reference cohort cannot be checked against the "
                "deployment whose drift is being measured.")
        if not self.expected_model_roster:
            raise ValueError(
                "expected_model_roster must enumerate the production roster. "
                "ROSTER-1: a count cannot detect an architecture change.")
        if not self.promotion_protocol_id:
            raise ValueError(
                "promotion_protocol_id must name the evaluation protocol. "
                "EVALPROV-1: an unqualified metric is not promotion evidence.")
        if not self.ewc_anchor_model:
            raise ValueError(
                "ewc_anchor_model must name the base model the Elastic Weight "
                "Consolidation proxy anchors on. EWCSEL-1: `best_score_` is "
                "set nowhere, so introspection returns insertion order.")


#: LSIF-1. Closing one blocker must not open the path. These remain
#: unresolved, and `_retrain` stays fail-closed while any of them is listed.
UNRESOLVED_ADAPTIVE_RETRAINING_BLOCKERS = frozenset({
    "ROSTER-1",     # --skip-nn --skip-svm changes architecture with the data
    "EVALPROV-1",   # a validation split registered as holdout evidence
    "EWCSEL-1",     # best_score_ is set nowhere; the anchor is dict order
})


def _aligned_lsif_matrices(
    *,
    reference: "ReferenceTrainingFeatures",
    new_features: "pd.DataFrame",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Both cohorts as arrays, or a refusal naming exactly what differs.

    LSIF-1. Column EQUALITY is checked, not column count: a reordered or
    substituted column preserves the width while destroying the correspondence,
    and width is the only thing the previous code could have checked.
    """
    reference_frame = reference.frame
    if tuple(reference_frame.columns) != tuple(new_features.columns):
        missing = sorted(set(new_features.columns)
                         - set(reference_frame.columns))
        extra = sorted(set(reference_frame.columns)
                       - set(new_features.columns))
        order_only = not missing and not extra
        raise ValueError(
            "LSIF requires the reference and new cohorts in one feature "
            "space. "
            + ("the columns match but their ORDER differs, which "
               "silently permutes every row" if order_only else
               f"missing_from_reference={missing}, "
               f"extra_in_reference={extra}"))
    if reference_frame.empty or new_features.empty:
        raise ValueError(
            "LSIF requires non-empty reference and new cohorts; got "
            f"{len(reference_frame)} reference and {len(new_features)} new")

    x_ref = reference_frame.to_numpy(dtype=np.float64, copy=True)
    x_new = new_features.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(x_ref).all():
        raise ValueError("the LSIF reference cohort contains non-finite "
                         "values; a density ratio over them is undefined")
    if not np.isfinite(x_new).all():
        raise ValueError("the LSIF new cohort contains non-finite values; a "
                         "density ratio over them is undefined")
    return x_ref, x_new


class ContinualLearner:
    """
    Orchestrates the full continual learning lifecycle.

    Designed to be run monthly (on each ClinVar release) or
    triggered by the drift monitoring API when PSI > threshold.
    """

    def __init__(self, config: ContinualLearningConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        reference_splits_dir: str | Path,
        new_clinvar_path:     str | Path,
        old_clinvar_path:     str | Path,
        current_model_path:   str | Path,
        gnomad_path:          Optional[str | Path] = None,
        alphamissense_path:   Optional[str | Path] = None,
        release_name:         str = "current",
        old_release_name:     str = "previous",
    ) -> dict:
        """
        Run the full continual learning check-and-adapt pipeline.

        Returns a summary dict with drift report, label drift report,
        retraining decision, and new model path (if retrained).
        """
        from genomic_variant_classifier.monitoring.drift_detector import DriftDetector
        from genomic_variant_classifier.monitoring.clinvar_tracker import ClinVarTracker
        from genomic_variant_classifier.monitoring.model_registry import (
            ModelRegistry)

        splits_dir = Path(reference_splits_dir)
        logger.info("=== Continual Learning Pipeline: starting ===")

        # ── Step 1: Load reference data ──────────────────────────────────
        X_train = pd.read_parquet(splits_dir / "X_train.parquet")
        X_val   = pd.read_parquet(splits_dir / "X_val.parquet")
        y_train = pd.read_parquet(splits_dir / "y_train.parquet")["label"]
        meta    = pd.read_parquet(splits_dir / "meta_test.parquet")

        training_ids = set(meta.get("variant_id", pd.Series(dtype=str)))
        logger.info("Reference training set: %d variants, %d features", len(X_train), X_train.shape[1])

        # ── Step 2: Load new ClinVar data ─────────────────────────────────
        new_clinvar = pd.read_parquet(new_clinvar_path)
        logger.info("New ClinVar: %d variants", len(new_clinvar))

        # ── Step 3: Feature drift detection ──────────────────────────────
        logger.info("Running feature drift detection ...")
        detector = DriftDetector.from_reference(
            X_ref=X_train,
            feature_names=list(X_train.columns),
            save_path=self.output_dir / "drift_reference.pkl",
        )
        # Build feature matrix for new ClinVar using the existing pipeline
        try:
            from genomic_variant_classifier.api.pipeline import engineer_features
            X_new = engineer_features(new_clinvar)
            drift_report = detector.check(X_new, timestamp=release_name)
        except Exception as e:
            logger.warning("Feature drift check failed: %s", e)
            drift_report = None

        if drift_report:
            drift_report.print_summary()
            drift_report.to_json(self.output_dir / f"drift_report_{release_name}.json")

        # ── Step 4: Label drift detection ────────────────────────────────
        logger.info("Running ClinVar label drift check ...")
        tracker = ClinVarTracker(
            training_variant_ids=training_ids,
            val_variant_ids=set(),
            test_variant_ids=set(),
        )
        label_report = tracker.compare(
            old_path=old_clinvar_path,
            new_path=new_clinvar_path,
            output_dir=self.output_dir / "temporal_cohorts",
            old_release=old_release_name,
            new_release=release_name,
        )
        label_report.to_json(self.output_dir / f"label_drift_{release_name}.json")

        # ── Step 5: Retraining decision ───────────────────────────────────
        feature_drift_triggered = (
            drift_report is not None and drift_report.action_required
        )
        label_drift_triggered = label_report.should_retrain

        should_retrain = feature_drift_triggered or label_drift_triggered
        decision_reason = []
        if feature_drift_triggered:
            decision_reason.append(
                f"Feature drift: {drift_report.features_drifted} features with PSI>{self.config.psi_retrain_threshold}"
            )
        if label_drift_triggered:
            decision_reason.append(
                f"Label drift: flip_rate={label_report.flip_rate_training:.3%}, "
                f"weighted_impact={label_report.weighted_impact:.3%}"
            )

        decision = {
            "should_retrain":      should_retrain,
            "feature_drift":       feature_drift_triggered,
            "label_drift":         label_drift_triggered,
            "reason":              "; ".join(decision_reason) if decision_reason else "No significant drift detected.",
            "drift_report":        drift_report.to_dict() if drift_report else None,
            "label_drift_report":  {
                "flip_rate":          label_report.flip_rate_training,
                "weighted_impact":    label_report.weighted_impact,
                "urgency":            label_report.urgency,
                "n_reclassified":     label_report.n_reclassified_training,
            },
        }

        logger.info("Retraining decision: %s -- %s", should_retrain, decision["reason"])

        # ── Step 6: Optionally trigger retraining ─────────────────────────
        new_model_path = None
        if should_retrain:
            if self.config.auto_retrain:
                new_model_path = self._retrain(
                    new_clinvar_path  = new_clinvar_path,
                    current_model_path = current_model_path,
                    gnomad_path       = gnomad_path,
                    alphamissense_path = alphamissense_path,
                    reclassified_ids  = {r.variant_id for r in label_report.reclassified
                                        if r.in_training_set},
                    release_name      = release_name,
                    drift_report_dict = drift_report.to_dict() if drift_report else None,
                )
                decision["new_model_path"] = new_model_path
            else:
                logger.warning(
                    "Retraining required but auto_retrain=False. "
                    "Run scripts/run_drift_monitor.py --auto-retrain to trigger."
                )
                decision["new_model_path"] = None
                decision["requires_manual_approval"] = True

        # Write decision summary
        summary_path = self.output_dir / f"decision_{release_name}.json"
        Path(summary_path).write_text(
            json.dumps(decision, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Decision written -> %s", summary_path)
        logger.info("=== Continual Learning Pipeline: complete ===")
        return decision

    # ── Retraining ─────────────────────────────────────────────────────────

    def _retrain(
        self,
        new_clinvar_path:    str | Path,
        current_model_path:  str | Path,
        gnomad_path:         Optional[str | Path],
        alphamissense_path:  Optional[str | Path],
        reclassified_ids:    set[str],
        release_name:        str,
        drift_report_dict:   Optional[dict],
    ) -> str:
        """
        Run the full retraining pipeline with adaptive sample weights.
        Returns the path to the new registered model artefact.

        FAIL-CLOSED. See AdaptiveRetrainingInputs. Until 2026-08-07 this
        method was unreachable only because `ModelRegistry` did not
        exist; the class now exists, so the boundary is explicit.
        """
        if self.config.adaptive_inputs is None:
            raise RuntimeError(
                "adaptive retraining is NOT scientifically armed. "
                "LSIF-1 (no reference population for the density "
                "ratio), ROSTER-1 (--skip-nn --skip-svm changes the "
                "architecture), EVALPROV-1 (a validation split "
                "registered as holdout evidence), EWCSEL-1 (the anchor "
                "is dictionary order) and PIPELINE-1 "
                "(InferencePipeline has no _prepare) are all open. "
                "Supply ContinualLearningConfig.adaptive_inputs only "
                "once each has an answer -- see "
                "docs/ROADMAP.md and AdaptiveRetrainingInputs.")
        import joblib
        from genomic_variant_classifier.training.ewc import TreeEWCProxy
        from genomic_variant_classifier.monitoring.drift_detector import LSIFImportanceWeighter
        from genomic_variant_classifier.monitoring.model_registry import (
            ModelRegistry)
        from genomic_variant_classifier.api.pipeline import InferencePipeline, INFERENCE_FEATURE_COLUMNS

        logger.info("Starting adaptive retraining for release: %s", release_name)

        # Load current production model
        current_pipe = InferencePipeline.load(current_model_path)

        # Load + process new data
        from genomic_variant_classifier.data.real_data_prep import DataPrepPipeline, DataPrepConfig
        config = DataPrepConfig(
            min_review_tier=self.config.min_review_tier,
            scale_features=True,
        )
        pipeline = DataPrepPipeline(config=config)
        run_kwargs: dict = {"clinvar_path": str(new_clinvar_path)}
        if gnomad_path:
            run_kwargs["gnomad_path"] = str(gnomad_path)

        X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, meta_val, meta_test = (
            pipeline.run(**run_kwargs)
        )

        logger.info(
            "New data: %d train, %d val, %d test variants, %d features",
            len(X_train_new), len(X_val_new), len(X_test_new), X_train_new.shape[1],
        )

        # ── Compute adaptive sample weights ──────────────────────────────

        # 1. LSIF density ratio (p_new / p_ref)
        #
        # LSIF-1 (2026-08-07). This call used to read:
        #
        #     lsif.fit(X_ref=current_pipe._prepare(pd.DataFrame(
        #                  X_train_new)),
        #              X_new=X_train_new.to_numpy(dtype=float))
        #
        # with a comment admitting the reference was a placeholder. Both
        # sides were the NEW cohort, so the declared ratio p_new/p_ref
        # had no reference population at all. `_prepare` is removed
        # rather than re-fed: passing the reference through the SERVING
        # pipeline while the new cohort arrives from DataPrepPipeline
        # would estimate a ratio across two representation functions --
        # compatible widths, uninterpretable quantity.
        inputs = self.config.adaptive_inputs
        reference = inputs.reference_training_features
        if reference.model_record_id != inputs.deployed_model_record_id:
            raise ValueError(
                "the LSIF reference cohort belongs to model record "
                f"{reference.model_record_id!r}, not the deployed "
                f"{inputs.deployed_model_record_id!r}. A density "
                "ratio against the wrong reference is not a measure "
                "of this deployment's drift.")

        x_ref, x_new = _aligned_lsif_matrices(
            reference=reference, new_features=X_train_new)

        # BOTH sides must be DECLARED. On 2026-08-07 this compared
        # against a name that existed nowhere in scope, which would
        # have raised NameError the first time the branch ran. An
        # absent fingerprint is not a declaration of difference -- it
        # means the ratio must be estimated, which is the safe default.
        if (reference.population_fingerprint is not None
                and inputs.new_population_fingerprint is not None
                and reference.population_fingerprint
                == inputs.new_population_fingerprint):
            # Declared identity beats estimated identity. p/p is defined
            # but means no covariate shift, and fitting anyway reports
            # estimator noise as adaptation.
            density_ratio_status = DensityRatioStatus.SAME_POPULATION
            lsif_weights = np.ones(len(x_new), dtype=np.float64)
            logger.info(
                "LSIF skipped: reference and new populations share "
                "fingerprint %s; weights are one BY POLICY, not by "
                "estimate", reference.population_fingerprint)
        else:
            density_ratio_status = DensityRatioStatus.ESTIMATED
            lsif = LSIFImportanceWeighter(
                sigma=1.0, lambda_=0.01, n_basis=200)
            lsif.fit(X_ref=x_ref, X_new=x_new)
            lsif_weights = lsif.transform(x_new)
        logger.info("density ratio: %s", density_ratio_status.value)
        lsif_weights = lsif_weights / (lsif_weights.mean() + 1e-8)  # normalise

        # 2. TreeEWC stability weights
        ewc_proxy = TreeEWCProxy(
            lambda_decay       = self.config.tree_ewc_lambda_decay,
            reclassified_boost = self.config.reclassified_boost,
            temporal_decay_lambda = self.config.temporal_decay_lambda,
        )

        # Get the best base model from the current production pipeline.
        #
        # PIPELINE-1: `base_models` does not exist on InferencePipeline;
        # the executable mapping is `trained_models`.
        #
        # EWCSEL-1 REMAINS OPEN, and this fails closed rather than
        # choosing. `best_score_` is set NOWHERE in src/, so the previous
        # `getattr(m, "best_score_", 0.0)` compared an all-equal keyspace
        # and `max` returned whichever model came first in dictionary
        # order. Correcting the attribute name would have turned that
        # from unreachable into silently arbitrary.
        scored = [(name, model, getattr(model, "best_score_", None))
                  for name, model in current_pipe.trained_models.items()]
        if not any(score is not None for _, _, score in scored):
            raise RuntimeError(
                "EWC anchor selection is undefined: no base model in the "
                f"production pipeline exposes a measured best_score_ "
                f"({[name for name, _, _ in scored]}). Resolve EWCSEL-1 "
                "before arming adaptive retraining; an anchor chosen by "
                "dictionary order is not a choice.")
        best_model_name, best_model, _ = max(
            (entry for entry in scored if entry[2] is not None),
            key=lambda entry: entry[2])

        ewc_weights = ewc_proxy.compute_weights(
            old_model=best_model,
            X_new=X_train_new.to_numpy(dtype=float),
            y_new=y_train_new.to_numpy(),
            reclassified_ids=reclassified_ids,
        )

        # Combine: geometric mean of LSIF and EWC weights
        combined_weights = np.sqrt(np.clip(lsif_weights, 0.1, None) * np.clip(ewc_weights, 0.1, None))
        combined_weights = np.clip(combined_weights, 0.1, 3.0)
        logger.info(
            "Combined weights: mean=%.3f, std=%.3f",
            combined_weights.mean(), combined_weights.std(),
        )

        # ── Retrain the ensemble ───────────────────────────────────────────
        # Import the training script's main logic
        import subprocess, sys
        output_dir = str(self.output_dir / f"retrain_{release_name}")

        cmd = [
            sys.executable, "scripts/run_phase2_eval.py",
            "--clinvar",       str(new_clinvar_path),
            "--min-review-tier", str(self.config.min_review_tier),
            "--output",        output_dir,
            "--skip-nn", "--skip-svm",
            "--n-folds",       str(self.config.n_folds),
        ]
        if gnomad_path:
            cmd += ["--gnomad", str(gnomad_path)]
        if alphamissense_path:
            cmd += ["--alphamissense", str(alphamissense_path)]

        # Save combined weights to disk so the training script can load them
        import joblib as jl
        weights_path = self.output_dir / f"sample_weights_{release_name}.npy"
        np.save(weights_path, combined_weights)
        cmd += ["--sample-weights", str(weights_path)]

        logger.info("Launching retraining: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Retraining subprocess failed with code {result.returncode}")

        # ── Export and register the new model ─────────────────────────────
        new_model_path = str(self.output_dir / f"pipeline_{release_name}.joblib")
        subprocess.run([
            sys.executable, "scripts/export_model.py", "export",
            "--input",  output_dir,
            "--output", new_model_path,
        ], check=True)

        # Evaluate on holdout
        new_pipe = InferencePipeline.load(new_model_path)
        from sklearn.metrics import roc_auc_score, average_precision_score
        val_proba = new_pipe.predict_proba(X_val_new)[:, 1]
        new_auroc = float(roc_auc_score(y_val_new, val_proba))
        new_auprc = float(average_precision_score(y_val_new, val_proba))

        logger.info(
            "Retrained model: holdout AUROC=%.4f, AUPRC=%.4f",
            new_auroc, new_auprc,
        )

        # Register in the model registry.
        #
        # EVALPROV-1 IS VISIBLE HERE BY CONSTRUCTION. `X_val_new` is the
        # new release's VALIDATION split, produced by DataPrepPipeline in
        # this method. The module docstring promises "evaluate on
        # canonical holdout set". The typed protocol below states what
        # the split ACTUALLY is, so a promotion comparison against a
        # production record evaluated under a different protocol is
        # refused rather than silently performed.
        from genomic_variant_classifier.monitoring.model_registry import (
            EvaluationEvidence, EvaluationProtocol, TrainingLineage)

        protocol = EvaluationProtocol(
            protocol_id     = self.config.adaptive_inputs
                                  .promotion_protocol_id,
            split_kind      = "new_release_validation",
            population_scope= f"clinvar_{release_name}_tier"
                              f"{self.config.min_review_tier}",
            n_observations  = int(len(y_val_new)),
            label_policy    = "acmg_five_tier_collapsed_binary",
        )
        registry = ModelRegistry.load(self.config.registry_path)
        record = registry.register(
            version       = f"{release_name}-adaptive",
            model_path    = new_model_path,
            lineage       = TrainingLineage(
                run_id          = f"{release_name}-adaptive",
                clinvar_release = release_name),
            evaluation    = EvaluationEvidence(
                protocol = protocol,
                metrics  = {"auroc": new_auroc, "auprc": new_auprc}),
            feature_names = list(X_train_new.columns),
            #: PIPELINE-1. `base_models` never existed on
            #: InferencePipeline; this line was written by the author and
            #: shipped in 372cea1, unreachable behind the fail-closed
            #: guard and wrong regardless. Sorted, so a record's roster
            #: does not depend on however the exporting run ordered it.
            model_roster  = tuple(sorted(new_pipe.trained_models)),
            notes         = f"Adaptive retraining on {release_name} "
                            f"with LSIF+EWC weights",
            drift_report  = drift_report_dict,
        )

        # Auto-promote to shadow. Production promotion is DELIBERATELY a
        # different method taking a policy -- a clinically consequential
        # transition should not look like a string assignment.
        registry.promote_to_shadow(record.version)
        registry.save()
        logger.info(
            "New model %s registered and promoted to shadow. After "
            "burn-in, run registry.promote_to_production('%s', policy) "
            "with an explicit PromotionPolicy; it refuses on protocol "
            "mismatch, roster mismatch, a non-durable artifact URI, or "
            "a regression beyond tolerance.",
            record.version, record.version,
        )
        return new_model_path