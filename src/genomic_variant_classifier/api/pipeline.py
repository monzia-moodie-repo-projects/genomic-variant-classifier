"""
src/genomic_variant_classifier/api/pipeline.py
===================
Serialisable InferencePipeline that bundles trained base models and a
stacking meta-learner into a single joblib artifact.

Build the artifact with scripts/export_model.py after a successful
run_phase2_eval.py run.  Load it in the API with InferencePipeline.load().

Usage (inference):
    pipe = InferencePipeline.load("models/phase2_pipeline.joblib")
    result = pipe.predict_single({
        "chrom": "17", "pos": 43094692, "ref": "G", "alt": "A",
        "consequence": "missense_variant",
        "gene_symbol": "BRCA1",
        "alphamissense_score": 0.94,
        "allele_freq": 0.0,
    })
    # -> {"pathogenicity_score": 0.97, "classification": "Pathogenic", "confidence": "high"}
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from genomic_variant_classifier.containment import (
    ContainmentError,
    load_after_admission,
    require_matrix,
    require_scientific_contract,
)
from genomic_variant_classifier.quarantine_policy import QUARANTINED_FEATURES
from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature contract — locked to the exact 79 columns from X_train.parquet
# ---------------------------------------------------------------------------

INFERENCE_FEATURE_COLUMNS: list[str] = list(TABULAR_FEATURES)
# INFERENCE_FEATURE_COLUMNS is derived from TABULAR_FEATURES above; its length
# is enforced by tests/unit/test_feature_count_contract.py rather than asserted
# at import time (a feature edit must not crash the API on import).


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class PipelineMetadata:
    """Immutable provenance stored alongside the model artifact."""

    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    val_auroc: float = 0.0
    n_train: int = 0
    n_features: int = 0
    feature_names: list[str] = field(
        default_factory=lambda: list(INFERENCE_FEATURE_COLUMNS)
    )
    model_version: str = "phase2"


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------


def _write_model_manifest(artifact_path, feature_names=None):
    """Write a JSON manifest: library versions, and (since 2026-09-23) the artifact's SHA-256 and the
    feature list its models consume -- what load() needs to admit the artifact BEFORE opening it."""
    import json, platform, importlib.metadata
    from datetime import datetime, timezone

    artifact_path = Path(artifact_path)
    libraries = [
        "numpy",
        "scikit-learn",
        "catboost",
        "lightgbm",
        "xgboost",
        "joblib",
        "pandas",
        "scipy",
    ]
    manifest = {
        "artifact": artifact_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {lib: importlib.metadata.version(lib) for lib in libraries},
    }
    if feature_names is not None:
        import hashlib
        h = hashlib.sha256()
        with artifact_path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        manifest["artifact_sha256"] = h.hexdigest()
        manifest["feature_names"] = list(feature_names)
    manifest_path = artifact_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


class InferencePipeline:
    """
    Wraps fitted tabular base models + stacking meta-learner for deployment.

    At inference time the pipeline:
      1. Optionally scores variants via GNNScorer → adds gnn_score column
      2. Calls engineer_features() to derive the INFERENCE_FEATURE_COLUMNS
      3. Applies the StandardScaler (if present)
      4. Drives each base model with a numpy array → stacks predictions
      5. Feeds the stack to the meta-learner
      6. Returns a structured result dict with score, classification, and confidence

    Sequence-based models (CNN) are excluded at export time — they require
    FASTA context that is not available at API inference time.

    If no scaler is provided (scaler=None) step 3 is skipped — safe for
    tree-based-only ensembles where scaling has no effect.

    If no gnn_scorer is provided (gnn_scorer=None) gnn_score defaults to 0.5
    (ambiguous / not available) for all variants.
    """

    def __init__(
        self,
        trained_models: dict,
        meta_learner,
        scaler=None,
        metadata: Optional[PipelineMetadata] = None,
        gnn_scorer=None,
        preprocessor=None,
    ) -> None:
        self.trained_models = trained_models
        self.meta_learner = meta_learner
        self.scaler = scaler
        self.metadata = metadata or PipelineMetadata()
        self.gnn_scorer = gnn_scorer  # Optional GNNScorer instance
        # The TRAINING missing-value policy, fitted on the training fold by
        # VariantEnsemble.fit. Serving must not fit one: a single variant offers
        # no basis for a median, and refitting per request would make the
        # imputed value depend on which variants happened to arrive together.
        # save() pickles self, so this travels with the artefact unchanged.
        self.preprocessor_ = preprocessor
        self._require_admissible()   # CONTAINMENT: a quarantine-era model cannot even be assembled

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_variant_ensemble(
        cls,
        ensemble,
        scaler=None,
        feature_names: Optional[list[str]] = None,
        val_auroc: float = 0.0,
        n_train: int = 0,
    ) -> "InferencePipeline":
        """
        Extract trained base models and meta-learner from a fitted VariantEnsemble.

        Sequence-based models ("cnn_1d") are excluded because they require a
        FASTA context window not available at API inference time.
        """
        trained_models = {
            name: model
            for name, model in ensemble.trained_models_.items()
            if name != "cnn_1d"
        }
        if not trained_models:
            raise ValueError(
                "No tabular models found in VariantEnsemble.trained_models_.  "
                "Ensure the ensemble was fitted before calling from_variant_ensemble()."
            )
        feature_names = feature_names or INFERENCE_FEATURE_COLUMNS
        metadata = PipelineMetadata(
            val_auroc=val_auroc,
            n_train=n_train,
            n_features=len(feature_names),
            feature_names=feature_names,
        )
        return cls(
            trained_models=trained_models,
            meta_learner=ensemble.meta_learner,
            scaler=scaler,
            metadata=metadata,
            # getattr, not attribute access: an ensemble pickled before the
            # fitted policy existed has no such attribute, and that is a
            # LEGACY ARTEFACT rather than an error. _apply_missing_value_policy
            # records the distinction; it does not paper over it.
            preprocessor=getattr(ensemble, "preprocessor_", None),
        )

    # ------------------------------------------------------------------
    # Missing-value rendering
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Containment boundary (serving, loading, construction)
    # ------------------------------------------------------------------

    def _declared_features(self) -> list[str]:
        """The feature list the fitted models consume: the scaler's (authoritative) or the contract's."""
        scaler_features = getattr(self.scaler, "feature_names_in_", None) if self.scaler is not None else None
        if scaler_features is None:
            return list(INFERENCE_FEATURE_COLUMNS)
        # numpy string subtypes -> str; any NON-string name is left as is so the contract check refuses it.
        return [str(c) if isinstance(c, str) else c for c in scaler_features]

    def _require_admissible(self) -> None:
        """Refuse a pipeline whose models, metadata or missing-value policy use a quarantined feature.

        Runs at construction, immediately after load, and before every prediction -- BEFORE feature
        engineering, so a quarantine-era policy cannot fail first with an unrelated library error.
        Every model trained before 2026-09-22 used the 95-feature contract and is refused here.
        """
        require_scientific_contract(self._declared_features(), QUARANTINED_FEATURES)
        meta_names = getattr(getattr(self, "metadata", None), "feature_names", None)
        if meta_names:
            require_scientific_contract(meta_names, QUARANTINED_FEATURES)
        policy_names = getattr(getattr(self, "preprocessor_", None), "feature_names", None)
        if policy_names:
            require_scientific_contract(policy_names, QUARANTINED_FEATURES)

    def _model_matrix(self, X: "pd.DataFrame") -> tuple["pd.DataFrame", list[str]]:
        """Exactly the declared features, in order, then scaled. NEVER zero-fills.

        Until 2026-09-23 every declared column absent from the input was set to 0.0 here, so a
        95-feature model was served with the four quarantined features fabricated as zeros.
        """
        model_features = self._declared_features()
        missing = [c for c in model_features if c not in X.columns]
        if missing:
            raise ContainmentError(
                f"The model expects {len(missing)} feature(s) the input does not provide: {missing[:10]}. "
                "They are never zero-filled; supply them, or re-export a model for the current contract.")
        frame = X[model_features]
        require_matrix(frame, model_features, QUARANTINED_FEATURES)
        if self.scaler is not None:
            frame = pd.DataFrame(self.scaler.transform(frame), columns=model_features, index=X.index)
        return frame, model_features

    def _apply_missing_value_policy(self, X: "pd.DataFrame") -> "pd.DataFrame":
        """Impute declared missingness using the TRAINING medians.

        PIPELINE-FILL-1 and the serving half of the missingness contract.

        engineer_features stopped fabricating values on 2026-08-11 (commit
        48985d6), so a variant whose gene has no gnomAD constraint entry now
        arrives with genuine NaN in gene_constraint_oe and gene_is_constrained.
        Measured the same day, by traceback:

            sklearn/utils/validation.py:182
            ValueError: Input X contains NaN.
            LogisticRegression does not accept missing values encoded as NaN.

        That is a SERVING-TIME failure on a model fitted from clean data --
        exactly the case a fit-time guard cannot see.

        The medians come from the TRAINING FOLD and travel with the artefact.
        Fitting a preprocessor here would compute a median from whatever rows
        happened to arrive, which for one variant is not a statistic at all.

        A LEGACY ARTEFACT PASSES THROUGH, DELIBERATELY. A pipeline pickled
        before the fitted policy existed has no preprocessor_, and this method
        returns X unchanged rather than raising. Two reasons: raising would
        refuse artefacts that scored correctly for months, and an estimator
        that cannot consume NaN raises its OWN error, with a clearer message
        than any we could substitute. The absence is logged once, not hidden.
        """
        preproc = getattr(self, "preprocessor_", None)
        if preproc is None:
            residual = [c for c in X.columns if X[c].isna().any()]
            if residual:
                logger.warning(
                    "This pipeline carries NO fitted missing-value policy (a "
                    "legacy artefact), and %d column(s) arrive with missing "
                    "values: %s. They are passed through UNCHANGED. An "
                    "estimator that cannot consume NaN will raise. Re-export "
                    "the pipeline from a VariantEnsemble fitted after "
                    "2026-08-11 to carry the training medians.",
                    len(residual), residual[:10],
                )
            return X
        # transform() emits the declared features PLUS their availability
        # masks. The masks are dropped downstream by the X[model_features]
        # selection, because these models were trained on the features alone
        # and must not be handed columns they have never seen. The three-arm
        # availability ablation decides whether masks ever become features.
        return preproc.transform(X)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def predict_single(self, variant: dict[str, Any]) -> dict[str, Any]:
        """
        Predict pathogenicity for one variant.

        Parameters
        ----------
        variant : dict
            Raw variant fields.  At minimum supply chrom, pos, ref, alt.

            An absent field does NOT receive a population median. It receives
            a FIXED CONSTANT chosen per feature by engineer_features (see
            src/genomic_variant_classifier/models/variant_ensemble.py):
            allele_freq 0.0, ref and alt "A", finngen_enrichment 1.0, and so
            on. MEASURED 2026-09-14: an absent allele_freq becomes af_raw 0.0,
            af_log10 -8.0 and af_is_absent 1 -- indistinguishable from an
            OBSERVED frequency of exactly zero.

            Training-fold medians enter only through the fitted
            missing-value policy (PIPELINE-FILL-1, _apply_missing_value_policy
            below), which imputes columns that are PRESENT WITH NaN. It does
            not act on fields absent from the input, and it does not act at
            all on a legacy artefact carrying no preprocessor_.

            Supply every field whose value matters. A constant is a fabricated
            measurement, not a neutral one.

        Returns
        -------
        dict with keys:
            pathogenicity_score  float in [0, 1]
            classification       5-tier ACMGish label
            confidence           "high" | "medium" | "low"
        """
        return self._predict_df(pd.DataFrame([variant]))[0]

    def predict_batch(self, variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict pathogenicity for a list of variant dicts."""
        if not variants:
            return []
        return self._predict_df(pd.DataFrame(variants))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return P(pathogenic) as a 1-D array of shape (n,).

        Parameters
        ----------
        df : pd.DataFrame
            Raw variant rows — same schema as predict_batch input dicts.
            If gnn_scorer is set, gnn_score is computed automatically.
            Otherwise gnn_score defaults to 0.5 (no GNN / ambiguous).
        """
        self._require_admissible()
        enriched = df.copy()

        # --- Optional GNN scoring (adds gnn_score column) ---
        if getattr(self, "gnn_scorer", None) is not None:
            try:
                gene_symbols = enriched.get(
                    "gene_symbol",
                    pd.Series([""] * len(enriched), index=enriched.index),
                ).fillna("")
                enriched["gnn_score"] = gene_symbols.map(
                    lambda g: self.gnn_scorer.score(g)
                )
            except ContainmentError:
                raise  # a refusal is not a scorer failure; never default it to 0.5
            except Exception as exc:
                logger.warning(
                    "GNNScorer failed (%s) -- defaulting gnn_score to 0.5.", exc
                )
                enriched["gnn_score"] = 0.5

        X = engineer_features(enriched)
        X = self._apply_missing_value_policy(X)

        X, model_features = self._model_matrix(X)

        X_np = X[model_features].values
        X_df_cat = None  # lazy — only built if catboost is in trained_models
        base_preds_list = []
        for name, model in self.trained_models.items():
            if name == "catboost":
                if X_df_cat is None:
                    X_df_cat = X[model_features]  # DataFrame preserves column names
                base_preds_list.append(model.predict_proba(X_df_cat)[:, 1])
            else:
                base_preds_list.append(model.predict_proba(X_np)[:, 1])
        base_preds = np.column_stack(base_preds_list)
        return self.meta_learner.predict_proba(base_preds)[:, 1]

    def predict_proba_with_uncertainty(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Return pathogenicity scores and uncertainty estimates.

        Returns
        -------
        dict with keys:
          proba                  (n,) mean pathogenicity probability
          uncertainty_epistemic  (n,) variance across base models (model uncertainty;
                                      high when base models disagree)
          uncertainty_aleatoric  (n,) binary entropy of meta-learner output
                                      (data uncertainty; high near decision boundary)

        Epistemic uncertainty is useful for flagging variants where the
        ensemble members strongly disagree -- a signal to collect more
        evidence before reporting.  Aleatoric uncertainty is inherent to
        the data; it cannot be reduced by training more models.
        """
        self._require_admissible()
        enriched = df.copy()

        if getattr(self, "gnn_scorer", None) is not None:
            try:
                gene_symbols = enriched.get(
                    "gene_symbol",
                    pd.Series([""] * len(enriched), index=enriched.index),
                ).fillna("")
                enriched["gnn_score"] = gene_symbols.map(
                    lambda g: self.gnn_scorer.score(g)
                )
            except ContainmentError:
                raise  # a refusal is not a scorer failure; never default it to 0.5
            except Exception as exc:
                logger.warning(
                    "GNNScorer failed (%s) -- defaulting gnn_score to 0.5.", exc
                )
                enriched["gnn_score"] = 0.5

        X = engineer_features(enriched)
        X = self._apply_missing_value_policy(X)

        X, model_features = self._model_matrix(X)

        X_np = X[model_features].values
        X_df_cat = None  # lazy — only built if catboost is in trained_models
        base_preds_list = []
        for name, model in self.trained_models.items():
            if name == "catboost":
                if X_df_cat is None:
                    X_df_cat = X[model_features]  # DataFrame preserves column names
                base_preds_list.append(model.predict_proba(X_df_cat)[:, 1])
            else:
                base_preds_list.append(model.predict_proba(X_np)[:, 1])
        base_preds = np.column_stack(
            base_preds_list
        )  # shape (n_variants, n_base_models)

        # Epistemic: variance across base model predictions
        uncertainty_epistemic = base_preds.var(axis=1)

        proba = self.meta_learner.predict_proba(base_preds)[:, 1]

        # Aleatoric: binary entropy of the final probability
        p = np.clip(proba, 1e-7, 1.0 - 1e-7)
        uncertainty_aleatoric = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

        return {
            "proba": proba,
            "uncertainty_epistemic": uncertainty_epistemic,
            "uncertainty_aleatoric": uncertainty_aleatoric,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict_df(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        result = self.predict_proba_with_uncertainty(df)
        proba = result["proba"]
        epistemic = result["uncertainty_epistemic"]
        aleatoric = result["uncertainty_aleatoric"]
        return [
            _score_to_result(float(p), float(e), float(a))
            for p, e, a in zip(proba, epistemic, aleatoric)
        ]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import joblib

        self._require_admissible()   # never export a quarantine-era pipeline
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        _write_model_manifest(path, feature_names=self._declared_features())
        logger.info("InferencePipeline saved -> %s", path)

    @classmethod
    def load(cls, path: str | Path, *, consumer: str, registry_path: str | Path,
             authority=None) -> "InferencePipeline":
        """Load through the ONE admission route (model_admission; owner ruling 2026-09-23, point 2).

        Refused BEFORE deserialization unless a deployment-registry record measured these bytes and the
        cutover authority ALLOWs this consumer -- today no authority can (gate C10 is not implemented), so
        every real load refuses explicitly. Then precisely the digest-verified snapshot is deserialized, and
        its declared features and executable roster must match the record. The manifest save() writes is
        integrity metadata; a checksum beside a model is not a binding, so it is not an admission input.
        """
        import joblib

        from genomic_variant_classifier.api.attribution import (
            RosterAlignment,
            roster_alignment,
            served_model_roster,
        )
        from genomic_variant_classifier.model_admission import DEFAULT_AUTHORITY, admit_artifact

        path = Path(path)
        admission = admit_artifact(path, consumer=consumer, registry_path=registry_path,
                                   authority=authority if authority is not None else DEFAULT_AUTHORITY)
        record = admission.record
        obj = load_after_admission(path, admit=lambda: None, artifact_sha256=record.artifact.sha256,
                                   deserialize=joblib.load)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected InferencePipeline, got {type(obj)}")
        obj._require_admissible()
        if tuple(obj._declared_features()) != tuple(record.feature_names):
            raise ContainmentError("the loaded pipeline's features differ from its registry record")
        state, detail = roster_alignment(record, served_model_roster(obj))
        if state not in (RosterAlignment.EXACT, RosterAlignment.SERVING_SUBSET):
            raise ContainmentError(f"the loaded pipeline's roster does not match its registry record: {detail}")
        logger.info(
            "InferencePipeline loaded for %s: record=%s  features=%d  created=%s",
            consumer, record.record_id, obj.metadata.n_features, obj.metadata.created_at,
        )
        return obj


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _score_to_result(
    score: float,
    uncertainty_epistemic: float = 0.0,
    uncertainty_aleatoric: float = 0.0,
) -> dict[str, Any]:
    """Convert a raw pathogenicity probability to a labelled result dict."""
    from genomic_variant_classifier.api.schemas import score_to_classification

    classification, confidence = score_to_classification(score)
    return {
        "pathogenicity_score": round(score, 4),
        "classification": classification,
        "confidence": confidence,
        "uncertainty_epistemic": round(uncertainty_epistemic, 6),
        "uncertainty_aleatoric": round(uncertainty_aleatoric, 6),
    }
