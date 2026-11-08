"""
src/genomic_variant_classifier/api/pipeline.py
===================
Serialisable InferencePipeline: the fitted members and the DECLARED combiner, scored through one shared path.

Build the artifact with scripts/export_model.py after a successful run_phase2_eval.py run.  Load it in the API with
InferencePipeline.load(), which refuses unless the deployment registry and the cutover authority admit it.

Package 1, set 1a (2026-11-07; frozen acceptance matrix PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07.md, gates G3-G6):

* The selected predictor is the stored convex blend (models/combiner.py): exact stored weights, the FULL ordered fitted
  roster and the member prediction stages. Until this change the pipeline served the logistic meta-learner while
  every evaluation reported the blend (measured on Run 15: 99th-percentile difference 0.237, 3,559 changed calls). The
  logistic stacker is a separately identified research candidate that this module never calls.
* A fitted member this process cannot execute (cnn_1d needs a sequence window serving does not have) makes the blend
  uncomputable, so construction REFUSES it. The previous twelve-model projection was a different, unevaluated
  predictor (and with 13 stored stacker inputs it raised at the first request anyway).
* score() is the ONE scoring path: each member called once, the combiner applied once, every payload an identity-bound
  ProbabilityBlock. predict_proba(), predict_single() and predict_batch() all reduce its result; the HTTP layer and
  offline callers share serialize_predictions().
* Response schema 3.0.0: full-precision score, one tie rule (positive_if_p_ge_0.5), identities, units and statuses.
  No rounding, clipping, five-tier relabelling or score-distance "confidence"; no decision policy is qualified.

Usage (inference):
    pipe = InferencePipeline.load(path, consumer="...", registry_path=...)
    record = pipe.predict_single({"chrom": "17", "pos": 43094692, "ref": "G", "alt": "A"})
    # -> a schema-3.0.0 record: pathogenicity_score, model_call, identities, quantities with units and statuses
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from genomic_variant_classifier.containment import (
    ContainmentError,
    load_after_admission,
    require_matrix,
    require_scientific_contract,
)
from genomic_variant_classifier.models.combiner import (
    BLEND_KERNEL,
    ConvexBlendSpec,
    SELECTED_PREDICTOR_SCOPE,
    MemberSlot,
    member_draw_schedule,
    member_modalities,
    require_supported_modalities,
    score_once,
)
from genomic_variant_classifier.models.disagreement import draw_decomposition
from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features
from genomic_variant_classifier.prediction_contract import (
    PREDICTION_SCHEMA_VERSION,
    TIE_RULE,
    Binding,
    ContractError,
    Observation,
    ObservationSet,
    ProbabilityBlock,
    Quantity,
    binary_entropy,
    canonical_digest,
    positive_column,
    variant_identity_of,
)
from genomic_variant_classifier.quarantine_policy import QUARANTINED_FEATURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature contract -- the tabular columns the fitted members consume
# ---------------------------------------------------------------------------

INFERENCE_FEATURE_COLUMNS: list[str] = list(TABULAR_FEATURES)
# INFERENCE_FEATURE_COLUMNS is derived from TABULAR_FEATURES above; its length
# is enforced by tests/unit/test_feature_count_contract.py rather than asserted
# at import time (a feature edit must not crash the API on import).

# ---------------------------------------------------------------------------
# Declared scope, modalities and identities of the serving path
# ---------------------------------------------------------------------------

#: What this process can supply to a member: the rendered tabular matrix. No sequence windows exist at serving time.
SERVING_MODALITIES: frozenset = frozenset({"tabular"})

#: The selected predictor's scope and each member's modalities: ONE definition each, in models/combiner.py (moved there
#: unchanged in package 1, set 1b, so the offline split scorer binds the same combiner identity serving does).
SERVING_SCOPE = SELECTED_PREDICTOR_SCOPE

_UNIT_CONTEXT_SHA256 = canonical_digest("gvc.unit-context.v1", {
    "unit": "one_pathogenicity_prediction_per_submitted_row", "positive_class": 1})
_SERVING_SPLIT_SHA256 = canonical_digest("gvc.prediction-split.v1", {"split": "serving_request_no_split"})
_COMBINER_EXECUTION_SHA256 = canonical_digest("gvc.combiner-execution-adapter.v1", {
    "adapter": "gvc.inference-pipeline.score.v1", "kernel": BLEND_KERNEL})
_RESULT_DRAW_ID = "selected_convex_blend"
_WEIGHTED_DIAGNOSTIC_UNAVAILABLE = (
    "weighted_mixture_kernel_not_yet_qualified_against_the_800_digit_oracle")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class PipelineMetadata:
    """Provenance stored alongside the model artifact."""

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


# ---------------------------------------------------------------------------
# Request identity
# ---------------------------------------------------------------------------

def _json_value(value: Any) -> Any:
    """A raw input cell as canonical JSON: missing (None or NaN) -> null; NumPy scalars -> Python; no other coercion."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ContractError("an infinite input value cannot be identified")
        return value
    if isinstance(value, (bool, int, str)):
        return value
    if value is pd.NA or value is pd.NaT:
        return None
    raise ContractError(f"input value of type {type(value).__name__} has no canonical identity")


def _variant_identity(record: Mapping[str, Any]) -> str:
    return variant_identity_of(record)   # one definition (prediction_contract), shared with the training writers


def rows_from_frame(df: pd.DataFrame) -> ObservationSet:
    """The ordered prediction units of a submitted frame (G1).

    dataset_sha256 binds the full submitted content (column names and every cell, missing as null) in submitted
    order; source_row_key is the position within that dataset; one prediction unit per row.
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        raise ContractError("a nonempty DataFrame of submitted variants is required")
    columns = [str(c) for c in df.columns]
    if len(set(columns)) != len(columns):
        raise ContractError("duplicate input column names")
    records = [{c: _json_value(v) for c, v in zip(columns, row)} for row in df.itertuples(index=False, name=None)]
    h = hashlib.sha256(b"gvc.submitted-rows.v1\x00")
    h.update(json.dumps(columns, ensure_ascii=True, separators=(",", ":")).encode("ascii"))
    for record in records:
        h.update(b"\n")
        h.update(json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                            allow_nan=False).encode("ascii"))
    dataset = h.hexdigest()
    return ObservationSet(tuple(
        Observation(dataset, f"position:{i}", _variant_identity(r), _UNIT_CONTEXT_SHA256)
        for i, r in enumerate(records)))


# ---------------------------------------------------------------------------
# The shared result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ScoredBatch:
    """One logical request's result: the selected predictor's block and the member blocks it binds."""

    result: ProbabilityBlock
    members: tuple[ProbabilityBlock, ...]
    combiner: ConvexBlendSpec
    preprocessing_status: str
    __hash__ = None  # type: ignore[assignment]

    def scores(self) -> np.ndarray:
        return np.array(self.result.positive_probability(), dtype=np.float64)

    def member_matrix(self) -> np.ndarray:
        return np.column_stack([b.positive_probability() for b in self.members])


def serialize_predictions(scored: ScoredBatch) -> list[dict[str, Any]]:
    """The ONE serializer for the HTTP single, batch and offline paths: full precision, no model calls, no triage."""
    if type(scored) is not ScoredBatch:
        raise ContractError("ScoredBatch required")
    p = scored.scores()
    entropy = binary_entropy(p)
    members = scored.member_matrix()
    roster = draw_decomposition(members.T) if members.shape[1] >= 2 else None
    records = []
    for i, row in enumerate(scored.result.rows.rows):
        if roster is None:
            spread = Quantity(None, "probability_squared", "unavailable", "fewer_than_two_fitted_members")
            disagreement = Quantity(None, "nats", "unavailable", "fewer_than_two_fitted_members")
            d_status, method = "unavailable", None
        else:
            spread = Quantity(float(roster["between_draw_spread_variance_probability_squared"][i]),
                              "probability_squared", "defined")
            d_value = roster["disagreement_nats"][i]
            disagreement = (Quantity(float(d_value), "nats", "defined") if d_value is not None
                            else Quantity(None, "nats", "unresolved", roster["disagreement_reason"][i]))
            d_status, method = roster["disagreement_status"][i], roster["disagreement_method"]
        records.append({
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "variant_id": row.variant_identity,
            "source_row_key": row.source_row_key,
            "source_observation_id": row.source_observation_id,
            "prediction_unit_id": row.prediction_unit_id,
            "ordered_rows_sha256": scored.result.rows.digest,
            "result_block_sha256": scored.result.digest,
            "combiner": {"kind": "convex_probability_blend", "combiner_sha256": scored.combiner.digest,
                         "kernel": scored.combiner.kernel, "members": list(scored.combiner.member_names)},
            "positive_class": 1,
            "pathogenicity_score": float(p[i]),
            "model_call": int(p[i] >= 0.5),
            "tie_rule": TIE_RULE,
            "predictive_entropy_nats": Quantity(float(entropy[i]), "nats", "defined").as_dict(),
            "roster_research_view": {
                "interpretation": "descriptive_uniform_view_of_the_fitted_members_not_the_selected_blend",
                "weighting": "uniform_over_fitted_members", "ddof": 0,
                "between_member_spread_variance": spread.as_dict(),
                "member_disagreement": disagreement.as_dict(),
                "disagreement_status": d_status,
                "method": method,
            },
            "weighted_blend_diagnostic": {"status": "unavailable", "reason": _WEIGHTED_DIAGNOSTIC_UNAVAILABLE},
            "preprocessing_status": scored.preprocessing_status,
            "calibration_status": "not_qualified_by_this_record",
            "decision_policy": {"status": "unavailable", "reason": "no_qualified_policy_bound"},
            "feature_attribution": {"status": "unavailable",
                                    "reason": "no_attribution_method_bound_to_the_selected_predictor"},
            "coverage_interval": {"status": "unavailable", "reason": "no_conformal_calibration_bound"},
        })
    json.dumps(records, allow_nan=False)   # the same finite serialisation policy for every transport
    return records


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------


class InferencePipeline:
    """
    The fitted members, the declared convex blend, and the training-time rendering they were fitted with.

    At inference time the pipeline:
      1. Optionally scores variants via GNNScorer -> adds gnn_score column
      2. Calls engineer_features() to derive the INFERENCE_FEATURE_COLUMNS
      3. Applies the TRAINING missing-value policy (reported when absent; never refitted)
      4. Applies the StandardScaler (if present)
      5. Calls every fitted member ONCE, in the fitted order, as identity-bound probability blocks
      6. Applies the stored blend ONCE: the exact stored-value dot product, rounded once

    If no gnn_scorer is provided (gnn_scorer=None) gnn_score defaults to 0.5 (ambiguous / not available) for all
    variants.
    """

    def __init__(
        self,
        trained_models: dict,
        combiner: ConvexBlendSpec,
        scaler=None,
        metadata: Optional[PipelineMetadata] = None,
        gnn_scorer=None,
        preprocessor=None,
        research_candidates: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.trained_models = trained_models
        self.combiner = combiner
        self.scaler = scaler
        self.metadata = metadata or PipelineMetadata()
        self.gnn_scorer = gnn_scorer  # Optional GNNScorer instance
        # The TRAINING missing-value policy, fitted on the training fold by
        # VariantEnsemble.fit. Serving must not fit one: a single variant offers
        # no basis for a median, and refitting per request would make the
        # imputed value depend on which variants happened to arrive together.
        # save() pickles self, so this travels with the artefact unchanged.
        self.preprocessor_ = preprocessor
        #: Separately identified candidates (for example the logistic stacker) carried for research only. No
        #: method of this class calls them (gate G3.3).
        self.research_candidates = dict(research_candidates or {})
        self._require_admissible()   # CONTAINMENT: a quarantine-era model cannot even be assembled
        self._require_combiner()     # the declared blend, the full roster, supported modalities (G3)

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
        The fitted ensemble's members (ALL of them, in the fitted order) and its declared convex blend.

        A member serving cannot execute (cnn_1d) is not dropped: construction refuses, because the stored blend
        needs every member (owner ruling 2026-11-07: unsupported projections are refused).
        """
        if not getattr(ensemble, "trained_models_", None):
            raise ValueError(
                "No fitted members found in VariantEnsemble.trained_models_.  "
                "Ensure the ensemble was fitted before calling from_variant_ensemble()."
            )
        spec = ensemble.convex_blend_spec(SERVING_SCOPE)
        feature_names = feature_names or INFERENCE_FEATURE_COLUMNS
        metadata = PipelineMetadata(
            val_auroc=val_auroc,
            n_train=n_train,
            n_features=len(feature_names),
            feature_names=feature_names,
        )
        return cls(
            trained_models=dict(ensemble.trained_models_),
            combiner=spec,
            scaler=scaler,
            metadata=metadata,
            # getattr, not attribute access: an ensemble pickled before the
            # fitted policy existed has no such attribute, and that is a
            # LEGACY ARTEFACT rather than an error. _apply_missing_value_policy
            # records the distinction; it does not paper over it.
            preprocessor=getattr(ensemble, "preprocessor_", None),
            research_candidates={"logistic_stacker": ensemble.meta_learner},
        )

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

    def preprocessing_identity(self) -> dict[str, Any]:
        """What the fitted input rendering IS: the missing-value policy and the scaler, as values (never pickles)."""
        preproc = getattr(self, "preprocessor_", None)
        if preproc is None:
            policy: Any = "absent"
        else:
            identity = getattr(preproc, "identity_", None)
            policy = {
                "class": f"{type(preproc).__module__}.{type(preproc).__qualname__}",
                "feature_names": [str(f) for f in getattr(preproc, "feature_names", ())],
                "identity": identity.as_dict() if hasattr(identity, "as_dict") else None,
                "medians_binary64_hex": {str(k): float(v).hex()
                                         for k, v in sorted(getattr(preproc, "medians_", {}).items())},
                "capability": str(getattr(getattr(preproc, "capability", None), "value", None)),
            }
        scaler = getattr(self, "scaler", None)
        if scaler is None:
            scaling: Any = "absent"
        else:
            def _hex(values: Any) -> Any:
                return None if values is None else [float(v).hex() for v in np.ravel(values)]
            scaling = {"class": f"{type(scaler).__module__}.{type(scaler).__qualname__}",
                       "mean_binary64_hex": _hex(getattr(scaler, "mean_", None)),
                       "scale_binary64_hex": _hex(getattr(scaler, "scale_", None))}
        return {"missing_value_policy": policy, "scaler": scaling}

    def predictor_contract_sha256(self) -> str:
        """The identity of the fitted predictor this object executes (owner ruling 2026-11-07, decision 38f15c7a).

        Binds the selected combiner (ordered member slots with fitted member identities and prediction stages, exact
        stored weights, numerical kernel, scope), the declared input features, the fitted rendering (scaler and
        missing-value policy) and the class order. A registry record must carry this digest for an inference load;
        matching member NAMES is necessary and never sufficient.
        """
        spec = self._require_combiner()
        return canonical_digest("gvc.predictor-contract.v1", {
            "combiner_sha256": spec.digest,
            "declared_features": [str(f) for f in self._declared_features()],
            "rendering": self.preprocessing_identity(),
            "class_order": [0, 1], "positive_class": 1})

    def _require_combiner(self) -> ConvexBlendSpec:
        """The declared blend over exactly the held members, in its fitted order, all executable here (G3)."""
        spec = getattr(self, "combiner", None)
        if type(spec) is not ConvexBlendSpec:
            raise ContractError(
                "this pipeline declares no authoritative combiner (a pipeline pickled before 2026-11-07 carries the "
                "logistic meta-learner, which is not the evaluated predictor and is never substituted). "
                "Re-export it from its VariantEnsemble.")
        models = getattr(self, "trained_models", None)
        if not isinstance(models, Mapping) or tuple(models) != spec.member_names:
            raise ContractError(f"the held members {list(models or ())} are not the declared blend's fitted roster "
                                f"in its fitted order {list(spec.member_names)}")
        require_supported_modalities(spec, required_by_member=member_modalities(spec.member_names),
                                     available=SERVING_MODALITIES)
        return spec

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

    def _apply_missing_value_policy(self, X: "pd.DataFrame") -> tuple["pd.DataFrame", str]:
        """Impute declared missingness using the TRAINING medians; return the frame and the policy status.

        PIPELINE-FILL-1 and the serving half of the missingness contract.

        engineer_features stopped fabricating values on 2026-08-11 (commit 48985d6), so a variant whose gene has
        no gnomAD constraint entry arrives with genuine NaN in gene_constraint_oe and gene_is_constrained. The
        medians come from the TRAINING FOLD and travel with the artefact; fitting a preprocessor here would compute
        a median from whatever rows happened to arrive, which for one variant is not a statistic at all.

        A LEGACY ARTEFACT PASSES THROUGH, DELIBERATELY, AND THE RESULT SAYS SO. A pipeline built without the fitted
        policy has no preprocessor_; X is returned unchanged (an estimator that cannot consume NaN raises its own
        error) and every record carries preprocessing_status "absent_legacy_artifact_passed_through".
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
            return X, "absent_legacy_artifact_passed_through"
        # transform() emits the declared features PLUS their availability
        # masks. The masks are dropped downstream by the X[model_features]
        # selection, because these models were trained on the features alone
        # and must not be handed columns they have never seen.
        return preproc.transform(X), "fitted_training_policy_applied"

    # ------------------------------------------------------------------
    # The ONE scoring path (G4)
    # ------------------------------------------------------------------

    def _member_schedule(self, name: str, model: Any) -> Any:
        return member_draw_schedule(name, model)   # one definition (combiner), shared with the offline split scorer

    def score(self, df: pd.DataFrame) -> ScoredBatch:
        """Render once, call each member once, apply the declared blend once; every payload identity-bound."""
        self._require_admissible()
        spec = self._require_combiner()
        rows = rows_from_frame(df)
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
        X, preprocessing_status = self._apply_missing_value_policy(X)
        frame, model_features = self._model_matrix(X)
        try:
            matrix = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"the rendered model matrix is not numeric: {exc}") from exc
        ordered_input = canonical_digest("gvc.member-ordered-input.v1", {
            "rows_sha256": rows.digest, "columns": list(model_features),
            "matrix_float64_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
            "preprocessing_status": preprocessing_status, "scaled": self.scaler is not None})
        X_np = frame.values

        bindings, draw_ids = [], []
        for slot in spec.members:
            model = self.trained_models[slot.name]
            rendering = "named_dataframe" if slot.name == "catboost" else "ndarray"
            bindings.append(Binding(
                producer_sha256=slot.artifact_sha256,
                input_contract_sha256=canonical_digest("gvc.member-input-contract.v1", {
                    "features": list(model_features), "rendering": rendering}),
                split_sha256=_SERVING_SPLIT_SHA256,
                draw_schedule_sha256=canonical_digest("gvc.member-draw-schedule.v1", {
                    "member": slot.name, "schedule": self._member_schedule(slot.name, model)}),
                ordered_input_sha256=ordered_input,
                execution_sha256=canonical_digest("gvc.member-execution.v1", {
                    "adapter": "gvc.inference-pipeline.member-adapter.v1", "member": slot.name,
                    "served_class": f"{type(model).__module__}.{type(model).__qualname__}",
                    "rendering": rendering})))
            draw_ids.append(f"{slot.name}:single_evaluation")
        slots = {m.name: m for m in spec.members}
        expected = dict(zip(spec.member_names, zip(bindings, draw_ids)))

        def predict_member(name: str, model: Any) -> tuple[MemberSlot, ProbabilityBlock]:
            X_input = frame if name == "catboost" else X_np
            values = model.predict_proba(X_input)
            _positive, classes, _conversion = positive_column(model, values, len(rows))
            binding, draw_id = expected[name]
            return slots[name], ProbabilityBlock.from_array(
                rows=rows, class_order=classes, positive_class=1, binding=binding, draw_id=draw_id,
                probabilities=values)

        result, members = score_once(
            spec, models=self.trained_models, predict_member=predict_member, rows=rows,
            member_bindings=tuple(bindings), member_draw_ids=tuple(draw_ids), result_draw_id=_RESULT_DRAW_ID,
            combiner_execution_sha256=_COMBINER_EXECUTION_SHA256)
        return ScoredBatch(result=result, members=members, combiner=spec,
                           preprocessing_status=preprocessing_status)

    # ------------------------------------------------------------------
    # Public inference API -- all reductions of score()
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(pathogenic) under the declared blend, a 1-D float64 array of shape (n,), full precision."""
        return self.score(df).scores()

    def predict_single(self, variant: dict[str, Any]) -> dict[str, Any]:
        """
        One schema-3.0.0 record for one variant.

        An absent field does NOT receive a population median. It receives a FIXED CONSTANT chosen per feature by
        engineer_features (see src/genomic_variant_classifier/models/variant_ensemble.py): allele_freq 0.0, ref and
        alt "A", finngen_enrichment 1.0, and so on. MEASURED 2026-09-14: an absent allele_freq becomes af_raw 0.0,
        af_log10 -8.0 and af_is_absent 1 -- indistinguishable from an OBSERVED frequency of exactly zero.
        Training-fold medians enter only through the fitted missing-value policy, which imputes columns that are
        PRESENT WITH NaN. Supply every field whose value matters. A constant is a fabricated measurement, not a
        neutral one.
        """
        return serialize_predictions(self.score(pd.DataFrame([variant])))[0]

    def predict_batch(self, variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Schema-3.0.0 records for a list of variant dicts, in submitted order (one shared serializer)."""
        if not variants:
            return []
        return serialize_predictions(self.score(pd.DataFrame(variants)))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import joblib

        self._require_admissible()   # never export a quarantine-era pipeline
        self._require_combiner()     # never export a pipeline without its declared, computable blend
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
        its declared features and executable roster must match the record, and it must declare its blend.
        The manifest save() writes is integrity metadata; a checksum beside a model is not a binding, so it
        is not an admission input.
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
        obj._require_combiner()
        if tuple(obj._declared_features()) != tuple(record.feature_names):
            raise ContainmentError("the loaded pipeline's features differ from its registry record")
        # EXACT ONLY (owner ruling 2026-11-07, decision 38f15c7a). A declared serving projection explains a
        # difference; it does not establish that the registered and the executable predictors are the same.
        state, detail = roster_alignment(record, served_model_roster(obj))
        if state is not RosterAlignment.EXACT:
            raise ContainmentError(
                "Loaded predictor roster differs from its registry record; serving projections do not authorize "
                f"this predictor ({state.value}: {detail})")
        # Names are necessary, never sufficient: the record must identify THIS fitted predictor.
        if record.predictor_contract_sha256 is None:
            raise ContainmentError("the registry record does not identify the fitted predictor "
                                   "(no predictor_contract_sha256); it cannot admit an inference load")
        if record.predictor_contract_sha256 != obj.predictor_contract_sha256():
            raise ContainmentError("the loaded predictor differs from the fitted predictor its registry record "
                                   "identifies (members, order, stages, weights, kernel, features, rendering "
                                   "or classes)")
        logger.info(
            "InferencePipeline loaded for %s: record=%s  features=%d  created=%s  combiner=%s",
            consumer, record.record_id, obj.metadata.n_features, obj.metadata.created_at, obj.combiner.digest[:16],
        )
        return obj
