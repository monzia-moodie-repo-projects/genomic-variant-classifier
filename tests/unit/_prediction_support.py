"""Test-only support for the package-1 prediction contract. Production code never imports this.

Builds InferencePipeline objects around SYNTHETIC members with an explicit ConvexBlendSpec, the way
InferencePipeline.from_variant_ensemble builds one from a fitted ensemble. The fit-input stage of a synthetic blend is
named for what it is -- a test fixture -- rather than a fabricated out-of-fold matrix.

Author: Monzia Moodie
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from genomic_variant_classifier.models.combiner import (
    BLEND_KERNEL,
    ConvexBlendSpec,
    MemberSlot,
    member_artifact_sha256,
    prediction_stage_sha256,
)
from genomic_variant_classifier.prediction_contract import canonical_digest


class FixedProba:
    """A fitted, PICKLABLE member returning a fixed positive-class probability, and recording what it received.

    Module-level so InferencePipeline.save can pickle it. classes_ is a real fitted attribute (integer 0 and 1);
    `classes` may reverse it to exercise class-order handling.
    """

    def __init__(self, proba: float = 0.85, classes=(0, 1), dtype=np.float64) -> None:
        self.proba = proba
        self.classes_ = np.array(classes, dtype=np.int64)
        self.dtype = dtype
        self.calls: list[Any] = []

    def predict_proba(self, X):
        self.calls.append(X)
        positive = np.full(len(X), self.proba, dtype=self.dtype)
        columns = [1.0 - positive, positive] if tuple(self.classes_.tolist()) == (0, 1) else [positive, 1.0 - positive]
        return np.column_stack(columns).astype(self.dtype)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["calls"] = []          # recorded inputs are test diagnostics, not model state
        return state


class WidthSensitive(FixedProba):
    """Refuses an input whose width differs from the width it was 'fitted' on (the constant mock never could)."""

    def __init__(self, n_features: int, proba: float = 0.85) -> None:
        super().__init__(proba)
        self.n_features_in_ = n_features

    def predict_proba(self, X):
        width = np.shape(X)[1]
        if width != self.n_features_in_:
            raise ValueError(f"X has {width} features, but the member was fitted with {self.n_features_in_}")
        return super().predict_proba(X)


def blend_spec(models: dict, weights: Optional[list[float]] = None, *, fixture: str = "synthetic") -> ConvexBlendSpec:
    """A ConvexBlendSpec over synthetic members (equal weights unless given), in the mapping's order."""
    from genomic_variant_classifier.api.pipeline import SERVING_SCOPE

    names = list(models)
    if weights is None:
        weights = [1.0 / len(names)] * len(names)
        weights[-1] = 1.0 - sum(weights[:-1])
    identities = {n: member_artifact_sha256(m) for n, m in models.items()}
    members = tuple(MemberSlot(n, identities[n], prediction_stage_sha256(n, models[n])) for n in names)
    stage = canonical_digest("gvc.test-fixture-fit-input-stage.v1", {"fixture": fixture, "members": names})
    source = canonical_digest("gvc.in-memory-fit-record.v1", {"fixture": fixture, "members": identities,
                                                               "weights": [float(w).hex() for w in weights]})
    return ConvexBlendSpec(source_kind="in_memory_fit_record_sha256", source_sha256=source,
                           scientific_scope_sha256=SERVING_SCOPE.digest, members=members,
                           weights=tuple(float(w) for w in weights), fit_input_stage_sha256=stage,
                           kernel=BLEND_KERNEL)


def blend_pipeline(models: dict, weights: Optional[list[float]] = None, **kwargs):
    from genomic_variant_classifier.api.pipeline import InferencePipeline

    return InferencePipeline(trained_models=dict(models), combiner=blend_spec(models, weights), **kwargs)


def ready_binding(tmp_path, pipe, *, version: str = "synthetic-ready"):
    """A REAL runtime binding for `pipe`: saved bytes, a registry record that identifies this fitted predictor
    (exact roster AND predictor_contract_sha256), promoted to production, resolved by the production resolver.

    Returns (binding, artifact_path, registry_path). Nothing here bypasses attribution: readiness comes from
    resolve_runtime_binding, exactly as at service startup.
    """
    from pathlib import Path

    from genomic_variant_classifier.api.attribution import resolve_runtime_binding
    from genomic_variant_classifier.monitoring.model_registry import (
        ArtifactIdentity, EvaluationEvidence, EvaluationProtocol, ModelRegistry, PromotionPolicy,
        TrainingLineage,
    )

    tmp_path = Path(tmp_path)
    artifact = tmp_path / f"{version}.joblib"
    pipe.save(artifact)
    registry = ModelRegistry(path=tmp_path / f"{version}.registry.v1.json")
    roster = tuple(pipe.trained_models)
    registry.register(
        version=version, model_path=artifact, lineage=TrainingLineage(run_id=version),
        evaluation=EvaluationEvidence(
            protocol=EvaluationProtocol(protocol_id="synthetic", split_kind="synthetic",
                                        population_scope="synthetic", n_observations=1, label_policy="synthetic"),
            # A synthetic number above the production policy's absolute floor (0.97): the promotion machinery is
            # exercised for real; the value is not evidence of anything.
            metrics={"auroc": 0.99}),
        feature_names=tuple(pipe._declared_features()), model_roster=roster,
        predictor_contract_sha256=pipe.predictor_contract_sha256())
    registry.promote_to_shadow(version)
    registry.promote_to_production(version, PromotionPolicy(
        policy_id="synthetic", require_durable_uri=False, expected_model_roster=roster))
    registry.save()
    return resolve_runtime_binding(pipe, ArtifactIdentity.measure(artifact), registry.path), artifact, registry.path


class CountingProba(FixedProba):
    """A FixedProba whose calls are counted (the recorded inputs live in .calls)."""

    @property
    def n_calls(self) -> int:
        return len(self.calls)
