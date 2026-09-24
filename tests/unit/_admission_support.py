"""Test-only support for the ONE admission route (model_admission). Production code never imports this.

A synthetic artifact a test created is registered in a registry under tmp_path, and loaded with an
EXPLICIT authority that allows it -- the way a future C10 cutover authority would. Production passes no
authority, so it gets the default, which denies everything while gate C10 is not implemented.

Author: Monzia Moodie
"""
from __future__ import annotations

from pathlib import Path

from genomic_variant_classifier.model_admission import Decision, Outcome
from genomic_variant_classifier.monitoring.model_registry import (
    EvaluationEvidence,
    EvaluationProtocol,
    ModelRegistry,
    TrainingLineage,
)


class AllowForTests:
    """An explicit ALLOW for synthetic artifacts, bound exactly as admission requires."""

    def decide(self, record, *, consumer, policy_sha256):
        return Decision(record.artifact.sha256, consumer, policy_sha256, Outcome.ALLOW,
                        "synthetic artifact created by this test")


def register(registry_path: Path, artifact: Path, *, feature_names, roster, version: str = "synthetic-v1",
             serving_projection=None) -> Path:
    """Add one record for `artifact` (its digest MEASURED by the registry) and save; returns the path."""
    registry = ModelRegistry.load(registry_path, create_if_missing=True)
    registry.register(
        version=version, model_path=artifact, lineage=TrainingLineage(run_id=version),
        evaluation=EvaluationEvidence(
            protocol=EvaluationProtocol(protocol_id="synthetic", split_kind="synthetic",
                                        population_scope="synthetic", n_observations=1,
                                        label_policy="synthetic"),
            metrics={"auroc": 0.5}),
        feature_names=tuple(feature_names), model_roster=tuple(roster),
        serving_projection=serving_projection)
    registry.save()
    return registry.path
