"""tests/unit/test_runtime_attribution.py

Author: Monzia Moodie
Written 2026-08-07. PROD-1, Commit A.

What is this process serving, and what may it claim about it?

WHY THIS FILE EXISTS. `api/main.py` published five constants dated 2026-03-25
under a comment reading "update after each training run". They were never
updated, through Runs 9 to 16, and four of the five were pinned by literal in
`test_api.py` -- so the suite DEFENDED them. `HOLDOUT_AUROC = 0.9847` fused a
Run-8 sixty-four-feature figure with 154,404, the validation split size of the
Runs 10-14 cohort. The endpoint published all of it regardless of which
artifact was loaded.

EVERY FIXTURE NAMES THE WORLD IT CONSTRUCTS. `client` is no longer an adequate
name once provenance is part of the test input: `unattributed_pipeline_client`
and `registered_production_client` construct genuinely different systems and
must be asserted differently.

THE FIXTURES SAVE AND RESTORE BOTH GLOBALS BY HAND. `_PIPELINE` and
`_RUNTIME_MODEL_BINDING` must never drift apart -- a stale binding surviving a
pipeline swap is a failure mode Commit A creates. None of these fixtures is
autouse and none requests `monkeypatch`, following the rule `tests/conftest.py`
arrived at after an autouse fixture requesting monkeypatch broke two passing
tests twice.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from genomic_variant_classifier.api.attribution import (
    ArtifactChangedDuringLoadError,
    ArtifactResolutionStatus,
    DeploymentAlignment,
    EvaluationApplicabilityStatus,
    RosterAlignment,
    RuntimeModelBinding,
    load_pipeline_with_identity,
    resolve_runtime_binding,
    served_model_roster,
)
from genomic_variant_classifier.monitoring.model_registry import (
    ArtifactIdentity,
    EvaluationEvidence,
    EvaluationProtocol,
    ModelRegistry,
    PromotionPolicy,
    ServingProjection,
    TrainingLineage,
    roster_fingerprint,
)

TRAINED_ROSTER = (
    "random_forest", "xgboost", "lightgbm", "catboost", "gradient_boosting",
    "logistic_regression", "svm", "svm_bagged_rbf", "kan", "tabular_nn",
    "cnn_1d", "mc_dropout", "deep_ensemble",
)
#: SERVEROSTER-1. `InferencePipeline.from_variant_ensemble` excludes `cnn_1d`
#: because the REST path has no FASTA context window, so the deployable
#: artifact executes twelve of the thirteen trained models.
SERVED_ROSTER = tuple(m for m in TRAINED_ROSTER if m != "cnn_1d")
PROJECTION = ServingProjection(
    excluded_models=("cnn_1d",),
    exclusion_reasons={"cnn_1d": "requires FASTA context unavailable in REST "
                                 "inference"})


class _ConstantModel:
    """A base model that always returns the same probability.

    MODULE-LEVEL AND PICKLABLE, deliberately. `InferencePipeline.save` is
    `joblib.dump`, and `MagicMock` raises PicklingError -- which meant the
    three tests built to drive the real serialisation path never ran.
    Pickle stores a class by qualified reference, so a class defined
    inside a fixture cannot be found on load either.
    """

    def __init__(self, proba: float = 0.92) -> None:
        self.proba = float(proba)

    def predict_proba(self, X):
        import numpy as _np
        return _np.array([[1.0 - self.proba, self.proba]] * len(X))


class _FakePipeline:
    """Something with an executable model mapping. Not an InferencePipeline.

    The served roster is read from the live object's mapping, so anything
    carrying one is a sufficient stand-in for the attribution logic.
    """

    def __init__(self, roster) -> None:
        self.trained_models = {name: MagicMock() for name in roster}


def _write_registry(tmp_path: Path, *, projection=None, promote=False,
                    roster=TRAINED_ROSTER, artifact: Path = None) -> Path:
    """A real registry file with one record for a real artifact."""
    path = tmp_path / "registry.v1.json"
    registry = ModelRegistry(path=path)
    record = registry.register(
        version="run15-ensemble",
        model_path=artifact,
        lineage=TrainingLineage(run_id="run15", clinvar_release="2026_06"),
        evaluation=EvaluationEvidence(
            protocol=EvaluationProtocol(
                protocol_id="unseen-gene-holdout-v1",
                split_kind="unseen_gene_holdout",
                population_scope="clinvar_tier2_grch38",
                n_observations=213_436,
                label_policy="acmg_five_tier_collapsed_binary"),
            metrics={"auroc": 0.9988}),
        feature_names=("af_raw", "af_log10"),
        model_roster=roster,
        serving_projection=projection)
    if promote:
        registry.promote_to_shadow(record.version)
        registry.promote_to_production(record.version, PromotionPolicy(
            policy_id="production-v1", require_durable_uri=False,
            expected_model_roster=roster))
    registry.save()
    return path


@pytest.fixture
def artifact(tmp_path) -> Path:
    path = tmp_path / "phase2_pipeline.joblib"
    path.write_bytes(b"serialised model bytes")
    return path


# --------------------------------------------------------------------------- #
# The four vocabularies
# --------------------------------------------------------------------------- #

def test_the_resolution_vocabulary_is_exactly_this():
    """NO_MODEL_LOADED and NO_ARTIFACT_IDENTITY are different worlds.

    A pipeline object injected directly into the process is loaded and usable
    and simply has no provenance. Collapsing the two would say something false
    about the model rather than something true about its identity.
    """
    assert [m.value for m in ArtifactResolutionStatus] == [
        "no_model_loaded", "no_artifact_identity",
        "artifact_not_registered", "registered"]


def test_the_alignment_vocabulary_is_exactly_this():
    assert [m.value for m in DeploymentAlignment] == [
        "unknown", "no_production_declared",
        "matches_declared_production", "differs_from_declared_production"]


def test_the_roster_vocabulary_is_exactly_this():
    assert [m.value for m in RosterAlignment] == [
        "unknown", "exact", "serving_subset", "inconsistent"]


def test_the_applicability_vocabulary_is_exactly_this():
    assert [m.value for m in EvaluationApplicabilityStatus] == [
        "applicable", "no_model_attributed", "no_sealed_evaluation",
        "roster_mismatch", "artifact_mismatch"]


# --------------------------------------------------------------------------- #
# The binding refuses incoherent states
# --------------------------------------------------------------------------- #

def test_a_resolved_binding_requires_identity():
    with pytest.raises(ValueError, match="not resolution"):
        RuntimeModelBinding(
            resolution_status=ArtifactResolutionStatus.REGISTERED,
            deployment_alignment=DeploymentAlignment.UNKNOWN,
            roster_alignment=RosterAlignment.UNKNOWN,
            evaluation_applicability=(
                EvaluationApplicabilityStatus.NO_SEALED_EVALUATION))


def test_an_unresolved_binding_must_not_carry_registry_identity():
    with pytest.raises(ValueError, match="must not carry registry identity"):
        RuntimeModelBinding(
            resolution_status=(
                ArtifactResolutionStatus.ARTIFACT_NOT_REGISTERED),
            deployment_alignment=DeploymentAlignment.UNKNOWN,
            roster_alignment=RosterAlignment.UNKNOWN,
            evaluation_applicability=(
                EvaluationApplicabilityStatus.NO_MODEL_ATTRIBUTED),
            record_id="run15-abc123456789")


def test_evidence_cannot_be_applicable_to_an_unresolved_artifact():
    with pytest.raises(ValueError, match="unresolved artifact"):
        RuntimeModelBinding(
            resolution_status=(
                ArtifactResolutionStatus.NO_ARTIFACT_IDENTITY),
            deployment_alignment=DeploymentAlignment.UNKNOWN,
            roster_alignment=RosterAlignment.UNKNOWN,
            evaluation_applicability=(
                EvaluationApplicabilityStatus.APPLICABLE))


# --------------------------------------------------------------------------- #
# The served roster comes from the live object
# --------------------------------------------------------------------------- #

def test_the_served_roster_is_read_from_the_executable_mapping():
    """Not from PipelineMetadata, not from a ModelRecord, not from a manifest.

    The question is what THIS OBJECT can invoke, and only the object answers
    it.
    """
    assert served_model_roster(_FakePipeline(SERVED_ROSTER)) == tuple(
        sorted(SERVED_ROSTER))


def test_an_object_without_a_model_mapping_serves_nothing():
    assert served_model_roster(object()) == ()
    assert served_model_roster(None) == ()


def test_the_roster_fingerprint_is_the_registry_s_own_rule():
    """One implementation, so a record's roster and a live pipeline's roster
    cannot disagree for reasons unrelated to the models."""
    binding = RuntimeModelBinding.unattributed(SERVED_ROSTER)
    assert binding.served_roster_fingerprint == roster_fingerprint(
        SERVED_ROSTER)
    assert roster_fingerprint(SERVED_ROSTER) == roster_fingerprint(
        tuple(reversed(SERVED_ROSTER)))


# --------------------------------------------------------------------------- #
# The artifact is measured around the load, not before or after it
# --------------------------------------------------------------------------- #

def test_an_artifact_changing_during_load_is_refused(artifact):
    """A digest taken only before describes bytes that may have been replaced;
    one taken only after describes bytes that may not be what was
    deserialised."""
    def mutating_loader(path):
        path.write_bytes(b"entirely different bytes")
        return _FakePipeline(SERVED_ROSTER)

    with pytest.raises(ArtifactChangedDuringLoadError, match="while it was"):
        load_pipeline_with_identity(artifact, mutating_loader)


def test_a_stable_artifact_returns_the_digest_of_what_was_loaded(artifact):
    pipeline, identity = load_pipeline_with_identity(
        artifact, lambda p: _FakePipeline(SERVED_ROSTER))
    assert identity.sha256 == ArtifactIdentity.measure(artifact).sha256
    assert served_model_roster(pipeline) == tuple(sorted(SERVED_ROSTER))


def test_a_missing_artifact_is_refused(tmp_path):
    with pytest.raises(Exception):
        load_pipeline_with_identity(tmp_path / "absent.joblib", lambda p: None)


# --------------------------------------------------------------------------- #
# The state matrix
# --------------------------------------------------------------------------- #

def test_no_pipeline_is_not_the_same_as_no_provenance(tmp_path, artifact):
    registry = _write_registry(tmp_path, artifact=artifact)
    binding = resolve_runtime_binding(None, None, registry)
    assert binding.resolution_status is (
        ArtifactResolutionStatus.NO_MODEL_LOADED)
    assert binding.model_loaded is False
    assert binding.is_ready is False


def test_an_injected_pipeline_has_no_artifact_identity(tmp_path, artifact):
    """The world every directly-injected test pipeline occupies, and the one
    the suite treated as healthy until 2026-08-07."""
    registry = _write_registry(tmp_path, artifact=artifact)
    binding = resolve_runtime_binding(
        _FakePipeline(SERVED_ROSTER), None, registry)
    assert binding.resolution_status is (
        ArtifactResolutionStatus.NO_ARTIFACT_IDENTITY)
    assert binding.model_loaded is True
    assert binding.is_ready is False
    assert binding.served_model_roster == tuple(sorted(SERVED_ROSTER))


def test_an_unreadable_registry_is_a_status_not_an_exception(tmp_path,
                                                             artifact):
    """An application that will not start because a declaration is missing is
    worse than one that starts and says it cannot identify itself."""
    binding = resolve_runtime_binding(
        _FakePipeline(SERVED_ROSTER),
        ArtifactIdentity.measure(artifact),
        tmp_path / "does_not_exist.json")
    assert binding.resolution_status is (
        ArtifactResolutionStatus.ARTIFACT_NOT_REGISTERED)
    assert binding.is_ready is False
    assert "registry" in (binding.detail or "")


def test_an_unregistered_digest_is_reported_as_such(tmp_path, artifact):
    registry = _write_registry(tmp_path, artifact=artifact)
    other = tmp_path / "other.joblib"
    other.write_bytes(b"a different artifact entirely")
    binding = resolve_runtime_binding(
        _FakePipeline(SERVED_ROSTER), ArtifactIdentity.measure(other),
        registry)
    assert binding.resolution_status is (
        ArtifactResolutionStatus.ARTIFACT_NOT_REGISTERED)
    assert binding.record_id is None


def test_a_registered_shadow_artifact_is_not_production(tmp_path, artifact):
    """The case a single status enum would hide: a perfectly identifiable,
    correctly registered SHADOW artifact being served by accident."""
    registry = _write_registry(tmp_path, artifact=artifact,
                               projection=PROJECTION)
    binding = resolve_runtime_binding(
        _FakePipeline(SERVED_ROSTER), ArtifactIdentity.measure(artifact),
        registry)
    assert binding.resolution_status is ArtifactResolutionStatus.REGISTERED
    assert binding.deployment_alignment is (
        DeploymentAlignment.NO_PRODUCTION_DECLARED)
    assert binding.roster_alignment is RosterAlignment.SERVING_SUBSET
    assert binding.is_ready is False


def test_an_artifact_registered_but_not_the_declared_production_differs(
        tmp_path, artifact):
    """The case a single status enum would hide, and the one that survived a
    mutation on 2026-08-07 because no test constructed it.

    Two registered artifacts; one is declared production; the OTHER is loaded.
    Its digest resolves perfectly -- identity is not the question. What it is
    not, is what the registry says should be serving.
    """
    other = tmp_path / "shadow.joblib"
    other.write_bytes(b"a second, differently registered artifact")

    path = tmp_path / "registry.v1.json"
    registry = ModelRegistry(path=path)
    evidence = EvaluationEvidence(
        protocol=EvaluationProtocol(
            protocol_id="unseen-gene-holdout-v1",
            split_kind="unseen_gene_holdout",
            population_scope="clinvar_tier2_grch38",
            n_observations=213_436,
            label_policy="acmg_five_tier_collapsed_binary"),
        metrics={"auroc": 0.9988})
    live = registry.register(
        version="run15-ensemble", model_path=artifact,
        lineage=TrainingLineage(run_id="run15"), evaluation=evidence,
        feature_names=("af_raw",), model_roster=TRAINED_ROSTER,
        serving_projection=PROJECTION)
    registry.register(
        version="run18-candidate", model_path=other,
        lineage=TrainingLineage(run_id="run18"), evaluation=evidence,
        feature_names=("af_raw",), model_roster=TRAINED_ROSTER,
        serving_projection=PROJECTION)
    registry.promote_to_shadow(live.version)
    registry.promote_to_production(live.version, PromotionPolicy(
        policy_id="production-v1", require_durable_uri=False,
        expected_model_roster=TRAINED_ROSTER))
    registry.save()

    binding = resolve_runtime_binding(
        _FakePipeline(SERVED_ROSTER), ArtifactIdentity.measure(other), path)

    assert binding.resolution_status is ArtifactResolutionStatus.REGISTERED
    assert binding.model_version == "run18-candidate"
    assert binding.deployment_alignment is (
        DeploymentAlignment.DIFFERS_FROM_DECLARED_PRODUCTION)
    assert binding.is_ready is False


def test_a_declared_production_artifact_with_its_projection_is_ready(
        tmp_path, artifact):
    registry = _write_registry(tmp_path, artifact=artifact,
                               projection=PROJECTION, promote=True)
    binding = resolve_runtime_binding(
        _FakePipeline(SERVED_ROSTER), ArtifactIdentity.measure(artifact),
        registry)
    assert binding.resolution_status is ArtifactResolutionStatus.REGISTERED
    assert binding.deployment_alignment is (
        DeploymentAlignment.MATCHES_DECLARED_PRODUCTION)
    assert binding.roster_alignment is RosterAlignment.SERVING_SUBSET
    assert binding.is_ready is True
    assert binding.record_id.startswith("run15-")
    assert binding.model_version == "run15-ensemble"


def test_an_undeclared_omission_is_not_a_projection(tmp_path, artifact):
    """SERVEROSTER-1's whole point. The same twelve-model roster against a
    record that declares NO projection is UNKNOWN, not SERVING_SUBSET -- an
    undeclared omission never passes as intentional."""
    registry = _write_registry(tmp_path, artifact=artifact, promote=True)
    binding = resolve_runtime_binding(
        _FakePipeline(SERVED_ROSTER), ArtifactIdentity.measure(artifact),
        registry)
    assert binding.roster_alignment is RosterAlignment.UNKNOWN
    assert binding.is_ready is False
    assert "declares no serving projection" in binding.detail


def test_a_model_missing_beyond_the_projection_is_inconsistent(tmp_path,
                                                               artifact):
    """Silent model loss. This project has already lost one model silently:
    the Kolmogorov-Arnold Network was absent from every Continuous Integration
    run until a repair was written for it."""
    registry = _write_registry(tmp_path, artifact=artifact,
                               projection=PROJECTION, promote=True)
    binding = resolve_runtime_binding(
        _FakePipeline(SERVED_ROSTER[:-1]), ArtifactIdentity.measure(artifact),
        registry)
    assert binding.roster_alignment is RosterAlignment.INCONSISTENT
    assert binding.evaluation_applicability is (
        EvaluationApplicabilityStatus.ROSTER_MISMATCH)
    assert binding.is_ready is False


def test_serving_the_full_trained_roster_is_exact(tmp_path, artifact):
    registry = _write_registry(tmp_path, artifact=artifact,
                               projection=PROJECTION, promote=True)
    binding = resolve_runtime_binding(
        _FakePipeline(TRAINED_ROSTER), ArtifactIdentity.measure(artifact),
        registry)
    assert binding.roster_alignment is RosterAlignment.EXACT
    assert binding.is_ready is True


# --------------------------------------------------------------------------- #
# Commit A publishes no evidence, in any state
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("projection,promote,roster", [
    (None, False, SERVED_ROSTER),
    (PROJECTION, False, SERVED_ROSTER),
    (PROJECTION, True, SERVED_ROSTER),
    (PROJECTION, True, TRAINED_ROSTER),
])
def test_no_binding_ever_reports_applicable_evidence(
        tmp_path, artifact, projection, promote, roster):
    """Resolving a digest authorises IDENTITY, not EVIDENCE.

    The record's metrics were measured on the thirteen-model ensemble; the
    REST artifact executes twelve. Until a sealed evaluation names this
    artifact digest AND this served-roster fingerprint, no metric here
    describes what is being served.
    """
    registry = _write_registry(tmp_path, artifact=artifact,
                               projection=projection, promote=promote)
    binding = resolve_runtime_binding(
        _FakePipeline(roster), ArtifactIdentity.measure(artifact), registry)
    assert binding.evaluation_applicability is not (
        EvaluationApplicabilityStatus.APPLICABLE)


def test_the_binding_carries_no_metric_field_at_all():
    """Structural, not behavioural. A future convenience property is how
    `record.auroc` would come back."""
    fields = set(RuntimeModelBinding.__dataclass_fields__)
    assert not {f for f in fields
                if "auroc" in f or "metric" in f or "score" in f}
    assert not hasattr(RuntimeModelBinding, "auroc")
    assert not hasattr(RuntimeModelBinding, "metrics")


# --------------------------------------------------------------------------- #
# The real startup path
# --------------------------------------------------------------------------- #

def _install(api_main, pipeline, binding):
    """Set BOTH globals. Neither may drift from the other."""
    api_main._PIPELINE = pipeline
    api_main._RUNTIME_MODEL_BINDING = binding


@pytest.fixture
def unattributed_pipeline_client():
    """A model object with no artifact provenance: loaded, not attributable."""
    from fastapi.testclient import TestClient
    import genomic_variant_classifier.api.main as api_main

    saved = (api_main._PIPELINE, api_main._RUNTIME_MODEL_BINDING)
    _install(api_main, _FakePipeline(SERVED_ROSTER),
             RuntimeModelBinding.unattributed(SERVED_ROSTER))
    try:
        yield TestClient(api_main.app)
    finally:
        api_main._PIPELINE, api_main._RUNTIME_MODEL_BINDING = saved


@pytest.fixture
def registered_production_client(tmp_path, monkeypatch):
    """Drives the ACTUAL lifespan against a real artifact and registry.

    Uses `with TestClient(...)`, because `TestClient(app)` alone does not run
    startup on starlette 1.0.0 -- measured 2026-08-07. Without the context
    manager this would test the response formatter rather than the binding.
    """
    from fastapi.testclient import TestClient
    import genomic_variant_classifier.api.main as api_main
    from genomic_variant_classifier.api.pipeline import (
        InferencePipeline, PipelineMetadata)

    artifact_path = tmp_path / "phase2_pipeline.joblib"
    pipeline = InferencePipeline(
        trained_models={n: _ConstantModel() for n in SERVED_ROSTER},
        meta_learner=_ConstantModel(), scaler=None,
        metadata=PipelineMetadata(n_features=2,
                                  feature_names=["af_raw", "af_log10"]))
    pipeline.save(artifact_path)
    registry = _write_registry(tmp_path, artifact=artifact_path,
                               projection=PROJECTION, promote=True)

    saved = (api_main._PIPELINE, api_main._RUNTIME_MODEL_BINDING,
             api_main.MODEL_PATH, api_main.DEPLOYMENT_REGISTRY_PATH)
    monkeypatch.setattr(api_main, "MODEL_PATH", artifact_path)
    monkeypatch.setattr(api_main, "DEPLOYMENT_REGISTRY_PATH", registry)
    try:
        with TestClient(api_main.app) as client:
            yield client
    finally:
        (api_main._PIPELINE, api_main._RUNTIME_MODEL_BINDING,
         api_main.MODEL_PATH, api_main.DEPLOYMENT_REGISTRY_PATH) = saved


def test_an_unattributed_model_is_alive_and_not_ready(
        unattributed_pipeline_client):
    body = unattributed_pipeline_client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["live"] is True
    assert body["ready"] is False
    assert body["status"] == "degraded"


def test_the_whole_chain_resolves_through_the_real_lifespan(
        registered_production_client):
    """serialised bytes -> digest before -> load -> digest after -> registry
    lookup -> production comparison -> roster reconciliation -> response."""
    health = registered_production_client.get("/health").json()
    assert health["ready"] is True
    assert health["status"] == "ok"

    info = registered_production_client.get("/info").json()
    attribution = info["attribution"]
    assert info["api_version"] == "2.0.0"
    assert attribution["resolution_status"] == "registered"
    assert attribution["deployment_alignment"] == "matches_declared_production"
    assert attribution["roster_alignment"] == "serving_subset"
    assert attribution["model_version"] == "run15-ensemble"
    assert len(attribution["artifact_sha256"]) == 64
    assert sorted(attribution["served_model_roster"]) == sorted(SERVED_ROSTER)
    assert sorted(attribution["registered_model_roster"]) == sorted(
        TRAINED_ROSTER)


def test_info_publishes_no_metric_and_no_free_text_description(
        registered_production_client):
    """PIPEMETA-1. `PipelineMetadata.val_auroc` is an unqualified scalar in the
    artifact format and must never become deployment evidence."""
    body = registered_production_client.get("/info").json()
    forbidden = {"holdout_auroc", "training_auroc", "training_auprc",
                 "pipeline_version", "description", "val_auroc"}
    assert not forbidden & set(body)
    assert not forbidden & set(body["attribution"])
    assert "0.9847" not in json.dumps(body)
    assert body["attribution"]["evaluation_applicability"] == (
        "no_sealed_evaluation")


def test_predictions_carry_the_serving_record_identity(
        registered_production_client):
    body = registered_production_client.post(
        "/predict",
        json={"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}).json()
    assert body["model_record_id"].startswith("run15-")
    assert body["model_version"] == "run15-ensemble"
    assert "pipeline_version" not in body
