"""tests/unit/test_model_registry.py

Author: Monzia Moodie
Written 2026-08-07. REGISTRY-1b.

Every refusal path is exercised, not only the accepting one. A promotion gate
whose refusals have never been observed to fire is an assertion, not evidence
-- the lesson the sabotage matrices of OP-1 steps 3 and 4 were built on.

WHAT THIS FILE DELIBERATELY ASSERTS ABOUT ABSENCE.
`ModelRecord` must not grow a bare `auroc` property. That is checked
structurally rather than left to review, because the pre-existing workflow read
`prod.auroc` and the whole point of GATE-1 is that a metric detached from its
protocol reads as comparable when it is not.
"""
from __future__ import annotations

import json

import pytest

from genomic_variant_classifier.monitoring.model_registry import (
    SCHEMA_VERSION,
    ArtifactIdentity,
    EvaluationEvidence,
    EvaluationProtocol,
    ModelRecord,
    ModelRegistry,
    PromotionPolicy,
    RegistryInvariantError,
    Stage,
)

ROSTER = ("random_forest", "xgboost", "lightgbm", "catboost",
          "gradient_boosting", "logistic_regression", "svm",
          "svm_bagged_rbf", "kan", "tabular_nn", "cnn_1d", "mc_dropout",
          "deep_ensemble")
FEATURES = ("af_raw", "af_log10", "cadd_phred")


def _protocol(protocol_id: str = "unseen-gene-holdout-v1",
              split_kind: str = "unseen_gene_holdout",
              n_observations: int = 213_436) -> EvaluationProtocol:
    return EvaluationProtocol(
        protocol_id=protocol_id,
        split_kind=split_kind,
        population_scope="clinvar_tier2_grch38",
        n_observations=n_observations,
        label_policy="acmg_five_tier_collapsed_binary",
        population_fingerprint="a1b2c3d4")


def _evidence(auroc: float = 0.9988, **protocol_kwargs) -> EvaluationEvidence:
    return EvaluationEvidence(protocol=_protocol(**protocol_kwargs),
                              metrics={"auroc": auroc, "auprc": 0.9712})


def _artifact(tmp_path, name: str = "pipeline.joblib", body: bytes = b"model"):
    path = tmp_path / name
    path.write_bytes(body)
    return path


def _lineage(run_id: str = "run15"):
    from genomic_variant_classifier.monitoring.model_registry import (
        TrainingLineage)
    return TrainingLineage(run_id=run_id, source_commit="032a2ab",
                           clinvar_release="2026_06")


def _register(registry, tmp_path, *, version="run15-ensemble",
              auroc=0.9988, roster=ROSTER, body=b"model", uri=None,
              protocol_kwargs=None):
    return registry.register(
        version=version,
        model_path=_artifact(tmp_path, f"{version}.joblib", body),
        lineage=_lineage(),
        evaluation=_evidence(auroc, **(protocol_kwargs or {})),
        feature_names=FEATURES,
        model_roster=roster,
        artifact_uri=uri)


# ------------------------------------------------------------------ identity

def test_the_artifact_digest_is_measured_not_accepted(tmp_path):
    """A caller-authored digest describes what the caller believed."""
    path = _artifact(tmp_path, body=b"exact bytes")
    identity = ArtifactIdentity.measure(path)
    import hashlib
    assert identity.sha256 == hashlib.sha256(b"exact bytes").hexdigest()
    assert identity.size_bytes == len(b"exact bytes")


def test_measuring_an_absent_artifact_is_refused(tmp_path):
    with pytest.raises(RegistryInvariantError, match="no artifact"):
        ArtifactIdentity.measure(tmp_path / "does_not_exist.joblib")


@pytest.mark.parametrize("bad", ["", "not-hex", "a" * 63, "A" * 64])
def test_a_malformed_digest_is_refused(bad):
    with pytest.raises(RegistryInvariantError, match="sha256"):
        ArtifactIdentity(uri="file:///x", sha256=bad, size_bytes=1)


def test_the_record_id_is_lineage_plus_content(tmp_path):
    """A rerun can reuse a version label; it cannot reuse a digest."""
    registry = ModelRegistry(path=tmp_path / "registry.v1.json")
    record = _register(registry, tmp_path)
    assert record.record_id == f"run15-{record.artifact.sha256[:12]}"


def test_two_artifacts_under_one_version_get_different_record_ids(tmp_path):
    first = ArtifactIdentity.measure(_artifact(tmp_path, "a.joblib", b"one"))
    second = ArtifactIdentity.measure(_artifact(tmp_path, "b.joblib", b"two"))
    assert first.sha256 != second.sha256


def test_a_local_path_is_not_durable(tmp_path):
    identity = ArtifactIdentity.measure(_artifact(tmp_path))
    assert identity.uri.startswith("file:")
    assert identity.is_durable is False
    assert ArtifactIdentity(uri="gs://bucket/m.joblib", sha256="0" * 64,
                            size_bytes=1).is_durable is True


# ------------------------------------------------------------ the absent API

def test_no_bare_auroc_property_exists_on_a_record():
    """GATE-1. `prod.auroc` is what made 0.9988 unseen-gene look comparable
    with 0.9984 test. The scalar must stay adjacent to its protocol."""
    assert not hasattr(ModelRecord, "auroc")
    assert not hasattr(ModelRecord, "metrics")


def test_evidence_without_metrics_is_refused():
    with pytest.raises(RegistryInvariantError, match="not evidence"):
        EvaluationEvidence(protocol=_protocol(), metrics={})


@pytest.mark.parametrize("field_name", ["protocol_id", "split_kind",
                                        "population_scope", "label_policy"])
def test_an_unnamed_protocol_field_is_refused(field_name):
    kwargs = dict(protocol_id="p", split_kind="test",
                  population_scope="scope", n_observations=10,
                  label_policy="binary")
    kwargs[field_name] = ""
    with pytest.raises(RegistryInvariantError, match="nonempty"):
        EvaluationProtocol(**kwargs)


def test_a_boolean_is_not_a_metric():
    with pytest.raises(RegistryInvariantError, match="real number"):
        EvaluationEvidence(protocol=_protocol(), metrics={"auroc": True})


# --------------------------------------------------------------- the record

def test_a_record_requires_its_roster_not_a_count(tmp_path):
    """ROSTER-1. A count cannot detect an architecture change."""
    registry = ModelRegistry(path=tmp_path / "r.json")
    with pytest.raises(RegistryInvariantError, match="model roster"):
        registry.register(
            version="v", model_path=_artifact(tmp_path),
            lineage=_lineage(), evaluation=_evidence(),
            feature_names=FEATURES, model_roster=())


def test_a_record_requires_its_feature_names(tmp_path):
    registry = ModelRegistry(path=tmp_path / "r.json")
    with pytest.raises(RegistryInvariantError, match="feature names"):
        registry.register(
            version="v", model_path=_artifact(tmp_path),
            lineage=_lineage(), evaluation=_evidence(),
            feature_names=(), model_roster=ROSTER)


def test_the_roster_fingerprint_is_order_independent(tmp_path):
    registry = ModelRegistry(path=tmp_path / "r.json")
    forward = _register(registry, tmp_path, version="a", roster=ROSTER)
    backward = _register(registry, tmp_path, version="b",
                         roster=tuple(reversed(ROSTER)))
    assert forward.roster_fingerprint == backward.roster_fingerprint


def test_a_different_roster_gives_a_different_fingerprint(tmp_path):
    registry = ModelRegistry(path=tmp_path / "r.json")
    full = _register(registry, tmp_path, version="a", roster=ROSTER)
    reduced = _register(registry, tmp_path, version="b",
                        roster=tuple(m for m in ROSTER
                                     if m not in {"svm", "tabular_nn"}))
    assert full.roster_fingerprint != reduced.roster_fingerprint


# -------------------------------------------------------------- persistence

def test_a_missing_registry_is_an_error_not_an_empty_one(tmp_path):
    """'No declaration exists' and 'a declaration declares nothing' are
    different statements. Conflating them is how a check reports health for
    something that never happened."""
    with pytest.raises(RegistryInvariantError, match="no registry at"):
        ModelRegistry.load(tmp_path / "absent.json")
    created = ModelRegistry.load(tmp_path / "absent.json",
                                 create_if_missing=True)
    assert created.records == []


def test_a_round_trip_preserves_every_field(tmp_path):
    path = tmp_path / "registry.v1.json"
    registry = ModelRegistry(path=path)
    original = _register(registry, tmp_path)
    registry.save()

    reloaded = ModelRegistry.load(path)
    assert len(reloaded.records) == 1
    restored = reloaded.records[0]
    assert restored == original
    assert restored.record_id == original.record_id
    assert restored.evaluation.protocol == original.evaluation.protocol


def test_the_written_file_declares_its_schema_version(tmp_path):
    path = tmp_path / "registry.v1.json"
    registry = ModelRegistry(path=path)
    _register(registry, tmp_path)
    registry.save()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert set(payload) == {"schema_version", "records", "promotion_history"}
    assert path.read_bytes().count(b"\r") == 0


def test_a_foreign_schema_version_is_refused(tmp_path):
    path = tmp_path / "registry.v1.json"
    path.write_text(json.dumps({"schema_version": 99, "records": [],
                                "promotion_history": []}), encoding="utf-8")
    with pytest.raises(RegistryInvariantError, match="schema_version"):
        ModelRegistry.load(path)


def test_duplicate_versions_are_refused_on_load(tmp_path):
    path = tmp_path / "registry.v1.json"
    registry = ModelRegistry(path=path)
    _register(registry, tmp_path, version="dup")
    registry.save()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"].append(dict(payload["records"][0]))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RegistryInvariantError, match="duplicate versions"):
        ModelRegistry.load(path)


def test_two_production_records_are_refused_on_load(tmp_path):
    path = tmp_path / "registry.v1.json"
    registry = ModelRegistry(path=path)
    _register(registry, tmp_path, version="a")
    _register(registry, tmp_path, version="b")
    registry.save()
    payload = json.loads(path.read_text(encoding="utf-8"))
    for record in payload["records"]:
        record["stage"] = "production"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RegistryInvariantError, match="more than one"):
        ModelRegistry.load(path)


def test_registering_a_version_twice_is_refused(tmp_path):
    registry = ModelRegistry(path=tmp_path / "r.json")
    _register(registry, tmp_path, version="same")
    with pytest.raises(RegistryInvariantError, match="already registered"):
        _register(registry, tmp_path, version="same", body=b"different")


# ---------------------------------------------------------------- promotion

def _shadowed(tmp_path, **kwargs):
    registry = ModelRegistry(path=tmp_path / "registry.v1.json")
    record = _register(registry, tmp_path, **kwargs)
    registry.promote_to_shadow(record.version)
    return registry


DURABLE = PromotionPolicy(policy_id="production-v1", metric_name="auroc",
                          minimum=0.97, max_regression=0.002,
                          require_durable_uri=False,
                          expected_model_roster=ROSTER)


def test_a_registered_record_starts_at_stage_registered(tmp_path):
    registry = ModelRegistry(path=tmp_path / "r.json")
    assert _register(registry, tmp_path).stage is Stage.REGISTERED


def test_promotion_refuses_a_candidate_not_in_shadow(tmp_path):
    registry = ModelRegistry(path=tmp_path / "r.json")
    record = _register(registry, tmp_path)
    decision = registry.evaluate_for_production(record.version, DURABLE)
    assert decision.accepted is False
    assert decision.reason == "candidate_not_in_shadow"


def test_promotion_refuses_a_non_durable_artifact_uri(tmp_path):
    registry = _shadowed(tmp_path)
    policy = PromotionPolicy(policy_id="p", require_durable_uri=True)
    decision = registry.evaluate_for_production("run15-ensemble", policy)
    assert decision.accepted is False
    assert decision.reason == "artifact_uri_not_durable"


def test_promotion_refuses_a_changed_roster(tmp_path):
    """ROSTER-1 as a gate: retraining with --skip-nn --skip-svm cannot be
    promoted against a thirteen-model expectation."""
    registry = _shadowed(tmp_path,
                         roster=tuple(m for m in ROSTER
                                      if m not in {"tabular_nn", "svm"}))
    decision = registry.evaluate_for_production("run15-ensemble", DURABLE)
    assert decision.accepted is False
    assert decision.reason == "model_roster_mismatch"


def test_promotion_refuses_an_absent_metric(tmp_path):
    registry = _shadowed(tmp_path)
    policy = PromotionPolicy(policy_id="p", metric_name="brier",
                             require_durable_uri=False)
    decision = registry.evaluate_for_production("run15-ensemble", policy)
    assert decision.accepted is False
    assert decision.reason == "metric_absent"


def test_promotion_refuses_below_the_absolute_minimum(tmp_path):
    registry = _shadowed(tmp_path, auroc=0.5)
    decision = registry.evaluate_for_production("run15-ensemble", DURABLE)
    assert decision.accepted is False
    assert decision.reason == "below_absolute_minimum"


def test_promotion_accepts_a_clean_candidate_and_moves_it(tmp_path):
    registry = _shadowed(tmp_path)
    decision = registry.evaluate_for_production("run15-ensemble", DURABLE)
    assert decision.accepted is True
    promoted = registry.promote_to_production("run15-ensemble", DURABLE)
    assert promoted.stage is Stage.PRODUCTION
    assert registry.current_production().version == "run15-ensemble"


def test_promotion_refuses_a_protocol_mismatch(tmp_path):
    """The whole of GATE-1: an unseen-gene number and a test number are not
    comparable however close they look."""
    registry = _shadowed(tmp_path)
    registry.promote_to_production("run15-ensemble", DURABLE)

    candidate = _register(
        registry, tmp_path, version="run18-adaptive", auroc=0.9990,
        protocol_kwargs={"protocol_id": "ordinary-test-v1",
                         "split_kind": "test", "n_observations": 146_329})
    registry.promote_to_shadow(candidate.version)
    decision = registry.evaluate_for_production(candidate.version, DURABLE)
    assert decision.accepted is False
    assert decision.reason == "evaluation_protocol_mismatch"


def test_promotion_refuses_a_regression_beyond_tolerance(tmp_path):
    registry = _shadowed(tmp_path, auroc=0.9988)
    registry.promote_to_production("run15-ensemble", DURABLE)
    candidate = _register(registry, tmp_path, version="run18-adaptive",
                          auroc=0.9950)
    registry.promote_to_shadow(candidate.version)
    decision = registry.evaluate_for_production(candidate.version, DURABLE)
    assert decision.accepted is False
    assert decision.reason == "regression_exceeds_tolerance"


def test_a_regression_within_tolerance_is_accepted(tmp_path):
    registry = _shadowed(tmp_path, auroc=0.9988)
    registry.promote_to_production("run15-ensemble", DURABLE)
    candidate = _register(registry, tmp_path, version="run18-adaptive",
                          auroc=0.9975)
    registry.promote_to_shadow(candidate.version)
    assert registry.evaluate_for_production(
        candidate.version, DURABLE).accepted is True


def test_promoting_archives_the_incumbent(tmp_path):
    registry = _shadowed(tmp_path, auroc=0.9988)
    registry.promote_to_production("run15-ensemble", DURABLE)
    candidate = _register(registry, tmp_path, version="run18-adaptive",
                          auroc=0.9985)
    registry.promote_to_shadow(candidate.version)
    registry.promote_to_production(candidate.version, DURABLE)

    assert registry.current_production().version == "run18-adaptive"
    archived = registry.by_stage(Stage.ARCHIVED)
    assert [r.version for r in archived] == ["run15-ensemble"]


def test_a_refused_promotion_raises_with_its_reason(tmp_path):
    registry = _shadowed(tmp_path, auroc=0.1)
    with pytest.raises(RegistryInvariantError,
                       match="below_absolute_minimum"):
        registry.promote_to_production("run15-ensemble", DURABLE)


def test_production_cannot_be_moved_sideways_into_shadow(tmp_path):
    registry = _shadowed(tmp_path)
    registry.promote_to_production("run15-ensemble", DURABLE)
    with pytest.raises(RegistryInvariantError, match="demote deliberately"):
        registry.promote_to_shadow("run15-ensemble")


def test_an_unknown_version_is_refused(tmp_path):
    registry = ModelRegistry(path=tmp_path / "r.json")
    with pytest.raises(RegistryInvariantError, match="no record with version"):
        registry.promote_to_shadow("never-registered")


# ------------------------------------------------------------------ history

def test_the_promotion_history_is_append_only_and_complete(tmp_path):
    registry = _shadowed(tmp_path)
    registry.promote_to_production("run15-ensemble", DURABLE)
    events = registry.history_for("run15-ensemble")
    assert [(e.from_stage, e.to_stage) for e in events] == [
        (Stage.REGISTERED, Stage.REGISTERED),
        (Stage.REGISTERED, Stage.SHADOW),
        (Stage.SHADOW, Stage.PRODUCTION)]


def test_history_survives_a_round_trip(tmp_path):
    path = tmp_path / "registry.v1.json"
    registry = ModelRegistry(path=path)
    _register(registry, tmp_path)
    registry.promote_to_shadow("run15-ensemble")
    registry.promote_to_production("run15-ensemble", DURABLE)
    registry.save()
    assert len(ModelRegistry.load(path).promotion_history) == 3


def test_what_was_production_before_is_answerable(tmp_path):
    registry = _shadowed(tmp_path, auroc=0.9988)
    registry.promote_to_production("run15-ensemble", DURABLE)
    second = _register(registry, tmp_path, version="run18-adaptive",
                       auroc=0.9985)
    registry.promote_to_shadow(second.version)
    registry.promote_to_production(second.version, DURABLE)

    superseded = [e for e in registry.promotion_history
                  if e.to_stage is Stage.ARCHIVED]
    assert len(superseded) == 1
    assert superseded[0].version == "run15-ensemble"
    assert "superseded by run18-adaptive" in superseded[0].reason


# ------------------------------------------------------------------ summary

def test_an_empty_registry_says_so_rather_than_looking_healthy(tmp_path):
    lines = ModelRegistry(path=tmp_path / "r.json").summary_lines()
    assert any("NO RECORDS" in line for line in lines)


def test_the_summary_names_the_protocol_beside_every_metric(tmp_path):
    registry = _shadowed(tmp_path)
    text = "\n".join(registry.summary_lines())
    assert "unseen-gene-holdout-v1" in text
    assert "auroc=0.998800" in text
    assert "n=213436" in text
