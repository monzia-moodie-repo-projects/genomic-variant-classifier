"""The ONE admission route for loading a classifier artifact.

Author: Monzia Moodie
Written 2026-09-23 (owner ruling 2026-09-23, point 2: "No trusted binding means no production load").

Three checks, kept distinct because they establish different things (ruling point 2):
  checksum            these are the expected bytes          -- the kernel's load_after_admission
  trusted binding     these bytes belong to a DECLARED       -- a deployment-registry record whose digest the
                      artifact with this feature contract       REGISTRY measured (never a sidecar beside the model)
  current admission   this consumer may use it NOW           -- a typed Decision from the cutover authority

Every step runs BEFORE deserialization: the file is only hashed here; InferencePipeline.load and
VariantEnsemble.load then deserialize precisely the digest-verified snapshot through the kernel.

THE AUTHORITY TODAY. Cohort-v2 gate C10 ("downstream cutover authorization (separate, explicit)",
docs/measurements/DECISION_2026-07-25_cohort-v2-authorization-and-phase-split.md section 8) is
AUTHORIZED_NOT_IMPLEMENTED. Nothing in the repository can grant scientific use, so the default authority
DENIES every load, and says why. That is the explicit unavailable state the ruling calls correct; it is
not a fault to work around. Tests pass an explicit authority for synthetic artifacts they create.

Not a loader, not a second policy: it reads quarantine_policy, the kernel and the registry.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from genomic_variant_classifier import quarantine_policy as _policy
from genomic_variant_classifier.containment import ContainmentError, require_scientific_contract
from genomic_variant_classifier.monitoring.model_registry import (
    ArtifactIdentity,
    ModelRecord,
    ModelRegistry,
    RegistryInvariantError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Admission", "C10NotImplemented", "CutoverAuthority", "DEFAULT_AUTHORITY", "Decision", "Outcome",
    "admit_artifact", "quarantine_policy_sha256", "repository_registry_path",
]


class Outcome(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class Decision:
    """An authorization decision BOUND to one artifact, one consumer and one policy."""

    artifact_sha256: str
    consumer: str
    policy_sha256: str
    outcome: Outcome
    reason: str


class CutoverAuthority(Protocol):
    def decide(self, record: ModelRecord, *, consumer: str, policy_sha256: str) -> Decision: ...


class C10NotImplemented:
    """The DEFAULT authority: gate C10 does not exist yet, so nothing is authorized."""

    REASON = ("cohort-v2 gate C10 (downstream cutover authorization) is AUTHORIZED_NOT_IMPLEMENTED; "
              "no model is authorized for use")

    def decide(self, record: ModelRecord, *, consumer: str, policy_sha256: str) -> Decision:
        return Decision(record.artifact.sha256, consumer, policy_sha256, Outcome.DENY, self.REASON)


DEFAULT_AUTHORITY: CutoverAuthority = C10NotImplemented()


def repository_registry_path() -> Path:
    """The repository's committed deployment registry, resolved from the package location (not the
    working directory). A deployed service configures its own (api.main DEPLOYMENT_REGISTRY_PATH)."""
    return Path(__file__).resolve().parents[2] / "deployments" / "registry.v1.json"


def quarantine_policy_sha256() -> str:
    """Digest of the CURRENT reviewed quarantine policy; a decision binds to it."""
    payload = {
        "QUARANTINED_FEATURES": list(_policy.QUARANTINED_FEATURES),
        "BLOCKED_PRODUCERS": list(_policy.BLOCKED_PRODUCERS),
        "STRUCTURAL_CONFIG_FIELDS": list(_policy.STRUCTURAL_CONFIG_FIELDS),
        "HISTORICAL_EXECUTION_DENIED": dict(_policy.HISTORICAL_EXECUTION_DENIED),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Admission:
    """What admission established: the registry record the bytes belong to, and the decision."""

    record: ModelRecord
    decision: Decision


def admit_artifact(path: str | Path, *, consumer: str, registry_path: str | Path,
                   authority: CutoverAuthority = DEFAULT_AUTHORITY) -> Admission:
    """Admit an artifact for `consumer`, or raise ContainmentError -- WITHOUT deserializing it."""
    if type(consumer) is not str or not consumer or consumer != consumer.strip():
        raise ContainmentError("a named consumer is required: a decision binds to who is loading")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no artifact at {path}")
    identity = ArtifactIdentity.measure(path)          # hashes bytes; deserializes nothing
    try:
        registry = ModelRegistry.load(registry_path)
    except (RegistryInvariantError, OSError, ValueError) as exc:
        raise ContainmentError(f"No trusted binding for {path.name}: {exc}") from exc
    record = registry.record_for_digest(identity.sha256)
    if record is None:
        raise ContainmentError(
            f"Unbound artifact {path.name} (sha256 {identity.sha256[:16]}): no deployment-registry record "
            f"measured these bytes, so it is refused BEFORE deserialization (no trusted binding)")
    require_scientific_contract(record.feature_names, _policy.QUARANTINED_FEATURES)
    policy = quarantine_policy_sha256()
    decision = authority.decide(record, consumer=consumer, policy_sha256=policy)
    # False, None, True, a stale ALLOW, or an ALLOW for another consumer or policy must all fail.
    if type(decision) is not Decision:
        raise ContainmentError("the authority must return a typed Decision; "
                               f"it returned {type(decision).__name__}")
    if not (decision.artifact_sha256 == record.artifact.sha256 and decision.consumer == consumer
            and decision.policy_sha256 == policy and type(decision.reason) is str and decision.reason.strip()):
        raise ContainmentError("the decision is not bound to this artifact, consumer and current policy")
    if decision.outcome is not Outcome.ALLOW:
        raise ContainmentError(f"{consumer} may not load {record.record_id}: {decision.reason}")
    logger.info("admitted %s for %s (record %s)", path.name, consumer, record.record_id)
    return Admission(record=record, decision=decision)
