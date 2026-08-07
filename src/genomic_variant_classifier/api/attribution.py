"""src/genomic_variant_classifier/api/attribution.py

Author: Monzia Moodie
Written 2026-08-07. PROD-1, Commit A.

What is this process actually serving, and what may it say about it?

WHY THIS MODULE EXISTS. Until today `api/main.py` published five module
constants dated 2026-03-25 -- `MODEL_VERSION = "phase2-v1"`,
`HOLDOUT_AUROC = 0.9847` and three others -- under a comment reading "update
after each training run". They were never updated, through Runs 9 to 16.
`0.9847` is a Run-8 sixty-four-feature figure fused with `154,404`, the
validation split size of the Runs 10-14 cohort: two measurements, two eras, one
line. The same digits are Run 15's unseen-gene F1, so a reader reconciling them
lands on a third quantity. The endpoint published all of this regardless of
which artifact was loaded, and the test suite pinned it.

THE INVARIANT THIS ESTABLISHES. There is no legal path from "a metric we know"
to "a metric advertised for the serving model" that does not traverse a
content-identified artifact, a registry record, and evidence explicitly
applicable to the SERVING PROJECTION. This module builds the first three links
and refuses the fourth until it is earned.

FOUR VOCABULARIES, BECAUSE THEY ANSWER FOUR DIFFERENT QUESTIONS.

    ArtifactResolutionStatus      can I identify these bytes?
    DeploymentAlignment           are they what the registry DECLARES?
    RosterAlignment               does the executable roster match the
                                  record's, allowing for a declared
                                  serving projection?
    EvaluationApplicabilityStatus may a metric measured on that record be
                                  shown as evidence for THIS artifact?

Collapsing any two would recreate the drift. A registered SHADOW artifact
served by accident must not look healthy merely because its digest resolves;
that is why alignment is a separate axis from resolution.

NO METRICS APPEAR IN THIS MODULE. `RuntimeModelBinding` answers identity and
applicability. Commit C attaches evidence, and only once a sealed evaluation
names this artifact digest and this served-roster fingerprint.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from genomic_variant_classifier.monitoring.model_registry import (
    ArtifactIdentity,
    ModelRecord,
    ModelRegistry,
    RegistryInvariantError,
    Stage,
    roster_fingerprint,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ArtifactResolutionStatus",
    "DeploymentAlignment",
    "EvaluationApplicabilityStatus",
    "RosterAlignment",
    "RuntimeModelBinding",
    "ArtifactChangedDuringLoadError",
    "load_pipeline_with_identity",
    "resolve_runtime_binding",
    "served_model_roster",
]


class ArtifactChangedDuringLoadError(RuntimeError):
    """The artifact file changed between the two measurements around a load.

    Raised rather than warned. If the bytes moved while they were being
    deserialised, the digest cannot describe the in-memory object, and a
    binding that cannot describe what it bound is worse than no binding.
    """


class ArtifactResolutionStatus(str, Enum):
    """Can these bytes be identified?

    `NO_MODEL_LOADED` and `NO_ARTIFACT_IDENTITY` are DIFFERENT WORLDS and are
    deliberately not collapsed. A pipeline object injected directly into the
    process -- which every existing API test does -- is loaded and usable, and
    simply has no provenance. Calling that "no artifact" would say something
    false about the model, rather than something true about its identity.
    """

    NO_MODEL_LOADED = "no_model_loaded"
    NO_ARTIFACT_IDENTITY = "no_artifact_identity"
    ARTIFACT_NOT_REGISTERED = "artifact_not_registered"
    REGISTERED = "registered"


class DeploymentAlignment(str, Enum):
    """Is what is loaded what the registry DECLARES as production?

    Orthogonal to resolution: a perfectly identifiable, correctly registered
    SHADOW artifact being served by accident is exactly the case a single
    status enum would hide.
    """

    UNKNOWN = "unknown"
    NO_PRODUCTION_DECLARED = "no_production_declared"
    MATCHES_DECLARED_PRODUCTION = "matches_declared_production"
    DIFFERS_FROM_DECLARED_PRODUCTION = "differs_from_declared_production"


class RosterAlignment(str, Enum):
    """Does the executable roster match the record's, given the projection?

    SERVING_SUBSET is legitimate ONLY because the record declares which models
    the serving artifact omits and why. INCONSISTENT is the state that catches
    silent model loss -- a component that vanished without being declared.
    """

    UNKNOWN = "unknown"
    EXACT = "exact"
    SERVING_SUBSET = "serving_subset"
    INCONSISTENT = "inconsistent"


class EvaluationApplicabilityStatus(str, Enum):
    """May evidence measured on the record be presented for THIS artifact?

    ROSTER_MISMATCH is the status this project needs most. A metric measured on
    a thirteen-model ensemble is not automatically evidence for a twelve-model
    serving projection of it, however intentional the projection. Resolving a
    digest authorises IDENTITY, not EVIDENCE.
    """

    APPLICABLE = "applicable"
    NO_MODEL_ATTRIBUTED = "no_model_attributed"
    NO_SEALED_EVALUATION = "no_sealed_evaluation"
    ROSTER_MISMATCH = "roster_mismatch"
    ARTIFACT_MISMATCH = "artifact_mismatch"


@dataclass(frozen=True)
class RuntimeModelBinding:
    """What this process is serving, and what it is permitted to claim.

    Computed ONCE, in the application lifespan, and immutable thereafter. Not
    recomputed per request: re-hashing the path later would describe whatever
    bytes are on disk now, which may not be the bytes that were deserialised.
    """

    resolution_status: ArtifactResolutionStatus
    deployment_alignment: DeploymentAlignment
    roster_alignment: RosterAlignment
    evaluation_applicability: EvaluationApplicabilityStatus

    served_model_roster: tuple[str, ...] = ()
    served_roster_fingerprint: Optional[str] = None

    artifact_sha256: Optional[str] = None
    artifact_uri: Optional[str] = None
    record_id: Optional[str] = None
    model_version: Optional[str] = None
    registry_stage: Optional[Stage] = None
    registered_model_roster: Optional[tuple[str, ...]] = None

    #: Supplementary human diagnostics ONLY. Never a machine contract: every
    #: decision a consumer makes must come from one of the four enums.
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "served_model_roster",
                           tuple(self.served_model_roster))
        if self.registered_model_roster is not None:
            object.__setattr__(self, "registered_model_roster",
                               tuple(self.registered_model_roster))
        resolved = (self.resolution_status
                    is ArtifactResolutionStatus.REGISTERED)
        if resolved and not all((self.record_id, self.model_version,
                                 self.artifact_sha256)):
            raise ValueError(
                "a REGISTERED binding requires record_id, model_version and "
                "artifact_sha256; identity without those is not resolution")
        if not resolved and self.record_id is not None:
            raise ValueError(
                f"resolution_status is {self.resolution_status.value!r} but a "
                "record_id is present; an unresolved binding must not carry "
                "registry identity")
        if (self.evaluation_applicability
                is EvaluationApplicabilityStatus.APPLICABLE and not resolved):
            raise ValueError(
                "evidence cannot be applicable to an unresolved artifact")

    @property
    def is_ready(self) -> bool:
        """Readiness, as distinct from liveness.

        A process that loaded some bytes it cannot identify is ALIVE and NOT
        READY. For a clinical-facing service, "I loaded a model but cannot
        establish what it is" must not be operationally green.
        """
        return (
            self.resolution_status is ArtifactResolutionStatus.REGISTERED
            and self.deployment_alignment
            is DeploymentAlignment.MATCHES_DECLARED_PRODUCTION
            and self.roster_alignment in (RosterAlignment.EXACT,
                                          RosterAlignment.SERVING_SUBSET)
        )

    @property
    def model_loaded(self) -> bool:
        return (self.resolution_status
                is not ArtifactResolutionStatus.NO_MODEL_LOADED)

    @classmethod
    def no_model_loaded(cls, detail: Optional[str] = None
                        ) -> "RuntimeModelBinding":
        return cls(
            resolution_status=ArtifactResolutionStatus.NO_MODEL_LOADED,
            deployment_alignment=DeploymentAlignment.UNKNOWN,
            roster_alignment=RosterAlignment.UNKNOWN,
            evaluation_applicability=(
                EvaluationApplicabilityStatus.NO_MODEL_ATTRIBUTED),
            detail=detail)

    @classmethod
    def unattributed(cls, roster: Sequence[str] = (),
                     detail: Optional[str] = None) -> "RuntimeModelBinding":
        """A model object with no artifact provenance.

        The state every directly-injected test pipeline occupies, and the one
        the suite treated as healthy until 2026-08-07.
        """
        roster = tuple(sorted(roster))
        return cls(
            resolution_status=ArtifactResolutionStatus.NO_ARTIFACT_IDENTITY,
            deployment_alignment=DeploymentAlignment.UNKNOWN,
            roster_alignment=RosterAlignment.UNKNOWN,
            evaluation_applicability=(
                EvaluationApplicabilityStatus.NO_MODEL_ATTRIBUTED),
            served_model_roster=roster,
            served_roster_fingerprint=roster_fingerprint(roster) if roster
            else None,
            detail=detail)


def served_model_roster(pipeline: Any) -> tuple[str, ...]:
    """The base models this in-memory object can actually invoke.

    Sourced from the live executable mapping, NOT from `PipelineMetadata`, not
    from a `ModelRecord`, and not from an export manifest. The question is what
    this object can run, and only the object answers it.

    `gnn_scorer` is deliberately excluded: it is a scorer contributing a
    feature, not a base model in the ensemble mapping, so the roster is not
    exhaustive of the served architecture.
    """
    models = getattr(pipeline, "trained_models", None)
    if not isinstance(models, Mapping):
        return ()
    return tuple(sorted(str(name) for name in models))


def load_pipeline_with_identity(
    path: str | Path,
    loader: Callable[[Path], Any],
) -> tuple[Any, ArtifactIdentity]:
    """Load an artifact and measure the bytes that were actually loaded.

    Measures BEFORE and AFTER and refuses on disagreement. A digest taken only
    before the load describes bytes that may have been replaced during it; one
    taken only after describes bytes that may not be what was deserialised.

    Exactly two measurements. Registry resolution then uses the returned
    identity rather than reading the file a third time.
    """
    resolved = Path(path)
    before = ArtifactIdentity.measure(resolved)
    pipeline = loader(resolved)
    after = ArtifactIdentity.measure(resolved)
    if before.sha256 != after.sha256 or before.size_bytes != after.size_bytes:
        raise ArtifactChangedDuringLoadError(
            f"{resolved} changed while it was being loaded: "
            f"{before.sha256[:12]} ({before.size_bytes} bytes) -> "
            f"{after.sha256[:12]} ({after.size_bytes} bytes). The digest "
            "cannot describe the object that was deserialised.")
    return pipeline, after


def _roster_alignment(record: ModelRecord,
                      served: tuple[str, ...]) -> tuple[RosterAlignment,
                                                        Optional[str]]:
    """Compare the executable roster against the record and its projection."""
    registered = frozenset(record.model_roster)
    actual = frozenset(served)

    if actual == registered:
        return RosterAlignment.EXACT, None

    projection = record.serving_projection
    if projection is None:
        return (RosterAlignment.UNKNOWN,
                "the record declares no serving projection, so a roster "
                f"difference of {sorted(registered ^ actual)} cannot be "
                "judged intentional or accidental")

    expected = projection.expected_served(record.model_roster)
    if actual == expected:
        return RosterAlignment.SERVING_SUBSET, None

    return (RosterAlignment.INCONSISTENT,
            f"expected the serving projection {sorted(expected)} but the "
            f"loaded object executes {sorted(actual)}; difference "
            f"{sorted(expected ^ actual)}")


def resolve_runtime_binding(
    pipeline: Any,
    artifact: Optional[ArtifactIdentity],
    registry_path: str | Path,
) -> RuntimeModelBinding:
    """Bind a loaded pipeline to a registry declaration, or state why not.

    Never raises for an absent or unreadable registry: an API that will not
    start because a declaration is missing is worse than one that starts and
    says it cannot identify itself. Every failure becomes a typed status.
    """
    if pipeline is None:
        return RuntimeModelBinding.no_model_loaded(
            detail="no artifact was loaded at startup")

    served = served_model_roster(pipeline)

    if artifact is None:
        return RuntimeModelBinding.unattributed(
            served,
            detail="the pipeline was supplied directly, so no artifact bytes "
                   "were measured and its identity cannot be established")

    unregistered = RuntimeModelBinding(
        resolution_status=ArtifactResolutionStatus.ARTIFACT_NOT_REGISTERED,
        deployment_alignment=DeploymentAlignment.UNKNOWN,
        roster_alignment=RosterAlignment.UNKNOWN,
        evaluation_applicability=(
            EvaluationApplicabilityStatus.NO_MODEL_ATTRIBUTED),
        served_model_roster=served,
        served_roster_fingerprint=roster_fingerprint(served) if served
        else None,
        artifact_sha256=artifact.sha256,
        artifact_uri=artifact.uri)

    try:
        registry = ModelRegistry.load(registry_path)
    except (RegistryInvariantError, OSError, ValueError) as exc:
        logger.warning("deployment registry unreadable at %s: %s",
                       registry_path, exc)
        return replace(
            unregistered,
            detail=f"the deployment registry could not be read: {exc}")

    match = next((r for r in registry.records
                  if r.artifact.sha256 == artifact.sha256), None)
    if match is None:
        return replace(
            unregistered,
            detail="the loaded artifact's digest appears in no record of "
                   f"{registry_path}")

    production = registry.current_production()
    if production is None:
        alignment = DeploymentAlignment.NO_PRODUCTION_DECLARED
    elif production.record_id == match.record_id:
        alignment = DeploymentAlignment.MATCHES_DECLARED_PRODUCTION
    else:
        alignment = DeploymentAlignment.DIFFERS_FROM_DECLARED_PRODUCTION

    roster_state, roster_detail = _roster_alignment(match, served)

    # COMMIT A PUBLISHES NO EVIDENCE. The served roster is a projection of the
    # evaluated roster, so the record's metrics are not automatically evidence
    # for these bytes. Commit C attaches a SealedEvaluation that names this
    # digest and this roster fingerprint, and only then can this become
    # APPLICABLE.
    if roster_state is RosterAlignment.INCONSISTENT:
        applicability = EvaluationApplicabilityStatus.ROSTER_MISMATCH
    else:
        # Even a linked sealed_evaluation_id is NOT enough here. Commit C
        # requires the evaluation to name this artifact digest AND this served
        # roster fingerprint; until that type exists there is no evidence this
        # module is entitled to call applicable.
        applicability = EvaluationApplicabilityStatus.NO_SEALED_EVALUATION

    return RuntimeModelBinding(
        resolution_status=ArtifactResolutionStatus.REGISTERED,
        deployment_alignment=alignment,
        roster_alignment=roster_state,
        evaluation_applicability=applicability,
        served_model_roster=served,
        served_roster_fingerprint=roster_fingerprint(served) if served
        else None,
        artifact_sha256=artifact.sha256,
        artifact_uri=artifact.uri,
        record_id=match.record_id,
        model_version=match.version,
        registry_stage=match.stage,
        registered_model_roster=match.model_roster,
        detail=roster_detail)
