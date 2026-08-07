"""src/genomic_variant_classifier/monitoring/model_registry.py

Author: Monzia Moodie
Written 2026-08-07. REGISTRY-1b.

The deployment control plane: which artifact is declared registered, in shadow,
in production, or archived -- and on what evidence.

WHY THIS IS A NEW MODULE RATHER THAN AN ADDITION TO `registry.py`.
`monitoring/registry.py` is a DATA-SOURCE registry: `Category`, `Check`,
`Source`, `Verdict`, `REGISTRY`, and five accessors over the external corpora
the classifier consumes. Four call sites imported `ModelRegistry` from it --
`continual_trainer.py` twice and `drift_monitor.yml` twice -- and the class was
never written. `git log --all -S "class ModelRegistry"` is empty: never
implemented, not deleted.

Those call sites specified an INTERFACE worth preserving. They did not
establish that their module placement was wise. Grafting deployment state onto
a data-source registry would resolve an ImportError by creating a semantic junk
drawer, so the imports are corrected instead.

WHAT THIS FILE REFUSES TO PROVIDE, AND WHY.

    record.auroc

does not exist, not even as a convenience property. A bare scalar is what makes

    0.9988 unseen-gene AUROC   vs   0.9984 ordinary test AUROC

look like a valid arithmetic comparison. Callers must write
`record.evaluation.metrics["auroc"]`, which keeps the number visibly adjacent
to `record.evaluation.protocol`. That is GATE-1's whole content, enforced by
the shape of the type rather than by a convention someone must remember.

WHAT A COMMITTED REGISTRY CAN AND CANNOT CLAIM. A file in version control is a
DECLARATION of deployment state. It is not the state serving predictions.
Continuous Integration can prove "the repository's declared production
deployment is structurally coherent"; it cannot prove "production is healthy".
Closing that gap requires the serving environment to attest that its loaded
artifact digest equals the declared one, which is DEPLOY-1's territory, not
this module's.

IDENTITY IS LINEAGE PLUS CONTENT. `version` is a human display label; a rerun
or re-export can produce a different artifact under the same label. `record_id`
is `f"{run_id}-{sha256[:12]}"`, so identity is immutable and a silently
regenerated artifact cannot masquerade as the one that was evaluated.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "ArtifactIdentity",
    "EvaluationEvidence",
    "EvaluationProtocol",
    "ModelRecord",
    "ModelRegistry",
    "PromotionDecision",
    "PromotionEvent",
    "PromotionPolicy",
    "RegistryInvariantError",
    "Stage",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = 1

#: Artifact locations a PRODUCTION declaration will not accept. A local
#: filesystem path is not deployment provenance: neither a hosted runner nor a
#: future serving host can resolve `C:\Projects\...\pipeline.joblib`.
_NON_DURABLE_URI_SCHEMES = ("file:",)


class RegistryInvariantError(RuntimeError):
    """A registry operation would produce an incoherent declaration.

    Raised rather than returned, because every caller of this module is
    performing a consequential state change and none of them has a sensible
    fallback for "the declaration is now incoherent".
    """


class Stage(str, Enum):
    """Where a record stands in the deployment lifecycle."""

    REGISTERED = "registered"
    SHADOW = "shadow"
    PRODUCTION = "production"
    ARCHIVED = "archived"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ArtifactIdentity:
    """WHERE the artifact is, and WHAT it is.

    `sha256` is computed by the registry from the bytes on disk, never accepted
    from a caller. A caller-authored digest describes what the caller believed,
    not what was registered.
    """

    uri: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.uri:
            raise RegistryInvariantError("an artifact identity requires a uri")
        if len(self.sha256) != 64 or not all(
                c in "0123456789abcdef" for c in self.sha256):
            raise RegistryInvariantError(
                f"sha256 must be 64 lowercase hexadecimal characters, got "
                f"{self.sha256!r}")
        if self.size_bytes <= 0:
            raise RegistryInvariantError(
                f"size_bytes must be positive, got {self.size_bytes}")

    @property
    def is_durable(self) -> bool:
        """False for a location only this machine can resolve."""
        return not self.uri.startswith(_NON_DURABLE_URI_SCHEMES)

    def to_dict(self) -> dict:
        return {"uri": self.uri, "sha256": self.sha256,
                "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ArtifactIdentity":
        return cls(uri=payload["uri"], sha256=payload["sha256"],
                   size_bytes=int(payload["size_bytes"]))

    @classmethod
    def measure(cls, path: str | Path, *, uri: Optional[str] = None
                ) -> "ArtifactIdentity":
        """Digest and size READ FROM DISK. The one constructor callers use."""
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise RegistryInvariantError(f"no artifact at {resolved}")
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return cls(uri=uri or resolved.as_uri(),
                   sha256=digest.hexdigest(),
                   size_bytes=resolved.stat().st_size)


@dataclass(frozen=True)
class EvaluationProtocol:
    """HOW a number was obtained. Without this, a metric is not comparable.

    `split_kind` is deliberately free text rather than an enumeration: the
    project's split vocabulary is still moving (validation, test, unseen-gene
    holdout, gene-disjoint holdout), and freezing a partial enumeration would
    force a future protocol into whichever member fits worst. Equality is what
    matters here, and equality works on strings.
    """

    protocol_id: str
    split_kind: str
    population_scope: str
    n_observations: int
    label_policy: str
    population_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("protocol_id", "split_kind", "population_scope",
                     "label_policy"):
            if not getattr(self, name):
                raise RegistryInvariantError(
                    f"an evaluation protocol requires a nonempty {name}; an "
                    "unnamed protocol makes its metrics incomparable, which "
                    "is the defect this type exists to prevent")
        if self.n_observations <= 0:
            raise RegistryInvariantError(
                f"n_observations must be positive, got {self.n_observations}")

    def to_dict(self) -> dict:
        return {"protocol_id": self.protocol_id,
                "split_kind": self.split_kind,
                "population_scope": self.population_scope,
                "n_observations": self.n_observations,
                "label_policy": self.label_policy,
                "population_fingerprint": self.population_fingerprint}

    @classmethod
    def from_dict(cls, payload: Mapping) -> "EvaluationProtocol":
        return cls(protocol_id=payload["protocol_id"],
                   split_kind=payload["split_kind"],
                   population_scope=payload["population_scope"],
                   n_observations=int(payload["n_observations"]),
                   label_policy=payload["label_policy"],
                   population_fingerprint=payload.get(
                       "population_fingerprint"))


@dataclass(frozen=True)
class EvaluationEvidence:
    """Metrics unreadable without their protocol in the same object."""

    protocol: EvaluationProtocol
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.metrics:
            raise RegistryInvariantError(
                "evaluation evidence with no metrics is not evidence")
        for name, value in self.metrics.items():
            if (not isinstance(value, (int, float))
                    or isinstance(value, bool)):
                raise RegistryInvariantError(
                    f"metric {name!r} must be a real number, got {value!r}")
        object.__setattr__(self, "metrics", dict(self.metrics))

    def to_dict(self) -> dict:
        return {"protocol": self.protocol.to_dict(),
                "metrics": dict(self.metrics)}

    @classmethod
    def from_dict(cls, payload: Mapping) -> "EvaluationEvidence":
        return cls(protocol=EvaluationProtocol.from_dict(payload["protocol"]),
                   metrics={k: float(v)
                            for k, v in payload["metrics"].items()})


@dataclass(frozen=True)
class TrainingLineage:
    """WHERE the artifact came from."""

    run_id: str
    source_commit: Optional[str] = None
    clinvar_release: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise RegistryInvariantError("a lineage requires a run_id")

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "source_commit": self.source_commit,
                "clinvar_release": self.clinvar_release}

    @classmethod
    def from_dict(cls, payload: Mapping) -> "TrainingLineage":
        return cls(run_id=payload["run_id"],
                   source_commit=payload.get("source_commit"),
                   clinvar_release=payload.get("clinvar_release"))


@dataclass(frozen=True)
class PromotionEvent:
    """One stage transition, kept forever.

    A mutable `record.stage = "production"` answers what is production now and
    destroys what was production before. A clinical result traced to a date
    needs the second question answered.
    """

    version: str
    from_stage: Stage
    to_stage: Stage
    occurred_at_utc: str
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {"version": self.version,
                "from_stage": self.from_stage.value,
                "to_stage": self.to_stage.value,
                "occurred_at_utc": self.occurred_at_utc,
                "reason": self.reason}

    @classmethod
    def from_dict(cls, payload: Mapping) -> "PromotionEvent":
        return cls(version=payload["version"],
                   from_stage=Stage(payload["from_stage"]),
                   to_stage=Stage(payload["to_stage"]),
                   occurred_at_utc=payload["occurred_at_utc"],
                   reason=payload.get("reason"))


@dataclass(frozen=True)
class ModelRecord:
    """A registered artifact and everything needed to judge it.

    THIS IS THE DEPLOYMENT PROVENANCE. There is deliberately no separate
    `DeploymentProvenance` type: two peer objects describing one artifact is
    how a fact acquires two sources of truth.

    `sealed_evaluation_id` links to PROD-1's future `SealedEvaluation`, which
    IS a different object -- what a scientifically sealed experiment
    established, as opposed to what is deployable. Those may legitimately
    diverge, which is exactly why they must not be the same record.
    """

    version: str
    artifact: ArtifactIdentity
    lineage: TrainingLineage
    evaluation: EvaluationEvidence
    feature_names: tuple[str, ...]
    model_roster: tuple[str, ...]
    stage: Stage
    registered_at_utc: str
    notes: Optional[str] = None
    sealed_evaluation_id: Optional[str] = None
    drift_report: Optional[Mapping] = None

    def __post_init__(self) -> None:
        if not self.version:
            raise RegistryInvariantError("a record requires a version")
        if not self.feature_names:
            raise RegistryInvariantError(
                "a record requires its feature names; a feature COUNT is a "
                "claim, an enumeration is a check")
        if not self.model_roster:
            raise RegistryInvariantError(
                "a record requires its model roster. ROSTER-1: retraining "
                "that changes the roster changes the intervention from 'new "
                "data plus adaptation' to 'new data plus adaptation plus "
                "architecture change', and any movement becomes confounded. "
                "A record without a roster cannot detect that.")
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "model_roster", tuple(self.model_roster))

    @property
    def record_id(self) -> str:
        """Immutable identity: human lineage plus content digest."""
        return f"{self.lineage.run_id}-{self.artifact.sha256[:12]}"

    @property
    def roster_fingerprint(self) -> str:
        """Order-independent digest of the roster, for equality without
        depending on however the exporting run happened to order it."""
        joined = "\n".join(sorted(self.model_roster))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def with_stage(self, stage: Stage) -> "ModelRecord":
        """A copy at a new stage. Records are frozen; the registry replaces."""
        return ModelRecord(
            version=self.version, artifact=self.artifact,
            lineage=self.lineage, evaluation=self.evaluation,
            feature_names=self.feature_names, model_roster=self.model_roster,
            stage=stage, registered_at_utc=self.registered_at_utc,
            notes=self.notes, sealed_evaluation_id=self.sealed_evaluation_id,
            drift_report=self.drift_report)

    def to_dict(self) -> dict:
        return {"version": self.version,
                "record_id": self.record_id,
                "artifact": self.artifact.to_dict(),
                "lineage": self.lineage.to_dict(),
                "evaluation": self.evaluation.to_dict(),
                "feature_names": list(self.feature_names),
                "model_roster": list(self.model_roster),
                "roster_fingerprint": self.roster_fingerprint,
                "stage": self.stage.value,
                "registered_at_utc": self.registered_at_utc,
                "notes": self.notes,
                "sealed_evaluation_id": self.sealed_evaluation_id,
                "drift_report": self.drift_report}

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ModelRecord":
        # `record_id` and `roster_fingerprint` are DERIVED and deliberately not
        # read back: a stored value that disagrees with the derivation would be
        # a second source of truth for one fact.
        return cls(
            version=payload["version"],
            artifact=ArtifactIdentity.from_dict(payload["artifact"]),
            lineage=TrainingLineage.from_dict(payload["lineage"]),
            evaluation=EvaluationEvidence.from_dict(payload["evaluation"]),
            feature_names=tuple(payload["feature_names"]),
            model_roster=tuple(payload["model_roster"]),
            stage=Stage(payload["stage"]),
            registered_at_utc=payload["registered_at_utc"],
            notes=payload.get("notes"),
            sealed_evaluation_id=payload.get("sealed_evaluation_id"),
            drift_report=payload.get("drift_report"))


@dataclass(frozen=True)
class PromotionPolicy:
    """What a production promotion demands. Typed, so that a calibration
    sanity floor and a promotion floor cannot be confused for one another --
    GATE-1 measured four thresholds in this repository answering four different
    questions, none of them named.
    """

    policy_id: str
    metric_name: str = "auroc"
    minimum: float = 0.97
    max_regression: float = 0.002
    require_durable_uri: bool = True
    expected_model_roster: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not self.policy_id or not self.metric_name:
            raise RegistryInvariantError(
                "a promotion policy requires an identifier and a metric name")
        if self.max_regression < 0.0:
            raise RegistryInvariantError("max_regression must not be negative")


@dataclass(frozen=True)
class PromotionDecision:
    """Accepted, or refused with a machine-readable reason. Never a bare bool.

    Mirrors `OperatingPointOutcome`: a refusal that cannot state why is the
    silent `None` the typed vocabulary exists to replace.
    """

    accepted: bool
    reason: Optional[str] = None
    detail: Optional[str] = None

    @classmethod
    def refused(cls, reason: str, detail: Optional[str] = None
                ) -> "PromotionDecision":
        return cls(accepted=False, reason=reason, detail=detail)

    @classmethod
    def approved(cls, detail: Optional[str] = None) -> "PromotionDecision":
        return cls(accepted=True, reason=None, detail=detail)


@dataclass
class ModelRegistry:
    """The declared deployment state, and the transitions that produced it.

    Persisted as JSON with an explicit `schema_version`, written atomically:
    a half-written registry is a state nothing can interpret, and `os.replace`
    is atomic on both POSIX and Windows for a same-directory rename.
    """

    path: Path
    records: list[ModelRecord] = field(default_factory=list)
    promotion_history: list[PromotionEvent] = field(default_factory=list)

    # ---------------------------------------------------------------- load

    @classmethod
    def load(cls, path: str | Path, *, create_if_missing: bool = False
             ) -> "ModelRegistry":
        """Read a registry. ABSENCE IS AN ERROR unless explicitly permitted.

        A missing registry is not an empty registry: one means "no declaration
        exists", the other means "a declaration exists and declares nothing".
        Conflating them is how a workflow reports health for a check that never
        happened.
        """
        resolved = Path(path)
        if not resolved.is_file():
            if not create_if_missing:
                raise RegistryInvariantError(
                    f"no registry at {resolved}. This is NOT 'the registry is "
                    "empty' -- it is 'no deployment declaration exists'. Pass "
                    "create_if_missing=True to declare one.")
            return cls(path=resolved)

        payload = json.loads(resolved.read_text(encoding="utf-8"))
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise RegistryInvariantError(
                f"{resolved} declares schema_version {version!r}; this build "
                f"understands {SCHEMA_VERSION}. Refusing to interpret a "
                "registry written by a different contract.")
        registry = cls(
            path=resolved,
            records=[ModelRecord.from_dict(r) for r in payload["records"]],
            promotion_history=[PromotionEvent.from_dict(e)
                               for e in payload["promotion_history"]])
        registry._validate()
        return registry

    def _validate(self) -> None:
        versions = [r.version for r in self.records]
        duplicates = sorted({v for v in versions if versions.count(v) > 1})
        if duplicates:
            raise RegistryInvariantError(
                f"duplicate versions in the registry: {duplicates}")
        production = [r for r in self.records if r.stage is Stage.PRODUCTION]
        if len(production) > 1:
            raise RegistryInvariantError(
                "more than one record is declared PRODUCTION: "
                f"{[r.version for r in production]}")

    # ---------------------------------------------------------------- save

    def save(self) -> None:
        """Atomic write. A torn registry is worse than no registry."""
        self._validate()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "records": [r.to_dict() for r in self.records],
            "promotion_history": [e.to_dict()
                                  for e in self.promotion_history],
        }
        text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", delete=False,
            dir=str(self.path.parent), suffix=".tmp")
        with handle:
            handle.write(text)
        os.replace(handle.name, self.path)
        logger.info("registry written: %s (%d records)",
                    self.path, len(self.records))

    # ------------------------------------------------------------ mutate

    def register(
        self,
        *,
        version: str,
        model_path: str | Path,
        lineage: TrainingLineage,
        evaluation: EvaluationEvidence,
        feature_names: Sequence[str],
        model_roster: Sequence[str],
        artifact_uri: Optional[str] = None,
        notes: Optional[str] = None,
        sealed_evaluation_id: Optional[str] = None,
        drift_report: Optional[Mapping] = None,
    ) -> ModelRecord:
        """Add a record at stage REGISTERED. The digest is MEASURED here."""
        if any(r.version == version for r in self.records):
            raise RegistryInvariantError(
                f"version {version!r} is already registered; a version is an "
                "identity, not a label to reuse")
        record = ModelRecord(
            version=version,
            artifact=ArtifactIdentity.measure(model_path, uri=artifact_uri),
            lineage=lineage,
            evaluation=evaluation,
            feature_names=tuple(feature_names),
            model_roster=tuple(model_roster),
            stage=Stage.REGISTERED,
            registered_at_utc=_utc_now(),
            notes=notes,
            sealed_evaluation_id=sealed_evaluation_id,
            drift_report=drift_report)
        self.records.append(record)
        self.promotion_history.append(PromotionEvent(
            version=version, from_stage=Stage.REGISTERED,
            to_stage=Stage.REGISTERED,
            occurred_at_utc=record.registered_at_utc,
            reason="registered"))
        self._validate()
        return record

    def _find(self, version: str) -> ModelRecord:
        for record in self.records:
            if record.version == version:
                return record
        raise RegistryInvariantError(f"no record with version {version!r}")

    def _transition(self, record: ModelRecord, to_stage: Stage,
                    reason: str) -> ModelRecord:
        moved = record.with_stage(to_stage)
        self.records[self.records.index(record)] = moved
        self.promotion_history.append(PromotionEvent(
            version=record.version, from_stage=record.stage,
            to_stage=to_stage, occurred_at_utc=_utc_now(), reason=reason))
        self._validate()
        return moved

    def promote_to_shadow(self, version: str,
                          reason: str = "shadow burn-in") -> ModelRecord:
        """Shadow carries no clinical consequence, so it stays cheap."""
        record = self._find(version)
        if record.stage is Stage.PRODUCTION:
            raise RegistryInvariantError(
                f"{version!r} is PRODUCTION; demote deliberately rather than "
                "moving it sideways into shadow")
        return self._transition(record, Stage.SHADOW, reason)

    def evaluate_for_production(
        self, version: str, policy: PromotionPolicy,
    ) -> PromotionDecision:
        """Judge a production promotion WITHOUT performing it.

        Separated so the decision can be reported, logged and reviewed before
        anything changes -- and so it can be tested without mutating state.
        """
        record = self._find(version)

        if record.stage is not Stage.SHADOW:
            return PromotionDecision.refused(
                "candidate_not_in_shadow",
                f"{version!r} is {record.stage.value}, not shadow")

        if policy.require_durable_uri and not record.artifact.is_durable:
            return PromotionDecision.refused(
                "artifact_uri_not_durable",
                f"{record.artifact.uri} cannot be resolved off this machine")

        if policy.expected_model_roster is not None:
            expected = tuple(sorted(policy.expected_model_roster))
            actual = tuple(sorted(record.model_roster))
            if expected != actual:
                return PromotionDecision.refused(
                    "model_roster_mismatch",
                    f"expected {list(expected)}, registered {list(actual)}")

        if policy.metric_name not in record.evaluation.metrics:
            return PromotionDecision.refused(
                "metric_absent",
                f"{policy.metric_name!r} is not in the candidate's evidence")
        candidate_metric = record.evaluation.metrics[policy.metric_name]

        if candidate_metric < policy.minimum:
            return PromotionDecision.refused(
                "below_absolute_minimum",
                f"{policy.metric_name}={candidate_metric:.6f} < "
                f"{policy.minimum}")

        production = self.current_production()
        if production is not None:
            if record.evaluation.protocol != production.evaluation.protocol:
                return PromotionDecision.refused(
                    "evaluation_protocol_mismatch",
                    "candidate protocol "
                    f"{record.evaluation.protocol.protocol_id!r} != "
                    "production protocol "
                    f"{production.evaluation.protocol.protocol_id!r}; the "
                    "two metrics are not comparable")
            if policy.metric_name not in production.evaluation.metrics:
                return PromotionDecision.refused(
                    "production_metric_absent",
                    f"production carries no {policy.metric_name!r}")
            production_metric = production.evaluation.metrics[
                policy.metric_name]
            if candidate_metric < production_metric - policy.max_regression:
                return PromotionDecision.refused(
                    "regression_exceeds_tolerance",
                    f"{candidate_metric:.6f} vs production "
                    f"{production_metric:.6f}, tolerance "
                    f"{policy.max_regression}")

        return PromotionDecision.approved(
            f"policy {policy.policy_id!r} satisfied")

    def promote_to_production(
        self, version: str, policy: PromotionPolicy,
        reason: str = "promoted after burn-in",
    ) -> ModelRecord:
        """Perform the promotion, or raise with the refusal's reason.

        DELIBERATELY NOT `promote(version, "production")`. A clinically
        consequential transition should not look like a string assignment, and
        the pre-existing logger instructed operators to run exactly that.
        """
        decision = self.evaluate_for_production(version, policy)
        if not decision.accepted:
            raise RegistryInvariantError(
                f"production promotion refused: {decision.reason} "
                f"({decision.detail})")
        incumbent = self.current_production()
        if incumbent is not None and incumbent.version != version:
            self._transition(incumbent, Stage.ARCHIVED,
                             f"superseded by {version}")
        return self._transition(self._find(version), Stage.PRODUCTION, reason)

    # ------------------------------------------------------------- query

    def current_production(self) -> Optional[ModelRecord]:
        for record in self.records:
            if record.stage is Stage.PRODUCTION:
                return record
        return None

    def by_stage(self, stage: Stage) -> list[ModelRecord]:
        return [r for r in self.records if r.stage is stage]

    def history_for(self, version: str) -> list[PromotionEvent]:
        return [e for e in self.promotion_history if e.version == version]

    def summary_lines(self) -> list[str]:
        """The summary as data, so it can be asserted as well as printed."""
        lines = [f"registry: {self.path}  schema_version={SCHEMA_VERSION}",
                 f"records: {len(self.records)}  "
                 f"transitions: {len(self.promotion_history)}"]
        if not self.records:
            lines.append("  NO RECORDS. No deployment is declared.")
        for record in self.records:
            metrics = ", ".join(
                f"{k}={v:.6f}"
                for k, v in sorted(record.evaluation.metrics.items()))
            lines.append(
                f"  {record.stage.value:<10} {record.version:<28} "
                f"{record.record_id}")
            protocol = record.evaluation.protocol
            lines.append(
                f"             protocol={protocol.protocol_id} "
                f"({protocol.split_kind}, n={protocol.n_observations})")
            lines.append(f"             {metrics}")
            lines.append(
                f"             roster={len(record.model_roster)} "
                f"fingerprint={record.roster_fingerprint} "
                f"features={len(record.feature_names)}")
        return lines

    def print_summary(self) -> None:
        for line in self.summary_lines():
            print(line)
