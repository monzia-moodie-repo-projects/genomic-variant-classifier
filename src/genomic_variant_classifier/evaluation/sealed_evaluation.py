"""A metric's origin is part of the metric. Commit C.

Created 2026-08-24, against two committed censuses read in full.

WHY THIS TYPE EXISTS
====================
`EvaluationEvidence.metrics` is `Mapping[str, float]` -- one flat mapping, with
`EvaluationProtocol` beside it rather than inside each entry. So a single
evidence object can hold a figure computed on held-out predictions and a figure
scraped from a training log under one protocol, with nothing in the type to tell
them apart.

MEASURED, `MEASUREMENT_2026-08-08_metricorigin-census.md`: Run 14's own manifest
holds FOUR figures a careless reader would call "Run 14's area under the
receiver operating characteristic curve", spanning 0.9975 to 0.9985, and
distinguishes them only by a suffix in their key names.

    0.9975  metrics.json -> auroc                     computed on the test split
    0.9975  manifest -> stacker_metrics_test.auroc    the same computation
    0.9984  manifest -> lr_stacker_auroc_from_log     scraped from a training log
    0.9985  manifest -> oof_blend_auroc_from_log      scraped from a training log

The scraped pair describe *a different quantity entirely*: out-of-fold
performance during training, not held-out performance after it. The census names
the consequence: **"That is precisely the mechanism by which 0.9847 came to be
published as a holdout metric."**

`docs/METRICS.md` has carried two named columns for months. Only the code cannot
represent the distinction. The `_from_log` suffix is, in the census's words,
**"a naming convention doing a type's job"**.

WHAT THE CENSUSES BOUND
=======================
`MEASUREMENT_2026-08-08_baseline1-provenance-census.md` establishes WHICH runs
may be sealed, and its ruling is quoted rather than paraphrased:

    Commit C's scope is bounded by section 9: seal Run 14, represent Run 10b's
    partiality honestly, and attribute nothing else.

`0.9847` is UNATTRIBUTABLE -- earliest appearance a commit subject line, no
Phase 2 or Run 8 artefact in the repository -- so no seal can ever be composed
for it. Run 10b's artefact declares its own incompleteness and is the test case
for a record that is honestly partial.

THE DESIGN IS RULED, NOT CHOSEN
===============================
    `EvaluationProtocol`, `EvaluationEvidence` and `TrainingLineage` already
    exist and already refuse the failures they were built for. The indicated
    design is a thin sealing layer over them, not a parallel hierarchy.

So this module composes those three. It does not replace them, does not
subclass them, and adds only what they cannot express.

FIVE REQUIREMENTS, EACH WITH A SOURCE
=====================================
1. ORIGIN IS A FIELD, not a key suffix.  metricorigin census section 8.1.
2. `artifact_sha256` is MANDATORY. It already exists in Run 14's manifest --
   "sealing does not need to introduce it; it needs to make it mandatory."
3. COERCION IS DECLARED. Run 14's per-model metrics are stored as `str`, and
   `EvaluationEvidence.__post_init__` refuses non-numbers. A sealing layer
   "must coerce explicitly and record that it coerced, never silently call
   float()". Note that `EvaluationEvidence.from_dict` DOES call `float(v)`
   silently, at model_registry.py:260.
4. PARTIALITY IS REPRESENTABLE. Run 10b declares
   `"status": "partial"` with three outputs `lost`.
5. THE SEAL NAMES A DIGEST AND A ROSTER FINGERPRINT. Not a design preference:
   `api/attribution.py:387` refuses to call a record applicable without both,
   because **"Resolving a digest authorises IDENTITY, not EVIDENCE"** -- a
   metric measured on a thirteen-model ensemble is not evidence for a
   twelve-model serving projection of it.

WHY ORIGIN IS AN ENUMERATION WHERE `split_kind` IS FREE TEXT
============================================================
`EvaluationProtocol.split_kind` is deliberately free text, and says why: the
split vocabulary is still moving, and "freezing a partial enumeration would
force a future protocol into whichever member fits worst". Metric ORIGIN is not
in that position. It is a closed distinction between a figure computed from
predictions and a figure read out of a log, and a third kind would be a new
fact requiring a new census -- not a vocabulary drift.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional

from genomic_variant_classifier.monitoring.model_registry import (
    EvaluationEvidence,
    EvaluationProtocol,
    RegistryInvariantError,
    TrainingLineage,
)

SCHEMA = "gvc.sealed-evaluation"
SCHEMA_VERSION = 1

#: The suffix Run 14's manifest uses to mark a scraped figure. A seal must not
#: contradict it: a key still carrying the suffix is asserting its own origin.
_FROM_LOG = "_from_log"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SealError(RegistryInvariantError):
    """A sealed record does not satisfy its own contract."""


class MetricOrigin(str, Enum):
    """WHERE a number came from. Part of the number, not of its key.

    The census found this fact already encoded in Run 14's key names --
    `oof_blend_auroc_from_log` -- because whoever wrote the manifest understood
    that origin belongs with the value and had nowhere to put it except the key.
    """

    COMPUTED_FROM_PREDICTIONS = "computed_from_predictions"
    SCRAPED_FROM_TRAINING_LOG = "scraped_from_training_log"


class SealCompleteness(str, Enum):
    """Whether the record covers everything the run was meant to produce.

    Run 10b is the reason this exists: its artefact declares
    `"status": "partial"` with `deep_ensemble.joblib`, the graph network and the
    cloud-computed test figure all `lost`. A seal that could not say so would
    have to either lie or refuse, and the census requires neither.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"


@dataclass(frozen=True)
class Coercion:
    """A declaration that a value was transformed on the way in.

    Never silent. Run 14 stores per-model metrics as strings, and a string that
    looks like a number "may be a rounded rendering of one, and rounding is a
    transformation a sealed record should declare".
    """

    original: str
    parsed_as: str = "float"

    def __post_init__(self) -> None:
        if not self.original.strip():
            raise SealError("a coercion must record the original text")


@dataclass(frozen=True)
class SealedMetric:
    """One number, with its origin, and any transformation applied to it."""

    name: str
    value: float
    origin: MetricOrigin
    coercion: Optional[Coercion] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SealError("a metric requires a name")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise SealError(
                "metric {!r} must be a real number, got {!r}. The same refusal "
                "EvaluationEvidence already makes.".format(self.name, self.value))
        if self.value != self.value:                       # NaN
            raise SealError(
                "metric {!r} is NaN. A metric that could not be computed is "
                "not sealed evidence; omit it and record the omission."
                .format(self.name))
        # THE SUFFIX MUST NOT CONTRADICT THE FIELD. Run 14's manifest asserts
        # origin in the key; a seal that disagreed with it would replace one
        # ambiguity with two.
        if (self.name.endswith(_FROM_LOG)
                and self.origin is not MetricOrigin.SCRAPED_FROM_TRAINING_LOG):
            raise SealError(
                "metric {!r} carries the {!r} suffix but declares origin {}. "
                "The suffix is the artefact's own statement of origin; a seal "
                "may replace it but must not contradict it."
                .format(self.name, _FROM_LOG, self.origin.value))

    @classmethod
    def from_manifest_value(cls, name: str, raw, origin: MetricOrigin
                            ) -> "SealedMetric":
        """Build from an artefact value, DECLARING any coercion.

        `EvaluationEvidence.from_dict` calls `float(v)` silently at
        model_registry.py:260. This does not: a string becomes a float only with
        a `Coercion` attached recording what it was.
        """
        if isinstance(raw, str):
            try:
                value = float(raw)
            except ValueError:
                raise SealError(
                    "metric {!r} holds {!r}, which is not a number. Sealing "
                    "refuses rather than guessing.".format(name, raw)) from None
            return cls(name=name, value=value, origin=origin,
                       coercion=Coercion(original=raw))
        return cls(name=name, value=raw, origin=origin)

    def as_record(self) -> dict:
        record = {"name": self.name, "value": self.value,
                  "origin": self.origin.value}
        if self.coercion is not None:
            record["coercion"] = {"original": self.coercion.original,
                                  "parsed_as": self.coercion.parsed_as}
        return record


@dataclass(frozen=True)
class SealedEvaluation:
    """What a scientifically sealed experiment established.

    Deliberately NOT the same object as `ModelRecord`, which is deployment
    provenance. model_registry.py:375 states why: the two "may legitimately
    diverge, which is exactly why they must not be the same record."
    """

    seal_id: str
    lineage: TrainingLineage
    protocol: EvaluationProtocol
    metrics: tuple
    artifact_sha256: Mapping[str, str]
    roster_fingerprint: str
    completeness: SealCompleteness = SealCompleteness.COMPLETE
    lost_outputs: tuple = ()
    findings: tuple = ()
    _checked: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not self.seal_id.strip():
            raise SealError("a seal requires an identifier")
        if not self.metrics:
            raise SealError(
                "a seal with no metrics is not evidence -- the same refusal "
                "EvaluationEvidence makes")
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise SealError(
                "duplicate metric name(s): {}. Two entries under one name is "
                "the flat-mapping defect this type exists to end."
                .format(sorted({n for n in names if names.count(n) > 1})))

        # REQUIREMENT 2: artifact_sha256 is MANDATORY, not optional.
        if not self.artifact_sha256:
            raise SealError(
                "a seal must name the artefact(s) it is evidence for. Run 14's "
                "manifest already carries artifact_sha256; sealing makes it "
                "mandatory rather than introducing it")
        for label, digest in self.artifact_sha256.items():
            if not _SHA256.match(str(digest)):
                raise SealError(
                    "artefact {!r} has digest {!r}; expected 64 lowercase "
                    "hexadecimal characters".format(label, digest))

        # REQUIREMENT 5: attribution.py cannot reach APPLICABLE without this.
        if not self.roster_fingerprint.strip():
            raise SealError(
                "a seal must name the roster it was measured on. Resolving a "
                "digest authorises IDENTITY, not EVIDENCE: a metric measured "
                "on a thirteen-model ensemble is not evidence for a "
                "twelve-model serving projection of it")

        # REQUIREMENT 4: partiality is representable, and must be stated.
        if self.completeness is SealCompleteness.PARTIAL and not self.lost_outputs:
            raise SealError(
                "a PARTIAL seal must name what was lost. Run 10b's artefact "
                "lists deep_ensemble.joblib, the graph network and the "
                "cloud-computed test figure; a partial record that does not "
                "say what is missing is indistinguishable from a complete one")
        if self.completeness is SealCompleteness.COMPLETE and self.lost_outputs:
            raise SealError(
                "a COMPLETE seal names {} lost output(s). Declare it PARTIAL."
                .format(len(self.lost_outputs)))
        if len(set(self.lost_outputs)) != len(self.lost_outputs):
            raise SealError("duplicate entry in lost_outputs")
        object.__setattr__(self, "artifact_sha256", dict(self.artifact_sha256))
        object.__setattr__(self, "_checked", True)

    # ---- what attribution.py asks -----------------------------------------

    def is_evidence_for(self, artifact_digest: str,
                        roster_fingerprint: str) -> bool:
        """Does this seal authorise EVIDENCE for these exact bytes and roster?

        `api/attribution.py:387`: "Even a linked sealed_evaluation_id is NOT
        enough here. Commit C requires the evaluation to name this artifact
        digest AND this served roster fingerprint."
        """
        return (artifact_digest in set(self.artifact_sha256.values())
                and roster_fingerprint == self.roster_fingerprint)

    # ---- origin, the whole point -------------------------------------------

    def metrics_by_origin(self, origin: MetricOrigin) -> tuple:
        return tuple(m for m in self.metrics if m.origin is origin)

    @property
    def has_mixed_origins(self) -> bool:
        """True when computed and scraped figures sit in one record.

        Not an error -- Run 14 legitimately holds both. The point is that the
        record can now SAY so, where the flat mapping could not.
        """
        return len({m.origin for m in self.metrics}) > 1

    # ---- serialization ------------------------------------------------------

    @property
    def payload(self) -> dict:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "seal_id": self.seal_id,
            "lineage": self.lineage.to_dict(),
            "protocol": self.protocol.to_dict(),
            "metrics": [m.as_record() for m in self.metrics],
            "artifact_sha256": dict(self.artifact_sha256),
            "roster_fingerprint": self.roster_fingerprint,
            "completeness": self.completeness.value,
            "lost_outputs": list(self.lost_outputs),
            "findings": list(self.findings),
        }

    def render(self) -> bytes:
        """Deterministic AND diffable, and authored, so it ends with a newline."""
        return (json.dumps(self.payload, indent=2, sort_keys=True,
                           ensure_ascii=True) + "\n").encode("utf-8")


def read_artifact_json(path) -> dict:
    """Read a committed artefact.

    MEASURED 2026-08-08 and again 2026-08-24: of nineteen tracked files under
    `outputs/`, EXACTLY ONE begins with a byte-order mark --
    `outputs/run14/reproducibility_manifest.json`, written by PowerShell. Python's
    `json.loads` refuses it outright.

    So `utf-8-sig` is not a workaround. It is the ONLY encoding that reads all of
    them, since it handles marked and unmarked files alike. A loader using plain
    `utf-8` reads the others and crashes on the one artefact this project can
    actually seal.
    """
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def evidence_from_seal(seal: SealedEvaluation) -> EvaluationEvidence:
    """Project a seal down onto the existing evidence type.

    LOSSY BY CONSTRUCTION, and that is the point: `EvaluationEvidence.metrics`
    is a flat `Mapping[str, float]` and cannot carry origin. This function exists
    so the loss is explicit and one-directional -- a seal can always be reduced
    to evidence; evidence can never be promoted to a seal.
    """
    return EvaluationEvidence(
        protocol=seal.protocol,
        metrics={m.name: m.value for m in seal.metrics},
    )
