"""Stage 1D-1c.1C: the fit context every ``VariantEnsemble.fit`` requires (owner ruling aa999b96 decision B).

``FitContext`` records what one ensemble fit is; the fitted ensemble keeps the record and serving reads it. It is
constructed only through its named constructors (direct construction is refused):

  ``historical_replay(reason)``                   epoch 1, unprojected: reproduction of historical behaviour only;
  ``engineering_qualification(semantics, projection, reason)``
                                                  synthetic qualification of the code; never a scientific artifact.

There is NO scientific constructor at this stage, and a context of protocol ``scientific`` cannot be constructed at
all: the readiness gates (``execution.READINESS_GATES``) have no certifying authority yet (ruling section 12), and
``execution.scientific_fit_context(intent)`` refuses, naming every one. The stage that certifies them adds the
constructor here, with the certificate it verifies.

Why a separate module (measured, stage 1D-1c.1C): ``VariantEnsemble`` is in the inference service's static import
closure, and ``execution.py`` imports the admission and allele-frequency authorities, which reach the data layer; the
API image would need 48 more modules (tests/unit/test_docker_image_covers_the_api.py). This module imports only the
models-layer enums and the prediction contract. ``execution.py`` re-exports both names.

In-process Python objects are not a security boundary; the constructors make an unauthorized fit an explicit, named act,
not an accident of a missing argument.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from genomic_variant_classifier.models.feature_semantics import FeatureSemantics
from genomic_variant_classifier.models.member import ProjectionProtocol
from genomic_variant_classifier.prediction_contract import canonical_digest

logger = logging.getLogger(__name__)

__all__ = ["ExecutionProtocol", "FitContext"]


class ExecutionProtocol(str, Enum):
    HISTORICAL_REPLAY = "historical_replay"
    ENGINEERING_QUALIFICATION = "engineering_qualification"
    SCIENTIFIC = "scientific"


_TOKEN = object()


@dataclass(frozen=True)
class FitContext:
    """What one ensemble fit is, recorded on the fitted object. Construct it through a named constructor only."""
    protocol: ExecutionProtocol
    semantics: FeatureSemantics
    projection: ProjectionProtocol
    reason: str
    regime: Any = None            # an execution.InformationRegime, on a scientific context only (none can exist yet)
    intent_sha256: str | None = None
    _token: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _TOKEN:
            raise TypeError("construct a FitContext through historical_replay() or engineering_qualification(); "
                            "direct construction is refused")
        if not isinstance(self.protocol, ExecutionProtocol) or not isinstance(self.semantics, FeatureSemantics) \
                or not isinstance(self.projection, ProjectionProtocol):
            raise TypeError("protocol, semantics and projection must be their enums")
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("a fit context states its reason")
        legacy = self.projection is ProjectionProtocol.LEGACY_UNPROJECTED
        if legacy != (self.semantics is FeatureSemantics.V1_LEGACY):
            raise ValueError("the legacy unprojected protocol is exactly epoch 1; every projected protocol needs epoch 2")
        if self.protocol is ExecutionProtocol.HISTORICAL_REPLAY and not legacy:
            raise ValueError("historical replay is epoch 1, unprojected")
        if self.protocol is ExecutionProtocol.SCIENTIFIC:
            raise PermissionError("no scientific fit context can be constructed at stage 1D-1c.1C: its readiness gates "
                                  "have no certifying authority (execution.scientific_fit_context names them)")
        if self.regime is not None or self.intent_sha256 is not None:
            raise ValueError("only a scientific context carries a regime and an intent")

    @classmethod
    def historical_replay(cls, reason: str) -> "FitContext":
        return cls(ExecutionProtocol.HISTORICAL_REPLAY, FeatureSemantics.V1_LEGACY,
                   ProjectionProtocol.LEGACY_UNPROJECTED, reason, _token=_TOKEN)

    @classmethod
    def engineering_qualification(cls, semantics: FeatureSemantics, projection: ProjectionProtocol,
                                  reason: str) -> "FitContext":
        return cls(ExecutionProtocol.ENGINEERING_QUALIFICATION, semantics, projection, reason, _token=_TOKEN)

    @property
    def scientific_artifact(self) -> bool:
        return self.protocol is ExecutionProtocol.SCIENTIFIC

    def as_record(self) -> dict:
        record = {"protocol": self.protocol.value, "semantics": self.semantics.value,
                  "projection": self.projection.value, "reason": self.reason,
                  "regime": None if self.regime is None else self.regime.value, "intent_sha256": self.intent_sha256}
        record["digest"] = canonical_digest("gvc.fit-context.v1", dict(record))
        return record
