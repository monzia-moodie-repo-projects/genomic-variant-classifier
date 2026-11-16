"""Stage 1D-1c.1C: the information regime, the experiment declaration and the fit context every ensemble fit requires.

Owner ruling aa999b96 (2026-11-16): decision B and revised ruling section 8; AUDIT_OF_RULING_1D1c1C_2026-11-16 (R5).

    Valid experiment specification  !=  authorized scientific execution.

* ``InformationRegime`` -- the regime VOCABULARY, built from the admission authority (``admission.REGIMES``); a test
  proves the two are the same set. Which features a regime admits stays in the census and admission report; nothing here
  restates it.
* ``ExperimentIntent`` -- a DECLARATION: regime, arm (``allele_frequency.Arm``), projection protocol, semantic epoch and
  the pinned identities of the source policy, roster, population and split. Built from a manifest with exactly its keys
  (no default, no extra key). It is not an authorization and cannot become one by being well formed.
* ``FitContext`` (defined in fit_context.py, re-exported here) -- what ``VariantEnsemble.fit(..., authorization=...)``
  requires: ``historical_replay(reason)`` or ``engineering_qualification(semantics, projection, reason)``.
* ``scientific_fit_context(intent)`` -- REFUSED at this stage: the readiness gates below have no certifying authority
  yet, so no scientific fit is reachable (ruling section 12).

In-process Python objects are not a security boundary; the constructors make an unauthorized fit an explicit, named act,
not an accident of a missing argument.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from genomic_variant_classifier.information_sources.admission import REGIMES
from genomic_variant_classifier.information_sources.allele_frequency import Arm
from genomic_variant_classifier.models.feature_semantics import FeatureSemantics
from genomic_variant_classifier.models.fit_context import ExecutionProtocol, FitContext
from genomic_variant_classifier.models.member import ProjectionProtocol
from genomic_variant_classifier.prediction_contract import canonical_digest

logger = logging.getLogger(__name__)

__all__ = ["ExecutionProtocol", "ExperimentIntent", "FitContext", "InformationRegime", "READINESS_GATES",
           "TrainingReadinessError", "add_execution_arguments", "fit_context_from_arguments",
           "require_scientific_readiness", "scientific_fit_context"]

_HEX64 = re.compile(r"[0-9a-f]{64}")

InformationRegime = Enum("InformationRegime", {name.upper(): name for name in sorted(REGIMES)}, type=str)
InformationRegime.__doc__ = "The information regimes of admission.REGIMES (one vocabulary, one authority)."
InformationRegime.__module__ = __name__


#: Every prerequisite of an authorized scientific fit (ruling aa999b96 section 12), with the stage that must first build
#: its CERTIFYING AUTHORITY. A caller-supplied boolean is never accepted for any of them.
READINESS_GATES: Mapping[str, str] = MappingProxyType({
    "source_bound": "1D-1c.1C set av: the qualified AF preparation path and its content-addressed reference receipt",
    "admission_enforced": "1D-1c.4: admission enforced at the fit boundary from the verified census report",
    "feature_semantics_certified": "1D-1c.2-1c.5: every legacy feature default retired or declared",
    "component_split_certified": "1D-1d: canonical biological-component isolation (grouping (a), every HGNC locus)",
    "train_serve_certified": "1D-1c.5: train-serve certification of the complete fitted predictor",
})


class TrainingReadinessError(PermissionError):
    """A scientific fit was requested while its prerequisites have no certifying authority."""


def require_scientific_readiness(regime: Any) -> None:
    """Refuse a scientific fit, naming every incomplete gate. At stage 1D-1c.1C no gate can be certified."""
    if not isinstance(regime, InformationRegime):
        raise ValueError("an explicit InformationRegime is required; none is assumed")
    raise TrainingReadinessError(
        f"scientific training under the {regime.value} regime is refused: no certifying authority exists yet for "
        + "; ".join(f"{gate} ({stage})" for gate, stage in READINESS_GATES.items())
        + ". Historical reproduction uses FitContext.historical_replay(reason) and is never a scientific artifact.")


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or not _HEX64.fullmatch(value):
        raise ValueError(f"{name} must be a lower-case SHA-256 digest")
    return value


@dataclass(frozen=True)
class ExperimentIntent:
    """A declared experiment (revised ruling section 8). Not an authorization."""
    experiment_id: str
    regime: InformationRegime
    arm: Arm
    projection: ProjectionProtocol
    feature_semantics: FeatureSemantics
    source_policy_sha256: str
    feature_roster_sha256: str
    population_sha256: str
    split_sha256: str

    def __post_init__(self) -> None:
        if type(self.experiment_id) is not str or not self.experiment_id.strip():
            raise ValueError("experiment_id is a non-empty str")
        for name, kind in (("regime", InformationRegime), ("arm", Arm), ("projection", ProjectionProtocol),
                           ("feature_semantics", FeatureSemantics)):
            if not isinstance(getattr(self, name), kind):
                raise ValueError(f"{name} must be declared explicitly as a {kind.__name__}")
        for name in ("source_policy_sha256", "feature_roster_sha256", "population_sha256", "split_sha256"):
            _sha(getattr(self, name), name)

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "ExperimentIntent":
        """Exactly the declared keys: a missing key (the regime above all) or an extra key is refused, never defaulted."""
        required = set(cls.__dataclass_fields__)
        if not isinstance(manifest, Mapping) or set(manifest) != required:
            got = set(manifest) if isinstance(manifest, Mapping) else set()
            raise ValueError(f"the manifest must hold exactly {sorted(required)}; missing {sorted(required - got)}, "
                             f"extra {sorted(got - required)}")
        values = dict(manifest)
        for name, kind in (("regime", InformationRegime), ("arm", Arm), ("projection", ProjectionProtocol),
                           ("feature_semantics", FeatureSemantics)):
            if type(values[name]) is not str:
                raise ValueError(f"{name} must be a string in the manifest")
            try:
                values[name] = kind(values[name])
            except ValueError as exc:
                raise ValueError(f"unknown {name} {values[name]!r}") from exc
        return cls(**values)

    @property
    def identity(self) -> str:
        return canonical_digest("gvc.experiment-intent.v1", {
            k: (v.value if isinstance(v, Enum) else v) for k, v in sorted(vars(self).items())})


def scientific_fit_context(intent: ExperimentIntent) -> FitContext:
    """The FitContext of a declared scientific experiment: REFUSED at this stage, naming every readiness gate without a
    certifying authority (ruling aa999b96 section 12). The stage that certifies them returns the context here."""
    if type(intent) is not ExperimentIntent:
        raise TypeError("a scientific fit needs a declared ExperimentIntent")
    require_scientific_readiness(intent.regime)
    raise AssertionError("unreachable: readiness refused above")   # pragma: no cover


# ---------------------------------------------------------------------------------------------------- launcher flags
def add_execution_arguments(parser: Any) -> None:
    """The two flags every training launcher carries (ruling aa999b96 section 11.3). Parsed without a default; the
    launcher calls ``fit_context_from_arguments`` immediately after parsing, so a missing declaration is refused before
    any data is read (parse-only callers that test other flags keep working)."""
    parser.add_argument(
        "--execution-protocol", choices=("historical_replay", "scientific"), default=None,
        help="REQUIRED. historical_replay: reproduce the epoch-1 legacy pipeline exactly (never a scientific artifact). "
             "scientific: refused at stage 1D-1c.1C, naming every readiness gate without a certifying authority.")
    parser.add_argument(
        "--information-regime", choices=tuple(r.value for r in InformationRegime), default=None,
        help="REQUIRED for --execution-protocol scientific (admission.REGIMES); refused for historical_replay, which "
             "has no information regime.")


def fit_context_from_arguments(args: Any, launcher: str) -> FitContext:
    """The FitContext a launcher's parsed flags declare, or SystemExit naming what is missing or refused."""
    protocol = getattr(args, "execution_protocol", None)
    regime = getattr(args, "information_regime", None)
    if protocol is None:
        raise SystemExit(f"{launcher}: --execution-protocol is required (historical_replay or scientific); none is "
                         "assumed (owner ruling aa999b96 decision B)")
    if protocol == "scientific":
        if regime is None:
            raise SystemExit(f"{launcher}: --information-regime is required for a scientific run; none is assumed")
        try:
            require_scientific_readiness(InformationRegime(regime))
        except TrainingReadinessError as exc:
            raise SystemExit(f"{launcher}: {exc}") from exc
    if protocol != "historical_replay":
        raise SystemExit(f"{launcher}: unknown execution protocol {protocol!r}")
    if regime is not None:
        raise SystemExit(f"{launcher}: historical replay has no information regime -- it reproduces the epoch-1 legacy "
                         "feature set exactly, which no regime admits; remove --information-regime")
    return FitContext.historical_replay(f"{launcher}: historical replay of the epoch-1 legacy pipeline")
