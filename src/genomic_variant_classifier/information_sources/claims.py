"""Stage 1D-1b.4: the claims authority (SHADOW MODE: nothing in production imports this module).

Owner ruling 68ca322f (2026-11-10: claims and evidence interfaces in 1D-1b.4; LABEL-ACCESS-DISCOVERY-SCOPE-1) and the
owner's reference packages 7b277d7c ("EvidenceScope") and 7ba10b3b ("TaskScoped"), audited on 2026-11-13
(AUDIT_1D1b4_REFERENCES_2026-11-13.md). What a result may be published as is decided here from pinned, scoped evidence --
never from a caller's assertion.

* A claim has a KIND (label reproduction, independent clinical evidence, molecular effect, mechanism) and a TIER
  (exploratory, confirmatory, external); each pair requires a fixed set of evidence kinds (``required_evidence``).
* Each evidence kind has a COMPLETE scope: every experiment dimension its proposition depends on, and no more
  (``EVIDENCE_SCOPE``). A split fact does not depend on the model; upstream independence and the statistical result DO
  depend on the evaluation population and split, so evidence for population A cannot certify population B (probes T2 and
  E1: both references let it).
* ``Experiment``: a dimension the experiment does not have is ``None``, and a requirement whose scope needs it is
  NOT_EVALUATED -- never satisfied by a placeholder string (probe E2).
* Evidence is trusted only through a ``TrustedPin`` keyed by (kind, content SHA-256), issued by an independently governed
  registry, with the same issuer and exactly the scope asked for. A ``Rule`` re-derives the status from the payload AND
  receives the scope, so it can check that the payload describes exactly that scope (probe E5).
* Eligibility (are the required facts supported?) and the scientific finding (the pinned statistical result) are separate
  axes; ``publish_supported`` needs both. Every assessment carries the policy digest and its own digest.
* Statistics: ``within_unit_auc`` (within split component AUROC -- area under the receiver operating characteristic curve --
  with both estimands, unit-macro and pair-weighted, and the excluded one-class units) and ``paired_unit_bootstrap``.

In-process objects are not a security boundary: pins and rules must come from a registry the claimant cannot write.
numpy is required; scipy for ``within_unit_auc``.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from genomic_variant_classifier.information_sources.census import HEX64
from genomic_variant_classifier.information_sources.population import digest

logger = logging.getLogger(__name__)


class ClaimKind(str, Enum):
    LABEL_REPRODUCTION = "label_reproduction"
    INDEPENDENT_CLINICAL_EVIDENCE = "independent_clinical_evidence"
    MOLECULAR_EFFECT = "molecular_effect"
    MECHANISM = "mechanism"


class Tier(str, Enum):
    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"
    EXTERNAL = "external"


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    NOT_EVALUATED = "not_evaluated"


class Finding(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"
    NOT_EVALUATED = "not_evaluated"


DIMENSIONS: tuple[str, ...] = ("experiment_lock", "fit", "model", "population", "split", "identity_policy",
                               "source_snapshot", "label_snapshot", "assay_snapshot", "endpoint", "regime")

#: Each kind's scope: every dimension its proposition depends on, and no more.
EVIDENCE_SCOPE: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "split": ("population", "split", "identity_policy"),
    "source": ("population", "regime", "source_snapshot"),
    "label_access": ("fit", "population", "split"),
    "model_input": ("fit", "model", "population", "split", "regime"),
    "evaluation": ("fit", "model", "population", "split", "endpoint", "label_snapshot"),
    "upstream": ("model", "fit", "population", "split", "endpoint", "regime"),
    "criterion": ("model", "fit", "population", "split", "endpoint", "label_snapshot", "regime"),
    "assay": ("model", "fit", "population", "split", "endpoint", "assay_snapshot"),
    "negative_control": ("fit", "model", "population", "split", "endpoint"),
    "preregistration": ("experiment_lock", "fit", "model", "population", "split", "endpoint"),
    "holdout_release": ("experiment_lock", "population", "split", "endpoint"),
    "external_population": ("population", "split", "endpoint", "label_snapshot"),
    "statistical_result": ("experiment_lock", "fit", "model", "population", "split", "endpoint", "label_snapshot"),
})
#: The evidence kind whose status is a scientific Finding; every other kind's status is an EvidenceStatus.
RESULT_KIND = "statistical_result"

BASE_REQUIREMENTS: Mapping[ClaimKind, frozenset[str]] = MappingProxyType({
    ClaimKind.LABEL_REPRODUCTION: frozenset({"split", "source", "label_access", "model_input", "evaluation"}),
    ClaimKind.INDEPENDENT_CLINICAL_EVIDENCE: frozenset({"split", "source", "label_access", "model_input", "evaluation",
                                                        "upstream", "criterion"}),
    ClaimKind.MOLECULAR_EFFECT: frozenset({"split", "source", "label_access", "model_input", "evaluation", "upstream",
                                           "assay"}),
    ClaimKind.MECHANISM: frozenset({"split", "source", "label_access", "model_input", "evaluation", "upstream", "assay",
                                    "negative_control"}),
})
TIER_REQUIREMENTS: Mapping[Tier, frozenset[str]] = MappingProxyType({
    Tier.EXPLORATORY: frozenset(),
    Tier.CONFIRMATORY: frozenset({"preregistration", "holdout_release"}),
    Tier.EXTERNAL: frozenset({"preregistration", "holdout_release", "external_population"}),
})


def required_evidence(claim: ClaimKind, tier: Tier) -> frozenset[str]:
    if not isinstance(claim, ClaimKind) or not isinstance(tier, Tier):
        raise TypeError("claim and tier must be a ClaimKind and a Tier")
    return BASE_REQUIREMENTS[claim] | TIER_REQUIREMENTS[tier]


POLICY_DIGEST = digest("claims-policy", {
    "dimensions": list(DIMENSIONS), "scope": {k: list(v) for k, v in EVIDENCE_SCOPE.items()}, "result_kind": RESULT_KIND,
    "requirements": {k.value: sorted(v) for k, v in BASE_REQUIREMENTS.items()},
    "tiers": {k.value: sorted(v) for k, v in TIER_REQUIREMENTS.items()}})


def _nonempty_str(value: object, what: str) -> None:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{what} must be a nonempty str, got {type(value).__name__}")


def _complete_scope(kind: str, scope: Mapping[str, str], what: str) -> MappingProxyType:
    if kind not in EVIDENCE_SCOPE:
        raise ValueError(f"unregistered evidence kind {kind!r}")
    snap = dict(scope)
    if set(snap) != set(EVIDENCE_SCOPE[kind]):
        raise ValueError(f"{what} scope for {kind} must name exactly {list(EVIDENCE_SCOPE[kind])}, got {sorted(snap)}")
    if any(type(v) is not str or not v.strip() for v in snap.values()):
        raise ValueError(f"{what} scope values are nonempty str")
    return MappingProxyType(snap)


@dataclass(frozen=True)
class Experiment:
    """The experiment a claim is about. A dimension it does not have is None (for example the assay snapshot of a
    label-reproduction claim); a requirement whose scope needs an undefined dimension is NOT_EVALUATED."""

    experiment_lock: str | None = None
    fit: str | None = None
    model: str | None = None
    population: str | None = None
    split: str | None = None
    identity_policy: str | None = None
    source_snapshot: str | None = None
    label_snapshot: str | None = None
    assay_snapshot: str | None = None
    endpoint: str | None = None
    regime: str | None = None

    def __post_init__(self) -> None:
        for k in DIMENSIONS:
            v = getattr(self, k)
            if v is not None and (type(v) is not str or not v.strip()):
                raise ValueError(f"dimension {k} must be None or a nonempty str")

    def as_dict(self) -> dict[str, str | None]:
        return {k: getattr(self, k) for k in DIMENSIONS}

    def scope_for(self, kind: str) -> dict[str, str] | None:
        if kind not in EVIDENCE_SCOPE:
            raise ValueError(f"unregistered evidence kind {kind!r}")
        s = {k: getattr(self, k) for k in EVIDENCE_SCOPE[kind]}
        return None if any(v is None for v in s.values()) else s


@dataclass(frozen=True)
class Evidence:
    """One artifact: its kind, the complete scope it claims, its bytes and its issuer."""

    kind: str
    scope: Mapping[str, str]
    payload: bytes
    issuer: str

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes:
            raise TypeError("payload must be bytes")
        _nonempty_str(self.issuer, "issuer")
        object.__setattr__(self, "scope", _complete_scope(self.kind, self.scope, "evidence"))

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


@dataclass(frozen=True)
class TrustedPin:
    """Issued by an independently governed registry, never by the caller asking for a claim."""

    kind: str
    issuer: str
    content_sha256: str
    scope: Mapping[str, str]

    def __post_init__(self) -> None:
        _nonempty_str(self.issuer, "issuer")
        if type(self.content_sha256) is not str or not HEX64.match(self.content_sha256):
            raise ValueError("content_sha256 is a lower-case SHA-256")
        object.__setattr__(self, "scope", _complete_scope(self.kind, self.scope, "pin"))


@dataclass(frozen=True)
class Rule:
    """``evaluate(payload, scope)`` re-derives the status from the payload AND checks that the payload describes exactly
    ``scope``; it returns an ``EvidenceStatus`` (a ``Finding`` for the statistical result). The version id names the
    released rule."""

    evaluate: Callable[[bytes, Mapping[str, str]], Enum]
    version_id: str

    def __post_init__(self) -> None:
        if not callable(self.evaluate):
            raise TypeError("evaluate must be callable")
        _nonempty_str(self.version_id, "version_id")


class ClaimsAuthority:
    """Assesses a claim from pinned evidence; the pins and rules are fixed when the authority is made."""

    __slots__ = ("_pins", "_rules")

    def __init__(self, *, pins: Mapping[tuple[str, str], TrustedPin], rules: Mapping[str, Rule]):
        pins = dict(pins)
        for key, pin in pins.items():
            if not isinstance(pin, TrustedPin) or key != (pin.kind, pin.content_sha256):
                raise ValueError(f"a pin is keyed by its own (kind, content_sha256): {key!r}")
        rules = dict(rules)
        for kind, rule in rules.items():
            if kind not in EVIDENCE_SCOPE or not isinstance(rule, Rule):
                raise ValueError(f"rules map registered evidence kinds to Rule: {kind!r}")
        object.__setattr__(self, "_pins", MappingProxyType(pins))
        object.__setattr__(self, "_rules", MappingProxyType(rules))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a claims authority is fixed when it is made: its pins and rules cannot be replaced")

    def _fact(self, kind: str, experiment: Experiment, item: Evidence | None) -> dict[str, Any]:
        want = Finding if kind == RESULT_KIND else EvidenceStatus
        scope = experiment.scope_for(kind)
        if scope is None:
            missing = [k for k in EVIDENCE_SCOPE[kind] if getattr(experiment, k) is None]
            return {"status": want.NOT_EVALUATED.value, "reason": f"undefined dimension(s) {missing}", "digest": None}
        rule = self._rules.get(kind)
        if item is None or rule is None:
            return {"status": want.NOT_EVALUATED.value, "reason": "no evidence" if item is None else "no rule",
                    "digest": None}
        if not isinstance(item, Evidence):
            raise TypeError(f"the {kind} artifact is not Evidence")
        pin = self._pins.get((kind, item.content_sha256))
        if pin is None or item.kind != kind or pin.issuer != item.issuer or dict(item.scope) != scope \
                or dict(pin.scope) != scope:
            # "unknown" for both kinds: the evidence is untrusted, so no status -- and for the statistical result no
            # finding -- is derived from it (the finding then stays NOT_EVALUATED).
            return {"status": EvidenceStatus.UNKNOWN.value,
                    "reason": "untrusted evidence, or a scope other than the experiment's", "digest": None}
        status = rule.evaluate(item.payload, MappingProxyType(dict(scope)))
        if not isinstance(status, want):
            raise TypeError(f"the {kind} rule returned {type(status).__name__}, not {want.__name__}")
        fact = {"kind": kind, "status": status.value, "scope": scope, "evidence_sha256": item.content_sha256,
                "issuer": item.issuer, "rule_version": rule.version_id}
        return {"status": status.value, "reason": None, "digest": digest("fact", fact)}

    def assess(self, claim: ClaimKind, tier: Tier, experiment: Experiment,
               artifacts: Mapping[str, Evidence]) -> dict[str, Any]:
        if not isinstance(experiment, Experiment):
            raise TypeError("experiment must be an Experiment")
        unknown_kinds = sorted(set(artifacts) - set(EVIDENCE_SCOPE))
        if unknown_kinds:
            raise ValueError(f"unregistered evidence kinds {unknown_kinds}")
        facts = {k: self._fact(k, experiment, artifacts.get(k)) for k in sorted(required_evidence(claim, tier))}
        states = {f["status"] for f in facts.values()}
        eligibility = "blocked" if "violated" in states else "eligible" if states == {"supported"} else "unknown"
        result = self._fact(RESULT_KIND, experiment, artifacts.get(RESULT_KIND))
        finding = result["status"] if result["digest"] else Finding.NOT_EVALUATED.value
        body = {"claim": claim.value, "tier": tier.value, "policy_digest": POLICY_DIGEST,
                "experiment": experiment.as_dict(), "eligibility": eligibility, "scientific_finding": finding,
                "facts": facts, "statistical_result": result,
                "publish_supported": eligibility == "eligible" and finding == Finding.SUPPORTED.value}
        body["assessment_digest"] = digest("assessment", body)
        return body


# ======================================================================================================= statistics
def within_unit_auc(labels: np.ndarray, scores: np.ndarray, unit_ids: Sequence[str]) -> dict[str, Any]:
    """Per unit (a split component): Mann-Whitney U / (n+ n-) with mid-ranks for ties. Two estimands: the unit-macro mean
    and the pair-weighted mean. One-class units are excluded and counted; one unit id per row."""
    from scipy.stats import rankdata

    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    u = list(unit_ids)
    if y.ndim != 1 or s.shape != y.shape or len(u) != len(y):
        raise ValueError("misaligned inputs")
    if y.dtype.kind not in "iu" or not np.isin(y, (0, 1)).all():
        raise TypeError("labels must be an integer array of 0 and 1 (never bool or float)")
    if not np.isfinite(s).all():
        raise ValueError("non-finite score")
    if any(type(x) is not str or not x for x in u):
        raise ValueError("unit ids must be nonempty str")
    idx: dict[str, list[int]] = defaultdict(list)
    for i, x in enumerate(u):
        idx[x].append(i)
    units, aucs, pairs, excluded = [], [], [], 0
    for x in sorted(idx):
        yy, ss = y[idx[x]], s[idx[x]]
        npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
        if not npos or not nneg:
            excluded += 1
            continue
        r = rankdata(ss, method="average")
        units.append(x)
        aucs.append(float((r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)))
        pairs.append(npos * nneg)
    return {"unit_macro": float(np.mean(aucs)) if aucs else None,
            "pair_weighted": float(np.average(aucs, weights=pairs)) if aucs else None,
            "mixed_class_units": len(aucs), "excluded_one_class_units": excluded, "eligible_pairs": int(sum(pairs)),
            "per_unit": MappingProxyType(dict(zip(units, aucs)))}


def paired_unit_bootstrap(model_auc: np.ndarray, baseline_auc: np.ndarray, *, replicates: int = 2000, seed: int = 2026,
                          alpha: float = 0.05) -> dict[str, Any]:
    """Percentile interval for the unit-macro mean of per-unit AUROC differences, resampling units (the same mixed-class
    units for both scorers)."""
    m, b = np.asarray(model_auc, dtype=float), np.asarray(baseline_auc, dtype=float)
    if m.shape != b.shape or m.ndim != 1 or len(m) < 2 or not (np.isfinite(m).all() and np.isfinite(b).all()):
        raise ValueError("need two aligned finite arrays of at least two units")
    if type(replicates) is not int or replicates < 100 or type(seed) is not int or type(alpha) is not float \
            or not 0 < alpha < 1:
        raise ValueError("replicates is an int >= 100, seed an int, alpha a float in (0, 1)")
    d = m - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(replicates, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"estimand": "unit-macro AUROC difference over mixed-class units", "estimate": float(d.mean()),
            "interval": (float(lo), float(hi)), "units": len(d), "replicates": replicates, "seed": seed, "alpha": alpha}


__all__ = ["BASE_REQUIREMENTS", "ClaimKind", "ClaimsAuthority", "DIMENSIONS", "EVIDENCE_SCOPE", "Evidence",
           "EvidenceStatus", "Experiment", "Finding", "POLICY_DIGEST", "RESULT_KIND", "Rule", "TIER_REQUIREMENTS", "Tier",
           "TrustedPin", "paired_unit_bootstrap", "required_evidence", "within_unit_auc"]
