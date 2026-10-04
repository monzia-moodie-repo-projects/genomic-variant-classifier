"""The analysis contract: what an experiment WILL do, fixed before any outcome is seen (owner ruling 2026-10-03, L226-248).

PLAN, NOT RESULT
================
`evaluation/sealed_evaluation.py` seals what an experiment ESTABLISHED (metrics with their origin, artifact digests, a
roster fingerprint). This module seals what an experiment WILL DO -- a pre-registration. They sit at opposite ends of an
experiment, and the word "sealed" must never let one be read as the other: a SealedContract is not evidence of any result.

TWO STATES
==========
DraftContract  -- intended sources and EXPLICITLY UNRESOLVED choices (each named; nothing silently missing).
SealedContract -- every identity actual and every decision needed to interpret results present. `DraftContract.seal()`
                  refuses while anything is unresolved. Its identity is the SHA-256 of its canonical rendering, so the
                  identity IS the content: an amendment cannot keep an old identity, and a changed element cannot hide.

Immutability is structural (tuples, frozensets, frozen dataclasses with exact types), not a frozen wrapper around
mutable dictionaries. An AMENDMENT is a new contract naming the superseded identity and the reason; it never overwrites
the meaning of results obtained under the earlier one.

INDEPENDENCE (ruling 2026-10-03b)
=================================
Every overlap is documented_disjoint, documented_overlap or unresolved -- "no matching cohort name found" is NOT
documented_disjoint. Unknown overlap does not block a contract; it blocks an INDEPENDENCE CLAIM: a contract claiming
independence seals only if every recorded relationship is documented_disjoint. Participant overlap and evidence reuse in
ascertainment are recorded as different relationship kinds.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, fields
from enum import Enum

from genomic_variant_classifier.inference.exact_confirmation import InferenceError

logger = logging.getLogger(__name__)

__all__ = ["Overlap", "Relationship", "Target", "InputSource", "Families", "Execution", "Evaluation", "Independence",
           "Amendment", "DraftContract", "SealedContract"]

SCHEMA = "gvc.analysis-contract"
SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise InferenceError(code, detail)


def _text(value, code: str) -> None:
    _require(type(value) is str and bool(value.strip()), code)


def _texts(values, code: str, *, nonempty: bool = True) -> None:
    _require(type(values) is tuple and (len(values) > 0 or not nonempty), code)
    for v in values:
        _text(v, code)
    _require(len(set(values)) == len(values), code, "duplicates")


class Overlap(str, Enum):
    DOCUMENTED_DISJOINT = "documented_disjoint"
    DOCUMENTED_OVERLAP = "documented_overlap"
    UNRESOLVED = "unresolved"


class Relationship(str, Enum):
    PARTICIPANT_OVERLAP = "participant_overlap"
    EVIDENCE_REUSE_IN_ASCERTAINMENT = "evidence_reuse_in_ascertainment"


@dataclass(frozen=True)
class Target:
    phenotype: str
    population: str
    hypothesis: str
    intended_claim: str

    def __post_init__(self) -> None:
        for f in fields(self):
            _text(getattr(self, f.name), "target_" + f.name)


@dataclass(frozen=True)
class InputSource:
    """An exact input: its identity is the digest of the bytes at their final location (a RECORDED identity, not proof
    of upstream authenticity)."""

    name: str
    release: str
    sha256: str
    size_bytes: int
    definition: str          # phenotype / mask / statistic definition, as the source states it
    numeric_meaning: str     # e.g. "decimal text p-values, unadjusted"

    def __post_init__(self) -> None:
        for name in ("name", "release", "definition", "numeric_meaning"):
            _text(getattr(self, name), "input_" + name)
        _require(type(self.sha256) is str and bool(_SHA256.fullmatch(self.sha256)), "input_sha256")
        _require(type(self.size_bytes) is int and self.size_bytes > 0, "input_size")


@dataclass(frozen=True)
class Families:
    genes: tuple                 # the complete contracted gene family, in a fixed order
    exposures: tuple             # ((gene, (exposure, ...)), ...) -- every gene's contracted exposure family
    missingness_policy: str

    def __post_init__(self) -> None:
        _texts(self.genes, "gene_family")
        _require(type(self.exposures) is tuple, "exposure_families")
        keys = [g for g, _ in self.exposures]
        _require(tuple(keys) == self.genes, "exposure_families", "one entry per gene, in the gene order")
        for _, exposures in self.exposures:
            _texts(exposures, "exposure_families")
        _text(self.missingness_policy, "missingness_policy")


@dataclass(frozen=True)
class Execution:
    code_commit: str
    package_path: str
    environment: str
    backend_policy: str

    def __post_init__(self) -> None:
        _require(type(self.code_commit) is str and bool(_COMMIT.fullmatch(self.code_commit)), "code_commit")
        for name in ("package_path", "environment", "backend_policy"):
            _text(getattr(self, name), "execution_" + name)


@dataclass(frozen=True)
class Evaluation:
    k: int
    primary_method: str
    primary_comparator: str
    ranking_and_tie_rule: str
    reference_id: str
    assay_rule_id: str
    secondary_analyses: tuple

    def __post_init__(self) -> None:
        _require(type(self.k) is int and self.k >= 1, "k_invalid")
        for name in ("primary_method", "primary_comparator", "ranking_and_tie_rule", "reference_id", "assay_rule_id"):
            _text(getattr(self, name), "evaluation_" + name)
        _require(self.primary_method != self.primary_comparator, "evaluation_comparator", "must differ from the method")
        _texts(self.secondary_analyses, "secondary_analyses", nonempty=False)


@dataclass(frozen=True)
class Independence:
    left: str
    right: str
    relationship: Relationship
    state: Overlap
    basis: str

    def __post_init__(self) -> None:
        _text(self.left, "independence_left")
        _text(self.right, "independence_right")
        _require(self.left != self.right, "independence_pair")
        _require(type(self.relationship) is Relationship, "independence_relationship")
        _require(type(self.state) is Overlap, "independence_state")
        _text(self.basis, "independence_basis")


@dataclass(frozen=True)
class Amendment:
    supersedes: str
    reason: str

    def __post_init__(self) -> None:
        _require(type(self.supersedes) is str and bool(_SHA256.fullmatch(self.supersedes)), "amendment_supersedes")
        _text(self.reason, "amendment_reason")


def _canonical(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {f.name: _canonical(getattr(value, f.name)) for f in fields(value)}
    if type(value) is tuple:
        return [_canonical(v) for v in value]
    if value is None or type(value) in (str, int, bool):
        return value
    raise InferenceError("contract_value_type", type(value).__name__)


@dataclass(frozen=True)
class SealedContract:
    """Created only by DraftContract.seal(). Its identity is the digest of its canonical rendering."""

    analysis_id: str
    target: Target
    inputs: tuple
    families: Families
    transformations: tuple
    execution: Execution
    evaluation: Evaluation
    independence: tuple
    independence_claimed: bool
    amendment: Amendment | None

    def __post_init__(self) -> None:
        _text(self.analysis_id, "analysis_id")
        _require(type(self.target) is Target, "target")
        _require(type(self.inputs) is tuple and len(self.inputs) > 0 and all(type(x) is InputSource for x in self.inputs),
                 "inputs")
        _require(len({x.name for x in self.inputs}) == len(self.inputs), "inputs", "duplicate input name")
        _require(type(self.families) is Families, "families")
        _texts(self.transformations, "transformations", nonempty=False)
        _require(type(self.execution) is Execution, "execution")
        _require(type(self.evaluation) is Evaluation, "evaluation")
        _require(type(self.independence) is tuple and all(type(x) is Independence for x in self.independence),
                 "independence")
        _require(type(self.independence_claimed) is bool, "independence_claimed")
        if self.independence_claimed:
            _require(len(self.independence) > 0, "independence_claim_unsupported", "no relationship recorded")
            undocumented = [(x.left, x.right) for x in self.independence if x.state is not Overlap.DOCUMENTED_DISJOINT]
            _require(not undocumented, "independence_claim_unsupported", repr(undocumented))
        _require(self.amendment is None or type(self.amendment) is Amendment, "amendment")

    def render(self) -> bytes:
        doc = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, **_canonical(self)}
        return (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")

    @property
    def contract_id(self) -> str:
        return hashlib.sha256(self.render()).hexdigest()


@dataclass(frozen=True)
class DraftContract:
    """A plan with its unresolved choices NAMED. Sealing refuses while any remain."""

    analysis_id: str
    unresolved: tuple                 # names of decisions not yet made, e.g. ("reference inclusion rule",)
    target: Target | None = None
    inputs: tuple = ()
    families: Families | None = None
    transformations: tuple = ()
    execution: Execution | None = None
    evaluation: Evaluation | None = None
    independence: tuple = ()
    independence_claimed: bool = False
    amendment: Amendment | None = None

    def __post_init__(self) -> None:
        _text(self.analysis_id, "analysis_id")
        _texts(self.unresolved, "unresolved", nonempty=False)

    def seal(self) -> SealedContract:
        _require(not self.unresolved, "contract_unresolved", repr(self.unresolved))
        for name in ("target", "families", "execution", "evaluation"):
            _require(getattr(self, name) is not None, "contract_incomplete", name)
        return SealedContract(self.analysis_id, self.target, self.inputs, self.families, self.transformations,
                              self.execution, self.evaluation, self.independence, self.independence_claimed,
                              self.amendment)
