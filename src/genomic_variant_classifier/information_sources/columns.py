"""Stage 1D-1b.2: columnar observations reconciled against an explicit population, and derived-feature propagation.

Owner ruling 7f4b69d0 (2026-11-09), section 5: for large data, one Python object per variant-feature cell is the wrong
shape; use numerical arrays with a parallel compact state array, bound to an exact row population, and require for every
feature

    N_observed + N_not_in_source + N_unsupported + N_source_absent + N_failed (+ ...) = N_population,

with any omitted or duplicated row refused. Here:

* ``Population`` is an ORDERED tuple of unique row identities; its SHA-256 binds the order, because a column's values are
  aligned to it by position (a population with the same rows in another order is another population). Since stage
  1D-1c.1 (ruling 482cd291) the identity is version 2 (``POPULATION_IDENTITY_SCHEMA``): canonical JSON of the schema, the
  population identifier and the ordered identities, so no two populations collide (version 1 joined identities with
  newlines and ignored the identifier: COLUMNS-POPULATION-DIGEST-COLLISION-1). ``sha256_v1`` reproduces version 1 only
  to verify records sealed under it, and refuses identities it cannot represent.
* ``ObservationColumn`` holds one float64 value (NaN exactly where not observed) and one uint8 state code per row, plus
  optional detail codes (why a row is in its state, e.g. "gene not modelled" against "substitution not scored") from a
  declared vocabulary. Every rule of observation.py holds for every row, checked vectorised. The arrays are bytes-backed
  read-only copies taken BEFORE validation (``readonly_array``), so what was validated is what is kept and a digest
  cannot go stale (COLUMNS-MUTABLE-ARRAYS-1). A derived column carries its ``lineage``: the derivation identity of its
  rule, implementation code and input columns, bound into its digest (COLUMNS-DERIVATION-LINEAGE-1).
* ``ObservationColumn.from_rows`` refuses rows that do not reconcile with the population (each missing and extra row
  counted), so the reconciliation is enforced, not assumed. ``counts()`` sums to the population by construction.
* ``derive(name, inputs)`` computes a derived column whose states are a DECLARED function of its inputs' states
  (``Propagation``). The rule, its implementation and its input features all come from ``DERIVED_RULES`` by name; the
  caller supplies only the input columns, which must be the declared features in the declared order
  (COLUMNS-CALLER-SUPPLIED-DERIVATION-1). ``DERIVED_RULES`` declares the rule for every derived feature the census
  declares (a test reconciles the two), and every implementation equals the production formula (a test shows it).

numpy is required; observation.py stays standard-library only.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from genomic_variant_classifier.information_sources.census import HEX64, canonical_bytes, sha256_bytes, symbol_closure_digest
from genomic_variant_classifier.information_sources.observation import (
    DERIVED_STATES,
    ObservationState,
    SOURCE_RULE,
    SourceIdentity,
    SourceRequirement,
    check_state_against_source,
    check_value,
)

logger = logging.getLogger(__name__)

#: The fixed uint8 code of each state (the enum's declaration order; append-only).
STATE_CODES: tuple[ObservationState, ...] = tuple(ObservationState)
CODE: dict[ObservationState, int] = {s: i for i, s in enumerate(STATE_CODES)}


#: Version 2 of the population identity (stage 1D-1c.1). Version 1 is ``Population.sha256_v1``.
POPULATION_IDENTITY_SCHEMA = "gvc.population-order/2"
#: The identity of one derivation: rule, implementation code, inputs (stage 1D-1c.1).
DERIVATION_SCHEMA = "gvc.derivation/1"


def readonly_array(values: Any, dtype: Any = None) -> np.ndarray:
    """``graph_identity.readonly_array`` (the one definition): a bytes-backed, read-only copy whose digest cannot go stale
    (COLUMNS-MUTABLE-ARRAYS-1). Imported at call time because graph_identity imports materialization, which imports this
    module; moving the definition would change graph_identity's sealed code identity (its committed record)."""
    from genomic_variant_classifier.information_sources.graph_identity import readonly_array as _readonly_array
    return _readonly_array(values, dtype)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Population:
    """An ordered set of unique row identities: the denominator every column of a materialisation reconciles to."""
    population_id: str
    row_ids: tuple[str, ...]
    _index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.population_id, str) or not self.population_id:
            raise ValueError("a population needs a non-empty identifier")
        if not isinstance(self.row_ids, tuple) or not self.row_ids:
            raise ValueError("a population is a non-empty tuple of row identities")
        if any(type(r) is not str or not r for r in self.row_ids):
            raise ValueError("row identities are non-empty strings (exactly str)")
        index = {r: i for i, r in enumerate(self.row_ids)}
        if len(index) != len(self.row_ids):
            raise ValueError(f"population {self.population_id}: {len(self.row_ids) - len(index)} duplicated row identities")
        object.__setattr__(self, "_index", index)

    def __len__(self) -> int:
        return len(self.row_ids)

    @cached_property
    def sha256(self) -> str:
        """Order-binding identity, version 2: SHA-256 of canonical JSON {schema, population_id, row_ids in order}. JSON
        escapes every character, so distinct populations never share it; the identifier is bound."""
        return sha256_bytes(canonical_bytes({"schema": POPULATION_IDENTITY_SCHEMA, "population_id": self.population_id,
                                             "row_ids": list(self.row_ids)}))

    @cached_property
    def sha256_v1(self) -> str:
        """Version 1 (stage 1D-1b.2): SHA-256 of the identities one per line. ONLY for verifying records sealed under it
        (materialisation receipt schema .../1). Refuses a population it cannot represent faithfully."""
        if any("\n" in r for r in self.row_ids):
            raise ValueError("a row identity containing a newline has no faithful version-1 digest")
        return _sha256(("\n".join(self.row_ids) + "\n").encode("utf-8"))

    def position(self, row_id: str) -> int:
        return self._index[row_id]


def _state_codes_valid(states: np.ndarray) -> bool:
    return bool((states < len(STATE_CODES)).all())


@dataclass(frozen=True)
class ObservationColumn:
    feature: str
    source: SourceIdentity
    population: Population
    values: np.ndarray                      # float64, NaN exactly where the state is not OBSERVED
    states: np.ndarray                      # uint8 codes into STATE_CODES
    details: np.ndarray | None = None       # uint16 codes into detail_vocabulary (0 = no detail), or None
    detail_vocabulary: tuple[str, ...] = ("",)
    lineage: str | None = None              # derivation_identity(...) of a derived column; None for a source column

    def __post_init__(self) -> None:
        n = len(self.population)
        if not isinstance(self.feature, str) or not self.feature:
            raise ValueError("a column names its feature")
        if not isinstance(self.source, SourceIdentity):
            raise ValueError("a column's source is a SourceIdentity")
        if self.source.is_derived:
            if not isinstance(self.lineage, str) or not HEX64.match(self.lineage):
                raise ValueError("a derived column carries its lineage (a derivation identity)")
        elif self.lineage is not None:
            raise ValueError("a source column has no derivation lineage")
        if not isinstance(self.values, np.ndarray) or self.values.dtype != np.float64 or self.values.shape != (n,):
            raise ValueError("one float64 value per population row")
        if not isinstance(self.states, np.ndarray) or self.states.dtype != np.uint8 or self.states.shape != (n,):
            raise ValueError("one uint8 state code per population row")
        if self.details is not None and (not isinstance(self.details, np.ndarray) or self.details.dtype != np.uint16
                                         or self.details.shape != (n,)):
            raise ValueError("one uint16 detail code per population row")
        # Own read-only copies BEFORE validating them: what is validated is what is kept.
        object.__setattr__(self, "values", readonly_array(self.values))
        object.__setattr__(self, "states", readonly_array(self.states))
        if self.details is not None:
            object.__setattr__(self, "details", readonly_array(self.details))
        if not _state_codes_valid(self.states):
            raise ValueError("unknown observation state code")
        present = {STATE_CODES[c] for c in np.unique(self.states)}
        for state in present:                       # every state present must be one this source can carry
            check_state_against_source(state, self.source)
        obs = self.states == CODE[ObservationState.OBSERVED]
        if not np.isfinite(self.values[obs]).all():
            raise ValueError("an observed value must be finite")
        if not np.isnan(self.values[~obs]).all():
            raise ValueError("an unobserved row carries a value (a sentinel)")
        if self.details is not None:
            if not self.detail_vocabulary or self.detail_vocabulary[0] != "" or \
                    len(set(self.detail_vocabulary)) != len(self.detail_vocabulary):
                raise ValueError("a detail vocabulary starts with '' (no detail) and has no duplicates")
            if int(self.details.max(initial=0)) >= len(self.detail_vocabulary):
                raise ValueError("unknown detail code")
        elif self.detail_vocabulary != ("",):
            raise ValueError("a detail vocabulary needs detail codes")

    # ------------------------------------------------------------------------------------------------ construction
    @classmethod
    def from_rows(cls, feature: str, source: SourceIdentity, population: Population,
                  rows: Mapping[str, tuple[ObservationState, object]]) -> "ObservationColumn":
        """Rows keyed by row identity; refused unless they are EXACTLY the population's rows."""
        missing = sum(1 for r in population.row_ids if r not in rows)
        extra = sum(1 for r in rows if r not in population._index)
        if missing or extra:
            raise ValueError(f"rows do not reconcile with population {population.population_id}: "
                             f"{missing} missing, {extra} extra")
        values = np.full(len(population), np.nan)
        states = np.empty(len(population), np.uint8)
        for i, r in enumerate(population.row_ids):
            state, value = rows[r]
            if not isinstance(state, ObservationState):
                raise ValueError(f"row {r}: state must be an ObservationState")
            v = check_value(state, value)
            states[i] = CODE[state]
            if v is not None:
                values[i] = v
        return cls(feature, source, population, values, states)

    # ------------------------------------------------------------------------------------------------------ reading
    def counts(self) -> dict[str, int]:
        c = {s.value: int((self.states == CODE[s]).sum()) for s in STATE_CODES}
        if sum(c.values()) != len(self.population):          # cannot fail for a valid column; never assumed
            raise AssertionError("state counts do not reconcile with the population")
        return c

    def detail_counts(self) -> dict[str, dict[str, int]]:
        """{state: {detail: rows}} for every (state, detail) pair present."""
        out: dict[str, dict[str, int]] = {}
        details = self.details if self.details is not None else np.zeros(len(self.population), np.uint16)
        pairs, n = np.unique(np.stack([self.states.astype(np.uint32), details.astype(np.uint32)]), axis=1,
                             return_counts=True)
        for (s, d), k in zip(pairs.T, n):
            out.setdefault(STATE_CODES[int(s)].value, {})[self.detail_vocabulary[int(d)]] = int(k)
        return out

    def observed_mask(self) -> np.ndarray:
        return self.states == CODE[ObservationState.OBSERVED]

    @cached_property
    def sha256(self) -> str:
        """Identity of the materialised column: feature, source, population (ordered, identity version 2), values, states,
        details and, for a derived column, its lineage."""
        return column_digest(self, population_sha256=self.population.sha256)

    @cached_property
    def sha256_v1(self) -> str:
        """The stage 1D-1b.2 column identity (population identity version 1), ONLY for verifying records sealed under it."""
        if self.lineage is not None:
            raise ValueError("version 1 column identities predate derivation lineage")
        return column_digest(self, population_sha256=self.population.sha256_v1)


def column_digest(column: ObservationColumn, *, population_sha256: str) -> str:
    """The column digest given a population identity. The head's keys are those of stage 1D-1b.2; ``lineage`` is added
    only for a derived column, so a source column's version-1 digest is reproduced exactly."""
    head_fields: dict[str, Any] = {"feature": column.feature, "logical_id": column.source.logical_id,
                                   "content_sha256": column.source.content_sha256, "release": column.source.release,
                                   "population_sha256": population_sha256,
                                   "detail_vocabulary": list(column.detail_vocabulary)}
    if column.lineage is not None:
        head_fields["lineage"] = column.lineage
    h = hashlib.sha256(json.dumps(head_fields, sort_keys=True).encode("utf-8"))
    h.update(np.ascontiguousarray(column.values, dtype="<f8").tobytes())
    h.update(column.states.tobytes())
    if column.details is not None:
        h.update(np.ascontiguousarray(column.details, dtype="<u2").tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------- derivation
class Propagation(str, Enum):
    STRICT = "strict"                   # any input not observed -> the derived value is input_unobserved
    PARTIAL_COUNT = "partial_count"     # a count over the inputs that ARE observed, reported with how many were


@dataclass(frozen=True)
class DerivedRule:
    inputs: tuple[str, ...]
    propagation: Propagation
    statement: str
    #: The qualified name of the module-level function in THIS module that computes the rule: one function of all
    #: inputs (STRICT), or one 0/1 indicator per input (PARTIAL_COUNT). None only for a retired feature.
    implementation: str | tuple[str, ...] | None = None
    retired: bool = False

    def __post_init__(self) -> None:
        if self.retired != (self.implementation is None):
            raise ValueError("a rule is retired exactly when it has no implementation")
        if self.propagation is Propagation.PARTIAL_COUNT and not self.retired and (
                not isinstance(self.implementation, tuple) or len(self.implementation) != len(self.inputs)):
            raise ValueError("a partial count names one indicator per input")
        if self.propagation is Propagation.STRICT and not self.retired and not isinstance(self.implementation, str):
            raise ValueError("a strict rule names one function of all inputs")


# --------------------------------------------------------------------------------- the registered implementations
# Each is the PRODUCTION formula (models/variant_ensemble.py engineer_features; data/finngen.py), applied to observed
# values only; a test evaluates both on the thresholds themselves.
def af_log10_of(af: np.ndarray) -> np.ndarray:
    return np.log10(af + 1e-8)


def af_is_absent_of(af: np.ndarray) -> np.ndarray:
    return (af == 0).astype(np.float64)


def af_is_ultra_rare_of(af: np.ndarray) -> np.ndarray:
    return (af < 0.0001).astype(np.float64)


def af_is_rare_of(af: np.ndarray) -> np.ndarray:
    return ((af >= 0.0001) & (af < 0.001)).astype(np.float64)


def af_is_common_of(af: np.ndarray) -> np.ndarray:
    return (af >= 0.01).astype(np.float64)


def is_indel_of(is_insertion: np.ndarray, is_deletion: np.ndarray) -> np.ndarray:
    return (is_insertion.astype(np.int64) | is_deletion.astype(np.int64)).astype(np.float64)


def cadd_high_of(cadd_phred: np.ndarray) -> np.ndarray:
    return (cadd_phred >= 20).astype(np.float64)


def sift_deleterious_of(sift_score: np.ndarray) -> np.ndarray:
    return (sift_score < 0.05).astype(np.float64)


def polyphen_probably_damaging_of(polyphen2_score: np.ndarray) -> np.ndarray:
    return (polyphen2_score >= 0.908).astype(np.float64)


def revel_pathogenic_of(revel_score: np.ndarray) -> np.ndarray:
    return (revel_score >= 0.5).astype(np.float64)


def indicator_is_one(indicator: np.ndarray) -> np.ndarray:
    return (indicator == 1.0).astype(np.float64)


def gene_is_constrained_of(loeuf: np.ndarray) -> np.ndarray:
    return (loeuf < 0.35).astype(np.float64)


def finngen_enrichment_of(af_fin: np.ndarray, af_nfsee: np.ndarray) -> np.ndarray:
    return np.clip(af_fin / (af_nfsee + 1e-9), None, 1000.0)


#: The propagation rule of every feature the census declares as computed from other features (its `depends_on`), except
#: the two GNN scores, which are model outputs (a versioned operation, 1D-3), not functions of observations. A test
#: reconciles this table with the census both ways.
DERIVED_RULES: Mapping[str, DerivedRule] = {
    **{name: DerivedRule(("af_raw",), Propagation.STRICT, statement, f"{name}_of") for name, statement in (
        ("af_log10", "log10(af + 1e-8) of an observed frequency"),
        ("af_is_absent", "af == 0 of an OBSERVED frequency: 'observed at zero', never 'not observed'"),
        ("af_is_ultra_rare", "af < 1e-4 of an observed frequency; an unobserved frequency is not rarity evidence"),
        ("af_is_rare", "1e-4 <= af < 1e-3 of an observed frequency"),
        ("af_is_common", "af >= 0.01 of an observed frequency"))},
    "is_indel": DerivedRule(("is_insertion", "is_deletion"), Propagation.STRICT, "insertion or deletion of observed alleles",
                            "is_indel_of"),
    "cadd_high": DerivedRule(("cadd_phred",), Propagation.STRICT, "CADD PHRED >= 20 of an observed score", "cadd_high_of"),
    "sift_deleterious": DerivedRule(("sift_score",), Propagation.STRICT, "SIFT < 0.05 of an observed score",
                                    "sift_deleterious_of"),
    "polyphen_probably_damaging": DerivedRule(("polyphen2_score",), Propagation.STRICT,
                                              "PolyPhen-2 HDIV >= 0.908 of an observed score",
                                              "polyphen_probably_damaging_of"),
    "revel_pathogenic": DerivedRule(("revel_score",), Propagation.STRICT,
                                    "REVEL >= 0.5 of an observed score; an unscored variant is not pathogenic evidence",
                                    "revel_pathogenic_of"),
    "n_tools_pathogenic": DerivedRule(("cadd_high", "sift_deleterious", "polyphen_probably_damaging", "revel_pathogenic"),
                                      Propagation.PARTIAL_COUNT,
                                      "the number of OBSERVED tools calling the variant damaging, reported with the "
                                      "number of tools observed", ("indicator_is_one",) * 4),
    "gene_is_constrained": DerivedRule(("loeuf",), Propagation.STRICT, "LOEUF < 0.35 of an observed gene constraint",
                                       "gene_is_constrained_of"),
    "gene_has_known_disease": DerivedRule(("n_pathogenic_in_gene",), Propagation.STRICT,
                                          "retired (retire_predictive); declared for completeness", None, retired=True),
    "finngen_enrichment": DerivedRule(("finngen_af_fin", "finngen_af_nfsee"), Propagation.STRICT,
                                      "a ratio of two observed FinnGen frequencies, af_fin / (af_nfsee + 1e-9) clipped "
                                      "at 1000", "finngen_enrichment_of"),
    "finngen_r13_enrichment": DerivedRule(("finngen_r13_af_fin", "finngen_r13_af_nfsee"), Propagation.STRICT,
                                          "as finngen_enrichment, release 13", "finngen_enrichment_of"),
}
#: Model outputs the census declares with measured dependencies; they are not derived by a propagation rule.
MODEL_OUTPUTS: frozenset[str] = frozenset({"gnn_score", "hetero_gnn_score"})

_THIS_FILE = "src/genomic_variant_classifier/information_sources/columns.py"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _implementation_names(rule: DerivedRule) -> tuple[str, ...]:
    impl = rule.implementation
    return impl if isinstance(impl, tuple) else (impl,)      # type: ignore[return-value]


def _registered_rule(name: str) -> DerivedRule:
    if name in MODEL_OUTPUTS:
        raise ValueError(f"{name}: a model output, not a derived feature (a versioned operation, 1D-3)")
    rule = DERIVED_RULES.get(name)
    if rule is None:
        raise ValueError(f"{name}: not a registered derived feature (columns.DERIVED_RULES)")
    if rule.retired:
        raise ValueError(f"{name}: retired; it is declared for completeness and is never derived")
    return rule


def derivation_identity(name: str, inputs: Sequence[ObservationColumn]) -> str:
    """The lineage of a derived column: SHA-256 of canonical JSON {schema, name, rule (inputs, propagation, statement),
    implementation (qualified name and semantic-closure digest of its code, measured now), input column identities in
    order}. Equal values from different inputs, rules or code have different lineage."""
    rule = _registered_rule(name)
    impl = [{"symbol": f"{_THIS_FILE}::{q}", "closure_sha256": symbol_closure_digest(_repo_root(), f"{_THIS_FILE}::{q}")}
            for q in _implementation_names(rule)]
    return sha256_bytes(canonical_bytes({
        "schema": DERIVATION_SCHEMA, "name": name,
        "rule": {"inputs": list(rule.inputs), "propagation": rule.propagation.value, "statement": rule.statement},
        "implementation": impl,
        "inputs": [{"feature": c.feature, "column_sha256": c.sha256} for c in inputs]}))


@dataclass(frozen=True)
class DerivedColumn:
    column: ObservationColumn
    inputs_observed: np.ndarray        # int64, read-only: how many inputs were observed for each row
    n_inputs: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs_observed", readonly_array(self.inputs_observed, np.int64))


def derive(name: str, inputs: Sequence[ObservationColumn]) -> DerivedColumn:
    """The registered derived feature ``name`` of ``inputs`` (the rule's declared features, in its declared order).

    STRICT: the implementation of all input values, on the rows where EVERY input is observed; every other row is
    input_unobserved. PARTIAL_COUNT: one 0/1 indicator per input; the value is the sum over the OBSERVED inputs, on rows
    where at least one input is observed, and ``inputs_observed`` says how many were. A row with no observed input is
    input_unobserved -- never a count of zero. The caller supplies no function and no propagation rule."""
    rule = _registered_rule(name)
    inputs = tuple(inputs)
    if tuple(c.feature for c in inputs) != rule.inputs:
        raise ValueError(f"{name}: the declared inputs are {rule.inputs}, got {tuple(c.feature for c in inputs)}")
    if len({c.population.sha256 for c in inputs}) != 1:
        raise ValueError("inputs are on different populations")
    fns = [globals()[q] for q in _implementation_names(rule)]
    pop = inputs[0].population
    n = len(pop)
    observed = np.stack([c.observed_mask() for c in inputs])
    inputs_observed = observed.sum(axis=0).astype(np.int64)
    values = np.full(n, np.nan)
    if rule.propagation is Propagation.STRICT:
        ok = observed.all(axis=0)
        if ok.any():
            values[ok] = np.asarray(fns[0](*[c.values[ok] for c in inputs]), dtype=np.float64)
    else:
        ok = inputs_observed > 0
        parts = []
        for f, c, o in zip(fns, inputs, observed):
            ind = np.zeros(n)
            if o.any():
                got = np.asarray(f(c.values[o]), dtype=np.float64)
                if not np.isin(got, (0.0, 1.0)).all():
                    raise ValueError("an indicator must be 0 or 1")
                ind[o] = got
            parts.append(ind)
        values[ok] = np.sum(parts, axis=0)[ok]
    states = np.where(ok, CODE[ObservationState.OBSERVED], CODE[ObservationState.INPUT_UNOBSERVED]).astype(np.uint8)
    column = ObservationColumn(name, SourceIdentity(f"derived:{name}"), pop, values, states,
                               lineage=derivation_identity(name, inputs))
    return DerivedColumn(column, inputs_observed, len(inputs))


__all__ = ["CODE", "DERIVATION_SCHEMA", "DERIVED_RULES", "DERIVED_STATES", "DerivedColumn", "DerivedRule", "MODEL_OUTPUTS",
           "ObservationColumn", "POPULATION_IDENTITY_SCHEMA", "Population", "Propagation", "SOURCE_RULE", "STATE_CODES",
           "SourceRequirement", "column_digest", "derivation_identity", "derive", "readonly_array"]
