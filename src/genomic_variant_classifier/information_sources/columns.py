"""Stage 1D-1b.2: columnar observations reconciled against an explicit population, and derived-feature propagation.

Owner ruling 7f4b69d0 (2026-11-09), section 5: for large data, one Python object per variant-feature cell is the wrong
shape; use numerical arrays with a parallel compact state array, bound to an exact row population, and require for every
feature

    N_observed + N_not_in_source + N_unsupported + N_source_absent + N_failed (+ ...) = N_population,

with any omitted or duplicated row refused. Here:

* ``Population`` is an ORDERED tuple of unique row identities; its SHA-256 binds the order, because a column's values are
  aligned to it by position (a population with the same rows in another order is another population).
* ``ObservationColumn`` holds one float64 value (NaN exactly where not observed) and one uint8 state code per row, plus
  optional detail codes (why a row is in its state, e.g. "gene not modelled" against "substitution not scored") from a
  declared vocabulary. Every rule of observation.py holds for every row, checked vectorised.
* ``ObservationColumn.from_rows`` refuses rows that do not reconcile with the population (each missing and extra row
  counted), so the reconciliation is enforced, not assumed. ``counts()`` sums to the population by construction.
* ``derive`` computes a derived column whose states are a DECLARED function of its inputs' states (``Propagation``), and
  ``DERIVED_RULES`` declares the rule for every derived feature the census declares (a test reconciles the two).

numpy is required; observation.py stays standard-library only.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

import numpy as np

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
        if any(not isinstance(r, str) or not r for r in self.row_ids):
            raise ValueError("row identities are non-empty strings")
        index = {r: i for i, r in enumerate(self.row_ids)}
        if len(index) != len(self.row_ids):
            raise ValueError(f"population {self.population_id}: {len(self.row_ids) - len(index)} duplicated row identities")
        object.__setattr__(self, "_index", index)

    def __len__(self) -> int:
        return len(self.row_ids)

    @property
    def sha256(self) -> str:
        """Order-binding identity: SHA-256 of the row identities, one per line, in population order."""
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

    def __post_init__(self) -> None:
        n = len(self.population)
        if not isinstance(self.feature, str) or not self.feature:
            raise ValueError("a column names its feature")
        if not isinstance(self.source, SourceIdentity):
            raise ValueError("a column's source is a SourceIdentity")
        if not isinstance(self.values, np.ndarray) or self.values.dtype != np.float64 or self.values.shape != (n,):
            raise ValueError("one float64 value per population row")
        if not isinstance(self.states, np.ndarray) or self.states.dtype != np.uint8 or self.states.shape != (n,):
            raise ValueError("one uint8 state code per population row")
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
            if not isinstance(self.details, np.ndarray) or self.details.dtype != np.uint16 or self.details.shape != (n,):
                raise ValueError("one uint16 detail code per population row")
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

    @property
    def sha256(self) -> str:
        """Identity of the materialised column: feature, source, population (ordered), values, states and details."""
        head = json.dumps({"feature": self.feature, "logical_id": self.source.logical_id,
                           "content_sha256": self.source.content_sha256, "release": self.source.release,
                           "population_sha256": self.population.sha256,
                           "detail_vocabulary": list(self.detail_vocabulary)}, sort_keys=True).encode("utf-8")
        h = hashlib.sha256(head)
        h.update(np.ascontiguousarray(self.values, dtype="<f8").tobytes())
        h.update(self.states.tobytes())
        if self.details is not None:
            h.update(np.ascontiguousarray(self.details, dtype="<u2").tobytes())
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


#: The propagation rule of every feature the census declares as computed from other features (its `depends_on`), except
#: the two GNN scores, which are model outputs (a versioned operation, 1D-3), not functions of observations. A test
#: reconciles this table with the census both ways.
DERIVED_RULES: Mapping[str, DerivedRule] = {
    **{name: DerivedRule(("af_raw",), Propagation.STRICT, statement) for name, statement in (
        ("af_log10", "log10(af + 1e-8) of an observed frequency"),
        ("af_is_absent", "af == 0 of an OBSERVED frequency: 'observed at zero', never 'not observed'"),
        ("af_is_ultra_rare", "af < 1e-4 of an observed frequency; an unobserved frequency is not rarity evidence"),
        ("af_is_rare", "1e-4 <= af < 1e-3 of an observed frequency"),
        ("af_is_common", "af >= 0.01 of an observed frequency"))},
    "is_indel": DerivedRule(("is_insertion", "is_deletion"), Propagation.STRICT, "insertion or deletion of observed alleles"),
    "cadd_high": DerivedRule(("cadd_phred",), Propagation.STRICT, "CADD PHRED >= 20 of an observed score"),
    "sift_deleterious": DerivedRule(("sift_score",), Propagation.STRICT, "SIFT < 0.05 of an observed score"),
    "polyphen_probably_damaging": DerivedRule(("polyphen2_score",), Propagation.STRICT,
                                              "PolyPhen-2 HDIV >= 0.908 of an observed score"),
    "revel_pathogenic": DerivedRule(("revel_score",), Propagation.STRICT,
                                    "REVEL >= 0.5 of an observed score; an unscored variant is not pathogenic evidence"),
    "n_tools_pathogenic": DerivedRule(("cadd_high", "sift_deleterious", "polyphen_probably_damaging", "revel_pathogenic"),
                                      Propagation.PARTIAL_COUNT,
                                      "the number of OBSERVED tools calling the variant damaging, reported with the "
                                      "number of tools observed"),
    "gene_is_constrained": DerivedRule(("loeuf",), Propagation.STRICT, "LOEUF < 0.35 of an observed gene constraint"),
    "gene_has_known_disease": DerivedRule(("n_pathogenic_in_gene",), Propagation.STRICT,
                                          "retired (retire_predictive); declared for completeness"),
    "finngen_enrichment": DerivedRule(("finngen_af_fin", "finngen_af_nfsee"), Propagation.STRICT,
                                      "a ratio of two observed FinnGen frequencies"),
    "finngen_r13_enrichment": DerivedRule(("finngen_r13_af_fin", "finngen_r13_af_nfsee"), Propagation.STRICT,
                                          "as finngen_enrichment, release 13"),
}
#: Model outputs the census declares with measured dependencies; they are not derived by a propagation rule.
MODEL_OUTPUTS: frozenset[str] = frozenset({"gnn_score", "hetero_gnn_score"})


@dataclass(frozen=True)
class DerivedColumn:
    column: ObservationColumn
    inputs_observed: np.ndarray        # int64: how many inputs were observed for each row
    n_inputs: int


def derive(name: str, inputs: Sequence[ObservationColumn], fn: Callable[..., np.ndarray] | Sequence[Callable],
           rule: Propagation) -> DerivedColumn:
    """A derived column whose states are a declared function of its inputs' states.

    STRICT: ``fn(*input_values)`` on the rows where EVERY input is observed; every other row is input_unobserved.
    PARTIAL_COUNT: ``fn`` is one 0/1 indicator per input; the value is the sum over the OBSERVED inputs, on rows where at
    least one input is observed, and ``inputs_observed`` says how many were. A row with no observed input is
    input_unobserved -- never a count of zero."""
    if not inputs:
        raise ValueError("a derived feature has at least one input")
    if len({c.population.sha256 for c in inputs}) != 1:
        raise ValueError("inputs are on different populations")
    pop = inputs[0].population
    n = len(pop)
    observed = np.stack([c.observed_mask() for c in inputs])
    inputs_observed = observed.sum(axis=0).astype(np.int64)
    values = np.full(n, np.nan)
    if rule is Propagation.STRICT:
        if not callable(fn):
            raise ValueError("a strict rule takes one function of all inputs")
        ok = observed.all(axis=0)
        if ok.any():
            values[ok] = np.asarray(fn(*[c.values[ok] for c in inputs]), dtype=np.float64)
    elif rule is Propagation.PARTIAL_COUNT:
        if callable(fn) or len(fn) != len(inputs):
            raise ValueError("a partial count takes one indicator function per input")
        ok = inputs_observed > 0
        parts = []
        for f, c, o in zip(fn, inputs, observed):
            ind = np.zeros(n)
            if o.any():
                got = np.asarray(f(c.values[o]), dtype=np.float64)
                if not np.isin(got, (0.0, 1.0)).all():
                    raise ValueError("an indicator must be 0 or 1")
                ind[o] = got
            parts.append(ind)
        values[ok] = np.sum(parts, axis=0)[ok]
    else:
        raise ValueError(f"unknown propagation rule {rule!r}")
    states = np.where(ok, CODE[ObservationState.OBSERVED], CODE[ObservationState.INPUT_UNOBSERVED]).astype(np.uint8)
    column = ObservationColumn(name, SourceIdentity(f"derived:{name}"), pop, values, states)
    return DerivedColumn(column, inputs_observed, len(inputs))


__all__ = ["CODE", "DERIVED_RULES", "DERIVED_STATES", "DerivedColumn", "DerivedRule", "MODEL_OUTPUTS",
           "ObservationColumn", "Population", "Propagation", "SOURCE_RULE", "STATE_CODES", "SourceRequirement", "derive"]
