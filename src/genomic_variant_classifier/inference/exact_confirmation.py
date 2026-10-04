"""Exact, conservative gene-level confirmation (owner rulings 2026-10-02 / 2026-10-03 / 2026-10-03b).

WHAT THIS CONFIRMS
==================
The gene-level claim: "the gene has burden association, AND at least one eligible exposure has trans association with
it". For gene g with burden p-value b_g and trans p-values t_eg over its CONTRACTED exposure family of size M_g:

    T_g = min(1, M_g * min_e t_eg)          multiple opportunities for a trans association (Bonferroni)
    P_g = max(b_g, T_g)                     the conjunction (an intersection-union test)

then Holm across the complete contracted gene family. If the burden null holds, P_g >= b_g; if every trans null holds,
P_g >= T_g, which Bonferroni makes valid. No independence between burden and trans evidence is required -- but VALID
component p-values and a VALID SELECTION procedure are: exactness cannot repair invalid p-values, outcome-dependent
selection or mismatched hypotheses.

The earlier construction min(1, M_g * min_e max(b_g, t_eg)) is SUPERSEDED (ruling 2026-10-03): it penalised the shared
burden evidence M_g times. P_g never exceeds it (verified on 100,000 exact-rational cases) and can be far smaller --
100 exposures, b = 0.001, min t = 1e-6 give 1/1000 against 1/10. The 2026-10-03 reference package still implemented the
superseded form; this module does not.

P_g >= b_g is a POWER CEILING: strong trans evidence cannot lower a gene below its burden p-value. That is the price of
the assumption-light guarantee, and why adaptive integration (DANDELION) is the primary RANKING method while this is the
conservative CONFIRMATION result. A confirmed gene is a gene-level conjunction, NOT a confirmed exposure-gene pathway:
the exposure attaining min t is not itself multiplicity-controlled.

EXACTNESS IS ABOUT REPRESENTATION, NOT ACCURACY
===============================================
`Probability.decimal("0.01")` is exactly 1/100. `Probability.binary64(0.01)` is the exact STORED binary value, which is
slightly above 1/100 -- the two are distinct, and both are exact. Neither recovers precision lost upstream. Log-only
evidence is REFUSED here: it needs certified enclosures through ordering and arithmetic (a separate, reviewable
component), never exponentiation to zero.

MISSINGNESS IS A STATE, NOT A NUMBER
====================================
A planned exposure without a measurement keeps its state (not_tested, excluded_by_quality_rule, mapping_unresolved,
source_unavailable) and contributes the conservative upper bound 1 -- never an imputed observation. The denominator stays
the CONTRACTED family size: dropping unavailable exposures and shrinking M would be selection-dependent.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction

logger = logging.getLogger(__name__)

__all__ = ["InferenceError", "Probability", "Missingness", "TransEvidence", "GeneTest", "HolmDecision",
           "checked_alpha", "gene_test", "holm", "partial_conjunction"]


class InferenceError(ValueError):
    """A refusal with a stable reason code (the reference packages' codes, kept verbatim)."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(code if not detail else "{}: {}".format(code, detail))


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise InferenceError(code, detail)


def _nonempty_str(value) -> bool:
    return type(value) is str and bool(value.strip())


@dataclass(frozen=True)
class Probability:
    """An exact probability in (0, 1] with its declared representation and a source description.

    The source is a DESCRIPTION, not authenticated provenance: production admission binds file hashes, source rows and
    the analysis-contract identity outside this type."""

    value: Fraction
    representation: str
    source: str

    def __post_init__(self) -> None:
        _require(type(self.value) is Fraction and 0 < self.value <= 1, "p_range")
        _require(self.representation in {"decimal", "binary64", "derived"}, "representation")
        _require(_nonempty_str(self.source), "source_required")

    @classmethod
    def decimal(cls, text: str, source: str) -> "Probability":
        """Exact as SUPPLIED, including scientific notation ("1e-1000" is exactly 10**-1000)."""
        _require(type(text) is str, "decimal_string_required")
        try:
            d = Decimal(text)
        except InvalidOperation:
            raise InferenceError("p_range", repr(text)) from None
        _require(d.is_finite(), "p_range", repr(text))
        _require(d != 0, "p_zero_requires_log_provenance")
        return cls(Fraction(d), "decimal", source)

    @classmethod
    def binary64(cls, value: float, source: str) -> "Probability":
        """The exact STORED binary value -- Fraction(0.01) is not 1/100."""
        _require(type(value) is float and math.isfinite(value), "binary64_required")
        _require(value != 0, "p_zero_requires_log_provenance")
        return cls(Fraction.from_float(value), "binary64", source)


def checked_alpha(alpha) -> Fraction:
    """Alpha's own validation runs BEFORE any p-value validation, so its reason code takes precedence."""
    _require(type(alpha) is Fraction and 0 < alpha < 1, "alpha_range")
    return alpha


class Missingness(str, Enum):
    """Why a planned exposure-gene pair has (or lacks) a measurement (ruling 2026-10-03, L127-133)."""

    MEASURED = "measured"
    NOT_TESTED = "not_tested"
    EXCLUDED_BY_QUALITY_RULE = "excluded_by_quality_rule"
    MAPPING_UNRESOLVED = "mapping_unresolved"
    SOURCE_UNAVAILABLE = "source_unavailable"


@dataclass(frozen=True)
class TransEvidence:
    """One planned exposure's trans evidence for a gene: a measurement, or an explicit reason there is none."""

    exposure: str
    state: Missingness
    evidence: Probability | None = None

    def __post_init__(self) -> None:
        _require(_nonempty_str(self.exposure), "exposure_identity")
        _require(type(self.state) is Missingness, "state")
        if self.state is Missingness.MEASURED:
            _require(type(self.evidence) is Probability, "measured_requires_evidence")
        else:
            _require(self.evidence is None, "unmeasured_has_no_measurement")

    @property
    def upper_bound(self) -> Fraction:
        """The measured p-value, or the conservative bound 1 -- which is NOT a measurement."""
        return self.evidence.value if self.state is Missingness.MEASURED else Fraction(1)


@dataclass(frozen=True)
class GeneTest:
    """P_g with its components and how much of the family was actually measured."""

    p: Probability
    burden: Fraction
    trans_global: Fraction
    family_size: int
    measured: int

    @property
    def unmeasured(self) -> int:
        return self.family_size - self.measured


def gene_test(burden: Probability, trans, planned: tuple) -> GeneTest:
    """P_g = max(b_g, min(1, M_g * min_e t_eg)) over the CONTRACTED exposure family `planned`.

    Every planned exposure must have exactly one record (measured or an explicit missingness state); none may be
    absent, extra or duplicated. The family itself must come from the sealed analysis contract -- a caller-supplied
    tuple does not establish that it is the right family."""
    _require(type(burden) is Probability, "burden_required")
    _require(type(planned) is tuple and len(planned) > 0
             and all(_nonempty_str(x) for x in planned) and len(set(planned)) == len(planned), "exposure_family")
    trans = tuple(trans)
    _require(all(type(x) is TransEvidence for x in trans), "trans_type")
    indexed = {x.exposure: x for x in trans}
    _require(len(indexed) == len(trans) and set(indexed) == set(planned), "exposure_membership")
    trans_global = min(Fraction(1), len(planned) * min(indexed[e].upper_bound for e in planned))
    measured = sum(1 for e in planned if indexed[e].state is Missingness.MEASURED)
    p = max(burden.value, trans_global)
    return GeneTest(Probability(p, "derived", "max(burden, Bonferroni over the contracted exposure family)"),
                    burden.value, trans_global, len(planned), measured)


@dataclass(frozen=True)
class HolmDecision:
    gene: str
    p: Fraction
    adjusted: Fraction
    reject: bool


def holm(rows, alpha: Fraction = Fraction(1, 20)) -> tuple:
    """Inclusive Holm over the COMPLETE contracted gene family, with exact sorting, products and running maxima.

    The running maximum of min(1, (m - i) * p_(i)) compared with alpha (inclusive) rejects exactly the prefix that the
    sequential step-down rejects, so a later step can never be rejected after an earlier one is not. Ties in p are
    ordered by gene identity, so the result is deterministic. Log-only input is refused (`exact_probability_required`)."""
    checked_alpha(alpha)
    rows = tuple(rows)
    _require(len(rows) > 0, "empty_gene_family")
    names = [g for g, _ in rows]
    _require(all(_nonempty_str(g) for g in names) and len(set(names)) == len(names), "gene_identity")
    _require(all(type(p) is Probability for _, p in rows), "exact_probability_required")
    ordered = sorted(rows, key=lambda item: (item[1].value, item[0]))
    running, decisions = Fraction(0), []
    for i, (gene, p) in enumerate(ordered):
        running = max(running, min(Fraction(1), (len(rows) - i) * p.value))
        decisions.append(HolmDecision(gene, p.value, running, running <= alpha))
    return tuple(decisions)


def partial_conjunction(studies, required: int) -> Probability:
    """'At least r of n studies are non-null' (Bonferroni partial conjunction: (n - r + 1) * p_(r), capped at 1).

    Valid under arbitrary dependence given distinct prespecified studies with valid study-level p-values. It does NOT
    turn repeated analyses of one cohort into independent replication -- study provenance is checked outside."""
    studies = tuple(studies)
    n = len(studies)
    _require(type(required) is int and 1 <= required <= n, "replication_count")
    ids = [s for s, _ in studies]
    _require(all(_nonempty_str(s) for s in ids) and len(set(ids)) == n, "study_identity")
    _require(all(type(v) is Probability for _, v in studies), "exact_probability_required")
    p = sorted(v.value for _, v in studies)[required - 1]
    return Probability(min(Fraction(1), (n - required + 1) * p), "derived", "Bonferroni partial conjunction")
