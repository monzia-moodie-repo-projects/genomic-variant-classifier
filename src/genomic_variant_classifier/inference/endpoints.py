"""Benchmark endpoints with honest missing labels (owner rulings 2026-10-03 / 2026-10-03b).

PRIMARY COMPUTATIONAL ENDPOINT (ruling 2026-10-03b): Delta H(20) = H_DANDELION(20) - H_burden-only(20), "known-positive
recovery at 20" -- the number of frozen reference-set positives among each method's top 20. It measures recovery of
ESTABLISHED evidence; genes absent from the reference set are NOT negatives, so it is never precision, a causal-gene
discovery rate or experimental validation. No confidence interval is attached by bootstrapping the 20 genes, and several
methods on one benchmark are not several experiments.

TWO LABEL SYSTEMS, NEVER COMBINED: reference membership (positive / not listed) and assay assessment (meets criterion /
does not meet / unresolved). "Not listed" is not an assay failure, and "does not meet" is not "no biological effect".

FUTURE ENDPOINT: Delta Y(20), "functional-assay success yield at 20", under one common prespecified protocol over the
union of the methods' top-20 sets. Unresolved assessments give BOUNDS (not confidence intervals), and shared nominations
CANCEL from a difference, so an unresolved shared gene widens neither contrast.

The calculations below come from the owner's reference (decision.txt generation 2a970b07, lines 929-1064), transformed
mechanically (its generic errors now raise InferenceError with the same codes); recovery_contrast is added for the primary
Delta H. The admission layer -- not these functions -- must verify that every assessment uses one assay-rule identity and
that the reference evidence is independent of the ranking inputs.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Mapping

from genomic_variant_classifier.inference.exact_confirmation import InferenceError

logger = logging.getLogger(__name__)

__all__ = ["Assessment", "EndpointContract", "top_k", "known_positive_recovery", "recovery_contrast", "assessment_for",
           "assay_yield_bounds", "comparative_yield_bounds"]


class Assessment(Enum):
    MEETS_CRITERION = "meets_criterion"
    DOES_NOT_MEET = "does_not_meet_criterion"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class EndpointContract:
    k: int
    eligible_genes: frozenset[str]
    reference_positives: frozenset[str]
    reference_id: str
    assay_rule_id: str

    def __post_init__(self):
        if type(self.k) is not int or self.k < 1:
            raise InferenceError("k_invalid")

        for values in (self.eligible_genes, self.reference_positives):
            if type(values) is not frozenset:
                raise InferenceError("immutable_gene_set_required")
            if any(type(g) is not str or not g for g in values):
                raise InferenceError("gene_identity")

        if len(self.eligible_genes) < self.k:
            raise InferenceError("eligible_universe_too_small")

        if not self.reference_positives <= self.eligible_genes:
            raise InferenceError("reference_outside_eligible_universe")

        for identity in (self.reference_id, self.assay_rule_id):
            if type(identity) is not str or not identity.strip():
                raise InferenceError("rule_identity_required")


def top_k(ranking, contract):
    """Ranking order, including tie-breaking, must already be frozen."""
    ranking = tuple(ranking)

    if any(type(g) is not str or not g for g in ranking):
        raise InferenceError("gene_identity")
    if len(set(ranking)) != len(ranking):
        raise InferenceError("duplicate_gene")
    if not set(ranking) <= contract.eligible_genes:
        raise InferenceError("ineligible_gene")
    if len(ranking) < contract.k:
        raise InferenceError("nomination_budget_incomplete")

    return frozenset(ranking[:contract.k])


def known_positive_recovery(ranking, contract):
    selected = top_k(ranking, contract)
    recovered = selected & contract.reference_positives
    count = len(recovered)

    return {
        "endpoint": "known_positive_recovery",
        "reference_id": contract.reference_id,
        "k": contract.k,
        "count": count,
        "fraction_of_nominations": Fraction(count, contract.k),
        "recovered_genes": tuple(sorted(recovered)),
        # Deliberately no "false_positive" count.
    }


def assessment_for(gene, assessments):
    state = assessments.get(gene, Assessment.UNRESOLVED)
    if type(state) is not Assessment:
        raise InferenceError("assessment_state")
    return state


def assay_yield_bounds(ranking, contract, assessments):
    selected = top_k(ranking, contract)
    states = [assessment_for(g, assessments) for g in selected]

    successes = states.count(Assessment.MEETS_CRITERION)
    unresolved = states.count(Assessment.UNRESOLVED)
    k = contract.k

    return {
        "endpoint": "functional_assay_success_yield",
        "assay_rule_id": contract.assay_rule_id,
        "k": k,
        "successes": successes,
        "unresolved": unresolved,
        "assessment_coverage": Fraction(k - unresolved, k),
        "lower": Fraction(successes, k),
        "upper": Fraction(successes + unresolved, k),
        "fully_assessed": unresolved == 0,
    }


def comparative_yield_bounds(
    integrated_ranking,
    baseline_ranking,
    contract,
    assessments: Mapping[str, Assessment],
):
    integrated = top_k(integrated_ranking, contract)
    baseline = top_k(baseline_ranking, contract)

    # Shared genes have coefficient zero in the difference.
    integrated_only = integrated - baseline
    baseline_only = baseline - integrated

    def counts(genes):
        states = [assessment_for(g, assessments) for g in genes]
        return (
            states.count(Assessment.MEETS_CRITERION),
            states.count(Assessment.UNRESOLVED),
        )

    success_i, unknown_i = counts(integrated_only)
    success_b, unknown_b = counts(baseline_only)

    observed_difference = success_i - success_b

    return {
        "contrast": "integrated_minus_baseline",
        "shared_nominations": len(integrated & baseline),
        "lower": Fraction(
            observed_difference - unknown_b, contract.k
        ),
        "upper": Fraction(
            observed_difference + unknown_i, contract.k
        ),
    }


def recovery_contrast(primary_ranking, comparator_ranking, contract):
    """The PRIMARY contrast Delta H(k) = H_primary(k) - H_comparator(k), decomposed exactly: shared nominations cancel,
    so Delta H = |(A - B) & R| - |(B - A) & R|. Reported with both counts and the exclusive genes behind them."""
    a = top_k(primary_ranking, contract)
    b = top_k(comparator_ranking, contract)
    r = contract.reference_positives
    primary_only = (a - b) & r
    comparator_only = (b - a) & r
    delta = len(a & r) - len(b & r)
    if delta != len(primary_only) - len(comparator_only):        # the identity, checked on every call
        raise InferenceError("recovery_identity")
    return {
        "endpoint": "known_positive_recovery_difference",
        "reference_id": contract.reference_id,
        "k": contract.k,
        "primary_count": len(a & r),
        "comparator_count": len(b & r),
        "delta": delta,
        "shared_nominations": len(a & b),
        "recovered_only_by_primary": tuple(sorted(primary_only)),
        "recovered_only_by_comparator": tuple(sorted(comparator_only)),
    }
