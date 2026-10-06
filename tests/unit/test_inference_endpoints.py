"""Benchmark endpoints (owner rulings 2026-10-03 / 2026-10-03b): the owner's essential case and eight listed properties,
plus the primary recovery contrast.

Author: Monzia Moodie
"""
from __future__ import annotations

from fractions import Fraction as F

import pytest

from genomic_variant_classifier.inference.endpoints import (
    Assessment as A, EndpointContract, assay_yield_bounds, comparative_yield_bounds, known_positive_recovery,
    recovery_contrast, top_k)
from genomic_variant_classifier.inference.exact_confirmation import InferenceError

GENES = frozenset({"shared", "a", "b", "c", "d"})
CONTRACT = EndpointContract(k=3, eligible_genes=GENES, reference_positives=frozenset({"a"}),
                            reference_id="reference-v1", assay_rule_id="assay-v1")
INTEGRATED, BASELINE = ("shared", "a", "b"), ("shared", "c", "d")
ASSESSED = {"shared": A.UNRESOLVED, "a": A.MEETS_CRITERION, "b": A.UNRESOLVED, "c": A.DOES_NOT_MEET, "d": A.DOES_NOT_MEET}


def code_of(call):
    with pytest.raises(InferenceError) as exc:
        call()
    return exc.value.code


def test_the_owners_essential_case():
    bounds = comparative_yield_bounds(INTEGRATED, BASELINE, CONTRACT, ASSESSED)
    assert (bounds["lower"], bounds["upper"]) == (F(1, 3), F(2, 3))
    assert known_positive_recovery(INTEGRATED, CONTRACT)["count"] == 1


def test_an_absent_assessment_remains_unresolved():
    assert assay_yield_bounds(INTEGRATED, CONTRACT, {})["unresolved"] == 3


def test_an_incomplete_nomination_budget_is_refused():
    assert code_of(lambda: top_k(("a", "b"), CONTRACT)) == "nomination_budget_incomplete"


def test_duplicate_genes_are_refused():
    assert code_of(lambda: top_k(("a", "a", "b"), CONTRACT)) == "duplicate_gene"


def test_reference_membership_never_generates_assay_labels():
    y = assay_yield_bounds(("a", "b", "c"), CONTRACT, {})
    assert (y["successes"], y["unresolved"]) == (0, 3)          # "a" is a reference positive, NOT an assay success
    assert "false_positive" not in known_positive_recovery(("a", "b", "c"), CONTRACT)


def test_identical_nomination_sets_give_zero_difference_bounds():
    same = comparative_yield_bounds(INTEGRATED, INTEGRATED, CONTRACT, ASSESSED)
    assert (same["lower"], same["upper"]) == (0, 0)


def test_complete_assessments_give_identical_bounds():
    full = {g: A.MEETS_CRITERION for g in GENES}
    y = assay_yield_bounds(INTEGRATED, CONTRACT, full)
    assert y["lower"] == y["upper"] and y["fully_assessed"]


def test_a_shared_genes_assessment_leaves_the_contrast_unchanged():
    changed = dict(ASSESSED, shared=A.MEETS_CRITERION)
    assert comparative_yield_bounds(INTEGRATED, BASELINE, CONTRACT, changed) == \
        comparative_yield_bounds(INTEGRATED, BASELINE, CONTRACT, ASSESSED)


def test_the_rule_identities_are_carried_into_every_result():
    """A changed rule identity requires a NEW evaluation record -- enforced by the admission layer; here, the identity
    travels with every result so the admission layer can compare it."""
    assert assay_yield_bounds(INTEGRATED, CONTRACT, ASSESSED)["assay_rule_id"] == "assay-v1"
    assert known_positive_recovery(INTEGRATED, CONTRACT)["reference_id"] == "reference-v1"


# ------------------------------------------------------------------ the primary recovery contrast

def test_recovery_contrast_decomposes_into_exclusive_recoveries():
    contract = EndpointContract(k=3, eligible_genes=frozenset("abcdefg"), reference_positives=frozenset("abe"),
                                reference_id="reference-v1", assay_rule_id="assay-v1")
    c = recovery_contrast(("a", "b", "c"), ("a", "d", "e"), contract)
    assert (c["primary_count"], c["comparator_count"], c["delta"]) == (2, 2, 0)
    assert (c["recovered_only_by_primary"], c["recovered_only_by_comparator"], c["shared_nominations"]) == (("b",), ("e",), 1)


def test_identical_rankings_have_zero_recovery_difference():
    assert recovery_contrast(INTEGRATED, INTEGRATED, CONTRACT)["delta"] == 0


def test_recovery_contrast_reports_the_complete_replacement_sets():
    """The owner's worked example (ruling 2026-10-07): integrated A C B D, baseline A B C D, R = {A, C}, k = 2."""
    contract = EndpointContract(k=2, eligible_genes=frozenset("ABCD"), reference_positives=frozenset("AC"),
                                reference_id="reference-v1", assay_rule_id="assay-v1")
    c = recovery_contrast(("A", "C", "B", "D"), ("A", "B", "C", "D"), contract)
    assert (c["delta"], c["gained_top_k"], c["lost_top_k"], c["shared_top_k"]) == (1, ("C",), ("B",), ("A",))
    assert (c["recovered_only_by_primary"], c["recovered_only_by_comparator"]) == (("C",), ())


def test_replacement_sets_are_complete_and_balanced():
    contract = EndpointContract(k=3, eligible_genes=frozenset("abcdefg"), reference_positives=frozenset("abe"),
                                reference_id="reference-v1", assay_rule_id="assay-v1")
    c = recovery_contrast(("a", "b", "c"), ("a", "d", "e"), contract)
    assert (c["gained_top_k"], c["lost_top_k"], c["shared_top_k"]) == (("b", "c"), ("d", "e"), ("a",))
    assert len(c["gained_top_k"]) == len(c["lost_top_k"])                       # both top-K sets have exactly k genes
    assert set(c["gained_top_k"]) | set(c["shared_top_k"]) == {"a", "b", "c"}   # the primary's top-K, partitioned
    assert c["shared_nominations"] == len(c["shared_top_k"])


@pytest.mark.parametrize("kwargs, code", [
    ({"reference_positives": frozenset({"z"})}, "reference_outside_eligible_universe"),
    ({"k": 0}, "k_invalid"),
    ({"k": 6}, "eligible_universe_too_small"),
    ({"reference_id": " "}, "rule_identity_required"),
    ({"eligible_genes": {"a", "b", "c"}}, "immutable_gene_set_required"),
])
def test_the_endpoint_contract_refuses_malformations(kwargs, code):
    args = dict(k=3, eligible_genes=GENES, reference_positives=frozenset({"a"}), reference_id="reference-v1",
                assay_rule_id="assay-v1")
    args.update(kwargs)
    assert code_of(lambda: EndpointContract(**args)) == code


def test_an_ineligible_gene_in_a_ranking_is_refused():
    assert code_of(lambda: top_k(("a", "b", "zz"), CONTRACT)) == "ineligible_gene"


def test_unresolved_genes_stay_in_the_denominator():
    """The ruling's worked example: 8 successes, 4 completed non-successes, 8 unresolved at K = 20 give [8/20, 16/20] --
    never 8/12, which drops unresolved genes and rewards methods whose difficult nominations stay unassessed."""
    genes = tuple("g{:02d}".format(i) for i in range(20))
    contract = EndpointContract(k=20, eligible_genes=frozenset(genes), reference_positives=frozenset(),
                                reference_id="reference-v1", assay_rule_id="assay-v1")
    states = {g: A.MEETS_CRITERION for g in genes[:8]}
    states.update({g: A.DOES_NOT_MEET for g in genes[8:12]})       # genes[12:] are absent -> unresolved
    y = assay_yield_bounds(genes, contract, states)
    assert (y["lower"], y["upper"], y["assessment_coverage"], y["fully_assessed"]) == (F(8, 20), F(16, 20), F(12, 20), False)
