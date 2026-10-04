"""Exact gene-level confirmation (owner rulings 2026-10-02 / 2026-10-03 / 2026-10-03b).

Ports the 2026-10-03 reference package's tests EXCEPT its pair-then-exposure test, which asserted the SUPERSEDED
construction (1/25); adds the revised formula, missingness, membership, boundary and ordering cases the rulings require.

Author: Monzia Moodie
"""
from __future__ import annotations

import random
from fractions import Fraction as F

import pytest

from genomic_variant_classifier.inference.exact_confirmation import (
    GeneTest, InferenceError, Missingness, Probability as P, TransEvidence, gene_test, holm, partial_conjunction)


def dec(text):
    return P.decimal(text, "fixture")


def measured(exposure, text):
    return TransEvidence(exposure, Missingness.MEASURED, dec(text))


def code_of(call):
    with pytest.raises(InferenceError) as exc:
        call()
    return exc.value.code


# ------------------------------------------------------------------ representations (reference tests, kept)

@pytest.mark.parametrize("text", ["0", "-0.1", "1.5", "NaN", "Infinity"])
def test_bad_probabilities_are_refused(text):
    assert code_of(lambda: dec(text)) in {"p_range", "p_zero_requires_log_provenance"}


def test_a_tiny_decimal_is_preserved_exactly():
    assert dec("1e-1000").value == F(1, 10**1000)


def test_binary_and_decimal_are_distinct_exact_values():
    assert P.binary64(0.01, "fixture").value == F.from_float(0.01) != dec("0.01").value == F(1, 100)


@pytest.mark.parametrize("alpha", [F(0), F(3, 2), True, 0.05])
def test_alpha_is_validated_before_anything_else(alpha):
    assert code_of(lambda: holm([], alpha)) == "alpha_range"


def test_log_only_input_is_refused_never_exponentiated():
    assert code_of(lambda: holm([("a", -1000)])) == "exact_probability_required"


# ------------------------------------------------------------------ the revised gene test

def test_the_worked_example_of_the_ruling():
    planned = tuple("exposure-{}".format(i) for i in range(100))
    t = gene_test(dec("0.001"), [measured(e, "0.000001") for e in planned], planned)
    assert (t.p.value, t.burden, t.trans_global, t.family_size, t.measured) == (F(1, 1000), F(1, 1000), F(1, 10000), 100, 100)


def test_the_revised_test_never_exceeds_the_superseded_construction():
    rng = random.Random(20261004)
    strictly = 0
    for _ in range(5000):
        m = rng.randint(1, 50)
        planned = tuple("e{}".format(i) for i in range(m))
        b = F(rng.randint(1, 10**6), 10**rng.randint(1, 9))
        ts = [F(rng.randint(1, 10**6), 10**rng.randint(1, 9)) for _ in planned]
        b, ts = min(b, F(1)), [min(x, F(1)) for x in ts]
        new = gene_test(P(b, "derived", "f"), [TransEvidence(e, Missingness.MEASURED, P(x, "derived", "f")) for e, x in zip(planned, ts)], planned).p.value
        old = min(F(1), m * min(max(b, x) for x in ts))
        assert new <= old
        strictly += new < old
    assert strictly > 0


@pytest.mark.parametrize("state", [s for s in Missingness if s is not Missingness.MEASURED])
def test_an_unmeasured_exposure_is_a_bound_of_one_and_keeps_its_state(state):
    planned = ("a", "b")
    t = gene_test(dec("0.001"), [measured("a", "0.0001"), TransEvidence("b", state)], planned)
    assert t.trans_global == F(2, 10000) and t.measured == 1 and t.unmeasured == 1     # denominator stays 2
    all_missing = gene_test(dec("0.001"), [TransEvidence("a", state), TransEvidence("b", state)], planned)
    assert all_missing.p.value == 1


@pytest.mark.parametrize("records, code", [
    ([measured("a", "0.01")], "exposure_membership"),                                   # b absent
    ([measured("a", "0.01"), measured("b", "0.01"), measured("c", "0.01")], "exposure_membership"),   # extra
    ([measured("a", "0.01"), measured("a", "0.02")], "exposure_membership"),           # duplicate
])
def test_the_exposure_family_must_match_exactly(records, code):
    assert code_of(lambda: gene_test(dec("0.01"), records, ("a", "b"))) == code


def test_measured_and_unmeasured_records_are_well_formed():
    assert code_of(lambda: TransEvidence("a", Missingness.MEASURED)) == "measured_requires_evidence"
    assert code_of(lambda: TransEvidence("a", Missingness.NOT_TESTED, dec("0.5"))) == "unmeasured_has_no_measurement"
    assert code_of(lambda: TransEvidence("a", "measured", dec("0.5"))) == "state"


# ------------------------------------------------------------------ exact Holm

@pytest.mark.parametrize("m, p", [(10, "0.005"), (100, "0.0005"), (5, "0.01")])
def test_decimal_ties_at_the_boundary_are_rejected(m, p):
    assert all(x.reject for x in holm([(str(i), dec(p)) for i in range(m)]))


def test_a_genuine_rational_tie_is_rejected():
    """3 * (1/60) == 1/20 exactly; the float log comparison gets this wrong (measured 2026-10-02)."""
    assert all(x.reject for x in holm([(str(i), P(F(1, 60), "derived", "exact")) for i in range(3)]))


def test_binary_float_near_boundaries_on_both_sides():
    """MEASURED 2026-10-04 (exact rationals of the stored floats): 5 x binary 0.01 lies BELOW binary 0.05 but ABOVE exact
    1/20; 3 x binary 0.01 lies ABOVE both binary 0.03 and exact 3/100. Alpha's representation matters as much as p's."""
    five = [(str(i), P.binary64(0.01, "f")) for i in range(5)]
    three = [(str(i), P.binary64(0.01, "f")) for i in range(3)]
    assert all(x.reject for x in holm(five, F.from_float(0.05)))        # below the stored binary alpha
    assert not holm(five, F(1, 20))[0].reject                            # above the exact alpha
    assert not holm(three, F.from_float(0.03))[0].reject
    assert not holm(three, F(3, 100))[0].reject
    assert holm([(str(i), dec("0.01")) for i in range(3)], F(3, 100))[0].reject                 # decimal: exactly equal


def test_values_immediately_either_side_of_the_boundary():
    eps = F(1, 10**30)
    assert holm([("a", P(F(1, 20) - eps, "derived", "f"))])[0].reject
    assert not holm([("a", P(F(1, 20) + eps, "derived", "f"))])[0].reject


def test_nearly_equal_probabilities_are_ordered_exactly():
    tiny = F(1, 10**40)
    rows = [("z", P(F(1, 100), "derived", "f")), ("a", P(F(1, 100) + tiny, "derived", "f"))]
    assert [d.gene for d in holm(rows)] == ["z", "a"]      # value first; the name would have put "a" first


def test_step_down_blocks_later_rejection():
    assert [x.reject for x in holm([("a", dec(".03")), ("b", dec(".04"))])] == [False, False]
    result = holm([("a", dec(".01")), ("b", dec(".04")), ("c", dec(".001"))])
    assert [(d.gene, d.reject) for d in result] == [("c", True), ("a", True), ("b", True)]


def test_duplicate_or_empty_gene_families_are_refused():
    assert code_of(lambda: holm([("a", dec(".01")), ("a", dec(".01"))])) == "gene_identity"
    assert code_of(lambda: holm([])) == "empty_gene_family"


# ------------------------------------------------------------------ replication

def test_partial_conjunction():
    studies = [("one", dec(".001")), ("two", dec(".02")), ("three", dec(".5"))]
    assert partial_conjunction(studies, 2).value == F(1, 25)
    assert code_of(lambda: partial_conjunction(studies, 4)) == "replication_count"
    assert code_of(lambda: partial_conjunction([("x", dec(".1")), ("x", dec(".2"))], 1)) == "study_identity"


def test_gene_test_result_is_typed():
    t = gene_test(dec("0.2"), [measured("a", "0.01")], ("a",))
    assert type(t) is GeneTest and t.p.representation == "derived"
