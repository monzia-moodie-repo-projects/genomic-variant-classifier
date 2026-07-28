"""The evaluation population contract, and the source identity it addresses.

Ruled 2026-07-27: no numerical kernel may select, filter, normalise or redefine
its evaluation population. Commit 2a enforced that for PREDICTIONS, which fail
closed. Label eligibility could not simply be deleted -- withheld labels are
first-class and selecting on them is a legitimate population decision -- so it
was parked behind a named transitional selector. This is its replacement.

WHAT EACH LAYER ANSWERS
=======================
    population_source_id      WHICH FRAME. Derived from cohort version, the
                              selected partition, and the ordered variant_id
                              sequence. Independent of predictions and of any
                              later restriction.
    membership_fingerprint    WHICH ROWS of that frame. Derived from the source
                              identity and the absolute indices.
    scope                     WHAT THE ROWS ARE CALLED. A name, not an identity:
                              two different row sets may share one.

Cardinality answers none of these. `n = 980` beside `n = 980` says nothing about
whether the same 980 rows were used, which is the defect the fingerprint exists
to reach.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.canonical import CanonicalVariantTable
from genomic_variant_classifier.evaluation.population import (
    EvaluationPopulation,
    PopulationComparison,
    PopulationError,
    PopulationTypeError,
)

SOURCE = "unit-test-frame:sha256:0000000000000000"


def _root(n=10, scope="attempted_cohort", source_id=SOURCE):
    return EvaluationPopulation.full(n, scope=scope, source_id=source_id)


def _table(vids, parts, cohort_version="v2", y=None):
    n = len(vids)
    return CanonicalVariantTable(
        {"variant_id": list(vids),
         "y_true": list(y) if y is not None else [i % 2 for i in range(n)],
         "partition": list(parts),
         "y_score": [0.5] * n},
        cohort_version=cohort_version)


# --------------------------------------------------------------------------- #
# 1. Narrowing, and only narrowing
# --------------------------------------------------------------------------- #
def test_a_root_population_covers_the_whole_frame():
    root = _root(10)
    assert root.n == 10
    assert root.is_complete
    assert root.n_excluded_from_parent == 0
    assert root.n_excluded_from_source == 0
    assert root.restriction_reason is None
    assert root.parent is None


def test_a_restriction_records_its_reason_and_its_parent():
    root = _root(10)
    keep = np.arange(10) < 7
    child = root.restrict(keep, scope="label_eligible", reason="reference_label_withheld")
    assert child.n == 7
    assert child.n_excluded_from_parent == 3
    assert child.n_excluded_from_source == 3
    assert child.restriction_reason == "reference_label_withheld"
    assert child.parent is root
    assert child.indices.tolist() == list(range(7))


def test_lineage_reads_oldest_first_and_carries_every_reason():
    root = _root(10)
    a = root.restrict(np.arange(10) < 8, scope="label_eligible", reason="withheld")
    b = a.restrict(a.take(np.arange(10)) % 2 == 0, scope="even", reason="subgroup")
    lineage = b.lineage()
    assert [s["scope"] for s in lineage] == ["attempted_cohort", "label_eligible", "even"]
    assert [s["reason"] for s in lineage] == [None, "withheld", "subgroup"]
    assert [s["n"] for s in lineage] == [10, 8, 4]
    assert "label_eligible(n=8, -2 withheld)" in b.describe()


def test_a_population_can_never_be_widened():
    """THE LOAD-BEARING INVARIANT. Widening would silently re-admit rows a named
    restriction had already removed, and no downstream assertion could detect it."""
    root = _root(10)
    child = root.restrict(np.arange(10) < 5, scope="half", reason="split")
    # A widening that GROWS is caught by the strict-narrowing guard, which fires
    # before the subset check is reached. The subset check catches the harder
    # case, where the child is smaller yet still re-admits foreign rows -- see
    # test_being_smaller_and_ordered_is_not_enough_to_be_a_subset. Two guards,
    # two failure modes, two tests: asserting one message for both would have
    # meant one of them was never exercised.
    with pytest.raises(PopulationError, match="must strictly narrow"):
        EvaluationPopulation(indices=np.arange(10, dtype=np.int64), scope="wider",
                             n_source=10, source_id=SOURCE,
                             restriction_reason="widening", parent=child)


def test_being_smaller_and_ordered_is_not_enough_to_be_a_subset():
    """Parent [0, 2, 4, 6] with child [1, 3] is smaller, strictly increasing,
    unique and inside the source bounds -- and re-admits rows the parent removed.
    Only an actual subset check catches it."""
    root = _root(8)
    evens = root.restrict(np.arange(8) % 2 == 0, scope="evens", reason="split")
    assert evens.indices.tolist() == [0, 2, 4, 6]
    with pytest.raises(PopulationError, match="not present in the parent"):
        EvaluationPopulation(indices=np.array([1, 3], dtype=np.int64), scope="odds",
                             n_source=8, source_id=SOURCE,
                             restriction_reason="fabricated", parent=evens)


def test_a_restriction_must_remove_at_least_one_row():
    """An unchanged population must not acquire artificial lineage claiming a
    restriction that never occurred."""
    root = _root(10)
    with pytest.raises(PopulationError, match="must remove at least one row"):
        root.restrict(np.ones(10, dtype=bool), scope="label_eligible", reason="withheld")


def test_a_root_must_contain_every_source_row():
    with pytest.raises(PopulationError, match="must contain every"):
        EvaluationPopulation(indices=np.array([0, 1], dtype=np.int64), scope="partial",
                             n_source=10, source_id=SOURCE)


@pytest.mark.parametrize("indices,match", [
    (np.array([0, 1, 1], dtype=np.int64), "duplicates"),
    (np.array([2, 1, 0], dtype=np.int64), "strictly increasing"),
    (np.array([0, 99], dtype=np.int64), r"must lie in"),
])
def test_membership_must_be_a_set_of_rows_inside_the_frame(indices, match):
    with pytest.raises(PopulationError, match=match):
        EvaluationPopulation(indices=indices, scope="s", n_source=10, source_id=SOURCE)


def test_reason_is_required_exactly_when_there_is_a_parent():
    root = _root(10)
    with pytest.raises(PopulationError, match="cannot carry a restriction reason"):
        EvaluationPopulation(indices=np.arange(10, dtype=np.int64), scope="s",
                             n_source=10, source_id=SOURCE, restriction_reason="why")
    with pytest.raises(PopulationError, match="requires a restriction reason"):
        EvaluationPopulation(indices=np.arange(5, dtype=np.int64), scope="s",
                             n_source=10, source_id=SOURCE, parent=root)


# --------------------------------------------------------------------------- #
# 2. Types that would corrupt membership silently
# --------------------------------------------------------------------------- #
def test_float_indices_are_refused_rather_than_truncated():
    """`np.array([1.7], dtype=np.int64)` yields [1] with no error. A truncating
    coercion inside a class that exists to prevent silent membership changes
    would be self-defeating."""
    with pytest.raises(PopulationTypeError, match="integer array"):
        EvaluationPopulation(indices=np.array([1.7, 2.9]), scope="s", n_source=10,
                             source_id=SOURCE)


def test_boolean_indices_are_refused_rather_than_read_as_positions():
    with pytest.raises(PopulationTypeError, match="integer array"):
        EvaluationPopulation(indices=np.array([True, False]), scope="s", n_source=10,
                             source_id=SOURCE)


def test_an_integer_mask_is_refused_even_when_it_holds_only_zero_and_one():
    with pytest.raises(PopulationTypeError, match="must be boolean"):
        _root(10).restrict(np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
                           scope="s", reason="r")


def test_the_mask_is_relative_to_the_population_not_the_source_frame():
    root = _root(10)
    child = root.restrict(np.arange(10) < 6, scope="six", reason="r")
    assert child.n == 6
    with pytest.raises(PopulationError, match="relative to THIS population"):
        child.restrict(np.arange(10) < 3, scope="three", reason="r")


def test_the_type_error_satisfies_both_exception_families():
    """A wrong dtype is genuinely a TypeError and also a contract violation.
    Callers should not have to know which spelling this module chose."""
    assert issubclass(PopulationTypeError, TypeError)
    assert issubclass(PopulationTypeError, ValueError)
    assert issubclass(PopulationTypeError, PopulationError)


# --------------------------------------------------------------------------- #
# 3. Immutability
# --------------------------------------------------------------------------- #
def test_the_caller_cannot_mutate_membership_after_construction():
    """Setting the write flag on a VIEW leaves the writable base reachable, so
    the population must own a copy."""
    base = np.arange(10, dtype=np.int64)
    pop = EvaluationPopulation(indices=base, scope="s", n_source=10, source_id=SOURCE)
    base[0] = 9
    assert pop.indices[0] == 0, "the population aliased the caller's array"
    with pytest.raises(ValueError):
        pop.indices[0] = 5


def test_the_dataclass_is_frozen():
    pop = _root(4)
    with pytest.raises(Exception):
        pop.scope = "renamed"   # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 4. Projection
# --------------------------------------------------------------------------- #
def test_take_projects_from_the_source_frame():
    root = _root(5)
    child = root.restrict(np.array([True, False, True, False, True]),
                          scope="odd_positions", reason="r")
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert child.take(values).tolist() == [10.0, 30.0, 50.0]


def test_take_refuses_an_already_projected_array():
    root = _root(5)
    child = root.restrict(np.array([True, False, True, False, True]),
                          scope="s", reason="r")
    once = child.take(np.arange(5.0))
    with pytest.raises(PopulationError, match="already been projected"):
        child.take(once)


def test_take_refuses_a_scalar():
    with pytest.raises(PopulationError, match="cannot project a scalar"):
        _root(4).take(np.float64(1.0))


# --------------------------------------------------------------------------- #
# 5. Membership fingerprint -- what cardinality cannot reach
# --------------------------------------------------------------------------- #
def test_equal_sized_but_different_subsets_are_distinguishable():
    root = _root(10)
    first = root.restrict(np.arange(10) < 5, scope="cohort", reason="split")
    second = root.restrict(np.arange(10) >= 5, scope="cohort", reason="split")
    assert first.n == second.n == 5
    assert first.scope == second.scope
    assert first.membership_fingerprint != second.membership_fingerprint


def test_the_fingerprint_tracks_rows_not_names():
    """Renaming a population does not change which rows it covers. A fingerprint
    that moved with the name could not support the report invariant that several
    metrics describe the same population."""
    a = _root(10, scope="attempted_cohort")
    b = _root(10, scope="a_different_name")
    assert a.membership_fingerprint == b.membership_fingerprint
    assert a.same_membership_as(b)


def test_identical_indices_over_different_frames_are_not_the_same_population():
    a = _root(10, source_id="frame-a:sha256:1111")
    b = _root(10, source_id="frame-b:sha256:2222")
    assert a.indices.tolist() == b.indices.tolist()
    assert a.membership_fingerprint != b.membership_fingerprint
    assert not a.same_membership_as(b)


def test_the_fingerprint_is_stable_across_repeated_access():
    pop = _root(50)
    assert pop.membership_fingerprint == pop.membership_fingerprint
    assert pop.membership_fingerprint.startswith("sha256:")


def test_a_child_must_inherit_the_source_identity_unchanged():
    root = _root(10)
    with pytest.raises(PopulationError, match="must equal the parent"):
        EvaluationPopulation(indices=np.arange(5, dtype=np.int64), scope="s",
                             n_source=10, source_id="a-different-frame",
                             restriction_reason="r", parent=root)


# --------------------------------------------------------------------------- #
# 6. Source identity derived from the canonical table
# --------------------------------------------------------------------------- #
def test_each_partition_receives_its_own_source_identity():
    """The failure this prevents: a test population and a calibration population
    both occupy indices [0, 1] within their own projections, so without a
    partition-specific identity their fingerprints would coincide."""
    table = _table(["a", "b", "c", "d"], ["test", "cal", "test", "cal"])
    test, cal = table.population_projection("test"), table.population_projection("cal")
    assert test.n == cal.n == 2
    assert test.source_indices.tolist() == [0, 2]
    assert cal.source_indices.tolist() == [1, 3]
    assert test.population_source_id != cal.population_source_id

    pt = EvaluationPopulation.full(test.n, scope="attempted_cohort",
                                   source_id=test.population_source_id)
    pc = EvaluationPopulation.full(cal.n, scope="attempted_cohort",
                                   source_id=cal.population_source_id)
    assert pt.indices.tolist() == pc.indices.tolist() == [0, 1]
    assert pt.membership_fingerprint != pc.membership_fingerprint


@pytest.mark.parametrize("vids,parts,version,expect_same", [
    (["a", "b", "c", "d"], ["test"] * 4, "v2", True),
    (["a", "b", "c", "e"], ["test"] * 4, "v2", False),      # different variants
    (["b", "a", "c", "d"], ["test"] * 4, "v2", False),      # different order
    (["a", "b", "c", "d"], ["test"] * 4, "v3", False),      # different version
])
def test_the_source_identity_discriminates_what_it_must(vids, parts, version, expect_same):
    baseline = _table(["a", "b", "c", "d"], ["test"] * 4).population_projection("test")
    other = _table(vids, parts, cohort_version=version).population_projection("test")
    same = baseline.population_source_id == other.population_source_id
    assert same is expect_same


def test_length_prefixing_prevents_concatenation_ambiguity():
    """['ab', 'c'] and ['a', 'bc'] must never serialise identically."""
    a = _table(["ab", "c", "d", "e"], ["test"] * 4).population_projection("test")
    b = _table(["a", "bc", "d", "e"], ["test"] * 4).population_projection("test")
    assert a.population_source_id != b.population_source_id


def test_a_partition_literally_named_all_cannot_collide_with_the_all_rows_projection():
    table = _table(["a", "b"], ["__all__", "__all__"])
    assert (table.population_projection("__all__").population_source_id
            != table.population_projection(None).population_source_id)


def test_the_source_identity_is_independent_of_model_predictions():
    """The same population evaluated by two models must yield the same
    fingerprint, or paired comparison becomes harder rather than safer."""
    a = CanonicalVariantTable(
        {"variant_id": ["a", "b"], "y_true": [0, 1], "partition": ["test"] * 2,
         "y_score": [0.1, 0.9]}, cohort_version="v2")
    b = CanonicalVariantTable(
        {"variant_id": ["a", "b"], "y_true": [0, 1], "partition": ["test"] * 2,
         "y_score": [0.7, 0.2]}, cohort_version="v2")
    assert (a.population_projection("test").population_source_id
            == b.population_projection("test").population_source_id)


def test_the_projection_does_not_require_a_score_column():
    """A population can be named before it is scored: identity is about which
    variants are evaluated, not what a model predicted for them."""
    table = CanonicalVariantTable(
        {"variant_id": ["a", "b"], "y_true": [0, 1], "partition": ["test"] * 2},
        cohort_version="v2")
    projection = table.population_projection("test")
    assert projection.n == 2
    assert projection.population_source_id.startswith("canonical-variant-table:sha256:")
    with pytest.raises(ValueError, match="requires a 'y_score' column"):
        table.arrays("test")


def test_an_absent_partition_is_refused_rather_than_returning_an_empty_frame():
    table = _table(["a", "b"], ["test", "test"])
    with pytest.raises(ValueError, match="not present"):
        table.population_projection("calibration")


def test_the_projection_is_memoised_per_partition():
    table = _table(["a", "b", "c", "d"], ["test", "cal", "test", "cal"])
    assert table.population_projection("test") is table.population_projection("test")
    assert table.population_projection("test") is not table.population_projection("cal")


# --------------------------------------------------------------------------- #
# 7. End to end
# --------------------------------------------------------------------------- #
def test_the_seam_sequence_produces_an_auditable_population():
    table = _table([f"v{i}" for i in range(10)], ["test"] * 10,
                   y=[0, 1, 0, 1, 0, 1, 0, 1, None, None])
    projection = table.population_projection("test")
    attempted = EvaluationPopulation.full(projection.n, scope="attempted_cohort",
                                          source_id=projection.population_source_id)
    y = table.arrays("test").y
    mask = np.isfinite(y)
    population = attempted if mask.all() else attempted.restrict(
        mask, scope="label_eligible", reason="reference_label_withheld")

    assert attempted.n == 10
    assert population.n == 8
    assert population.n_excluded_from_parent == 2
    assert population.take(y).shape[0] == population.n
    assert np.isfinite(population.take(y)).all()
    assert "reference_label_withheld" in population.describe()


def test_a_fully_labelled_cohort_acquires_no_lineage():
    """The `mask.all()` branch is not a workaround for the strict-narrowing rule.
    It prevents a false claim that a restriction occurred."""
    table = _table([f"v{i}" for i in range(4)], ["test"] * 4, y=[0, 1, 0, 1])
    projection = table.population_projection("test")
    attempted = EvaluationPopulation.full(projection.n, scope="attempted_cohort",
                                          source_id=projection.population_source_id)
    mask = np.isfinite(table.arrays("test").y)
    assert mask.all()
    population = attempted if mask.all() else attempted.restrict(
        mask, scope="label_eligible", reason="reference_label_withheld")
    assert population is attempted
    assert population.restriction_reason is None
    assert len(population.lineage()) == 1


# --------------------------------------------------------------------------- #
# 8. The context must carry arrays that match its population
# --------------------------------------------------------------------------- #
def test_a_context_refuses_arrays_that_do_not_match_its_population():
    """CLOSES A GAP THE SABOTAGE MATRIX FOUND.

    The arrays in a MetricContext are ALREADY PROJECTED, so they carry the length
    of the population, not of the source frame. Without this check, unprojected
    arrays would be computed over in full while the result reported the narrower
    population count -- the exact divergence between a number and its stated
    denominator that this whole stack exists to remove, reintroduced one layer up.
    """
    from genomic_variant_classifier.evaluation.registry import MetricContext

    y = np.array([0.0, 1.0, np.nan, 1.0, 0.0])
    attempted = EvaluationPopulation.full(5, scope="attempted_cohort", source_id=SOURCE)
    eligible = attempted.restrict(np.isfinite(y), scope="label_eligible",
                                  reason="reference_label_withheld")
    assert eligible.n == 4

    with pytest.raises(ValueError, match="ALREADY\\s+PROJECTED|already\\s+projected"):
        MetricContext(y_true=y, y_score=y, population=eligible)

    projected = eligible.take(y)
    ctx = MetricContext(y_true=projected, y_score=projected, population=eligible)
    assert ctx.n == 4
    assert ctx.population_scope == "label_eligible"


def test_a_context_derives_its_scope_from_the_population():
    """Scope is not stored beside the population. Two sources of truth for one
    fact eventually disagree, and the artifact could not say which was right."""
    from genomic_variant_classifier.evaluation.registry import MetricContext

    pop = EvaluationPopulation.full(4, scope="attempted_cohort", source_id=SOURCE)
    ctx = MetricContext(y_true=np.array([0.0, 1.0, 0.0, 1.0]), population=pop)
    assert ctx.population_scope == pop.scope
    assert not hasattr(type(ctx), "__dataclass_fields__") or \
        "population_scope" not in type(ctx).__dataclass_fields__, (
            "population_scope is still a stored field; it must be derived")


# --------------------------------------------------------------------------- #
# 9. Attribution: an identity that is absent, never faked
#
# `evaluate()` receives arrays, not a canonical table, so it has no source
# identity to give a population. Three ways of inventing one were measured and
# all three fail:
#
#   a fixed sentinel   two DIFFERENT equal-length cohorts share a fingerprint,
#                      certifying an equivalence nobody established -- the exact
#                      defect the fingerprint exists to prevent
#   derived from y_true  ruled out 2026-07-27: a label-policy change must be
#                      visible through cohort_version, not embedded in an opaque
#                      row digest
#   per-call unique id safe but NON-DETERMINISTIC, breaking reproducibility and
#                      every byte-identity oracle in this project
#
# So attribution is optional and absence is represented as absence.
# --------------------------------------------------------------------------- #
def test_a_population_may_be_explicitly_unattributed():
    pop = EvaluationPopulation.full(80, scope="attempted_cohort", source_id=None)
    assert pop.source_id is None
    assert pop.is_attributed is False
    assert pop.n == 80


def test_an_unattributed_population_has_NO_fingerprint():
    """Not a fingerprint of nothing -- the ABSENCE of one. A digest here would
    let `a.fingerprint == b.fingerprint` answer True for two populations whose
    equivalence is unknown."""
    pop = EvaluationPopulation.full(80, scope="attempted_cohort", source_id=None)
    assert pop.membership_fingerprint is None


def test_a_blank_source_id_is_still_refused():
    """None states 'unattributed'. A blank string states nothing at all, and
    admitting it would give two ways to spell absence, one of them accidental."""
    with pytest.raises(PopulationError, match="or None to state explicitly"):
        EvaluationPopulation.full(10, scope="s", source_id="   ")


def test_two_unattributed_populations_cannot_be_proven_equal():
    """THE POINT OF THE WHOLE DESIGN.

    Equal-size unattributed calls genuinely cannot be distinguished -- that is an
    epistemic limit, not a defect. What matters is that the implementation does
    not convert the limit into a FALSE EQUALITY. Comparison returns UNKNOWN,
    which is the true answer, rather than SAME, which would be a claim.
    """
    first = EvaluationPopulation.full(4, scope="cohort", source_id=None)
    second = EvaluationPopulation.full(4, scope="cohort", source_id=None)
    assert first.compare_membership(second) is PopulationComparison.UNKNOWN
    assert second.compare_membership(first) is PopulationComparison.UNKNOWN
    assert first.same_membership_as(second) is False


def test_comparison_is_three_valued_and_each_value_is_reachable():
    """A boolean cannot express 'not knowable'. Collapsing UNKNOWN into False
    would read as 'different rows', which is itself a claim."""
    a = EvaluationPopulation.full(10, scope="c", source_id="frame:1")
    b = EvaluationPopulation.full(10, scope="c", source_id="frame:1")
    c = EvaluationPopulation.full(10, scope="c", source_id="frame:2")
    u = EvaluationPopulation.full(10, scope="c", source_id=None)

    assert a.compare_membership(b) is PopulationComparison.SAME
    assert a.compare_membership(c) is PopulationComparison.DIFFERENT
    assert a.compare_membership(u) is PopulationComparison.UNKNOWN
    assert u.compare_membership(a) is PopulationComparison.UNKNOWN
    assert {v.value for v in PopulationComparison} == {"same", "different", "unknown"}


def test_direct_fingerprint_comparison_cannot_be_used_as_evidence():
    """`None == None` is True in Python. A caller comparing two absent
    fingerprints directly would conclude sameness. `compare_membership` is the
    only comparison that gets this right, which is why it exists."""
    first = EvaluationPopulation.full(4, scope="c", source_id=None)
    second = EvaluationPopulation.full(4, scope="c", source_id=None)

    naive = first.membership_fingerprint == second.membership_fingerprint
    assert naive is True, "None == None; this is exactly the trap"
    assert first.compare_membership(second) is PopulationComparison.UNKNOWN, (
        "the authoritative comparator must not repeat the naive answer")


def test_attribution_is_inherited_by_restriction():
    mask = np.arange(10) < 6
    unattributed = EvaluationPopulation.full(10, scope="attempted", source_id=None)
    child = unattributed.restrict(mask, scope="label_eligible", reason="withheld")
    assert child.source_id is None
    assert child.is_attributed is False
    assert child.membership_fingerprint is None

    attributed = EvaluationPopulation.full(10, scope="attempted", source_id=SOURCE)
    kid = attributed.restrict(mask, scope="label_eligible", reason="withheld")
    assert kid.is_attributed is True
    assert kid.membership_fingerprint is not None


def test_a_restriction_cannot_invent_an_identity_its_parent_lacked():
    root = EvaluationPopulation.full(10, scope="attempted", source_id=None)
    with pytest.raises(PopulationError, match="must equal the parent"):
        EvaluationPopulation(indices=np.arange(6, dtype=np.int64), scope="child",
                             n_source=10, source_id=SOURCE,
                             restriction_reason="r", parent=root)


def test_an_attributed_restriction_cannot_discard_its_identity():
    root = EvaluationPopulation.full(10, scope="attempted", source_id=SOURCE)
    with pytest.raises(PopulationError, match="must equal the parent"):
        EvaluationPopulation(indices=np.arange(6, dtype=np.int64), scope="child",
                             n_source=10, source_id=None,
                             restriction_reason="r", parent=root)


def test_comparison_refuses_a_non_population():
    pop = EvaluationPopulation.full(4, scope="c", source_id=SOURCE)
    with pytest.raises(TypeError, match="only compare against another"):
        pop.compare_membership("not a population")

