"""The identity-law kernel, sabotaged against its own most dangerous refinement.

Phase 1C Unit 3A++.4c. Created 2026-09-03.

The kernel's most important invariant is NEGATIVE: it must not normalise. The
dangerous future edit is a one-line "robustness" improvement --

    token = token.strip().lower()

-- which would make these tests pass more often while silently changing what
scientific identity means. The law-authority census found exactly that
semantics in `partitions_equivalent`, which collapses `None`, `""`, whitespace
and `NaN`; those are correct for cluster resolution and wrong for provenance.

So this file sabotages the kernel from outside: distinct-but-similar tokens
must stay distinct, and equal-but-not-identical tokens must compare equal.

Author: Monzia Moodie
"""
from __future__ import annotations

import pytest

from tests.support.identity_laws import (
    _pairwise_relation,
    assert_all_identities_distinct,
    assert_identity_equivalence_preserved,
    assert_orthogonal_change,
)


# ---------------------------------------------------------------------------
# 1. NO NORMALISATION -- the invariant the census exists to protect
# ---------------------------------------------------------------------------

def test_the_laws_do_NOT_strip_or_casefold():
    """`token.strip().lower()` inside the kernel must fail this."""
    assert_all_identities_distinct({
        "exact": "ABC", "case": "abc", "leading": " ABC", "trailing": "ABC ",
        "inner": "A BC",
    })


def test_the_laws_do_NOT_collapse_missing_LIKE_strings():
    """`None`, `""` and whitespace are collapsed by `partitions_equivalent`.

    They are DIFFERENT scientific values here. The empty string is refused
    outright as a token, so the near-misses are what must stay distinct.
    """
    assert_all_identities_distinct({
        "none_word": "None", "none_lower": "none", "null_word": "null",
        "nan_word": "NaN", "space": " ", "zero": "0",
    })


def test_a_token_must_be_a_NON_EMPTY_STRING():
    for bad in ({"a": ""}, {"a": None}, {"a": 0}, {"a": b"x"}, {"a": ["x"]}):
        with pytest.raises(TypeError):
            assert_all_identities_distinct(bad)


def test_a_LABEL_must_be_a_non_empty_string():
    with pytest.raises(TypeError):
        assert_all_identities_distinct({"": "x"})
    with pytest.raises(TypeError):
        assert_all_identities_distinct({7: "x"})


def test_an_EMPTY_case_set_is_refused():
    with pytest.raises(ValueError):
        assert_all_identities_distinct({})


def test_a_NON_MAPPING_is_refused():
    with pytest.raises(TypeError):
        assert_all_identities_distinct(["a", "b"])


# ---------------------------------------------------------------------------
# 2. EQUALITY, NOT OBJECT IDENTITY
# ---------------------------------------------------------------------------

def test_equal_but_DISTINCT_string_objects_compare_equal():
    """Three times in this arc a defect hid behind accidental object identity:
    a reused singleton, a short interned literal, and a constant-folded
    `"a" * 64`. Build the tokens at runtime so `is` cannot hold."""
    left = "".join(["a", "b", "c"])
    right = "".join(["ab", "c"])
    assert left == right and left is not right
    with pytest.raises(AssertionError, match="one == two"):
        assert_all_identities_distinct({"one": left, "two": right})


def test_the_pairwise_relation_is_symmetric_and_complete():
    relation = _pairwise_relation({"a": "x", "b": "x", "c": "y"})
    assert relation == {("a", "b"): True, ("a", "c"): False,
                        ("b", "c"): False}
    assert len(relation) == 3


# ---------------------------------------------------------------------------
# 3. EQUIVALENCE PRESERVATION
# ---------------------------------------------------------------------------

def test_values_may_move_while_the_relation_holds():
    """The v5 migration in miniature: every value changed, no class did."""
    assert_identity_equivalence_preserved(
        before={"a": "x", "b": "y", "c": "x"},
        after={"a": "p", "b": "q", "c": "p"})


def test_a_FALSE_MERGE_is_detected():
    with pytest.raises(AssertionError, match=r"a != b became =="):
        assert_identity_equivalence_preserved(
            before={"a": "x", "b": "y"}, after={"a": "z", "b": "z"})


def test_a_FALSE_SPLIT_is_detected():
    with pytest.raises(AssertionError, match=r"a == b became !="):
        assert_identity_equivalence_preserved(
            before={"a": "x", "b": "x"}, after={"a": "p", "b": "q"})


def test_ALL_VALUES_CHANGED_is_not_sufficient():
    """Both digests moved AND the migration created a collision."""
    with pytest.raises(AssertionError):
        assert_identity_equivalence_preserved(
            before={"a": "x", "b": "y"}, after={"a": "z", "b": "z"})


def test_the_CLASS_COUNT_alone_is_not_sufficient():
    """Two partitions of four cases, both with two classes, but different."""
    with pytest.raises(AssertionError):
        assert_identity_equivalence_preserved(
            before={"a": "1", "b": "1", "c": "2", "d": "2"},
            after={"a": "1", "b": "2", "c": "1", "d": "2"})


def test_a_POPULATION_change_is_refused_not_intersected():
    """Silently intersecting would turn a missing case into a passing test."""
    with pytest.raises(AssertionError, match="population changed"):
        assert_identity_equivalence_preserved(
            before={"a": "x", "b": "y"}, after={"a": "p"})
    with pytest.raises(AssertionError, match="population changed"):
        assert_identity_equivalence_preserved(
            before={"a": "x"}, after={"a": "p", "b": "q"})


def test_a_RENAMED_case_is_a_population_change():
    with pytest.raises(AssertionError, match="population changed"):
        assert_identity_equivalence_preserved(
            before={"a": "x", "b": "y"}, after={"a": "x", "B": "y"})


# ---------------------------------------------------------------------------
# 4. DISCRIMINATION
# ---------------------------------------------------------------------------

def test_distinct_cases_pass():
    assert_all_identities_distinct({"a": "x", "b": "y", "c": "z"})


def test_a_COLLISION_names_the_scientific_cases():
    with pytest.raises(AssertionError, match=r"clinvar_grch37 == clinvar_grch38"):
        assert_all_identities_distinct({
            "clinvar_grch37": "x", "clinvar_grch38": "x"})


def test_a_TOTAL_collapse_reports_every_pair():
    """The measured sabotage: one payload, every manifest identical."""
    with pytest.raises(AssertionError) as exc:
        assert_all_identities_distinct({"a": "x", "b": "x", "c": "x"})
    message = str(exc.value)
    for pair in ("a == b", "a == c", "b == c"):
        assert pair in message


# ---------------------------------------------------------------------------
# 5. ORTHOGONALITY -- both directions
# ---------------------------------------------------------------------------

def test_exactly_the_authorized_family_moves():
    assert_orthogonal_change(
        before={"source_evidence": "aaa", "transformation": "bbb"},
        after={"source_evidence": "ccc", "transformation": "bbb"},
        changed=frozenset({"source_evidence"}))

    # Section 4: an EMPTY frozenset is valid and means no family may move.
    assert_orthogonal_change(
        before={"source_evidence": "aaa", "transformation": "bbb"},
        after={"source_evidence": "aaa", "transformation": "bbb"},
        changed=frozenset())

    # ...and it is an exact expectation, so a move under an empty set fails.
    with pytest.raises(AssertionError, match="should NOT have"):
        assert_orthogonal_change(
            before={"source_evidence": "aaa", "transformation": "bbb"},
            after={"source_evidence": "ccc", "transformation": "bbb"},
            changed=frozenset())


def test_a_PROTECTED_family_that_moved_is_detected():
    with pytest.raises(AssertionError, match="should NOT have"):
        assert_orthogonal_change(
            before={"source_evidence": "aaa", "transformation": "bbb"},
            after={"source_evidence": "ccc", "transformation": "ddd"},
            changed=frozenset({"source_evidence"}))


def test_a_NO_OP_migration_is_detected():
    """Asserting only that the protected family held would pass this."""
    with pytest.raises(AssertionError, match="did NOT"):
        assert_orthogonal_change(
            before={"source_evidence": "aaa", "transformation": "bbb"},
            after={"source_evidence": "aaa", "transformation": "bbb"},
            changed=frozenset({"source_evidence"}))


def test_an_unknown_family_name_is_refused():
    with pytest.raises(ValueError, match="unknown identity families"):
        assert_orthogonal_change(
            before={"a": "x"}, after={"a": "y"},
            changed=frozenset({"nonexistent"}))


def test_a_family_POPULATION_change_is_refused():
    with pytest.raises(AssertionError, match="population changed"):
        assert_orthogonal_change(
            before={"a": "x", "b": "y"}, after={"a": "z"},
            changed=frozenset({"a"}))


def test_changed_must_be_a_FROZENSET_of_NON_EMPTY_STRINGS():
    """Section 4: exactly `frozenset`, and every member a non-empty string.

    A mutable `set` is refused even though `frozenset({"a"}) == {"a"}` is True
    in Python -- which is precisely why accepting one would be invisible to a
    value comparison and must be refused by TYPE.
    """
    for bad_container in ({"a"}, ["a"], ("a",), "a", None):
        with pytest.raises(TypeError, match="must be a frozenset"):
            assert_orthogonal_change(
                before={"a": "x"}, after={"a": "y"},
                changed=bad_container)

    for bad_members in (frozenset({""}), frozenset({1}), frozenset({b"a"}),
                        frozenset({"a", 1})):
        with pytest.raises(
                TypeError,
                match="family names must be non-empty strings"):
            assert_orthogonal_change(
                before={"a": "x"}, after={"a": "y"},
                changed=bad_members)
