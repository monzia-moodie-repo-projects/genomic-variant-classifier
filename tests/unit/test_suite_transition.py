"""A suite change is a change of identity. Proven, not assumed.

SUITE-NEUTRAL-IDENTITY-1, 2026-08-22.

WHY
---
ADR-0003 established that a count is not an identity. The ADDITION installers
honoured it and compared node-identity sets. The NEUTRAL installers did not:
they verified `collected == expected` and `ratchet == collected` and nothing
more, so a transition removing one test and adding another would have passed.

Two units were published under that weaker check -- `e1a5297` and `ba9060d` --
whose commit messages state the neutral transition was "verified inside the
transaction, not assumed". The claim was stronger than the check. That is the
same shape as an acceptance line recording zeroes because it was rendered before
the gate ran: a true-looking record produced by a check that could not have
established it.

WHAT THIS FILE PROVES
---------------------
That `suite_transition` REFUSES every malformed transition, not merely that it
accepts well-formed ones. Fourteen of the twenty-two tests below are negative
controls. The first of them reproduces the exact defect:

    before  {test_a, test_b, test_c}
    after   {test_a, test_b, test_d}

equal counts, different suite. The old check accepts it. This one must not.

Author: Monzia Moodie
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.transactions.suite_transition import (
    InvariantMigration,
    SuiteSnapshot,
    SuiteTransition,
    SuiteTransitionError,
    SuiteTransitionKind,
    suite_digest,
)

A, B, C, D = ("t.py::test_a", "t.py::test_b", "t.py::test_c", "t.py::test_d")


def snap(*ids: str) -> SuiteSnapshot:
    return SuiteSnapshot(frozenset(ids))


# ---------------------------------------------------------------------------
# 1. NEUTRAL -- the defect this module exists for
# ---------------------------------------------------------------------------

def test_neutral_refuses_a_swap_that_preserves_the_count():
    """THE DEFECT. Equal counts, different membership. The old check accepted it."""
    before, after = snap(A, B, C), snap(A, B, D)
    assert before.count == after.count, "fixture precondition: counts are equal"
    with pytest.raises(SuiteTransitionError) as exc:
        SuiteTransition(kind=SuiteTransitionKind.NEUTRAL).verify(before, after)
    assert "test_d" in str(exc.value), (
        "the refusal must NAME the identity, not merely report a mismatch"
    )


def test_neutral_accepts_an_unchanged_suite():
    before = after = snap(A, B, C)
    ev = SuiteTransition(kind=SuiteTransitionKind.NEUTRAL).verify(before, after)
    assert ev.added_nodeids == () and ev.removed_nodeids == ()
    assert ev.before_digest == ev.after_digest
    assert ev.before_count == ev.after_count == 3


def test_neutral_refuses_a_pure_addition():
    with pytest.raises(SuiteTransitionError):
        SuiteTransition(kind=SuiteTransitionKind.NEUTRAL).verify(
            snap(A, B), snap(A, B, C))


def test_neutral_refuses_a_pure_removal():
    with pytest.raises(SuiteTransitionError):
        SuiteTransition(kind=SuiteTransitionKind.NEUTRAL).verify(
            snap(A, B, C), snap(A, B))


# ---------------------------------------------------------------------------
# 2. ADDITION
# ---------------------------------------------------------------------------

def test_addition_accepts_exactly_the_declared_identities():
    ev = SuiteTransition(kind=SuiteTransitionKind.ADDITION,
                         expected_added_nodeids=frozenset({C})).verify(
        snap(A, B), snap(A, B, C))
    assert ev.added_nodeids == (C,)
    assert ev.after_count - ev.before_count == 1


def test_addition_refuses_an_undeclared_extra():
    """Two tests appear where one was declared. The count is +2 either way."""
    with pytest.raises(SuiteTransitionError) as exc:
        SuiteTransition(kind=SuiteTransitionKind.ADDITION,
                        expected_added_nodeids=frozenset({C})).verify(
            snap(A), snap(A, B, C))
    assert "test_b" in str(exc.value)


def test_addition_refuses_a_declared_identity_that_did_not_appear():
    with pytest.raises(SuiteTransitionError) as exc:
        SuiteTransition(kind=SuiteTransitionKind.ADDITION,
                        expected_added_nodeids=frozenset({C, D})).verify(
            snap(A), snap(A, C))
    assert "test_d" in str(exc.value)


def test_addition_refuses_a_silent_removal_alongside_the_addition():
    """The count still rises. Only identity detects the loss."""
    with pytest.raises(SuiteTransitionError) as exc:
        SuiteTransition(kind=SuiteTransitionKind.ADDITION,
                        expected_added_nodeids=frozenset({C, D})).verify(
            snap(A, B), snap(A, C, D))
    assert "test_b" in str(exc.value)


# ---------------------------------------------------------------------------
# 3. DELIBERATE_RETIREMENT
# ---------------------------------------------------------------------------

def test_retirement_accepts_exactly_the_declared_removals():
    ev = SuiteTransition(
        kind=SuiteTransitionKind.DELIBERATE_RETIREMENT,
        expected_removed_nodeids=frozenset({C}),
        justification="superseded by a domain-level owner",
    ).verify(snap(A, B, C), snap(A, B))
    assert ev.removed_nodeids == (C,)
    assert ev.after_count < ev.before_count


def test_retirement_refuses_an_undeclared_removal():
    with pytest.raises(SuiteTransitionError) as exc:
        SuiteTransition(
            kind=SuiteTransitionKind.DELIBERATE_RETIREMENT,
            expected_removed_nodeids=frozenset({C}),
            justification="one only",
        ).verify(snap(A, B, C), snap(A))
    assert "test_b" in str(exc.value)


def test_retirement_carries_its_invariant_migrations():
    mig = InvariantMigration(
        invariant_id="INV-MODEL-ROSTER-COMPLETENESS",
        old_owners=("tests/unit/test_readme_claims.py::x",),
        new_owners=("tests/unit/test_invariant_ownership.py::y",),
        proof_test="test_the_roster_check_detects_a_silently_dropped_model")
    t = SuiteTransition(
        kind=SuiteTransitionKind.DELIBERATE_RETIREMENT,
        expected_removed_nodeids=frozenset({C}),
        invariant_migrations=(mig,),
        justification="relocated to the domain boundary")
    assert t.invariant_migrations[0].invariant_id.startswith("INV-")


# ---------------------------------------------------------------------------
# 4. Construction-time refusals -- a malformed declaration never reaches verify
# ---------------------------------------------------------------------------

def test_neutral_may_not_declare_additions():
    with pytest.raises(SuiteTransitionError):
        SuiteTransition(kind=SuiteTransitionKind.NEUTRAL,
                        expected_added_nodeids=frozenset({C}))


def test_neutral_may_not_declare_removals():
    with pytest.raises(SuiteTransitionError):
        SuiteTransition(kind=SuiteTransitionKind.NEUTRAL,
                        expected_removed_nodeids=frozenset({C}))


def test_addition_must_name_what_it_adds():
    with pytest.raises(SuiteTransitionError):
        SuiteTransition(kind=SuiteTransitionKind.ADDITION)


def test_addition_may_not_declare_a_removal():
    with pytest.raises(SuiteTransitionError):
        SuiteTransition(kind=SuiteTransitionKind.ADDITION,
                        expected_added_nodeids=frozenset({C}),
                        expected_removed_nodeids=frozenset({D}))


def test_retirement_must_name_what_it_retires():
    with pytest.raises(SuiteTransitionError):
        SuiteTransition(kind=SuiteTransitionKind.DELIBERATE_RETIREMENT,
                        justification="because")


def test_retirement_requires_a_justification():
    """The ratchet catches ACCIDENTAL loss; a deliberate one must say why."""
    for blank in ("", "   ", "\n"):
        with pytest.raises(SuiteTransitionError):
            SuiteTransition(kind=SuiteTransitionKind.DELIBERATE_RETIREMENT,
                            expected_removed_nodeids=frozenset({C}),
                            justification=blank)


def test_an_identity_may_not_be_both_added_and_removed():
    with pytest.raises(SuiteTransitionError):
        SuiteTransition(kind=SuiteTransitionKind.DELIBERATE_RETIREMENT,
                        expected_added_nodeids=frozenset({C}),
                        expected_removed_nodeids=frozenset({C}),
                        justification="contradictory")


def test_an_invariant_may_not_migrate_to_no_owner():
    """That is removal, not migration. INVARIANT-HANDOFF-1."""
    with pytest.raises(SuiteTransitionError):
        InvariantMigration(invariant_id="INV-X", old_owners=("a",),
                           new_owners=(), proof_test="t")


# ---------------------------------------------------------------------------
# 5. Snapshots refuse to be built from the wrong thing
# ---------------------------------------------------------------------------

def test_a_snapshot_refuses_lines_that_are_not_node_identities():
    with pytest.raises(SuiteTransitionError):
        SuiteSnapshot(frozenset({"5237 tests collected", A}))


def test_parsing_refuses_when_the_listing_and_the_summary_disagree():
    """Two measurements of one quantity. If they differ, neither may be used."""
    with pytest.raises(SuiteTransitionError) as exc:
        SuiteSnapshot.from_pytest_output(
            "t.py::test_a\nt.py::test_b\n\n3 tests collected in 0.1s\n")
    assert "disagree" in str(exc.value)


def test_parsing_refuses_a_listing_with_no_summary_witness():
    with pytest.raises(SuiteTransitionError):
        SuiteSnapshot.from_pytest_output("t.py::test_a\nt.py::test_b\n")


def test_parsing_accepts_a_consistent_collection():
    s = SuiteSnapshot.from_pytest_output(
        "t.py::test_a\nt.py::test_b\n\n2 tests collected in 0.10s\n")
    assert s.count == 2 and A in s.nodeids


# ---------------------------------------------------------------------------
# 6. The digest is an identity proof, not a size proof
# ---------------------------------------------------------------------------

def test_the_digest_distinguishes_suites_of_the_same_size():
    assert suite_digest(frozenset({A, B, C})) != suite_digest(frozenset({A, B, D}))


def test_the_digest_is_independent_of_collection_order():
    assert suite_digest(frozenset({A, B})) == suite_digest(frozenset({B, A}))
