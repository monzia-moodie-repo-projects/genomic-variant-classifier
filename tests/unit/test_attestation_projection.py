"""An installer may request a record; it may not author one.

PROOF-AFTER-IRREVERSIBILITY-1. Created 2026-08-25.

WHAT THIS GUARDS
----------------
On 2026-08-25 the DRIFT-1 installer committed `abcb22e` and then refused:

    ATTESTATION INVALID AFTER A SUCCESSFUL COMMIT:
        a deliberate retirement requires a justification

The omission was not carelessness. MEASURED: `TransitionEvidence` carries
`kind`, counts, digests and OBSERVED identities. `SuiteTransition` owns the
EXPECTED identities and the `justification`. The installer hand-built the
attestation's `suite_transition` record mostly from the evidence object, which
has no structural route to `justification` at all.

Three vocabularies were being joined by hand, and they genuinely differ:

    PlannedTarget.as_record   relpath, action, sha256, size, reason
    attestation target        path,    action, post_sha256, post_size

    TransitionEvidence.as_record   kind, before{count,digest},
                                   after{count,digest}, added_nodeids,
                                   removed_nodeids
    attestation suite_transition   kind, expected_added_nodeids,
                                   expected_removed_nodeids,
                                   observed_added_nodeids,
                                   observed_removed_nodeids, before_count,
                                   after_count, before_digest, after_digest
                                   [+ justification]

`_exact_keys` refuses unknown keys as hard as missing ones, so neither
`as_record` output can be used directly. Every installer performed the
translation by hand. One of them dropped a field, after the commit.

WHY THE PROJECTION LIVES ON THE DECLARATION
-------------------------------------------
The attestation record is a JOIN of declaration and observation. Putting it on
the evidence would read as though evidence owned it and merely needed a
declaration supplied -- and would permit `evidence_from_a.as_attestation_record(
declaration_b)`. The pairing guards make that unconstructible instead, and they
are REACHABLE, unlike the three checks this module deleted from `verify()` as
provably unreachable.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from genomic_variant_classifier.transactions.install_attestation import (
    OPTIONAL_TRANSITION,
    REQUIRED_TARGET,
    REQUIRED_TRANSITION,
)
from genomic_variant_classifier.transactions.install_plan import (
    PlannedTarget,
    TargetAction,
)
from genomic_variant_classifier.transactions.suite_transition import (
    SuiteSnapshot,
    SuiteTransition,
    SuiteTransitionError,
    SuiteTransitionKind,
    TransitionEvidence,
)

ADDED = frozenset({"tests/unit/test_a.py::test_one",
                   "tests/unit/test_a.py::test_two"})
REMOVED = frozenset({"tests/unit/test_b.py::test_gone"})
KEPT = "tests/unit/test_c.py::test_kept"


def a_retirement(**over):
    kw = {"kind": SuiteTransitionKind.DELIBERATE_RETIREMENT,
          "expected_added_nodeids": ADDED,
          "expected_removed_nodeids": REMOVED,
          "justification": "sixteen identities renamed by a workflow edit"}
    kw.update(over)
    return SuiteTransition(**kw)


def verified(transition):
    before = SuiteSnapshot(frozenset(transition.expected_removed_nodeids)
                           | {KEPT})
    after = SuiteSnapshot(frozenset(transition.expected_added_nodeids) | {KEPT})
    return transition.verify(before, after)


def refuses(fn, fragment):
    with pytest.raises(SuiteTransitionError) as exc:
        fn()
    assert fragment in str(exc.value), (
        "refused, but on the WRONG check.\n  expected: {!r}\n  actual: {}"
        .format(fragment, exc.value))


# ---------------------------------------------------------------------------
# 1. THE PROJECTION SATISFIES THE ATTESTATION CONTRACT EXACTLY
# ---------------------------------------------------------------------------

def test_the_transition_projection_has_exactly_the_required_keys():
    """Not "roughly the right shape". `_exact_keys` refuses unknown keys as
    hard as missing ones, so this must be exact in both directions."""
    transition = a_retirement()
    record = transition.as_attestation_record(verified(transition))
    keys = set(record)
    assert not REQUIRED_TRANSITION - keys, sorted(REQUIRED_TRANSITION - keys)
    assert not keys - REQUIRED_TRANSITION - OPTIONAL_TRANSITION, sorted(
        keys - REQUIRED_TRANSITION - OPTIONAL_TRANSITION)


def test_a_deliberate_retirement_projects_its_justification():
    """The field whose absence produced PROOF-AFTER-IRREVERSIBILITY-1.

    It was unreachable from the object the installer was serialising. It is
    reachable from the object that OWNS it.
    """
    transition = a_retirement()
    record = transition.as_attestation_record(verified(transition))
    assert record["justification"] == transition.justification
    assert record["justification"].strip(), "a blank justification is not one"


def test_an_addition_does_not_project_a_justification():
    """`justification` is OPTIONAL_TRANSITION. Emitting an empty one for a kind
    that has no use for it would put an undeclared-looking field in every
    addition attestation."""
    transition = SuiteTransition(kind=SuiteTransitionKind.ADDITION,
                                 expected_added_nodeids=ADDED)
    before = SuiteSnapshot(frozenset({KEPT}))
    after = SuiteSnapshot(frozenset(ADDED) | {KEPT})
    record = transition.as_attestation_record(transition.verify(before, after))
    assert "justification" not in record


def test_the_projection_separates_expected_from_observed():
    """Four distinct keys, not two. A record collapsing them could not show a
    declaration and its evidence disagreeing -- which is the entire point of
    verifying by identity."""
    transition = a_retirement()
    record = transition.as_attestation_record(verified(transition))
    assert set(record["expected_added_nodeids"]) == ADDED
    assert set(record["observed_added_nodeids"]) == ADDED
    assert set(record["expected_removed_nodeids"]) == REMOVED
    assert set(record["observed_removed_nodeids"]) == REMOVED


def test_the_target_projection_has_exactly_the_required_keys():
    target = PlannedTarget(relpath="README.md", action=TargetAction.PATCH,
                           payload=b"x\n", reason="a reason")
    record = target.as_attestation_record()
    assert set(record) == REQUIRED_TARGET, sorted(set(record) ^ REQUIRED_TARGET)
    assert record["path"] == "README.md"
    assert record["post_sha256"] == target.digest
    assert record["post_size"] == 2


def test_the_target_projection_is_not_the_plan_record():
    """Two contracts, deliberately different. Overloading `as_record` would
    make one of its two callers silently wrong."""
    target = PlannedTarget(relpath="README.md", action=TargetAction.PATCH,
                           payload=b"x\n", reason="a reason")
    plan_record = target.as_record()
    attestation_record = target.as_attestation_record()
    assert plan_record != attestation_record
    assert set(plan_record) != set(attestation_record)
    assert "reason" in plan_record and "reason" not in attestation_record
    assert "relpath" in plan_record and "path" in attestation_record


# ---------------------------------------------------------------------------
# 2. THE PAIRING GUARDS ARE REACHABLE
# ---------------------------------------------------------------------------

def test_evidence_of_a_different_kind_is_refused():
    """Reachable, unlike the three checks deleted from `verify()`.

    `as_attestation_record` is PUBLIC and may be handed evidence built
    elsewhere, so this can fire -- which is the standard this module set when
    it removed defence that could not.
    """
    addition = SuiteTransition(kind=SuiteTransitionKind.ADDITION,
                               expected_added_nodeids=ADDED)
    before = SuiteSnapshot(frozenset({KEPT}))
    after = SuiteSnapshot(frozenset(ADDED) | {KEPT})
    foreign = addition.verify(before, after)
    refuses(lambda: a_retirement().as_attestation_record(foreign),
            "does not belong to")


def test_evidence_with_different_added_identities_is_refused():
    """Otherwise one declaration's justification could be projected beside
    another's observations -- a record describing something that never
    happened."""
    evidence = verified(a_retirement())
    other = a_retirement(
        expected_added_nodeids=frozenset({"tests/unit/test_z.py::test_other"}))
    refuses(lambda: other.as_attestation_record(evidence),
            "added identities are not this declaration")


def test_evidence_with_different_removed_identities_is_refused():
    evidence = verified(a_retirement())
    other = a_retirement(
        expected_removed_nodeids=frozenset({"tests/unit/test_z.py::test_gone"}))
    refuses(lambda: other.as_attestation_record(evidence),
            "removed identities are not this")


def test_hand_built_evidence_is_still_checked():
    """The guard does not depend on evidence having come from `verify()`."""
    forged = TransitionEvidence(
        kind=SuiteTransitionKind.DELIBERATE_RETIREMENT,
        before_count=1, after_count=2,
        before_digest="a" * 64, after_digest="b" * 64,
        added_nodeids=("tests/unit/test_forged.py::test_x",),
        removed_nodeids=tuple(sorted(REMOVED)))
    refuses(lambda: a_retirement().as_attestation_record(forged),
            "added identities are not this declaration")


# ---------------------------------------------------------------------------
# 3. NO INSTALLER MAY HAND-BUILD EITHER RECORD
# ---------------------------------------------------------------------------

def _package_sources():
    root = Path("src/genomic_variant_classifier")
    assert root.is_dir(), root
    return sorted(root.rglob("*.py"))


def test_no_module_hand_builds_a_suite_transition_record():
    """The defect class, not the defect instance.

    A dictionary literal carrying the transition key set is a second author of
    a vocabulary that has an owner. Parsed, not grepped: a docstring naming
    these keys is prose, and a substring search cannot tell the difference.

    `suite_transition.py` is exempt -- it IS the owner.
    """
    signature = {"kind", "expected_added_nodeids", "observed_added_nodeids",
                 "before_digest", "after_digest"}
    offenders = []
    for path in _package_sources():
        if path.name == "suite_transition.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if signature <= keys:
                offenders.append("{}:{}".format(path.as_posix(), node.lineno))
    assert not offenders, (
        "these reconstruct the suite-transition record instead of asking "
        "SuiteTransition.as_attestation_record for it: {}".format(offenders))


def test_no_module_hand_builds_an_attestation_target_record():
    signature = {"path", "action", "post_sha256", "post_size"}
    offenders = []
    for path in _package_sources():
        if path.name == "install_plan.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if signature <= keys:
                offenders.append("{}:{}".format(path.as_posix(), node.lineno))
    assert not offenders, (
        "these reconstruct the attestation target record instead of asking "
        "PlannedTarget.as_attestation_record for it: {}".format(offenders))


def test_the_static_guard_can_actually_find_an_offender():
    """Guards the guard.

    A structural search that matched nothing would pass over an empty result
    and report green forever. This proves the predicate fires on a synthetic
    offender, so its silence on the real package means something.
    """
    source = (
        "record = {\n"
        '    "kind": k,\n'
        '    "expected_added_nodeids": a,\n'
        '    "expected_removed_nodeids": r,\n'
        '    "observed_added_nodeids": oa,\n'
        '    "observed_removed_nodeids": orr,\n'
        '    "before_count": bc, "after_count": ac,\n'
        '    "before_digest": bd, "after_digest": ad,\n'
        "}\n")
    signature = {"kind", "expected_added_nodeids", "observed_added_nodeids",
                 "before_digest", "after_digest"}
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if signature <= keys:
                found.append(node.lineno)
    assert found, "the structural predicate matched nothing on a real offender"
