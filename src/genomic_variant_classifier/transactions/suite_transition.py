"""A suite change is a change of IDENTITY, not of count.

SUITE-NEUTRAL-IDENTITY-1, 2026-08-22.

WHY THIS MODULE EXISTS
======================
ADR-0003 established that a count is not an identity: a delta of +9 cannot
distinguish nine intended tests appearing from four appearing beside five
unrelated ones. The ADDITION installers honoured that and compared node-identity
sets. The NEUTRAL installers did not. They verified:

    collected == expected      and      ratchet == collected

and nothing more. A transition that REMOVED one test and ADDED another satisfies
both and is not neutral:

    before  {test_a, test_b, test_c}      count 3
    after   {test_a, test_b, test_d}      count 3

Two units were published under that weaker check -- `e1a5297` and `ba9060d` --
whose commit messages state "Suite transition: NEUTRAL. Verified inside the
transaction, not assumed." That claim was stronger than what was verified. It is
the same shape as the acceptance line that recorded zeroes because it was
rendered before the gate ran: a true-looking record produced by a check that
could not have established it.

    NEUTRAL  ==  delta count 0  AND  delta identity empty

not merely the first conjunct.

WHY IT LIVES IN THE PACKAGE AND NOT IN AN INSTALLER
===================================================
Four installers had each reimplemented collection, identity parsing, transition
classification and ratchet validation. The definitions DRIFTED between ADDITION
and NEUTRAL -- which is exactly how semantic drift re-enters a system that has
just removed it. One semantic concept, one typed owner. Every installer now
consumes this primitive rather than carrying a private notion of "neutral".

WHY A DIGEST AS WELL AS THE SETS
================================
Storing 5,237 identities in every neutral attestation is absurd; storing only a
count reintroduces the defect. A canonical digest over the sorted identity set
is a cheap identity proof: two suites of the same size with different membership
have different digests. Attestations therefore record the before and after
digests always, and the explicit added and removed sets -- which are small --
when they are non-empty.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum


class SuiteTransitionError(RuntimeError):
    """A declared suite transition does not describe what was observed."""


class SuiteTransitionKind(str, Enum):
    """What kind of change to the collected suite a unit declares.

    The kind is DECLARED before the change and PROVEN after it. A unit that
    declares ADDITION and removes a test fails, even though the count rose.
    """

    ADDITION = "addition"
    NEUTRAL = "neutral"
    DELIBERATE_RETIREMENT = "deliberate_retirement"


def suite_digest(nodeids: frozenset[str]) -> str:
    """A canonical digest over a set of node identities.

    Sorted, newline-joined, newline-terminated, UTF-8. Order-independent by
    construction, so two collections of the same suite agree regardless of the
    order pytest happened to report them in.
    """
    payload = "\n".join(sorted(nodeids)).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SuiteSnapshot:
    """The collected suite at one moment, as identities rather than a number."""

    nodeids: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.nodeids, frozenset):
            object.__setattr__(self, "nodeids", frozenset(self.nodeids))
        bad = sorted(n for n in self.nodeids if "::" not in n)
        if bad:
            raise SuiteTransitionError(
                "these are not pytest node identities: {}. A snapshot built "
                "from summary lines rather than from the collection listing "
                "would compare the wrong thing.".format(bad[:5]))

    @property
    def count(self) -> int:
        return len(self.nodeids)

    @property
    def digest(self) -> str:
        return suite_digest(self.nodeids)

    @classmethod
    def from_pytest_output(cls, text: str) -> "SuiteSnapshot":
        """Parse `pytest --collect-only -q` output, cross-checking two readings.

        The listing and the summary line are two independent measurements of
        the same quantity. If they disagree, NEITHER is used: a parser that
        silently prefers one has chosen which measurement to believe.
        """
        ids = frozenset(
            line.strip().replace("\\", "/")
            for line in text.splitlines()
            if "::" in line and not line.startswith(" ")
        )
        m = re.search(r"^(\d+)\s+tests?\s+collected", text, re.M)
        if m is None:
            raise SuiteTransitionError(
                "no collection summary found; refusing to build a snapshot "
                "from the listing alone, because the listing has no witness.")
        reported = int(m.group(1))
        if reported != len(ids):
            raise SuiteTransitionError(
                "pytest reported {} tests collected but {} distinct node "
                "identities were parsed. Two measurements of the same quantity "
                "disagree and neither may be used.".format(reported, len(ids)))
        return cls(nodeids=ids)


@dataclass(frozen=True)
class InvariantMigration:
    """Where a guarantee moved. INVARIANT-HANDOFF-1.

    The dangerous moment in a refactor is not when code moves; it is when a
    GUARANTEE moves. `proof_test` names the negative control that demonstrated
    the new owner can reject.
    """

    invariant_id: str
    old_owners: tuple[str, ...]
    new_owners: tuple[str, ...]
    proof_test: str
    description: str = ""

    def __post_init__(self) -> None:
        for label, value in (("invariant_id", self.invariant_id),
                             ("proof_test", self.proof_test)):
            if not str(value).strip():
                raise SuiteTransitionError(
                    "an invariant migration requires a non-empty {}".format(label))
        if not self.new_owners:
            raise SuiteTransitionError(
                "{}: an invariant may not be migrated to no owner. That is "
                "removal, not migration.".format(self.invariant_id))


@dataclass(frozen=True)
class TransitionEvidence:
    """What was actually observed. Recorded in the install attestation."""

    kind: SuiteTransitionKind
    before_count: int
    after_count: int
    before_digest: str
    after_digest: str
    added_nodeids: tuple[str, ...]
    removed_nodeids: tuple[str, ...]

    def as_record(self) -> dict:
        return {
            "kind": self.kind.value,
            "before": {"count": self.before_count, "digest": self.before_digest},
            "after": {"count": self.after_count, "digest": self.after_digest},
            "added_nodeids": list(self.added_nodeids),
            "removed_nodeids": list(self.removed_nodeids),
        }


@dataclass(frozen=True)
class SuiteTransition:
    """A declared suite change, validated at construction and proven on use."""

    kind: SuiteTransitionKind
    expected_added_nodeids: frozenset[str] = frozenset()
    expected_removed_nodeids: frozenset[str] = frozenset()
    invariant_migrations: tuple[InvariantMigration, ...] = ()
    justification: str = ""
    _checked: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_added_nodeids",
                           frozenset(self.expected_added_nodeids))
        object.__setattr__(self, "expected_removed_nodeids",
                           frozenset(self.expected_removed_nodeids))
        overlap = self.expected_added_nodeids & self.expected_removed_nodeids
        if overlap:
            raise SuiteTransitionError(
                "these identities are declared both added and removed: {}"
                .format(sorted(overlap)))

        if self.kind is SuiteTransitionKind.NEUTRAL:
            if self.expected_added_nodeids or self.expected_removed_nodeids:
                raise SuiteTransitionError(
                    "a NEUTRAL transition declares no added and no removed "
                    "identities. Declaring either means the suite changed, "
                    "which is not neutral however the count behaves.")
        elif self.kind is SuiteTransitionKind.ADDITION:
            if not self.expected_added_nodeids:
                raise SuiteTransitionError(
                    "an ADDITION transition must name what it adds")
            if self.expected_removed_nodeids:
                raise SuiteTransitionError(
                    "an ADDITION transition may not remove. A unit that both "
                    "adds and removes is a DELIBERATE_RETIREMENT, and must "
                    "justify itself as one.")
        elif self.kind is SuiteTransitionKind.DELIBERATE_RETIREMENT:
            if not self.expected_removed_nodeids:
                raise SuiteTransitionError(
                    "a DELIBERATE_RETIREMENT must name every identity it retires")
            if not self.justification.strip():
                raise SuiteTransitionError(
                    "a DELIBERATE_RETIREMENT requires a justification. The "
                    "suite ratchet exists to catch ACCIDENTAL loss; a "
                    "deliberate one must say why it is deliberate.")
        object.__setattr__(self, "_checked", True)

    def verify(self, before: SuiteSnapshot,
               after: SuiteSnapshot) -> TransitionEvidence:
        """Prove the observed change is exactly the declared one, or refuse."""
        added = after.nodeids - before.nodeids
        removed = before.nodeids - after.nodeids

        if added != self.expected_added_nodeids:
            raise SuiteTransitionError(
                "ADDED IDENTITIES ARE NOT THE DECLARED SET.\n"
                "  observed but not declared: {}\n"
                "  declared but not observed: {}\n"
                "  a count of {:+d} cannot distinguish these.".format(
                    sorted(added - self.expected_added_nodeids),
                    sorted(self.expected_added_nodeids - added),
                    after.count - before.count))
        if removed != self.expected_removed_nodeids:
            raise SuiteTransitionError(
                "REMOVED IDENTITIES ARE NOT THE DECLARED SET.\n"
                "  observed but not declared: {}\n"
                "  declared but not observed: {}\n"
                "  a count of {:+d} cannot distinguish these.".format(
                    sorted(removed - self.expected_removed_nodeids),
                    sorted(self.expected_removed_nodeids - removed),
                    after.count - before.count))

        # THREE FURTHER CHECKS WERE WRITTEN HERE AND REMOVED AS DEAD CODE.
        #
        # A sabotage matrix on 2026-08-22 disabled each guard in turn and
        # required the suite to detect it. Three were NOT detected, and the
        # reason was not missing tests -- they are provably unreachable:
        #
        #   NEUTRAL set-equality. Reached only after both comparisons above
        #   pass. For NEUTRAL both expected sets are empty, so added == {} and
        #   removed == {}; hence after is a subset of before and before of
        #   after, hence after == before. It cannot fire.
        #
        #   Count/identity cross-check. |after| - |before| == |added| -
        #   |removed| is a set identity, true for all finite sets. The comment
        #   accompanying it even said it could not fail.
        #
        #   ADDITION rising-count. Construction enforces a non-empty added set
        #   and an empty removed set; verify proves the observed sets equal
        #   them; so the count must rise.
        #
        # Defence in depth that cannot fire is not defence. It is the exact
        # shape this project keeps finding -- a vacuous iterator, a gate whose
        # default invocation cannot fail, an alert never observed to alert --
        # and it is worse than absence because it reads as protection.
        #
        # The two comparisons above are the whole contract. Both are proven
        # detectable by the sabotage matrix.

        return TransitionEvidence(
            kind=self.kind,
            before_count=before.count, after_count=after.count,
            before_digest=before.digest, after_digest=after.digest,
            added_nodeids=tuple(sorted(added)),
            removed_nodeids=tuple(sorted(removed)),
        )
