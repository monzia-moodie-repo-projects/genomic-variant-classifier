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


#: A SHA-256 digest, as the projection requires it.
_DIGEST = re.compile(r"\A[0-9a-f]{64}\Z")

#: Characters a line-oriented identity may never contain. MEASURED 2026-09-10:
#: SuiteSnapshot(frozenset({"a::x\nb::y"})) has count 1 and
#: SuiteSnapshot(frozenset({"a::x", "b::y"})) has count 2, and the two share one
#: digest, because `suite_digest` joins with a newline so both serialise to the
#: same bytes. NOT a SHA-256 collision: an ambiguous encoding over inputs the
#: domain wrongly accepted. The repair validates the domain, not the hash.
_FORBIDDEN_IN_NODEID = ("\n", "\r", "\x00")


def require_nodeids(values, *, label: str) -> frozenset:
    """Validate an identity collection BEFORE it becomes a set.

    Duplicates are refused HERE because a constructor cannot recover duplicates
    a caller already discarded: frozenset(["a", "a"]) has lost the evidence.
    The producer must pass its original sequence.

    MEASURED 2026-09-10 against the previous implementation: three identities
    supplied to SuiteSnapshot with two identical yielded a count of 2, in
    silence. The reported-count cross-check existed only on the text route.
    """
    if isinstance(values, (str, bytes)):
        raise SuiteTransitionError(
            "{}: a single string is not an identity collection".format(label))
    try:
        items = tuple(values)
    except TypeError as exc:
        raise SuiteTransitionError(
            "{}: not an iterable of identities".format(label)) from exc
    for nodeid in items:
        if type(nodeid) is not str:
            raise SuiteTransitionError(
                "{}: identity is not a string: {!r}".format(label, nodeid))
        if "::" not in nodeid:
            raise SuiteTransitionError(
                "{}: not a pytest node identity: {!r}. A snapshot built from "
                "summary lines rather than from the collection listing would "
                "compare the wrong thing.".format(label, nodeid))
        for character in _FORBIDDEN_IN_NODEID:
            if character in nodeid:
                raise SuiteTransitionError(
                    "{}: identity contains {!r}, which the newline-joined "
                    "digest encoding cannot represent unambiguously: "
                    "{!r}".format(label, character, nodeid[:80]))
    if len(items) != len(set(items)):
        seen, duplicated = set(), []
        for nodeid in items:
            if nodeid in seen and nodeid not in duplicated:
                duplicated.append(nodeid)
            seen.add(nodeid)
        raise SuiteTransitionError(
            "{}: duplicate identities: {}".format(label,
                                                  sorted(duplicated)[:5]))
    return frozenset(items)


def require_observed_nodeids(value, *, label: str) -> frozenset:
    """Validate REPORTED OBSERVATIONS, which must preserve their multiplicity.

    MEASURED 2026-09-11: DELIBERATE_RETIREMENT evidence with added=(B, B) and
    removed=(A, A) at counts 1 and 1 was EMITTED. Membership comparison
    collapsed the duplicates to sets while the arithmetic compared 2-2 against
    1-1 -- two incompatible interpretations of one field, whose errors
    cancelled.

    A set is refused HERE because an observation that arrives already
    deduplicated has destroyed the evidence this check exists to read. That is
    the opposite of a declaration, where a frozenset is the correct object.
    """
    if type(value) not in (list, tuple):
        raise SuiteTransitionError(
            "{}: expected an observation list or tuple, not {}".format(
                label, type(value).__name__))
    return require_nodeids(value, label=label)


def require_document_nodeids(value, *, label: str) -> frozenset:
    """Validate a SERIALISED declaration, which must arrive as a JSON array.

    Prevents a document loader from silently treating an arbitrary
    representation as the declared schema. Strict JSON parsing must separately
    reject duplicate object keys and nonstandard numeric constants.
    """
    if type(value) is not list:
        raise SuiteTransitionError(
            "{}: expected a JSON array, not {}".format(
                label, type(value).__name__))
    return require_nodeids(value, label=label)


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
        # A frozenset argument has ALREADY lost duplicate evidence; only a
        # sequence can be checked for them. Both routes are validated for the
        # identity domain, and from_pytest_output checks duplicates on the raw
        # listing before it reaches here.
        object.__setattr__(self, "nodeids",
                           require_nodeids(self.nodeids, label="snapshot"))

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
        # NO SEPARATOR REWRITING.
        #
        # MEASURED 2026-09-10: the previous body applied .replace(chr(92), "/")
        # to the WHOLE line, parameter text included. Two collections, each
        # internally consistent --
        #     before  test_path[a\b]     1 test collected
        #     after   test_path[a/b]     1 test collected
        # -- both mapped to ONE identity, so snapshots and digests compared
        # equal and NEUTRAL was ACCEPTED for a changed suite. The
        # listing/summary cross-check cannot see it: it catches colliding
        # identities appearing TOGETHER, not one replacing the other ACROSS
        # collections.
        #
        # If path-prefix normalisation is ever required it belongs to the
        # portion before the first "::", with its own contract and tests. It is
        # never applied to parameter text.
        listed = [line.strip()
                  for line in text.splitlines()
                  if "::" in line and not line.startswith(" ")]
        ids = require_nodeids(listed, label="collection listing")
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
        # DECLARATIONS ARE MEMBERSHIP. A frozenset is the correct object for
        # "these identities are approved", and its inability to hold duplicates
        # is its semantics, not a defect. require_nodeids validates the members
        # and rejects duplicates WHEN THE REPRESENTATION PRESERVES THEM.
        #
        # A sequence-only requirement was considered and REJECTED, because it
        # would not establish what it appears to. MEASURED 2026-09-11:
        #     tuple(set([A, A, B]))  ->  PASSES a list-or-tuple check
        # The duplicates vanished before the boundary. Provenance needs a
        # controlled producer and retained execution evidence, not a type.
        object.__setattr__(self, "expected_added_nodeids",
                           require_nodeids(self.expected_added_nodeids,
                                           label="declared additions"))
        object.__setattr__(self, "expected_removed_nodeids",
                           require_nodeids(self.expected_removed_nodeids,
                                           label="declared removals"))
        # THE KIND MUST BE THE ENUM.
        #
        # MEASURED 2026-09-10: SuiteTransition(kind="addition") CONSTRUCTED and
        # then VERIFIED. A plain string is not identical to any member, so every
        # branch below was bypassed and `_checked` was set to True regardless.
        # Failure would have surfaced later at serialisation -- the wrong
        # boundary. Annotations do not enforce anything and dataclasses do not
        # inspect them.
        if type(self.kind) is not SuiteTransitionKind:
            raise SuiteTransitionError(
                "kind must be a SuiteTransitionKind, not {!r}. A domain "
                "constructor does not interpret arbitrary values; convert at "
                "the serialisation boundary with "
                "SuiteTransitionKind(document['kind']).".format(self.kind))
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
        else:
            # A FUTURE MEMBER WHOSE SEMANTICS ARE NOT IMPLEMENTED HERE.
            #
            # SUITE-TRANSITION-KIND-INCOMPLETE-1 records that
            # IDENTITY_REPLACEMENT is missing: a pure rename is expressible
            # only as DELIBERATE_RETIREMENT, which records a retirement where
            # nothing was retired. Re-executed 2026-09-10 and still true.
            #
            # Adding that member WITHOUT this branch would let it construct
            # with no validation at all -- the latent defect firing. This
            # branch prepares for it; it does not add it.
            raise SuiteTransitionError(
                "unsupported transition kind: {!r}. Its validation rules are "
                "not implemented here.".format(self.kind))
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

    def _assert_evidence_belongs_here(self,
                                      evidence: TransitionEvidence) -> None:
        """Refuse evidence produced against a DIFFERENT declaration.

        These checks are REACHABLE, unlike the three removed from `verify` as
        provably unreachable. `as_attestation_record` is public and may be
        handed evidence constructed elsewhere or by hand, so each of these can
        be demonstrated firing by a negative control -- which is the standard
        this module set for itself when it deleted defence that could not fire.
        """
        if evidence.kind is not self.kind:
            raise SuiteTransitionError(
                "evidence of kind {!r} does not belong to a {!r} declaration"
                .format(evidence.kind.value, self.kind.value))
        if frozenset(evidence.added_nodeids) != self.expected_added_nodeids:
            raise SuiteTransitionError(
                "the evidence's added identities are not this declaration's. "
                "Projecting one declaration's justification beside another's "
                "observations would produce a record that never happened.")
        if frozenset(evidence.removed_nodeids) != self.expected_removed_nodeids:
            raise SuiteTransitionError(
                "the evidence's removed identities are not this "
                "declaration's.")

    def _assert_evidence_is_internally_consistent(
            self, evidence: "TransitionEvidence") -> None:
        """Refuse evidence whose own fields contradict each other.

        ONE VALIDATED INTERPRETATION. MEASURED 2026-09-11: added=(B, B) and
        removed=(A, A) at counts 1 and 1 was EMITTED, because membership
        checks read the tuples as sets while the arithmetic read their
        lengths. Every check below now operates on the SAME validated sets and
        their sizes.

        These checks reject contradictions. They cannot prove that plausible
        digests correspond to real collections; `verified_attestation_record`
        is the route that recomputes from snapshots.
        """
        for name, value in (("before_count", evidence.before_count),
                            ("after_count", evidence.after_count)):
            if type(value) is not int or type(value) is bool or value < 0:
                raise SuiteTransitionError(
                    "{} must be a nonnegative integer, not {!r}".format(
                        name, value))
        for name, value in (("before_digest", evidence.before_digest),
                            ("after_digest", evidence.after_digest)):
            if type(value) is not str or not _DIGEST.fullmatch(value):
                raise SuiteTransitionError(
                    "{} must be 64 lowercase hexadecimal digits, not "
                    "{!r}".format(name, value))
        added = require_observed_nodeids(evidence.added_nodeids,
                                         label="observed additions")
        removed = require_observed_nodeids(evidence.removed_nodeids,
                                           label="observed removals")
        both = added & removed
        if both:
            raise SuiteTransitionError(
                "an identity cannot be both added and removed: {}".format(
                    sorted(both)[:5]))
        if len(removed) > evidence.before_count:
            raise SuiteTransitionError(
                "removals ({}) exceed the before population ({})".format(
                    len(removed), evidence.before_count))
        if len(added) > evidence.after_count:
            raise SuiteTransitionError(
                "additions ({}) exceed the after population ({})".format(
                    len(added), evidence.after_count))
        observed = evidence.after_count - evidence.before_count
        declared = len(added) - len(removed)
        if observed != declared:
            raise SuiteTransitionError(
                "the counts move by {:+d} while the identity transition moves "
                "by {:+d}".format(observed, declared))
        if evidence.before_digest == evidence.after_digest and (
                added or removed):
            raise SuiteTransitionError(
                "the digests are equal while identities changed")
        if evidence.before_digest != evidence.after_digest and not (
                added or removed):
            raise SuiteTransitionError(
                "the digests differ while no identity changed")

    def verified_attestation_record(self, before: SuiteSnapshot,
                                    after: SuiteSnapshot) -> dict:
        """Project from SNAPSHOTS, so the record cannot disagree with them."""
        return self.as_attestation_record(self.verify(before, after))

    def as_attestation_record(self, evidence: TransitionEvidence) -> dict:
        """Project one declared-AND-verified transition for an attestation.

        THE DECLARATION OWNS THIS PROJECTION, and that is the point.

        The attestation's `suite_transition` record is a JOIN: the declaration
        owns the expected identities and the justification; the evidence owns
        the observed identities and the measured before/after state. Neither
        alone holds the whole truth.

        PROOF-AFTER-IRREVERSIBILITY-1, 2026-08-25. The DRIFT-1 installer built
        this record by hand from `TransitionEvidence`, which carries `kind` but
        NOT `justification`. The omission was therefore not an oversight -- the
        field was structurally unreachable from the object supplying most of the
        serialization. The installer committed, then refused to write its own
        attestation, and the repository crossed an irreversible boundary
        without its publication evidence.

        Placed on the DECLARATION rather than on the evidence because
        `evidence.as_attestation_record(declaration)` would read as though
        evidence owned the projection and merely needed a declaration supplied
        -- and would permit an arbitrary pairing unless it reproduced
        verification logic. `_assert_evidence_belongs_here` makes the pairing
        provable instead.

        NO INSTALLER MAY WRITE THIS DICTIONARY. That is enforced by a test.
        """
        # ORDER MATTERS, AND THE GATE PROVED IT.
        #
        # MEASURED 2026-09-10 at f808944: with the consistency check first,
        # tests/unit/test_attestation_projection.py::
        # test_hand_built_evidence_is_still_checked failed --
        #     expected: "added identities are not this declaration"
        #     actual:   "the counts move by +1 while the difference sets
        #                move by +0"
        # -- because the new check fired before the established one. That test
        # asserts the refusal REASON, not merely that a refusal happened, and
        # it was right to fail.
        #
        # BELONGING IS SETTLED FIRST. Whether this evidence is THIS
        # declaration's to judge must be answered before its internal
        # coherence, or a coherence complaint is raised about evidence that
        # was never in scope.
        self._assert_evidence_belongs_here(evidence)
        self._assert_evidence_is_internally_consistent(evidence)

        record = {
            "kind": self.kind.value,
            "expected_added_nodeids": sorted(self.expected_added_nodeids),
            "expected_removed_nodeids": sorted(self.expected_removed_nodeids),
            "observed_added_nodeids": list(evidence.added_nodeids),
            "observed_removed_nodeids": list(evidence.removed_nodeids),
            "before_count": evidence.before_count,
            "after_count": evidence.after_count,
            "before_digest": evidence.before_digest,
            "after_digest": evidence.after_digest,
        }
        if self.kind is SuiteTransitionKind.DELIBERATE_RETIREMENT:
            record["justification"] = self.justification
        return record
