"""What population a measurement inspected, and which members it realised.

ADR-0005. The central distinction:

    selector meaning  !=  selection result

A `CorpusSpec` declares what an instrument INTENDED to inspect. A
`CorpusSnapshot` records what was ACTUALLY selected at a particular repository
state. Collapsing them into one object loses the difference between a selector
that matched nothing and a selector that could not be resolved.

MEASURED 2026-09-05, the failure this prevents: `git grep -- tests/foo.py`
where `tests/foo.py` does not exist exits 0 with NO OUTPUT, which is
indistinguishable from an existing file containing zero matches. The silence
was read as absence.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class CorpusKind(str, Enum):
    """Which population a selector is drawn from.

    These are NOT interchangeable. Filesystem membership is not Git identity;
    Git identity is not package membership; package membership is not runtime
    reachability.
    """

    TRACKED = "tracked"
    WORKTREE = "worktree"
    PACKAGE = "package"
    HISTORICAL = "historical"
    EXPLICIT = "explicit"


class EnumerationMode(str, Enum):
    """How the members were obtained from the population."""

    COMPLETE = "complete"
    SAMPLED = "sampled"
    DISCOVERY = "discovery"


#: Domain separation. The repository already has several digest-bearing
#: systems -- suite identity, payload identity, manifest identity, evidence
#: identity -- and two of them producing the same bytes for different meanings
#: would be a collision in MEANING even where the encoding is innocent.
_MEMBERSHIP_DOMAIN = b"gvc.repository-measurement.membership.v1\x00"


def corpus_membership_digest(members: Tuple[str, ...]) -> str:
    """Hash member IDENTITIES, not file contents.

    Named `membership` and never `corpus_digest`, because a corpus can retain
    identical membership while every file inside it changes. Length-delimited
    so no delimiter ambiguity exists: two different member lists cannot encode
    to the same byte stream.

    |A| == |B| does not imply A == B. Two corpora can each hold 357 files and
    hold different 357 files.
    """
    h = hashlib.sha256()
    h.update(_MEMBERSHIP_DOMAIN)
    for member in members:
        raw = member.encode("utf-8")
        h.update(len(raw).to_bytes(8, byteorder="big"))
        h.update(raw)
    return h.hexdigest()


@dataclass(frozen=True)
class CorpusSpec:
    """Semantic declaration of the population a measurement intends to inspect.

    This is not the realised member set.

    `minimum_members` exists so that a legitimate absence query (0) is
    distinguishable from a probe whose semantics require a non-empty universe.
    """

    kind: CorpusKind
    selector: str
    enumerator: str

    includes_untracked: bool = False
    includes_ignored: bool = False

    minimum_members: int = 0
    required_roots: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.selector.strip():
            raise ValueError("selector must be non-empty")
        if not self.enumerator.strip():
            raise ValueError("enumerator must be non-empty")
        if self.minimum_members < 0:
            raise ValueError("minimum_members must be >= 0")
        if self.kind is CorpusKind.TRACKED:
            if self.includes_untracked:
                raise ValueError(
                    "TRACKED corpus cannot include untracked artifacts")
            if self.includes_ignored:
                raise ValueError(
                    "TRACKED corpus cannot include ignored worktree artifacts")


@dataclass(frozen=True)
class SelectionCoverage:
    """Did enumeration cover its declared population?

    Distinct from analysis completeness, which asks whether every enumerated
    member was successfully processed. A probe may enumerate completely and
    fail to parse half of what it found, or enumerate partially and parse
    everything it got. One overloaded `coverage` field cannot say which.
    """

    requested_roots: Tuple[str, ...] = ()
    resolved_roots: Tuple[str, ...] = ()
    missing_roots: Tuple[str, ...] = ()
    enumeration_complete: bool = False

    def __post_init__(self) -> None:
        overlap = set(self.resolved_roots) & set(self.missing_roots)
        if overlap:
            raise ValueError(
                "a root cannot be both resolved and missing: "
                "{!r}".format(sorted(overlap)))
        accounted = set(self.resolved_roots) | set(self.missing_roots)
        unaccounted = set(self.requested_roots) - accounted
        if unaccounted:
            raise ValueError(
                "requested roots neither resolved nor missing: {!r}. A root "
                "whose fate is unrecorded is exactly the silent-pathspec "
                "failure this type exists to prevent."
                .format(sorted(unaccounted)))


@dataclass(frozen=True)
class RepositoryState:
    """The repository state a snapshot was taken against.

    For a WORKTREE corpus, `head_oid` is only the BASE state. Printing
    "measured at c18a1df" for a dirty worktree would attribute uncommitted
    bytes to that commit.
    """

    head_oid: str | None = None
    worktree_dirty: bool = False


@dataclass(frozen=True)
class CorpusSnapshot:
    """Concrete corpus realised from a CorpusSpec at a particular state."""

    spec: CorpusSpec
    members: Tuple[str, ...]

    repository_head: str | None = None
    worktree_dirty: bool | None = None
    selection: SelectionCoverage | None = None

    def __post_init__(self) -> None:
        if len(self.members) != len(set(self.members)):
            raise ValueError("corpus contains duplicate members")
        if self.members != tuple(sorted(self.members)):
            raise ValueError(
                "corpus members must be canonicalized in sorted order")
        if len(self.members) < self.spec.minimum_members:
            raise ValueError(
                "corpus has {} member(s), minimum is {}".format(
                    len(self.members), self.spec.minimum_members))
        if self.spec.kind is CorpusKind.TRACKED:
            if self.repository_head is None:
                raise ValueError("TRACKED corpus requires repository_head")
            if self.worktree_dirty not in (False, None):
                raise ValueError(
                    "TRACKED corpus cannot claim dirty-worktree membership; "
                    "the snapshot refers to HEAD and uncommitted bytes are "
                    "not part of it")
        if self.spec.required_roots and self.selection is None:
            raise ValueError(
                "a spec declaring required_roots must carry SelectionCoverage, "
                "or a missing root is invisible")

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def membership_sha256(self) -> str:
        return corpus_membership_digest(self.members)
