"""An installer is one atomic repository state transition.

INSTALLER-TRANSACTION-1, step 4.

    Before the first byte of the repository changes, the system must possess an
    immutable description of the complete transition it intends to perform;
    after the last byte changes, it must prove that the observed repository
    equals that description plus no unintended transition.

WHY A PLAN RATHER THAN A SEQUENCE OF CALLS
Every installer before this one performed its writes imperatively: patch a
file, write another, edit the ratchet, edit the README. Nothing declared the
complete set in advance, so nothing could detect an installer that touched one
file more than it meant to.

    There are no transactional "payload files" and nontransactional
    "bookkeeping files". The transaction owns the entire declared mutation set,
    including tests, ratchet, README, and same-unit metadata. PowerShell owns
    no repository state.

MEASURED 2026-08-20, before this module was written:

    RepositoryTransaction exposed NO public accessor for its target set;
    `_targets` was private, so a runner comparing declared against actual would
    have had to reach into the transaction's internals -- the coupling the
    accessor now prevents.

    The transaction journal resolves to
    <LOCALAPPDATA>/GenomicVariantClassifier/transactions, OUTSIDE the
    repository, so the repository hygiene invariant needs no journal exception.

THE RATCHET IS DERIVED, NEVER TRANSCRIBED
Today's workflow measured pytest's collection, derived a new ratchet by hand,
and encoded the number in an installer. That produced repeated stale-count
incidents: on 2026-08-20 alone, a file expected to collect 52 cases collected
54, and another expected 38 collected 44.

So a plan for a test-adding unit carries a DELTA, not a total. The runner
measures collection before and after, asserts the delta matches, and renders
BOTH the ratchet and the README badge from that single measured number.

    Never independently write expected_suite_size = N and readme_badge = N.

WHAT THIS MODULE DOES NOT DO
It implements no backup, rollback, path-ownership, recovery, preimage or
journal semantics. Those belong to RepositoryTransaction. A plan knows only
what should change, what must be true afterward, and which gate to run.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PlanError(RuntimeError):
    """A refusal by the plan machinery, before any repository write."""


class WriteSetViolation(PlanError):
    """The transaction's actual writes are not the declared writes."""


class TargetAction(str, Enum):
    PATCH = "patch"
    CREATE = "create"


@dataclass(frozen=True)
class PlannedTarget:
    """One declared repository mutation.

    `relpath` is POSIX-style and relative to the repository root. `payload` is
    the complete intended postimage: the plan never describes a mutation it
    cannot fully materialise before the repository is touched.
    """

    relpath: str
    action: TargetAction
    payload: bytes
    reason: str = ""

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def as_record(self) -> dict:
        """The PLAN vocabulary. Not the attestation vocabulary."""
        return {
            "relpath": self.relpath,
            "action": self.action.value,
            "sha256": self.digest,
            "size": len(self.payload),
            "reason": self.reason,
        }

    def as_attestation_record(self) -> dict:
        """Projection into the `gvc.install-attestation` target vocabulary.

        A SEPARATE projection, not a rename of `as_record`. MEASURED
        2026-08-25, the two contracts genuinely differ:

            as_record            relpath, action, sha256, size, reason
            attestation target   path,    action, post_sha256, post_size

        Five keys against four, three differently named. `_exact_keys` in
        install_attestation.py rejects unknown keys as hard as missing ones, so
        `as_record()` output would be refused on seven counts.

        Every installer performed that translation BY HAND, which is how
        PROOF-AFTER-IRREVERSIBILITY-1 became possible: a hand-built record can
        omit a field, and on 2026-08-25 one did -- after the commit.
        """
        return {
            "path": self.relpath,
            "action": self.action.value,
            "post_sha256": self.digest,
            "post_size": len(self.payload),
        }


@dataclass(frozen=True)
class DerivedCount:
    """A test-count measurement used to render BOTH counters.

    Holding the number once, in one object, is what prevents the ratchet and
    the README badge from becoming two independently transcribed authorities.
    """

    before: int
    after: int

    @property
    def delta(self) -> int:
        return self.after - self.before


@dataclass
class InstallPlan:
    """The complete intended repository transition, fixed before mutation."""

    unit: str
    targets: tuple = ()
    expected_delta: int | None = None

    _frozen: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.targets = tuple(self.targets)
        seen = {}
        for t in self.targets:
            if not isinstance(t, PlannedTarget):
                raise PlanError("every target must be a PlannedTarget, got {}"
                                .format(type(t).__name__))
            if t.relpath in seen:
                raise PlanError(
                    "{} is declared twice ({} then {}). A plan describes one "
                    "transition per path.".format(
                        t.relpath, seen[t.relpath].value, t.action.value))
            seen[t.relpath] = t.action
        self._frozen = True

    @property
    def declared_write_set(self) -> frozenset:
        return frozenset(t.relpath for t in self.targets)

    @property
    def digest(self) -> str:
        """One identifier binding a dry-run, an apply and a journal entry.

        Any change to any target's path, action or bytes changes this.
        """
        h = hashlib.sha256()
        h.update(self.unit.encode("utf-8"))
        for t in sorted(self.targets, key=lambda x: x.relpath):
            h.update(t.relpath.encode("utf-8"))
            h.update(t.action.value.encode("utf-8"))
            h.update(t.digest.encode("utf-8"))
        return h.hexdigest()

    def describe(self) -> dict:
        return {
            "unit": self.unit,
            "plan_digest": self.digest,
            "expected_delta": self.expected_delta,
            "targets": [t.as_record() for t in self.targets],
        }

    # ---- validation, all of it BEFORE any repository write --------------
    def validate_against(self, repo: Path) -> None:
        """Refuse a plan that cannot be executed exactly as declared.

        Every check here runs while nothing has been written, so a refusal
        costs nothing and leaves nothing to undo.
        """
        repo = Path(repo).resolve()
        for t in self.targets:
            resolved = (repo / t.relpath).resolve()
            if resolved != repo and repo not in resolved.parents:
                raise PlanError(
                    "{} resolves outside the repository".format(t.relpath))
            exists = resolved.exists()
            if t.action is TargetAction.PATCH and not exists:
                raise PlanError(
                    "{} is declared as a patch but does not exist".format(t.relpath))
            if t.action is TargetAction.CREATE and exists:
                raise PlanError(
                    "{} is declared as a create but already exists".format(t.relpath))
            if not t.payload:
                raise PlanError("{} has an empty payload".format(t.relpath))

    def assert_write_set(self, transaction) -> None:
        """WRITE-SET COMPLETENESS, checked while rollback is still available.

        An installer that unexpectedly touches a sixth file fails here even if
        every test passes. That is a strictly stronger property than "the
        suite is green afterwards".
        """
        actual = transaction.write_set
        declared = self.declared_write_set
        if actual != declared:
            extra = sorted(actual - declared)
            missing = sorted(declared - actual)
            raise WriteSetViolation(
                "the transaction wrote a different set than the plan declared. "
                "UNDECLARED: {}. NOT WRITTEN: {}.".format(extra or "none",
                                                          missing or "none"))
        mutated = transaction.mutated_set
        if mutated != declared:
            raise WriteSetViolation(
                "declared targets that were never mutated: {}".format(
                    sorted(declared - mutated)))


def render_ratchet(existing: bytes, entry: str, count: int) -> bytes:
    """Rewrite the ratchet from ONE measured count.

    The file carries an append-only comment log and exactly one bare integer.
    Both the log entry and the number come from the same measurement, so they
    cannot disagree.
    """
    text = existing.decode("utf-8")
    lines = [l for l in text.split("\n")]
    bare = [l for l in lines if l.strip().isdigit()]
    if len(bare) != 1:
        raise PlanError(
            "the ratchet holds {} bare number(s); expected exactly 1"
            .format(len(bare)))
    kept = "\n".join(l for l in lines if not l.strip().isdigit())
    if not kept.endswith("\n"):
        kept += "\n"
    return (kept + entry + str(count) + "\n").encode("utf-8")


def render_readme(existing: bytes, old_count: int, new_count: int) -> bytes:
    """Move the README badge to the SAME measured count.

    Refuses rather than silently producing an unchanged file, so a badge that
    has drifted from the ratchet is caught before the transaction opens.
    """
    text = existing.decode("utf-8")
    needle = "badge/tests-{}-".format(old_count)
    if needle not in text:
        raise PlanError(
            "the README badge does not read tests-{}-; refusing to guess"
            .format(old_count))
    return text.replace(needle, "badge/tests-{}-".format(new_count)).encode("utf-8")
