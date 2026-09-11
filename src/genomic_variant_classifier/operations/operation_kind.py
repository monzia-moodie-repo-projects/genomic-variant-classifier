"""What kind of operation this is, established rather than declared.

Author: Monzia Moodie

THE TERMINATING RULE
====================
    An archive-admission operation creates a MAINTENANCE-EVIDENCE obligation.
    Preserving that evidence completes the obligation; it does not create
    another installation-archive admission obligation.

Without that boundary every admission generates the next one forever: A admits
88 attestations, A's own attestation becomes an 89th obligation, admitting it
produces a 90th, and no batch ever closes. The boundary is a policy decision,
not an exemption from preservation -- A's evidence is still preserved, still
validated, still discoverable. It is preserved in a DIFFERENT CHANNEL whose
own preservation is not itself an installation-archive admission.

ROUTING IS BY VERIFIED KIND, NEVER BY DECLARATION
=================================================
An operation does not obtain maintenance-only treatment by calling itself an
admission. A filename, a caller-supplied Boolean and an unverified
`operation_kind` string are all inadmissible as evidence of kind.

The kind is ESTABLISHED from the approved plan and the OBSERVED transition:

    the observed write set equals the approved one, exactly
    every target lies under the archive's own subtree
    every predecessor entry outside that subtree is untouched
    the additions are complete and approved

A unit that also changes executable code fails the second test and is an
ordinary installation, whatever it calls itself.

WHAT THIS MODULE DOES NOT DO
============================
It does not publish, store, or retain anything. It classifies, and it refuses.
The channel's storage, retention, cleanup exclusion and recovery are the
evidence owner's contract; a classifier that also stored things would make its
own answer true.

EXIT STATUS when run as a script
  0  every self-test held
  2  a self-test did not hold
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from enum import Enum

#: The archive's own subtree. A target outside it is not archive maintenance.
ARCHIVE_SUBTREE = "records/attestations/installations/"
MANIFEST_PATH = ARCHIVE_SUBTREE + "manifest.json"
ARTIFACTS_SUBTREE = ARCHIVE_SUBTREE + "artifacts/"

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

#: Ordinary file only. MEASURED 2026-09-08: mode "120000" -- a symbolic link --
#: was accepted, because the observed values were never examined at all.
SUPPORTED_MODES = frozenset({"100644"})
SUPPORTED_ACTIONS = frozenset({"create", "patch"})


@dataclass(frozen=True)
class FileEffect:
    """One committed effect, with every field validated at construction.

    Ordinary dataclass equality does not solve Boolean/integer equivalence, so
    the CONSTRUCTOR refuses invalid values rather than the comparison trying to
    catch them later: `size_bytes=True` must never reach a field where
    `True == 1` would make it equal an approved size of one.
    """

    action: str
    mode: str
    size_bytes: int
    content_sha256: str

    def __post_init__(self) -> None:
        if self.action not in SUPPORTED_ACTIONS:
            raise ClassificationError(
                "unsupported action {!r}; supported are {}".format(
                    self.action, sorted(SUPPORTED_ACTIONS)))
        if self.mode not in SUPPORTED_MODES:
            raise ClassificationError(
                "unsupported mode {!r}; supported are {}".format(
                    self.mode, sorted(SUPPORTED_MODES)))
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ClassificationError(
                "size_bytes must be a nonnegative integer, not {!r} of type "
                "{}".format(self.size_bytes, type(self.size_bytes).__name__))
        if type(self.content_sha256) is not str or \
                not _SHA256.fullmatch(self.content_sha256):
            raise ClassificationError(
                "content_sha256 must be 64 lowercase hexadecimal digits")


def require_canonical_repository_path(path) -> str:
    r"""A repository-relative path with no way to mean something else.

    MEASURED 2026-09-08, both ACCEPTED by the previous string checks:

        records/.../artifacts/..\..\..\escape.json
        records/.../artifacts/a.json:stream

    The first passed because the code split only on "/" while Windows also
    recognises "\" as a separator. The second passed because nothing examined
    the basename. Windows additionally has reserved device names and
    alternate-data-stream syntax that a prefix comparison cannot address.

    This mirrors the plan owner's canonical-path rule rather than replacing
    it; the plan owner remains the authority for repository paths generally,
    and its own empty-relative-path defect is recorded separately.
    """
    if type(path) is not str or not path.strip():
        raise ClassificationError(
            "a target path must be a nonempty string, not {!r}".format(path))
    if "\\" in path:
        raise ClassificationError(
            "{!r} contains a backslash, which Windows treats as a path "
            "separator".format(path))
    if path.startswith("/") or re.match(r"\A[A-Za-z]:", path):
        raise ClassificationError(
            "{!r} is not repository-relative".format(path))
    segments = path.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        raise ClassificationError(
            "{!r} is not canonical: empty, '.' or '..' segment".format(path))
    # EVERY SEGMENT, not only the basename. MEASURED 2026-09-08, both
    # ACCEPTED when only the last segment was examined:
    #
    #     .../artifacts/sub:stream/x.json
    #     .../artifacts/CON/x.json
    #
    # A directory component carries the same Windows meaning as a file one.
    for segment in segments:
        if ":" in segment:
            raise ClassificationError(
                "{!r} uses alternate-data-stream syntax".format(segment))
        if segment != segment.strip() or segment.endswith("."):
            raise ClassificationError(
                "{!r} has trailing whitespace or a trailing dot, which "
                "Windows silently strips".format(segment))
        reserved = segment.split(".")[0].upper()
        if reserved in {"CON", "PRN", "AUX", "NUL"} or \
                re.fullmatch(r"(COM|LPT)[1-9]", reserved):
            raise ClassificationError(
                "{!r} is a Windows reserved device name".format(segment))
        if any(ch in segment for ch in '<>"|?*') or \
                any(ord(ch) < 32 for ch in segment):
            raise ClassificationError(
                "{!r} contains a character Windows reserves".format(segment))
    return path


def normalize_effects(mapping) -> dict:
    """One validated representation, produced once and compared exactly.

    Comparing raw dictionaries differently in several places is how the
    value-blindness below arose. Both sides go through this.
    """
    if not isinstance(mapping, dict) or not mapping:
        raise ClassificationError("an effect mapping must be a nonempty object")
    out, seen = {}, {}
    for path, spec in mapping.items():
        canonical = require_canonical_repository_path(path)
        folded = canonical.lower()
        if folded in seen and seen[folded] != canonical:
            raise ClassificationError(
                "{!r} and {!r} collide on a case-insensitive filesystem"
                .format(seen[folded], canonical))
        seen[folded] = canonical
        if not isinstance(spec, dict):
            raise ClassificationError("{}: effect must be an object".format(path))
        out[canonical] = FileEffect(
            action=spec.get("action"),
            mode=spec.get("mode", "100644"),
            size_bytes=spec.get("size_bytes"),
            content_sha256=spec.get("content_sha256"))
    return out


def require_exact_effects(approved: dict, observed: dict) -> None:
    """Paths AND values. MEASURED 2026-09-08: comparing only keys accepted a
    changed observed digest, a changed size, a symlink mode and an invalid
    manifest action -- four transitions that were not the approved one."""
    if approved.keys() != observed.keys():
        raise ClassificationError(
            "the observed changed-path set is not the approved one. "
            "UNDECLARED: {}. APPROVED BUT ABSENT: {}.".format(
                sorted(set(observed) - set(approved))[:5] or "none",
                sorted(set(approved) - set(observed))[:5] or "none"))
    differing = [p for p in sorted(approved) if approved[p] != observed[p]]
    if differing:
        first = differing[0]
        raise ClassificationError(
            "{} committed effect(s) differ from approval; first is {}: "
            "approved {} observed {}".format(
                len(differing), first, approved[first], observed[first]))


class ClassificationError(RuntimeError):
    """The operation is not what it claims, or cannot be established."""


class OperationKind(Enum):
    INSTALLATION = "installation"
    ARCHIVE_ADMISSION = "archive_admission"


class Obligation(Enum):
    INSTALLATION_ARCHIVE = "installation_archive"
    MAINTENANCE_EVIDENCE = "maintenance_evidence"


def obligations_for(kind: OperationKind) -> frozenset:
    """The FINITE obligation set for a kind. No recursion anywhere in it."""
    if kind is OperationKind.INSTALLATION:
        return frozenset({Obligation.INSTALLATION_ARCHIVE})
    if kind is OperationKind.ARCHIVE_ADMISSION:
        return frozenset({Obligation.MAINTENANCE_EVIDENCE})
    raise ClassificationError(
        "unsupported operation kind: {!r}".format(kind))


def require_exact_approved_write_set(approved, observed) -> None:
    """The observed transition equals the approved one -- both directions.

    A subset check would let an undeclared file ride along; a superset check
    would let an approved one go missing.
    """
    approved_paths = frozenset(approved)
    observed_paths = frozenset(observed)
    if approved_paths != observed_paths:
        raise ClassificationError(
            "the observed write set is not the approved one. UNDECLARED: {}. "
            "APPROVED BUT ABSENT: {}.".format(
                sorted(observed_paths - approved_paths)[:5] or "none",
                sorted(approved_paths - observed_paths)[:5] or "none"))


def require_archive_only_targets(approved) -> None:
    """Every target lies under the archive subtree, and nothing else does.

    This is what a unit changing executable code cannot satisfy, and it is
    checked by ACTION as well as path: the manifest is patched, artifacts are
    created, and nothing is deleted.
    """
    if not approved:
        raise ClassificationError(
            "an archive admission with no targets admits nothing")
    for path, spec in sorted(approved.items()):
        require_canonical_repository_path(path)
        action = spec.get("action")
        if path == MANIFEST_PATH:
            if action != "patch":
                raise ClassificationError(
                    "the manifest must be PATCHED, not {!r}".format(action))
            # NO early continue. MEASURED 2026-09-08: the manifest escaped
            # payload validation here while every artifact was checked.
            digest = spec.get("content_sha256")
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ClassificationError(
                    "{}: content_sha256 must be 64 lowercase hexadecimal "
                    "digits".format(path))
            size = spec.get("size_bytes")
            if type(size) is not int or size < 0:
                raise ClassificationError(
                    "{}: size_bytes must be a nonnegative integer".format(path))
            continue
        if not path.startswith(ARTIFACTS_SUBTREE):
            raise ClassificationError(
                "{!r} lies outside the archive subtree {!r}. An operation that "
                "changes anything else is an ordinary installation, whatever "
                "it calls itself.".format(path, ARCHIVE_SUBTREE))
        if action != "create":
            raise ClassificationError(
                "an admitted artifact must be CREATED, not {!r}: {}".format(
                    action, path))
        digest = spec.get("content_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ClassificationError(
                "{}: content_sha256 must be 64 lowercase hexadecimal "
                "digits".format(path))
        size = spec.get("size_bytes")
        if type(size) is not int or size < 0:
            raise ClassificationError(
                "{}: size_bytes must be a nonnegative integer".format(path))
    if MANIFEST_PATH not in approved:
        raise ClassificationError(
            "an archive admission must patch {}".format(MANIFEST_PATH))


def require_preserved_predecessor(predecessor_paths, observed) -> None:
    """Nothing outside the archive subtree changed, and nothing was deleted."""
    for path, spec in sorted(observed.items()):
        if path != MANIFEST_PATH and not path.startswith(ARTIFACTS_SUBTREE):
            raise ClassificationError(
                "{!r} changed and lies outside the archive subtree".format(
                    path))
        if spec.get("action") == "delete":
            raise ClassificationError(
                "{!r} was DELETED. Admission preserves; it never removes."
                .format(path))
        if spec.get("action") == "create" and path in predecessor_paths:
            raise ClassificationError(
                "{!r} is declared a create and already existed in the "
                "predecessor".format(path))
        if spec.get("action") == "patch" and path not in predecessor_paths:
            raise ClassificationError(
                "{!r} is declared a patch and did not exist in the "
                "predecessor".format(path))


def require_complete_approved_additions(expected_count, observed) -> None:
    """Exactly the approved number of artifact creates, no more and no fewer."""
    creates = [p for p, s in observed.items()
               if p.startswith(ARTIFACTS_SUBTREE) and s.get("action") == "create"]
    if type(expected_count) is not int or expected_count < 1:
        raise ClassificationError(
            "the approved addition count must be a positive integer")
    if len(creates) != expected_count:
        raise ClassificationError(
            "{} artifact create(s) observed and {} approved".format(
                len(creates), expected_count))


def classify_admission_candidate(*, approved_targets, observed_transition,
                                 predecessor_paths, approved_addition_count):
    """STRUCTURAL PREFLIGHT ONLY. It identifies an archive-admission CANDIDATE.

    RENAMED from classify_operation deliberately. The old name implied this
    established the operation kind, and it does not: it receives no manifest
    bytes and invokes no archive-preservation predicate, so it cannot know
    that an existing entry survived unchanged or that the added records'
    complete projections were approved.

    Maintenance-only routing must follow the COMPOSED semantic verification --
    plan owner, Git verifier, archive owner, evidence validator - not this
    function. What it does establish is the transition's SHAPE, exactly:
    canonical paths, supported actions and modes, validated payload
    identities, and observed effects equal to approved ones by VALUE.

    Both sides are normalised through the same constructor, so a caller
    cannot compare two differently-shaped dictionaries and call it agreement.
    """
    approved = normalize_effects(approved_targets)
    observed = normalize_effects(observed_transition)
    require_exact_effects(approved, observed)
    require_archive_only_targets(approved_targets)
    require_preserved_predecessor(predecessor_paths, observed_transition)
    require_complete_approved_additions(approved_addition_count,
                                        observed_transition)
    return OperationKind.ARCHIVE_ADMISSION, obligations_for(
        OperationKind.ARCHIVE_ADMISSION)


#: The old name, retained so no caller silently gets a different meaning.
def classify_operation(**kwargs):
    raise ClassificationError(
        "classify_operation was renamed to classify_admission_candidate "
        "because it establishes the transition's SHAPE, not the operation's "
        "kind. Maintenance-only routing must follow composed semantic "
        "verification.")


#: The channel's contract, recorded as data so the installer can check it
#: rather than a reader having to trust prose.
MAINTENANCE_CHANNEL_CONTRACT = {
    "ownership": "the existing evidence and publication owner, extended",
    "identity": ["operation_id", "attempt_id", "evidence_role",
                 "content_sha256"],
    "binding": ["plan_sha256", "candidate_commit", "integrated_commit",
                "preimage_manifest_sha256", "postimage_manifest_sha256"],
    "publication": "complete validated bytes, staged and installed without "
                   "replacing conflicting evidence",
    "retention": "permanent while the archive history it explains is retained",
    "discovery": "enumerated independently of transaction journals",
    "recovery": "existing valid evidence is reconciled; incomplete or "
                "conflicting evidence stays visible",
    "cleanup": "EXCLUDED from transaction cleanup and ordinary cache "
               "eviction. The name cache_root confers no durability; the "
               "configured directory's cleanup, backup and recovery behaviour "
               "must satisfy this contract.",
    "replication": "an explicit backup policy with a verified restoration "
                   "path -- NOT YET IMPLEMENTED",
    "recursion": "successful preservation creates no further "
                 "installation-admission obligation",
    "location": "operations/<operation-id>/evidence/",
    "not_included": [
        "The 106-entry installation manifest gains no 107th entry.",
        "This module classifies and refuses; it stores nothing.",
    ],
}
