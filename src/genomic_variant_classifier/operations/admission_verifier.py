"""One admission decision, composed from independent owners.

Author: Monzia Moodie

WHY COMPOSED, AND WHY NOT A CLASSIFIER
======================================
`classify_admission_candidate` establishes the transition's SHAPE. It receives
no manifest bytes and invokes no archive predicate, so it cannot know that an
existing entry survived unchanged or that an added record's projection was
approved. Routing evidence on its answer would route on a partial check.

    authorized admission =
        valid policy AND bound evidence AND exact candidate transition
        AND archive semantics AND required validation

Each conjunct is delegated to the owner that already owns it. This module
composes; it does not reimplement.

THE OBSERVATION MUST COME FROM THE COMMIT
=========================================
`observed = dict(approved)` is a positive fixture and proves nothing about
whether execution is bound to approval: it compares approval with itself.
Every observation here is read from the CANDIDATE COMMIT'S OBJECTS -- modes
and blob identifiers from `git ls-tree`, bytes from `git cat-file` -- so a
filter, a line-ending conversion or a hand-edited working tree cannot pass
unnoticed.

WHAT A VerifiedAdmission IS NOT
===============================
Not an unforgeable capability. Within one process a dataclass can be
constructed by anyone. Its value is that production routing must obtain it
through this path, and that boundary is what the tests exercise.

EXIT STATUS when run as a script
  0  every check held
  2  a check did not hold
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .operation_kind import (                                    # noqa: E402
    ARTIFACTS_SUBTREE, MANIFEST_PATH, ClassificationError, FileEffect,
    OperationKind, classify_admission_candidate, normalize_effects,
    obligations_for, require_canonical_repository_path, require_exact_effects)


_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


class AdmissionError(RuntimeError):
    """The admission is not authorized. Composed owners raise their own."""


class BindingError(AdmissionError):
    """A claimed identity does not identify the bytes actually used."""


def load_bound(raw: bytes, expected_sha256: str, what: str):
    """Parse ONLY after the bytes match the approved digest.

    MEASURED 2026-09-08 against the previous entry point: `plan_sha256` was a
    caller-supplied STRING that nothing checked -- passing "not-a-digest" was
    accepted and copied verbatim into the result. A claimed identity must
    identify the bytes actually parsed, or it is decoration.
    """
    if type(expected_sha256) is not str or \
            not _SHA256.fullmatch(expected_sha256):
        raise BindingError(
            "the approved {} digest must be 64 lowercase hexadecimal digits, "
            "not {!r}".format(what, expected_sha256))
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise BindingError(
            "{} bytes are {} and the approved digest is {}".format(
                what, observed, expected_sha256))
    return json.loads(raw.decode("utf-8"))


def require_dependency_bindings(plan, *, census_bytes, citation_bytes,
                                approved_entries_bytes, postimage_bytes):
    """Every dependency the plan DECLARES must be the bytes supplied.

    MEASURED 2026-09-08: `census["_bound_sha256"]` was a field the caller
    invented, so the census digest asserted nothing about the census.
    """
    declared = plan.get("derived_from") or {}
    supplied = {"census": census_bytes, "citations": citation_bytes,
                "approved_entries": approved_entries_bytes}
    parsed = {}
    for name, raw in sorted(supplied.items()):
        binding = declared.get(name)
        if not isinstance(binding, dict) or "sha256" not in binding:
            raise BindingError(
                "the plan declares no digest for its {} dependency".format(
                    name))
        parsed[name] = load_bound(raw, binding["sha256"], name)
    approved_postimage = plan.get("postimage_manifest", {}).get("sha256")
    if type(approved_postimage) is not str or \
            not _SHA256.fullmatch(approved_postimage):
        raise BindingError("the plan declares no postimage digest")
    observed = hashlib.sha256(postimage_bytes).hexdigest()
    if observed != approved_postimage:
        raise BindingError(
            "postimage bytes are {} and the plan declares {}".format(
                observed, approved_postimage))
    return parsed


def require_acceptance(validation_evidence, *, candidate_commit: str,
                       candidate_tree: str, expected_delta: int) -> dict:
    """A gate result must name the exact subject it ran against.

    A passing gate proves nothing about THIS candidate unless it says which
    commit and tree it examined. Without this, evidence from any successful
    run could be presented for any candidate.

    `expected_delta` is compared by TYPE as well as value: a Boolean True
    would otherwise satisfy an expected delta of one.
    """
    if not isinstance(validation_evidence, dict):
        raise BindingError("the gate result is not an object")
    subject = validation_evidence.get("subject")
    if not isinstance(subject, dict):
        raise BindingError("the gate result names no subject")
    if subject.get("candidate_commit") != candidate_commit:
        raise BindingError(
            "the gate ran against candidate {!r} and this candidate is "
            "{!r}".format(subject.get("candidate_commit"), candidate_commit))
    if subject.get("candidate_tree") != candidate_tree:
        raise BindingError(
            "the gate ran against tree {!r} and this candidate's tree is "
            "{!r}".format(subject.get("candidate_tree"), candidate_tree))
    execution = validation_evidence.get("execution")
    if not isinstance(execution, dict):
        raise BindingError("the gate result records no execution")
    if execution.get("exit_code") != 0 or \
            type(execution.get("exit_code")) is not int:
        raise BindingError(
            "the gate did not exit successfully: {!r}".format(
                execution.get("exit_code")))
    collection = validation_evidence.get("collection")
    if not isinstance(collection, dict):
        raise BindingError("the gate result records no collection")
    added = collection.get("added")
    removed = collection.get("removed")
    if type(added) is not list or type(removed) is not list:
        raise BindingError(
            "collection added and removed must both be lists")
    if type(expected_delta) is not int or type(expected_delta) is bool:
        raise BindingError(
            "expected_delta must be an integer, not {!r}".format(
                expected_delta))
    observed = len(added) - len(removed)
    if observed != expected_delta:
        raise BindingError(
            "the gate observed a delta of {} and the plan declares "
            "{}".format(observed, expected_delta))
    return {"candidate_commit": candidate_commit,
            "candidate_tree": candidate_tree,
            "expected_delta": expected_delta,
            "added": sorted(added), "removed": sorted(removed)}


def require_manifest_citation_agreement(after_entries, approved_entries,
                                        citation_report) -> None:
    """An admitted record's cited_by must be what the citation evidence says.

    MEASURED 2026-09-08: changing one citation row to a different but
    structurally valid citing commit, leaving the approved entries and the
    committed manifest untouched, was ACCEPTED. The evidence and the manifest
    could disagree and nothing noticed.
    """
    derived = citation_report.get("derived") or {}
    disagreeing = []
    for entry in after_entries:
        if entry["record_id"] not in approved_entries:
            continue
        aliases = entry.get("legacy_aliases") or []
        if not aliases:
            disagreeing.append((entry["record_id"], "no alias to cite by"))
            continue
        row = derived.get(aliases[0])
        if row is None:
            disagreeing.append((aliases[0], "absent from the citation report"))
            continue
        if sorted(entry.get("cited_by") or []) != sorted(row.get("cited_by")
                                                         or []):
            disagreeing.append(
                (aliases[0], "manifest {} vs evidence {}".format(
                    entry.get("cited_by"), row.get("cited_by"))))
    if disagreeing:
        raise AdmissionError(
            "{} admitted record(s) disagree with the citation evidence: "
            "{}".format(len(disagreeing), disagreeing[:3]))


@dataclass(frozen=True)
class VerifiedTransition:
    """What the TRANSITION verifier established, named for what it proves.

    RENAMED from VerifiedAdmission. That name promised authorization, and this
    value does not carry it: no independently selected policy and no gate
    result contributed to it. A consumer that routed evidence on this alone
    would be routing on the transition conjunct only.
    """

    plan_sha256: str
    predecessor_commit: str
    candidate_commit: str
    candidate_tree: str
    postimage_manifest_sha256: str
    admitted_record_ids: frozenset
    kind: OperationKind
    obligations: frozenset


def git_bytes(repo: Path, *args: str) -> bytes:
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, timeout=900)
    if done.returncode != 0:
        raise AdmissionError("git {} exited {}: {}".format(
            " ".join(args), done.returncode,
            done.stderr.decode("utf-8", "replace").strip()))
    return done.stdout


def tree_entries(repo: Path, commit: str) -> dict:
    """path -> (mode, kind, oid), read from the COMMIT, not the worktree."""
    result = {}
    raw = git_bytes(repo, "ls-tree", "-r", "-z", "--full-tree", commit)
    for item in raw.split(b"\x00"):
        if not item:
            continue
        metadata, path = item.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        result[path.decode("utf-8")] = (mode.decode("ascii"),
                                        kind.decode("ascii"),
                                        oid.decode("ascii"))
    return result


def observed_effects(repo: Path, predecessor: str, candidate: str) -> dict:
    """The committed transition, derived from the two commits themselves.

    Nothing here consults the plan, so the result is an OBSERVATION rather
    than a restatement of what was approved.
    """
    before = tree_entries(repo, predecessor)
    after = tree_entries(repo, candidate)
    effects = {}
    for path in sorted(before.keys() | after.keys()):
        if before.get(path) == after.get(path):
            continue
        if path not in after:
            raise AdmissionError(
                "{!r} was DELETED by the candidate. Admission preserves."
                .format(path))
        mode, kind, oid = after[path]
        if kind != "blob":
            raise AdmissionError(
                "{!r} is committed as a {}, not a blob".format(path, kind))
        payload = git_bytes(repo, "cat-file", "blob", oid)
        effects[path] = {
            "action": "patch" if path in before else "create",
            "mode": mode,
            "size_bytes": len(payload),
            "content_sha256": hashlib.sha256(payload).hexdigest()}
    return effects


def require_sole_parent(repo: Path, candidate: str, predecessor: str) -> str:
    fields = git_bytes(repo, "rev-list", "--parents", "-n", "1",
                       candidate).decode("ascii").split()
    if fields != [candidate, predecessor]:
        raise AdmissionError(
            "expected exactly commit {} with sole parent {}; observed "
            "{}".format(candidate, predecessor, fields))
    return git_bytes(repo, "rev-parse",
                     candidate + "^{tree}").decode("ascii").strip()


def require_created_artifact_correspondence(after_entries, effects) -> None:
    """One added record to one created artifact, both ways.

    An added record with no file, or a created file with no record, is a
    manifest that describes something other than what was committed.
    """
    created = {p for p, e in effects.items()
               if e["action"] == "create" and p.startswith(ARTIFACTS_SUBTREE)}
    recorded = {e["canonical_path"] for e in after_entries}
    unrecorded = sorted(created - recorded)
    if unrecorded:
        raise AdmissionError(
            "{} created artifact(s) have no manifest record: {}".format(
                len(unrecorded), unrecorded[:3]))
    return created, recorded


def verify_transition(*, repo: Path, plan: dict, plan_sha256: str,
                      candidate_commit: str, approved_entries: dict,
                      postimage_bytes: bytes, evidence_report: dict,
                      census: dict, census_sha256: str, owners):
    """The TRANSITION verifier. RENAMED from verify_admission.

    It establishes that the committed transition is the approved one and that
    the archive semantics hold. It does NOT establish authorization: it
    receives no independently selected policy and no gate result, so its
    result is named for what it proves.

    `authorize_admission` is the boundary that binds bytes, policy and
    validation evidence; this remains its transition conjunct.
    """
    predecessor = plan["predecessor_commit"]

    # 1. EVIDENCE -- bound, current-shape, and internally consistent.
    owners.evidence.require_current_citation_shape(
        evidence_report,
        expected_basenames=[r["basename"] for r in census["candidates"]],
        expected_accepted_count=census["manifest"]["entries"])
    owners.evidence.require_citation_evidence(
        evidence_report, predecessor=predecessor,
        census_sha256=census_sha256)

    # 2. GIT -- the actual candidate, its sole parent, its committed effects.
    tree = require_sole_parent(repo, candidate_commit, predecessor)
    effects = observed_effects(repo, predecessor, candidate_commit)

    approved = {t["to_path"]: {"action": "create",
                              "mode": "100644",
                              "size_bytes": t["size_bytes"],
                              "content_sha256": t["content_sha256"]}
                for t in plan["artifacts_to_copy"]}
    approved[MANIFEST_PATH] = {
        "action": "patch", "mode": "100644",
        "size_bytes": len(postimage_bytes),
        "content_sha256": hashlib.sha256(postimage_bytes).hexdigest()}
    require_exact_effects(normalize_effects(approved),
                          normalize_effects(effects))

    # 3. SHAPE -- structural preflight, now over observations from the commit.
    kind, obligations = classify_admission_candidate(
        approved_targets=approved, observed_transition=effects,
        predecessor_paths=frozenset(tree_entries(repo, predecessor)),
        approved_addition_count=len(plan["artifacts_to_copy"]))

    # 4. ARCHIVE SEMANTICS -- from the manifests the COMMITS actually hold.
    before_manifest = owners.archive.parse(
        git_show(repo, predecessor, MANIFEST_PATH))
    after_raw = git_show(repo, candidate_commit, MANIFEST_PATH)
    if after_raw != postimage_bytes:
        raise AdmissionError(
            "the committed manifest is not the approved postimage")
    after_manifest = owners.archive.parse(after_raw)
    owners.archive.require_archive_preservation(before_manifest,
                                                after_manifest)
    report = owners.archive.require_approved_addition(
        before_manifest, after_manifest, approved_entries=approved_entries)

    # 5. CORRESPONDENCE -- records and files describe each other.
    after_entries = json.loads(after_raw.decode("utf-8"))["entries"]
    created, recorded = require_created_artifact_correspondence(
        after_entries, effects)
    admitted_paths = {e["canonical_path"] for e in after_entries
                      if e["record_id"] in approved_entries}
    if admitted_paths != created:
        raise AdmissionError(
            "the approved records and the created artifacts are not in "
            "one-to-one correspondence: {} recorded, {} created".format(
                len(admitted_paths), len(created)))
    for entry in after_entries:
        if entry["record_id"] not in approved_entries:
            continue
        effect = effects[entry["canonical_path"]]
        if entry["content_sha256"] != effect["content_sha256"] or \
                entry["size_bytes"] != effect["size_bytes"]:
            raise AdmissionError(
                "{}: the manifest record disagrees with the committed "
                "bytes".format(entry["canonical_path"]))

    if report["admitted_count"] != len(plan["artifacts_to_copy"]):
        raise AdmissionError(
            "{} records admitted and {} artifacts planned".format(
                report["admitted_count"], len(plan["artifacts_to_copy"])))

    require_manifest_citation_agreement(after_entries, approved_entries,
                                        evidence_report)

    return VerifiedTransition(
        plan_sha256=plan_sha256, predecessor_commit=predecessor,
        candidate_commit=candidate_commit, candidate_tree=tree,
        postimage_manifest_sha256=hashlib.sha256(postimage_bytes).hexdigest(),
        admitted_record_ids=frozenset(approved_entries),
        kind=kind, obligations=obligations)


def git_show(repo: Path, commit: str, path: str) -> bytes:
    return git_bytes(repo, "show", "{}:{}".format(commit, path))


#: `census_digest` is DELETED. MEASURED 2026-09-08: it returned
#: `census["_bound_sha256"]`, a field the caller invented, so a "bound" digest
#: asserted nothing about the census bytes. The digest now comes from
#: `require_dependency_bindings`, which hashes the supplied bytes and compares
#: them with what the plan declares.
