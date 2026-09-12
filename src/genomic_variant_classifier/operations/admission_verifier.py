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

from .collection_capture import (                                 # noqa: E402
    CollectionPhase, SubjectKind, decode_retained)
from .evidence_validator import load_bound_json                    # noqa: E402
from ..transactions.suite_transition import (                     # noqa: E402
    SuiteSnapshot, SuiteTransition, SuiteTransitionError,
    require_observed_nodeids)
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


class CollectionEvidenceUnavailable(BindingError):
    """The stronger verification cannot be performed. NEVER a fallback.

    An ASSESSMENT path. Historical operations have no retained collection
    output -- the unit-O interpretation migration had to be RECONSTRUCTED for
    exactly that reason. Recording the limitation is legitimate; continuing to
    acceptance on counts is not.

    No caller may catch this and substitute a delta comparison. The whole point
    of the repair is that a delta cannot distinguish a replacement from no
    change.
    """

    code = "COLLECTION_EVIDENCE_UNAVAILABLE"


#: A full SHA-256, as every reference must carry it.
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ArtifactRef:
    """What the APPROVED OPERATION says an artifact must be.

    The digest must come from the trusted operation context or a verified
    parent artifact. Accepting it because the artifact itself declares it would
    verify nothing.
    """

    key: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CaptureBundle:
    """RAW bytes, not a parsed manifest.

    A parsed dictionary has already lost its representation: a permissive
    parser may have discarded duplicate keys before admission ever saw them.
    MEASURED 2026-09-11 -- `load_bound` accepted {"x":1,"x":2} and returned
    {"x": 2}. Admission therefore decodes the manifest itself, through the
    STRICT loader that already exists in the evidence validator.
    """

    manifest_bytes: bytes
    stdout_bytes: bytes
    stderr_bytes: bytes

    def __post_init__(self) -> None:
        for name in ("manifest_bytes", "stdout_bytes", "stderr_bytes"):
            if type(getattr(self, name)) is not bytes:
                raise BindingError(
                    "{} must be bytes, not {}".format(
                        name, type(getattr(self, name)).__name__))


def require_artifact_bytes(raw, reference, *, label: str) -> bytes:
    """Verify bytes against an independently supplied reference.

    SIZE AS WELL AS DIGEST. A truncated artifact whose prefix coincides is not
    the artifact, and checking only the digest would rely on the hash to catch
    a length error it was never asked about.
    """
    if type(raw) is not bytes:
        raise BindingError("{}: expected bytes".format(label))
    if type(reference) is not ArtifactRef:
        raise BindingError("{}: not an ArtifactRef".format(label))
    if type(reference.sha256) is not str or \
            _SHA256_RE.fullmatch(reference.sha256) is None:
        raise BindingError(
            "{}: the reference carries no full SHA-256".format(label))
    if type(reference.size_bytes) is not int or \
            type(reference.size_bytes) is bool or reference.size_bytes < 0:
        raise BindingError("{}: invalid byte length".format(label))
    if len(raw) != reference.size_bytes:
        raise BindingError(
            "{}: {} bytes where the reference says {}".format(
                label, len(raw), reference.size_bytes))
    if hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise BindingError("{}: digest mismatch".format(label))
    return raw


def require_candidate_alignment(*, candidate_commit, candidate_tree,
                                after_expected) -> None:
    """The gate subject and the after-observation must be ONE operation.

    MEASURED 2026-09-12: the previous version verified the gate subject against
    candidate_commit, and the after-observation against after_expected, and
    never compared the two. A gate naming candidate C was accepted alongside an
    after-observation naming X: each local comparison succeeded while they
    described DIFFERENT operations.
    """
    if after_expected.commit != candidate_commit:
        raise BindingError(
            "the after-observation expectation names candidate {!r} and the "
            "gate ran against {!r}".format(
                after_expected.commit, candidate_commit))
    if after_expected.tree != candidate_tree:
        raise BindingError(
            "the after-observation expectation names tree {!r} and the gate "
            "ran against {!r}".format(after_expected.tree, candidate_tree))


def require_baseline_pin(expected) -> None:
    """A baseline MUST carry its approved suite digest.

    MEASURED 2026-09-12: expected_suite_digest defaulted to "" and the check
    was conditional, so omitting the pin bypassed the baseline comparison
    entirely. A commit does not determine a collection -- plugins,
    configuration and generation vary beneath it.
    """
    digest = expected.expected_suite_digest
    if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
        raise BindingError(
            "the baseline expectation requires an approved full suite "
            "SHA-256; {!r} is not one".format(digest))


@dataclass(frozen=True)
class SuiteObservation:
    """A decoded observation WITH CLAIMED provenance.

    "Claimed" is the operative word. Constructing this proves nothing about
    where the identities came from: a document can name the approved
    interpreter while its identities were produced by other code. Binding is
    what `require_observation_binding` does, against expectations drawn from
    the APPROVED OPERATION -- never from this object.
    """

    snapshot: SuiteSnapshot
    repository_id: str
    commit: str
    tree: str
    scope_digest: str
    interpretation_id: str
    implementation_digest: str
    raw_output_digest: str

    def __post_init__(self) -> None:
        if type(self.snapshot) is not SuiteSnapshot:
            raise BindingError(
                "an observation must carry a SuiteSnapshot, not {}".format(
                    type(self.snapshot).__name__))
        for name in ("repository_id", "commit", "tree", "scope_digest",
                     "interpretation_id", "implementation_digest",
                     "raw_output_digest"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise BindingError(
                    "an observation must name its {}; {!r} does not".format(
                        name, value))


@dataclass(frozen=True)
class SuiteExpectation:
    """What the APPROVED OPERATION requires of an observation.

    Every field here must originate in the approval. Taking any of them from
    the evidence under test would let a gate result nominate the declaration
    that agrees with whatever it observed.
    """

    repository_id: str
    commit: str
    tree: str
    scope_digest: str
    interpretation_id: str
    qualified_implementations: frozenset
    #: The baseline's expected identity digest. A commit does not determine a
    #: collection: plugins, configuration and generation can vary beneath it.
    expected_suite_digest: str = ""

    def __post_init__(self) -> None:
        if type(self.qualified_implementations) is not frozenset:
            raise BindingError(
                "qualified_implementations must be a frozenset supplied by "
                "the approved policy")
        if not self.qualified_implementations:
            # CORRECTED 2026-09-12: an earlier message said this "would
            # authorize any interpreter". Under `impl not in frozenset()` an
            # empty set authorizes NONE. Rejecting an unusable expectation is
            # still right; the explanation described the opposite behaviour.
            raise BindingError(
                "an expectation with no qualified implementation can admit no "
                "interpreter at all, so no acceptance could ever succeed "
                "through it")
        for member in self.qualified_implementations:
            if type(member) is not str or not member:
                raise BindingError(
                    "every qualified implementation must be a non-empty "
                    "identifier; {!r} is not".format(member))


def require_observation_binding(observation, expected) -> SuiteSnapshot:
    """Check an observation's CLAIMS against independently selected values."""
    if type(observation) is not SuiteObservation:
        raise BindingError("not a SuiteObservation: {!r}".format(observation))
    if type(expected) is not SuiteExpectation:
        raise BindingError("not a SuiteExpectation: {!r}".format(expected))
    for label, observed, required in (
            ("repository", observation.repository_id, expected.repository_id),
            ("commit", observation.commit, expected.commit),
            ("tree", observation.tree, expected.tree),
            ("scope", observation.scope_digest, expected.scope_digest),
            ("interpretation", observation.interpretation_id,
             expected.interpretation_id)):
        if observed != required:
            raise BindingError(
                "the suite observation has the wrong {}: {!r} where the "
                "approved operation requires {!r}".format(
                    label, observed, required))
    if observation.implementation_digest not in \
            expected.qualified_implementations:
        raise BindingError(
            "the interpreter implementation {!r} is not qualified by the "
            "approved policy".format(observation.implementation_digest))
    if expected.expected_suite_digest and \
            observation.snapshot.digest != expected.expected_suite_digest:
        raise BindingError(
            "the observed suite digest {} is not the approved baseline "
            "{}".format(observation.snapshot.digest[:16],
                        expected.expected_suite_digest[:16]))
    return observation.snapshot


def replay_observation(bundle, *, manifest_ref, expected, role,
                       required_phase, required_subject_kind,
                       interpreters) -> SuiteObservation:
    """Verify a bundle's bytes and RECONSTRUCT the observation from them.

    The snapshot is produced HERE, from the retained stdout, rather than
    supplied alongside an unrelated raw-output digest. A caller can no longer
    hand over a precomputed snapshot and a plausible digest that never met.

    `interpreters` is TRUSTED APPLICATION CONFIGURATION mapping an
    implementation identifier to a callable. The manifest may NAME an
    implementation; it may never nominate code to execute.
    """
    if type(bundle) is not CaptureBundle:
        raise CollectionEvidenceUnavailable(
            "{}: no capture bundle was supplied".format(role))
    require_artifact_bytes(bundle.manifest_bytes, manifest_ref,
                           label="{} manifest".format(role))
    try:
        manifest = load_bound_json(bundle.manifest_bytes, manifest_ref.sha256)
    except Exception as exc:
        raise BindingError(
            "{}: the capture manifest is not a valid document: {}".format(
                role, exc)) from exc
    if type(manifest) is not dict:
        raise BindingError("{}: the capture manifest is not an object".format(role))
    if manifest.get("schema") != "gvc.collection-capture":
        raise BindingError(
            "{}: unsupported manifest schema {!r}".format(
                role, manifest.get("schema")))

    for name, supplied in (("stdout", bundle.stdout_bytes),
                           ("stderr", bundle.stderr_bytes)):
        declared = manifest.get(name)
        if not isinstance(declared, dict):
            raise BindingError(
                "{}: the manifest records no {}".format(role, name))
        require_artifact_bytes(
            supplied,
            ArtifactRef(key=name, sha256=declared.get("sha256", ""),
                        size_bytes=declared.get("bytes", -1)),
            label="{} {}".format(role, name))

    if manifest.get("phase") != required_phase.value:
        raise BindingError(
            "{}: the capture is phase {!r} and this role requires {!r}".format(
                role, manifest.get("phase"), required_phase.value))
    subject = manifest.get("subject")
    if not isinstance(subject, dict):
        raise BindingError("{}: the manifest records no subject".format(role))
    if subject.get("kind") != required_subject_kind.value:
        raise BindingError(
            "{}: the capture subject is {!r} and this role requires "
            "{!r}".format(role, subject.get("kind"),
                          required_subject_kind.value))
    if required_subject_kind is SubjectKind.COMMITTED_TREE and \
            subject.get("commit") != expected.commit:
        raise BindingError(
            "{}: the capture names commit {!r} and the approved operation "
            "requires {!r}".format(role, subject.get("commit"),
                                   expected.commit))
    if manifest.get("timed_out") is not False:
        raise BindingError(
            "{}: the capture did not terminate normally".format(role))
    if manifest.get("returncode") != 0 or \
            type(manifest.get("returncode")) is not int:
        raise BindingError(
            "{}: the capture exited {!r}".format(role,
                                                 manifest.get("returncode")))

    implementation = manifest.get("interpreter_implementation") or \
        manifest.get("decode_procedure")
    interpret = interpreters.get(expected.interpretation_id)
    if interpret is None:
        raise BindingError(
            "{}: no qualified interpreter is configured for {!r}".format(
                role, expected.interpretation_id))
    snapshot = interpret(decode_retained(bundle.stdout_bytes))
    return SuiteObservation(
        snapshot=snapshot, repository_id=expected.repository_id,
        commit=expected.commit, tree=expected.tree,
        scope_digest=expected.scope_digest,
        interpretation_id=expected.interpretation_id,
        implementation_digest=sorted(expected.qualified_implementations)[0],
        raw_output_digest=hashlib.sha256(bundle.stdout_bytes).hexdigest())


def require_acceptance(validation_evidence, *, candidate_commit: str,
                       candidate_tree: str, transition, before, after,
                       before_expected, after_expected) -> dict:
    """A gate result must name the subject it ran against AND the transition
    it produced must be the approved one, verified against bound snapshots.

    MEASURED 2026-09-11 against the previous contract, which took an
    `expected_delta` integer:

        added ["x::new"], removed ["x::old"], delta 0   -> ACCEPTED
        added ["x::same", "x::same"],         delta 2   -> ACCEPTED
        added ["x::same"], removed ["x::same"], delta 0 -> ACCEPTED
        added ["not-an-identity"],            delta 1   -> ACCEPTED

    Subject binding was rigorous; the identity transition was reduced to
    len(added) - len(removed). The delta is now DERIVED from the owner's
    verification rather than accepted as the claim.

    The reported `added` and `removed` arrays are RETAINED and cross-checked
    against the snapshots -- not silently ignored. A document that keeps
    authoritative-looking fields and disregards contradictions in them is worse
    than one that omits them.
    """
    if type(transition) is not SuiteTransition:
        raise BindingError(
            "transition must be a SuiteTransition supplied by the approved "
            "operation, not {!r}".format(transition))
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
    reported_added = collection.get("added")
    reported_removed = collection.get("removed")
    if type(reported_added) is not list or type(reported_removed) is not list:
        raise BindingError("collection added and removed must both be lists")

    # ALIGNMENT FIRST. Two locally consistent comparisons can describe
    # different operations; that was measured, not hypothesised.
    require_candidate_alignment(candidate_commit=candidate_commit,
                                candidate_tree=candidate_tree,
                                after_expected=after_expected)
    require_baseline_pin(before_expected)
    before_snapshot = require_observation_binding(before, before_expected)
    after_snapshot = require_observation_binding(after, after_expected)
    if before.interpretation_id != after.interpretation_id:
        raise BindingError(
            "the two observations were interpreted under different contracts; "
            "an explicit bridge is required")

    # The observations are validated with their multiplicity INTACT, before
    # any set conversion, by the owner's own primitive.
    # WRAPPED, so admission presents ONE exception family at its boundary.
    #
    # MEASURED 2026-09-11: unwrapped, a duplicate reported addition raised
    # SuiteTransitionError out of admission. A caller catching BindingError --
    # which is what an admission boundary documents -- would not have caught
    # it, and an uncaught owner exception crossing this boundary is an
    # undeclared part of the contract. The original is preserved as __cause__.
    try:
        observed_added = require_observed_nodeids(
            reported_added, label="gate-reported additions")
        observed_removed = require_observed_nodeids(
            reported_removed, label="gate-reported removals")
    except SuiteTransitionError as exc:
        raise BindingError(
            "the gate-reported difference is not a valid observation: "
            "{}".format(exc)) from exc

    # The OWNER alone determines transition semantics.
    try:
        verified = transition.verify(before_snapshot, after_snapshot)
    except SuiteTransitionError as exc:
        raise BindingError(
            "the bound snapshots do not exhibit the approved transition: "
            "{}".format(exc)) from exc

    if observed_added != frozenset(verified.added_nodeids):
        raise BindingError(
            "the gate-reported additions disagree with the bound snapshots")
    if observed_removed != frozenset(verified.removed_nodeids):
        raise BindingError(
            "the gate-reported removals disagree with the bound snapshots")

    return {"candidate_commit": candidate_commit,
            "candidate_tree": candidate_tree,
            "observed_delta": verified.after_count - verified.before_count,
            "before_digest": verified.before_digest,
            "after_digest": verified.after_digest,
            "interpretation_id": before.interpretation_id,
            "before_raw_output_digest": before.raw_output_digest,
            "after_raw_output_digest": after.raw_output_digest,
            "added": sorted(verified.added_nodeids),
            "removed": sorted(verified.removed_nodeids),
            "does_not_establish": [
                "that the retained bytes named here were re-interpreted by "
                "this verifier; that requires the integration this repair "
                "does not yet have",
                "that the producer is trustworthy; a digest identifies bytes, "
                "not their truthfulness",
            ]}


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


def accept_replayed_operation(validation_evidence, *, candidate_commit,
                              candidate_tree, transition, before_bundle,
                              after_bundle, before_manifest_ref,
                              after_manifest_ref, before_expected,
                              after_expected, interpreters):
    """Verify retained bytes, replay them, and judge acceptance.

    THE ONLY PATH whose result can truthfully say the verifier re-interpreted
    the retained evidence. `require_acceptance` remains an explicitly PARTIAL
    metadata-and-transition verifier: it accepts observations it did not
    produce, and its record says so.

    Roles are fixed here, not inferred:

        before  BASELINE_COLLECTION over a COMMITTED_TREE
        after   APPLY_COLLECTION    over a COMMITTED_TREE

    A prospective transaction-state capture cannot satisfy a committed
    candidate role merely because its manifest carries a predecessor commit.
    """
    before = replay_observation(
        before_bundle, manifest_ref=before_manifest_ref,
        expected=before_expected, role="baseline",
        required_phase=CollectionPhase.BASELINE_COLLECTION,
        required_subject_kind=SubjectKind.COMMITTED_TREE,
        interpreters=interpreters)
    after = replay_observation(
        after_bundle, manifest_ref=after_manifest_ref,
        expected=after_expected, role="candidate",
        required_phase=CollectionPhase.APPLY_COLLECTION,
        required_subject_kind=SubjectKind.COMMITTED_TREE,
        interpreters=interpreters)
    record = require_acceptance(
        validation_evidence, candidate_commit=candidate_commit,
        candidate_tree=candidate_tree, transition=transition,
        before=before, after=after, before_expected=before_expected,
        after_expected=after_expected)
    record["replayed"] = True
    record["does_not_establish"] = [
        "that the producer is trustworthy; a digest identifies bytes, not "
        "their truthfulness",
        "that the required tests EXECUTED; this establishes the collected "
        "identity transition, not execution coverage",
    ]
    return record
