"""The transition verifier, against a REAL repository and REAL commits.

Author: Monzia Moodie

TWO DISCIPLINES THIS MODULE ENFORCES ON ITSELF
==============================================
Every negative control asserts the SPECIFIC owner exception, and where useful
a reason. A prototype for these tests caught bare `Exception`, which would let
a NameError, a broken import or a wrong adapter count as a successful refusal.

Every sabotage commit is a SIBLING of the candidate -- its sole parent is the
approved predecessor. An earlier fixture built them as CHILDREN, so
`require_sole_parent` and the evidence predecessor check fired before the byte
and archive checks were ever reached, and two refusals passed for reasons
unrelated to the property under test.

The repository precedent for creating repositories and launching processes in
this tier is `test_a_real_process_KILL_is_recoverable`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import types

import pytest

from genomic_variant_classifier.operations import evidence_validator
from genomic_variant_classifier.operations.admission_verifier import (
    AdmissionError, BindingError, load_bound, observed_effects,
    require_acceptance, require_dependency_bindings,
    require_manifest_citation_agreement, require_sole_parent,
    verify_transition)
from genomic_variant_classifier.operations.operation_kind import (
    ARTIFACTS_SUBTREE, MANIFEST_PATH, ClassificationError)
from genomic_variant_classifier.repository_records import archive_guard
from genomic_variant_classifier.repository_records.archive_manifest import (
    ArchiveManifest, ArchiveManifestError)

CENSUS_SHA = "c" * 64


def _render(document):
    return ArchiveManifest.parse(
        (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True)
         + "\n").encode("utf-8"))


def _entry(record_id, name, payload, cited):
    return {"record_id": record_id,
            "canonical_path": ARTIFACTS_SUBTREE + name,
            "content_sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload), "legacy_aliases": [name],
            "cited_by": [cited], "role": "installation_attestation",
            "disclosure": "public_verbatim",
            "preservation": "admitted_verbatim",
            "retention": "permanent_evidence",
            "provenance": ["emitted_by_installer", "imported_from_staging"],
            "artifact_schema_version": 3}


class Candidate:
    """A real repository with a real approved candidate commit."""

    def __init__(self, root):
        self.repo = root / "repo"
        (self.repo / ARTIFACTS_SUBTREE).mkdir(parents=True)
        (self.repo / "src").mkdir()
        (self.repo / "src" / "module.py").write_bytes(b"x = 1\n")
        self.env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_CONFIG_SYSTEM=os.devnull)

        genesis = []
        for i in range(2):
            name = "install-attestation-GENESIS-{}.json".format(i)
            payload = json.dumps({"g": i}, sort_keys=True).encode()
            (self.repo / ARTIFACTS_SUBTREE / name).write_bytes(payload)
            genesis.append(_entry("REC-" + "{:032x}".format(i), name, payload,
                                  "aaaaaaa"))
        base = {"schema": "gvc.installation-attestation-archive",
                "schema_version": 1,
                "artifact_class": "installation_attestation",
                "genesis_cardinality": 2,
                "genesis_aliases": sorted(e["legacy_aliases"][0]
                                          for e in genesis),
                "entries": sorted(genesis, key=lambda r: r["canonical_path"])}
        self.pre_bytes = _render(base).render()
        (self.repo / MANIFEST_PATH).write_bytes(self.pre_bytes)
        for command in (["init", "-q"], ["config", "user.email", "t@t"],
                        ["config", "user.name", "t"], ["add", "-A"],
                        ["commit", "-qm", "preimage"]):
            self.git(*command)
        self.predecessor = self.git("rev-parse", "HEAD")

        self.payloads, added, copies = {}, [], []
        for i in range(3):
            name = "install-attestation-CANDIDATE-{}.json".format(i)
            payload = json.dumps({"c": i}, sort_keys=True).encode()
            path = ARTIFACTS_SUBTREE + name
            (self.repo / path).write_bytes(payload)
            self.payloads[path] = payload
            record = _entry("REC-" + "{:032x}".format(100 + i), name, payload,
                            "bbbbbb{}".format(i))
            added.append(record)
            copies.append({"from_basename": name, "to_path": path,
                           "content_sha256": record["content_sha256"],
                           "size_bytes": record["size_bytes"]})
        post = json.loads(self.pre_bytes.decode("utf-8"))
        post["entries"] = sorted(post["entries"] + added,
                                 key=lambda r: r["canonical_path"])
        self.post_bytes = _render(post).render()
        (self.repo / MANIFEST_PATH).write_bytes(self.post_bytes)
        rendered = {e["record_id"]: e for e
                    in json.loads(self.post_bytes.decode("utf-8"))["entries"]}
        self.approved_entries = {r["record_id"]: rendered[r["record_id"]]
                                 for r in added}
        self.git("add", "-A")
        self.git("commit", "-qm", "candidate")
        self.candidate = self.git("rev-parse", "HEAD")
        self.git("checkout", "-q", "-b", "candidate-branch")

        self.census_bytes = (json.dumps(
            {"candidates": [{"basename": c["from_basename"]} for c in copies],
             "manifest": {"entries": 2}},
            indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.citation_bytes = (json.dumps(
            {"schema": "gvc.citation-derivation", "schema_version": 1,
             "checks_status": "passed", "failed_checks": [],
             "repository_is_shallow": False,
             "measured_at_head": self.predecessor,
             "census_sha256": CENSUS_SHA,
             "accepted_records_differing": [],
             "accepted_records_skipped_no_alias": [],
             "candidates_without_citation": [],
             "accepted_records_in_manifest": 2,
             "accepted_records_examined": 2,
             "accepted_records_reproduced": 2,
             "derived": {c["from_basename"]: {
                 "cited_by": ["bbbbbb{}".format(i)],
                 "cited_by_oids": ["bbbbbb{}".format(i) + "0" * 33]}
                 for i, c in enumerate(copies)}},
            indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.approved_bytes = (json.dumps(self.approved_entries, indent=2,
                                          sort_keys=True) + "\n").encode()
        self.plan = {
            "predecessor_commit": self.predecessor,
            "artifacts_to_copy": copies,
            "expected_delta": 0,
            "postimage_manifest": {
                "sha256": hashlib.sha256(self.post_bytes).hexdigest()},
            "derived_from": {
                "census": {"sha256":
                           hashlib.sha256(self.census_bytes).hexdigest()},
                "citations": {"sha256":
                              hashlib.sha256(self.citation_bytes).hexdigest()},
                "approved_entries": {
                    "sha256": hashlib.sha256(self.approved_bytes).hexdigest()}}}
        self.plan_bytes = (json.dumps(self.plan, indent=2, sort_keys=True)
                           + "\n").encode("utf-8")
        self.owners = types.SimpleNamespace(
            evidence=evidence_validator,
            archive=types.SimpleNamespace(
                parse=ArchiveManifest.parse,
                require_archive_preservation=(
                    archive_guard.require_archive_preservation),
                require_approved_addition=(
                    archive_guard.require_approved_addition)))

    def git(self, *args, check=True):
        done = subprocess.run(["git", "-C", str(self.repo), *args],
                              capture_output=True, env=self.env)
        if check and done.returncode != 0:
            raise RuntimeError(done.stderr.decode())
        return done.stdout.decode().strip()

    def sibling(self, name, mutate):
        """A sabotage commit whose SOLE PARENT is the approved predecessor."""
        self.git("checkout", "-q", "-b", name, self.predecessor)
        for path, payload in self.payloads.items():
            (self.repo / path).write_bytes(payload)
        (self.repo / MANIFEST_PATH).write_bytes(self.post_bytes)
        mutate()
        self.git("add", "-A")
        self.git("commit", "-qm", "sabotage " + name)
        commit = self.git("rev-parse", "HEAD")
        self.git("checkout", "-q", "candidate-branch")
        parents = self.git("rev-list", "--parents", "-n", "1", commit).split()
        assert parents == [commit, self.predecessor], (
            "a sabotage commit must be a SIBLING, or the refusal may come "
            "from require_sole_parent instead of the check under test")
        return commit

    def run(self, **over):
        kwargs = dict(repo=self.repo, plan=self.plan,
                      plan_sha256=hashlib.sha256(self.plan_bytes).hexdigest(),
                      candidate_commit=self.candidate,
                      approved_entries=self.approved_entries,
                      postimage_bytes=self.post_bytes,
                      evidence_report=json.loads(
                          self.citation_bytes.decode("utf-8")),
                      census=json.loads(self.census_bytes.decode("utf-8")),
                      census_sha256=CENSUS_SHA, owners=self.owners)
        kwargs.update(over)
        return verify_transition(**kwargs)


@pytest.fixture
def candidate(tmp_path):
    return Candidate(tmp_path)


def test_the_approved_candidate_verifies(candidate):
    result = candidate.run()
    assert result.candidate_commit == candidate.candidate
    assert result.predecessor_commit == candidate.predecessor
    assert len(result.candidate_tree) == 40
    assert len(result.admitted_record_ids) == 3


def test_effects_are_read_from_the_commit_not_the_plan(candidate):
    effects = observed_effects(candidate.repo, candidate.predecessor,
                               candidate.candidate)
    assert set(effects) == {MANIFEST_PATH} | set(candidate.payloads)
    for path, payload in candidate.payloads.items():
        assert effects[path]["content_sha256"] == \
            hashlib.sha256(payload).hexdigest()
        assert effects[path]["mode"] == "100644"


def test_plan_bytes_must_match_the_approved_digest(candidate):
    with pytest.raises(BindingError, match="digest"):
        load_bound(candidate.plan_bytes + b" ",
                   hashlib.sha256(candidate.plan_bytes).hexdigest(), "plan")


def test_a_malformed_approved_digest_is_refused(candidate):
    with pytest.raises(BindingError, match="hexadecimal"):
        load_bound(candidate.plan_bytes, "not-a-digest", "plan")


@pytest.mark.parametrize("name", ["census", "citations", "approved_entries"])
def test_substituted_dependency_bytes_are_refused(candidate, name):
    supplied = {"census_bytes": candidate.census_bytes,
                "citation_bytes": candidate.citation_bytes,
                "approved_entries_bytes": candidate.approved_bytes}
    key = {"census": "census_bytes", "citations": "citation_bytes",
           "approved_entries": "approved_entries_bytes"}[name]
    supplied[key] = supplied[key] + b" "
    with pytest.raises(BindingError, match="digest"):
        require_dependency_bindings(candidate.plan,
                                    postimage_bytes=candidate.post_bytes,
                                    **supplied)


def test_a_plan_declaring_no_dependency_digest_is_refused(candidate):
    plan = copy.deepcopy(candidate.plan)
    plan["derived_from"].pop("citations")
    with pytest.raises(BindingError, match="declares no digest"):
        require_dependency_bindings(
            plan, census_bytes=candidate.census_bytes,
            citation_bytes=candidate.citation_bytes,
            approved_entries_bytes=candidate.approved_bytes,
            postimage_bytes=candidate.post_bytes)


def test_substituted_postimage_bytes_are_refused(candidate):
    with pytest.raises(BindingError, match="postimage"):
        require_dependency_bindings(
            candidate.plan, census_bytes=candidate.census_bytes,
            citation_bytes=candidate.citation_bytes,
            approved_entries_bytes=candidate.approved_bytes,
            postimage_bytes=candidate.pre_bytes)


def test_a_citation_row_contradicting_the_manifest_is_refused(candidate):
    """The row must remain internally WELL FORMED.

    MEASURED: changing only `cited_by` leaves it inconsistent with its own
    `cited_by_oids`, and the current-shape gate refuses it first -- a correct
    refusal, but for a different reason than the one under test. The short
    identifier and its full identifier are changed together so the report is
    structurally valid and contradicts only the manifest.
    """
    report = json.loads(candidate.citation_bytes.decode("utf-8"))
    first = sorted(report["derived"])[0]
    report["derived"][first]["cited_by"] = ["9999999"]
    report["derived"][first]["cited_by_oids"] = ["9999999" + "0" * 33]
    with pytest.raises(AdmissionError, match="disagree with the citation"):
        candidate.run(evidence_report=report)


def test_an_internally_inconsistent_citation_row_is_refused_first(candidate):
    """And the shape gate's own refusal is asserted, not left implicit."""
    report = json.loads(candidate.citation_bytes.decode("utf-8"))
    first = sorted(report["derived"])[0]
    report["derived"][first]["cited_by"] = ["9999999"]
    with pytest.raises(evidence_validator.EvidenceError, match="prefix"):
        candidate.run(evidence_report=report)


def test_a_citation_row_absent_from_the_report_is_refused(candidate):
    report = json.loads(candidate.citation_bytes.decode("utf-8"))
    report["derived"].pop(sorted(report["derived"])[0])
    with pytest.raises(Exception) as caught:
        candidate.run(evidence_report=report)
    assert isinstance(caught.value, (AdmissionError,
                                     evidence_validator.EvidenceError))


def test_an_undeclared_committed_file_is_refused(candidate):
    commit = candidate.sibling(
        "undeclared",
        lambda: (candidate.repo / "src" / "extra.py").write_bytes(b"y = 1\n"))
    with pytest.raises(ClassificationError, match="not the approved one"):
        candidate.run(candidate_commit=commit)


def test_a_changed_committed_blob_is_refused(candidate):
    victim = sorted(candidate.payloads)[0]
    commit = candidate.sibling(
        "changed-blob",
        lambda: (candidate.repo / victim).write_bytes(b'{"tampered": 1}\n'))
    with pytest.raises(ClassificationError, match="differ from approval"):
        candidate.run(candidate_commit=commit)


def test_a_changed_mode_is_refused(candidate):
    victim = sorted(candidate.payloads)[0]

    def make_executable():
        candidate.git("update-index", "--chmod=+x", "--add", victim,
                      check=False)
        os.chmod(candidate.repo / victim, 0o755)

    commit = candidate.sibling("changed-mode", make_executable)
    effects = observed_effects(candidate.repo, candidate.predecessor, commit)
    if effects[victim]["mode"] == "100644":
        pytest.skip("this filesystem did not record an executable mode")
    with pytest.raises(ClassificationError):
        candidate.run(candidate_commit=commit)


def test_a_deletion_of_a_PREDECESSOR_file_is_refused(candidate):
    """A real deletion removes something the PREDECESSOR carries.

    MEASURED: unlinking a CANDIDATE artifact is not a deletion at all -- that
    path never existed in the predecessor, so it is an omission and is
    reported as APPROVED BUT ABSENT. The distinction matters because only the
    first exercises the deletion refusal.
    """
    victim = ARTIFACTS_SUBTREE + "install-attestation-GENESIS-0.json"
    assert (candidate.repo / victim).is_file()
    commit = candidate.sibling(
        "deletion", lambda: (candidate.repo / victim).unlink())
    with pytest.raises(AdmissionError, match="DELETED"):
        candidate.run(candidate_commit=commit)


def test_an_omitted_approved_artifact_is_refused_as_absent(candidate):
    """The other case, asserted for what it actually is."""
    victim = sorted(candidate.payloads)[0]
    commit = candidate.sibling(
        "omitted", lambda: (candidate.repo / victim).unlink())
    with pytest.raises(ClassificationError, match="APPROVED BUT ABSENT"):
        candidate.run(candidate_commit=commit)


def test_a_merge_parent_is_refused(candidate):
    candidate.git("checkout", "-q", "-b", "side", candidate.predecessor)
    (candidate.repo / "src" / "side.py").write_bytes(b"s = 1\n")
    candidate.git("add", "-A")
    candidate.git("commit", "-qm", "side")
    candidate.git("checkout", "-q", "candidate-branch")
    candidate.git("merge", "--no-ff", "-q", "-m", "merge", "side")
    merge = candidate.git("rev-parse", "HEAD")
    with pytest.raises(AdmissionError, match="sole parent"):
        require_sole_parent(candidate.repo, merge, candidate.predecessor)


def test_a_preserved_record_changed_in_the_manifest_is_refused(candidate):
    document = json.loads(candidate.post_bytes.decode("utf-8"))
    for entry in document["entries"]:
        if entry["record_id"] == "REC-" + "{:032x}".format(0):
            entry["artifact_schema_version"] = 2
    altered = _render(document).render()
    commit = candidate.sibling(
        "altered-entry",
        lambda: (candidate.repo / MANIFEST_PATH).write_bytes(altered))
    with pytest.raises(ArchiveManifestError, match="changed 1 existing"):
        candidate.run(candidate_commit=commit, postimage_bytes=altered)


def test_a_created_artifact_without_its_record_is_refused(candidate):
    """Correspondence, one direction."""
    document = json.loads(candidate.post_bytes.decode("utf-8"))
    victim = sorted(candidate.approved_entries)[0]
    document["entries"] = [e for e in document["entries"]
                           if e["record_id"] != victim]
    altered = _render(document).render()
    commit = candidate.sibling(
        "record-missing",
        lambda: (candidate.repo / MANIFEST_PATH).write_bytes(altered))
    with pytest.raises((AdmissionError, ArchiveManifestError)):
        candidate.run(candidate_commit=commit, postimage_bytes=altered)


def test_an_approved_record_without_its_artifact_is_refused(candidate):
    """Correspondence, the other direction.

    Refused by the TRANSITION check rather than the correspondence check,
    because an approved artifact that was never committed is an absent
    approved effect. ClassificationError is not a subclass of AdmissionError,
    so the expectation names the exception actually raised.
    """
    victim = sorted(candidate.payloads)[0]
    commit = candidate.sibling(
        "artifact-missing", lambda: (candidate.repo / victim).unlink())
    with pytest.raises(ClassificationError, match="APPROVED BUT ABSENT"):
        candidate.run(candidate_commit=commit)


def test_a_code_changing_candidate_cannot_obtain_maintenance_treatment(
        candidate):
    commit = candidate.sibling(
        "code-change",
        lambda: (candidate.repo / "src" / "module.py").write_bytes(b"x = 2\n"))
    with pytest.raises(ClassificationError, match="not the approved one"):
        candidate.run(candidate_commit=commit)


def test_a_gate_result_for_another_commit_is_refused(candidate):
    result = candidate.run()
    evidence = {"subject": {"candidate_commit": "0" * 40,
                            "candidate_tree": result.candidate_tree},
                "execution": {"exit_code": 0},
                "collection": {"added": [], "removed": []}}
    with pytest.raises(BindingError, match="candidate"):
        require_acceptance(evidence,
                           candidate_commit=result.candidate_commit,
                           candidate_tree=result.candidate_tree,
                           expected_delta=0)


def test_a_gate_result_for_another_tree_is_refused(candidate):
    result = candidate.run()
    evidence = {"subject": {"candidate_commit": result.candidate_commit,
                            "candidate_tree": "0" * 40},
                "execution": {"exit_code": 0},
                "collection": {"added": [], "removed": []}}
    with pytest.raises(BindingError, match="tree"):
        require_acceptance(evidence,
                           candidate_commit=result.candidate_commit,
                           candidate_tree=result.candidate_tree,
                           expected_delta=0)


def test_a_gate_result_naming_this_subject_is_accepted(candidate):
    result = candidate.run()
    evidence = {"subject": {"candidate_commit": result.candidate_commit,
                            "candidate_tree": result.candidate_tree},
                "execution": {"exit_code": 0},
                "collection": {"added": [], "removed": []}}
    assert require_acceptance(
        evidence, candidate_commit=result.candidate_commit,
        candidate_tree=result.candidate_tree,
        expected_delta=0)["expected_delta"] == 0


def test_a_wrong_predecessor_is_refused(candidate):
    plan = copy.deepcopy(candidate.plan)
    plan["predecessor_commit"] = "0" * 40
    with pytest.raises(evidence_validator.EvidenceError, match="predecessor"):
        candidate.run(plan=plan)


def test_manifest_citation_agreement_accepts_agreeing_evidence():
    entries = [{"record_id": "REC-" + "a" * 32, "legacy_aliases": ["x.json"],
                "cited_by": ["1111111"]}]
    approved = {"REC-" + "a" * 32: {}}
    report = {"derived": {"x.json": {"cited_by": ["1111111"]}}}
    require_manifest_citation_agreement(entries, approved, report)
