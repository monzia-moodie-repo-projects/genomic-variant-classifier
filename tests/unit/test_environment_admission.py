"""The shared admission layer (owner rulings 2026-10-08, 2026-10-08b): one definition of admission, reused by every script.

Author: Monzia Moodie
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from genomic_variant_classifier.environment_qualification.admission import (
    ABSENT, CheckRequirement, CheckResult, EvidenceState, RunRecord, admit_artifact_set, admit_lock_transition, decide_qualification,
    exact_field_diff, library_digest, plan_digest, verify_plan)
from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError


def _reason(fn):
    with pytest.raises(AdmissionError) as error:
        fn()
    return str(error.value)


# ------------------------------------------------------------------ admit_artifact_set

def _artifact(tmp_path, package, kind, decision="accepted", data=b"bytes"):
    path = tmp_path / "{}_{}.bin".format(package, kind)
    path.write_bytes(data)
    return {"package": package, "version": "1.0", "kind": kind, "decision": decision, "accepted_path": str(path),
            "sha256": hashlib.sha256(data).hexdigest()}


def _plan(*packages):
    return [{"package": p, "version": "1.0", "kind": k} for p in packages for k in ("source", "windows_binary")]


def test_accepted_and_already_accepted_are_both_admitted(tmp_path):
    rows = [_artifact(tmp_path, "a", "source"), _artifact(tmp_path, "a", "windows_binary", decision="already_accepted")]
    assert len(admit_artifact_set(_plan("a"), rows)) == 2


@pytest.mark.parametrize("case, reason", [
    ("empty_plan", "artifact.plan_empty"), ("missing", "artifact.coverage_mismatch"), ("unexpected", "artifact.coverage_mismatch"),
    ("duplicate_kind", "artifact.manifest_duplicate"), ("plan_duplicate", "artifact.plan_duplicate"), ("failed", "artifact.not_admitted"),
    ("altered", "artifact.bytes_changed"), ("digest_invalid", "artifact.digest_invalid"), ("file_missing", "artifact.file_missing"),
    ("kind_invalid", "artifact.kind_invalid"), ("substituted", "artifact.coverage_mismatch"),
])
def test_artifact_set_refusals(tmp_path, case, reason):
    rows = [_artifact(tmp_path, "a", "source"), _artifact(tmp_path, "a", "windows_binary")]
    plan = _plan("a")
    if case == "empty_plan":
        plan = []
    elif case == "missing":
        rows = rows[:1]
    elif case == "unexpected":
        rows.append(_artifact(tmp_path, "b", "source"))
    elif case == "duplicate_kind":
        rows.append(dict(rows[0]))
    elif case == "plan_duplicate":
        plan = plan + plan[:1]
    elif case == "failed":
        rows[0]["decision"] = "conflict_new_candidate"
    elif case == "altered":
        Path(rows[0]["accepted_path"]).write_bytes(b"changed")
    elif case == "digest_invalid":
        rows[0]["sha256"] = "ABC"
    elif case == "file_missing":
        os.remove(rows[0]["accepted_path"])
    elif case == "kind_invalid":
        rows[0]["kind"] = "linux_binary"
    elif case == "substituted":                 # SAME count, different identity: b's source replaces a's binary
        rows = [rows[0], _artifact(tmp_path, "b", "source")]
    assert _reason(lambda: admit_artifact_set(plan, rows)) == reason


# ------------------------------------------------------------------ plan_digest / verify_plan

def test_a_modified_plan_body_keeping_its_label_is_refused():
    plan = {"schema": "x/1", "install": [{"package": "a"}]}
    plan["plan_sha256"] = plan_digest(plan)
    assert verify_plan(plan, plan["plan_sha256"]) == plan["plan_sha256"]
    tampered = copy.deepcopy(plan)
    tampered["install"].append({"package": "b"})
    assert _reason(lambda: verify_plan(tampered, plan["plan_sha256"])) == "plan.body_digest_mismatch"
    relabelled = copy.deepcopy(plan)
    relabelled["plan_sha256"] = "0" * 64
    assert _reason(lambda: verify_plan(relabelled, plan["plan_sha256"])) == "plan.label_mismatch"


# ------------------------------------------------------------------ library_digest

def test_library_digest_is_content_only_and_detects_changes(tmp_path):
    for root in (tmp_path / "x", tmp_path / "y"):
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "DESCRIPTION").write_bytes(b"Package: pkg\n")
    assert library_digest(tmp_path / "x") == library_digest(tmp_path / "y")       # location differs, content identical
    (tmp_path / "y" / "pkg" / "DESCRIPTION").write_bytes(b"Package: pkg2\n")
    assert library_digest(tmp_path / "x") != library_digest(tmp_path / "y")
    assert _reason(lambda: library_digest(tmp_path / "missing")) == "library.missing"


# ------------------------------------------------------------------ exact_field_diff + admit_lock_transition

def _lock(r="4.6.0", **packages):
    base = {"renv": {"Package": "renv", "Version": "1.2.3"}}
    base.update(packages)
    return {"R": {"Version": r, "Repositories": [{"Name": "CRAN", "URL": "https://cran"}]}, "Bioconductor": {"Version": "3.23"}, "Packages": base}


BEFORE = _lock(A={"Package": "A", "Version": "1.0", "Repository": "CRAN", "RemoteSha": "abc", "Flag": None})


def test_exact_field_diff_distinguishes_absent_from_null_and_types():
    after = copy.deepcopy(BEFORE)
    a = after["Packages"]["A"]
    del a["RemoteSha"]                        # present -> ABSENT
    a["Flag"] = 0                             # null -> 0 (a TYPE change, not equality)
    a["git_url"] = None                       # ABSENT -> present null
    diff = {(p, f): (o, n) for p, f, o, n in exact_field_diff(BEFORE, after)}
    assert diff[("A", "RemoteSha")] == ("abc", ABSENT)
    assert diff[("A", "Flag")] == (None, 0)
    assert diff[("A", "git_url")] == (ABSENT, None)
    assert ("A", "Repository") not in diff


def _t(package, field, old, new):
    side = lambda v: {"absent": True} if v is ABSENT else {"value": v}
    return {"package": package, "field": field, "old": side(old), "new": side(new), "classification": "test", "evidence": "test",
            "restoration_effect": "test"}


def _candidate():
    after = _lock(r="4.6.1", A={"Package": "A", "Version": "1.0", "Repository": "RSPM", "Flag": None},
                  B={"Package": "B", "Version": "2.0"})
    approved = [_t("A", "Repository", "CRAN", "RSPM"), _t("A", "RemoteSha", "abc", ABSENT)]
    return after, approved


def test_exactly_approved_transitions_are_admitted():
    after, approved = _candidate()
    result = admit_lock_transition(BEFORE, after, observed_version="4.6.1", additions={"B": "2.0"}, approved_transitions=approved)
    assert result["transitions"] == 2 and result["packages_after"] == 3


@pytest.mark.parametrize("mutate, reason", [
    (lambda a, t: a["Packages"]["A"].update(Extra="x"), "lockfile.unapproved_transition"),           # an unexpected difference
    (lambda a, t: t.append(_t("A", "Other", "x", "y")), "lockfile.unapproved_transition"),           # an approved change that did not occur
    (lambda a, t: t.append(dict(t[0])), "transition.duplicate"),
    (lambda a, t: t.__setitem__(1, _t("A", "RemoteSha", "abc", None)), "lockfile.unapproved_transition"),   # approved null, actual ABSENT
    (lambda a, t: a["Packages"].pop("B"), "lock.additions_mismatch"),
    (lambda a, t: a["Packages"]["B"].update(Version="2.1"), "lock.addition_version:B"),
    (lambda a, t: a["Packages"].pop("renv"), "lock.removed:renv"),
    (lambda a, t: a["Packages"]["A"].update(Version="1.1"), "lock.version_changed:A"),
    (lambda a, t: a["R"]["Repositories"].append({"Name": "X", "URL": "https://x"}), "change_outside_R.Version"),
    (lambda a, t: t[0].pop("evidence"), "transition.missing:evidence"),
    # ruling 2026-10-08c: every transition states its effect on a later renv::restore (required since the lockfile migration)
    (lambda a, t: t[0].pop("restoration_effect"), "transition.missing:restoration_effect"),
    (lambda a, t: t[0].update(restoration_effect="  "), "transition.restoration_effect"),
    (lambda a, t: t[0].update(restoration_effect=None), "transition.restoration_effect"),
])
def test_lock_transition_refusals(mutate, reason):
    after, approved = _candidate()
    mutate(after, approved)
    assert _reason(lambda: admit_lock_transition(BEFORE, after, observed_version="4.6.1", additions={"B": "2.0"},
                                                 approved_transitions=approved)) == reason


# ------------------------------------------------------------------ boundary defects reproduced 2026-10-08 (ruling 2026-10-08c)

@pytest.mark.parametrize("absent", [{"absent": 1}, {"absent": 1.0}, {"absent": False}, {"absent": None}, {"absent": True, "x": 1}, ["absent"]])
def test_only_the_exact_absent_representation_is_accepted(absent):
    """`v == {"absent": True}` accepted 1 and 1.0 (numeric equality) -- measured before the fix."""
    after, approved = _candidate()
    approved[1] = dict(approved[1], new=absent)
    assert _reason(lambda: admit_lock_transition(BEFORE, after, observed_version="4.6.1", additions={"B": "2.0"},
                                                 approved_transitions=approved)) == "transition.value_shape"


def test_a_non_string_decision_is_refused_not_a_type_error(tmp_path):
    rows = [_artifact(tmp_path, "a", "source"), _artifact(tmp_path, "a", "windows_binary")]
    rows[0]["decision"] = []
    assert _reason(lambda: admit_artifact_set(_plan("a"), rows)) == "artifact.not_admitted"


def test_a_traversal_error_refuses_the_digest_instead_of_omitting_content(tmp_path, monkeypatch):
    """Measured before the fix: an injected scanning error produced the digest of NOTHING (e3b0c442...)."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "f").write_bytes(b"x")
    real_walk = os.walk

    def failing_walk(top, *args, **kwargs):
        onerror = kwargs.get("onerror")
        assert onerror is not None, "library_digest must pass an error handler to os.walk"
        onerror(OSError(13, "injected", str(top)))
        return real_walk(top, *args, **kwargs)
    monkeypatch.setattr(os, "walk", failing_walk)
    assert _reason(lambda: library_digest(tmp_path)).startswith("library.traversal_failed:")


def test_library_digest_encoding_is_unchanged(tmp_path):
    """Valid historical library identities must stay comparable: the encoding is the tools' original one."""
    (tmp_path / "b").mkdir()
    (tmp_path / "a.txt").write_bytes(b"1")
    (tmp_path / "b" / "c.txt").write_bytes(b"22")
    h = hashlib.sha256()
    for rel, data in (("a.txt", b"1"), ("b/c.txt", b"22")):
        h.update(rel.encode() + b"\0" + hashlib.sha256(data).hexdigest().encode() + b"\n")
    assert library_digest(tmp_path) == h.hexdigest()


# ------------------------------------------------------------------ RunRecord

def test_run_record_separates_execution_from_admission(tmp_path):
    with RunRecord(tmp_path / "ok") as rec:
        rec.stage("checking")
        rec.decide(False, "artifact.coverage_mismatch")         # executed perfectly, correctly REFUSED
    data = json.loads((tmp_path / "ok" / "run_record.json").read_text())
    assert (data["execution_status"], data["admission_status"], data["reason"]) == ("completed", "refused", "artifact.coverage_mismatch")
    assert rec.exit_code == 1
    with RunRecord(tmp_path / "good") as rec2:
        rec2.decide(True)
    assert rec2.exit_code == 0
    with pytest.raises(ZeroDivisionError):
        with RunRecord(tmp_path / "bad") as rec3:
            rec3.stage("dividing")
            1 / 0
    bad = json.loads((tmp_path / "bad" / "run_record.json").read_text())
    assert (bad["execution_status"], bad["admission_status"], bad["failed_stage"], bad["exception_class"]) == ("failed", "undetermined", "dividing", "ZeroDivisionError")
    assert rec3.exit_code == 1
    assert _reason(lambda: RunRecord(tmp_path / "ok").__enter__()) == "record.run_directory_exists"     # EXCLUSIVE ownership
    assert _reason(lambda: rec2.decide(False)) == "record.refusal_reason_required"


# ------------------------------------------------------------------ decide_qualification (ruling 2026-10-08c)

SPEC, VERIFIER, EVIDENCE = "a" * 64, "b" * 64, "c" * 64
SUBJECTS = {"replay": "1" * 64, "fixtures": "2" * 64}
REQUIRED = tuple(CheckRequirement(k, v) for k, v in SUBJECTS.items())


def _result(check_id, state=EvidenceState.MATCH, **changes):
    fields = dict(check_id=check_id, subject_sha256=SUBJECTS.get(check_id, "9" * 64), specification_sha256=SPEC, verifier_sha256=VERIFIER,
                  state=state, evidence_sha256=EVIDENCE, reason="measured")
    fields.update(changes)
    return CheckResult(**fields)


def _decide(results, requirements=REQUIRED):
    return decide_qualification(requirements, results, specification_sha256=SPEC, verifier_sha256=VERIFIER)


def test_every_required_check_matching_is_admitted():
    decision = _decide((_result("replay"), _result("fixtures")))
    assert decision.admitted and decision.reasons == ()


@pytest.mark.parametrize("results, reason", [
    ((_result("replay"),), "qualification.coverage_mismatch"),                                            # a missing check
    ((_result("replay"), _result("other")), "qualification.coverage_mismatch"),                           # EQUAL COUNT, substituted identity
    ((_result("replay"), _result("fixtures", subject_sha256="3" * 64)), "fixtures:subject_mismatch"),    # substituted subject
    ((_result("replay"), _result("fixtures", specification_sha256="d" * 64)), "fixtures:specification_mismatch"),
    ((_result("replay"), _result("fixtures", verifier_sha256="e" * 64)), "fixtures:verifier_mismatch"),
    ((_result("replay"), _result("fixtures", EvidenceState.UNAVAILABLE)), "fixtures:unavailable"),
    ((_result("replay"), _result("fixtures", EvidenceState.MISMATCH)), "fixtures:mismatch"),
    ((_result("replay"), _result("fixtures", EvidenceState.INVALID)), "fixtures:invalid_evidence"),
])
def test_insufficient_results_are_refused_with_reasons(results, reason):
    decision = _decide(results)
    assert not decision.admitted and reason in decision.reasons


@pytest.mark.parametrize("build, reason", [
    (lambda: _decide((_result("replay"), _result("fixtures")), requirements=()), "qualification.requirements_empty"),
    (lambda: _decide((_result("replay"),), requirements=REQUIRED + REQUIRED[:1]), "qualification.requirement_duplicate"),
    (lambda: _decide((_result("replay"), _result("replay"), _result("fixtures"))), "qualification.result_duplicate"),
    (lambda: _decide((_result("replay"), _result("fixtures", state="match"))), "qualification.state_invalid"),       # a string is not the enum
    (lambda: _decide((_result("replay"), _result("fixtures", reason=""))), "qualification.reason_missing"),
    (lambda: _decide((_result("replay"), _result("fixtures", evidence_sha256="ABC"))), "qualification.result_digest_invalid"),
    (lambda: decide_qualification(REQUIRED, (), specification_sha256="x", verifier_sha256=VERIFIER), "qualification.specification_digest_invalid"),
])
def test_malformed_qualification_inputs_refuse(build, reason):
    assert _reason(build) == reason


# ------------------------------------------------------------------ ported from the retired source_repair tests (ADR-0004 authority succession)

def _preserved_baseline_lock() -> Path:
    """The REAL baseline lockfile (R 4.6.0, 87 packages): since the lockfile migration of 2026-10-09 the live renv.lock is its
    successor, and the predecessor is preserved verbatim in the migration record (ADR-0004 AUTHORITY-SUCCESSION-1)."""
    found = sorted((Path(__file__).resolve().parents[2] / "records" / "migrations" / "environment-qualification" / "lockfile")
                   .glob("REC-*/artifacts/renv.lock"))
    assert len(found) == 1, found
    return found[0]


REAL_LOCK = json.loads(_preserved_baseline_lock().read_text(encoding="utf-8"))


def test_real_lockfile_one_exact_transition_plus_the_runtime_change_is_admitted():
    after = copy.deepcopy(REAL_LOCK)
    after["R"]["Version"] = "4.6.1"
    old = after["Packages"]["S4Arrays"]["Repository"]
    after["Packages"]["S4Arrays"]["Repository"] = "BioCsoft"
    approved = [_t("S4Arrays", "Repository", old, "BioCsoft")]
    result = admit_lock_transition(REAL_LOCK, after, observed_version="4.6.1", additions={}, approved_transitions=approved)
    assert result["transitions"] == 1 and result["packages_after"] == len(REAL_LOCK["Packages"])


def test_real_lockfile_version_change_is_refused():
    after = copy.deepcopy(REAL_LOCK)
    after["R"]["Version"] = "4.6.1"
    after["Packages"]["S4Arrays"]["Version"] = "1.12.1"
    assert _reason(lambda: admit_lock_transition(REAL_LOCK, after, observed_version="4.6.1", additions={},
                                                 approved_transitions=[])) == "lock.version_changed:S4Arrays"


def test_an_extra_top_level_key_is_refused():
    after, approved = _candidate()
    after["Extra"] = {}
    assert _reason(lambda: admit_lock_transition(BEFORE, after, observed_version="4.6.1", additions={"B": "2.0"},
                                                 approved_transitions=approved)) == "change_outside_R.Version"


def test_a_true_to_one_type_change_is_detected():
    """The retired source_repair compared with != and reported True -> 1 as NO change (measured 2026-10-08)."""
    before = _lock(A={"Package": "A", "Version": "1.0", "Flag": True})
    after = _lock(A={"Package": "A", "Version": "1.0", "Flag": 1})
    assert exact_field_diff(before, after) == [("A", "Flag", True, 1)]
