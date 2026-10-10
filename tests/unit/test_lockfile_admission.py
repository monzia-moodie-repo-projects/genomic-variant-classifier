"""The lockfile-migration admission policy (owner rulings 2026-10-08c/d/e/f), exercised on the REAL preserved evidence.

Two kinds of negative control:

  * BINDING controls change one input and leave the approval contract as it is: the first digest naming that input must refuse.
  * SEMANTIC controls change one input and then RE-BIND every digest that names it (plans, difference record, proposal inputs,
    regenerated proposal, equivalence record, approval contract) -- a consistent forgery. The semantic check itself must refuse.
    Without the re-binding, every semantic control would be stopped by a digest and prove nothing about the check it names.

The forger is proven non-vacuous first: with no change it reproduces the preserved bytes exactly and is admitted.

Author: Monzia Moodie
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from genomic_variant_classifier.environment_qualification.admission import plan_digest
from genomic_variant_classifier.environment_qualification.lockfile_admission import (
    APPROVED, ApprovedMigration, MigrationEvidence, admit_lockfile_migration, canonical_lf, derive_artifact_selection)
from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError
from genomic_variant_classifier.repository_records.artifact_inventory import ArtifactInventoryRecord, scan_records
from genomic_variant_classifier.repository_records.lockfile_migration import LockfileMigrationManifest, family_root

ROOT = Path(__file__).resolve().parents[2]


def _manifest() -> LockfileMigrationManifest:
    found = sorted(ROOT.joinpath(*family_root().parts).glob("REC-*/manifest.json"))
    assert len(found) == 1, found
    return LockfileMigrationManifest.parse(found[0].read_bytes())


MANIFEST = _manifest()
PRESERVED = MANIFEST.read_preserved(ROOT)
INVENTORY = [r for r in scan_records(ROOT) if r.record_id.value == "REC-ece23653bf8e45dbad3da26303870dc6"][0]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def evidence(parts=None) -> MigrationEvidence:
    p = PRESERVED if parts is None else parts
    return MigrationEvidence(baseline_lock=p["baseline_lockfile"], candidate_lock=p["candidate_lockfile"], proposal=p["approved_proposal"],
                             regenerated_proposal=p["regenerated_proposal"], equivalence=p["equivalence_record"],
                             replay_plan=p["replay_plan"], candidate_plan=p["candidate_plan"], candidate_difference=p["candidate_difference"])


def reason(call) -> str:
    with pytest.raises(AdmissionError) as exc:
        call()
    return str(exc.value)


# The preserved files' own serialisations (measured): plans, difference and equivalence record indent 2; the proposal indent 1.
def _dump2(doc) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _dump1(doc) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8")


def forge(content=None, plan_binding=None, proposal=None, regenerated=None, equivalence=None, *, replay_dump=_dump2, stale_labels=False):
    """A CONSISTENT forgery: apply `content` to the parsed documents / raw lockfiles, then re-bind every digest that names them.
    Hooks: content(ctx), plan_binding(candidate_plan) after the replay plan is re-bound, proposal(proposal) after its inputs are
    re-bound, regenerated(doc), equivalence(doc). A document is re-serialised only if it changed (unchanged bytes stay exact).
    stale_labels=True keeps each plan's ORIGINAL plan_sha256 label (and every binding to it) while its body changes: only a
    recomputation of the body digest can then notice."""
    raw = dict(PRESERVED)
    ctx = {"baseline": raw["baseline_lockfile"], "candidate": raw["candidate_lockfile"],
           "replay": json.loads(raw["replay_plan"]), "cand_plan": json.loads(raw["candidate_plan"]),
           "difference": json.loads(raw["candidate_difference"]), "proposal": json.loads(raw["approved_proposal"])}
    original = copy.deepcopy(ctx)
    if content:
        content(ctx)
    replay = ctx["replay"]
    replay["plan_sha256"] = original["replay"]["plan_sha256"] if stale_labels else plan_digest(replay)
    replay_raw = raw["replay_plan"] if replay == original["replay"] else replay_dump(replay)
    cand = ctx["cand_plan"]
    cand["binding"]["replay"]["replay_plan_file_sha256"] = sha(replay_raw)
    cand["binding"]["replay"]["replay_plan_sha256"] = replay["plan_sha256"]
    if plan_binding:
        plan_binding(cand)
    cand["plan_sha256"] = original["cand_plan"]["plan_sha256"] if stale_labels else plan_digest(cand)
    cand_raw = raw["candidate_plan"] if cand == original["cand_plan"] else _dump2(cand)
    diff = ctx["difference"]
    diff["baseline_canonical_sha256"] = sha(canonical_lf(ctx["baseline"]))
    diff["candidate_sha256"] = sha(ctx["candidate"])
    diff_raw = raw["candidate_difference"] if diff == original["difference"] else _dump2(diff)
    prop = ctx["proposal"]
    prop["inputs"].update(baseline_canonical_sha256=sha(canonical_lf(ctx["baseline"])), candidate_sha256=sha(ctx["candidate"]),
                          candidate_canonical_sha256=sha(canonical_lf(ctx["candidate"])), candidate_plan_sha256=cand["plan_sha256"],
                          difference_record_sha256=sha(diff_raw), replay_plan_sha256=replay["plan_sha256"])
    if proposal:
        proposal(prop)
    prop_raw = raw["approved_proposal"] if prop == original["proposal"] else _dump1(prop)
    regen = copy.deepcopy(prop)
    regen["generator_sha256"] = APPROVED.regenerating_generator_sha256
    if regenerated:
        regenerated(regen)
    regen_raw = raw["regenerated_proposal"] if prop == original["proposal"] and not regenerated else _dump1(regen)
    eq = json.loads(raw["equivalence_record"])
    eq.update(approved_proposal_sha256=sha(prop_raw), regenerated_proposal_sha256=sha(regen_raw))
    if equivalence:
        equivalence(eq)
    eq_raw = _dump2(eq)
    contract = dataclasses.replace(APPROVED, proposal_sha256=sha(prop_raw), regenerated_proposal_sha256=sha(regen_raw),
                                   equivalence_sha256=sha(eq_raw), replay_plan_file_sha256=sha(replay_raw),
                                   candidate_plan_file_sha256=sha(cand_raw))
    parts = {"baseline_lockfile": ctx["baseline"], "candidate_lockfile": ctx["candidate"], "approved_proposal": prop_raw,
             "regenerated_proposal": regen_raw, "equivalence_record": eq_raw, "replay_plan": replay_raw, "candidate_plan": cand_raw,
             "candidate_difference": diff_raw}
    return evidence(parts), contract


# ------------------------------------------------------------------ the real migration

def test_the_real_migration_is_admitted_and_equals_the_committed_record():
    result = admit_lockfile_migration(evidence(), contract=APPROVED, inventory_record=INVENTORY)
    assert result == MANIFEST.admission
    assert (result["transition"]["transitions"], len(result["transition"]["additions"]), result["transition"]["packages_after"]) == (90, 15, 102)
    assert (result["baseline"]["r_version"], result["candidate"]["r_version"]) == ("4.6.0", "4.6.1")
    assert result["selection"] == {"packages": 102, "routes": {"bootstrap": 1, "local_build": 13, "runtime": 3, "upstream_binary": 85}}


def test_admission_is_deterministic():
    a = admit_lockfile_migration(evidence(), contract=APPROVED, inventory_record=INVENTORY)
    b = admit_lockfile_migration(evidence(), contract=APPROVED, inventory_record=INVENTORY)
    assert a == b and json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_the_forger_is_not_vacuous():
    """With no change, the forger reproduces the preserved bytes EXACTLY and the approval contract unchanged."""
    ev, contract = forge()
    assert ev == evidence() and contract == APPROVED
    assert admit_lockfile_migration(ev, contract=contract, inventory_record=INVENTORY) == MANIFEST.admission


def test_a_forgery_with_a_neutral_change_is_still_admitted():
    """Re-binding works: a changed explanatory note in the proposal changes the proposal, regeneration and equivalence digests, and
    nothing refuses. (A change to a PLAN is not neutral: the committed inventory measured those exact plan files -- see the
    "plans are not the ones the inventory measured" control.)"""
    ev, contract = forge(content=lambda c: c["proposal"].update(restoration_note="neutral wording"))
    assert contract.proposal_sha256 != APPROVED.proposal_sha256 and contract.equivalence_sha256 != APPROVED.equivalence_sha256
    assert admit_lockfile_migration(ev, contract=contract, inventory_record=INVENTORY)["transition"]["transitions"] == 90


# ------------------------------------------------------------------ binding controls (the approval contract unchanged)

def _with(part, raw):
    p = dict(PRESERVED)
    p[part] = raw
    return evidence(p)


@pytest.mark.parametrize("part, code", [
    ("baseline_lockfile", "migration.baseline_digest"),
    ("candidate_lockfile", "migration.candidate_exact_digest"),
    ("candidate_difference", "migration.difference_digest"),
    ("replay_plan", "migration.replay_plan_file_digest"),
    ("candidate_plan", "migration.candidate_plan_file_digest"),
    ("approved_proposal", "equivalence.approved_digest"),
    ("regenerated_proposal", "equivalence.regenerated_digest"),
    ("equivalence_record", "equivalence.record_digest"),
])
def test_one_changed_byte_in_any_input_is_refused_by_its_binding(part, code):
    assert reason(lambda: admit_lockfile_migration(_with(part, PRESERVED[part] + b" "), contract=APPROVED, inventory_record=INVENTORY)) == code


def test_a_candidate_whose_line_endings_were_normalised_is_not_the_admitted_bytes():
    """Exact bytes and canonical text are two domains: LF-normalising the CRLF candidate keeps its canonical digest and still refuses."""
    lf = canonical_lf(PRESERVED["candidate_lockfile"])
    assert sha(lf) == MANIFEST.admission["candidate"]["canonical_sha256"]
    assert reason(lambda: admit_lockfile_migration(_with("candidate_lockfile", lf), contract=APPROVED, inventory_record=INVENTORY)) \
        == "migration.candidate_exact_digest"


def test_a_lone_carriage_return_has_no_canonical_form():
    """Canonicalisation precedes every digest comparison, so a lone CR refuses as itself, not as a digest mismatch."""
    lone = PRESERVED["candidate_lockfile"].replace(b"\r\n", b"\r", 1)
    assert lone.count(b"\r") == PRESERVED["candidate_lockfile"].count(b"\r")
    assert reason(lambda: admit_lockfile_migration(_with("candidate_lockfile", lone), contract=APPROVED, inventory_record=INVENTORY)) \
        == "migration.lone_cr"


def test_the_contract_is_not_taken_from_the_evidence():
    other = dataclasses.replace(APPROVED, proposal_sha256="0" * 64)
    assert reason(lambda: admit_lockfile_migration(evidence(), contract=other, inventory_record=INVENTORY)) == "equivalence.approved_digest"


@pytest.mark.parametrize("call, code", [
    (lambda: admit_lockfile_migration(dict(PRESERVED), contract=APPROVED, inventory_record=INVENTORY), "migration.evidence_type"),
    (lambda: admit_lockfile_migration(evidence(), contract=dataclasses.asdict(APPROVED), inventory_record=INVENTORY), "migration.contract_type"),
    (lambda: admit_lockfile_migration(evidence(), contract=APPROVED, inventory_record=INVENTORY.render()), "migration.inventory_record_type"),
    (lambda: admit_lockfile_migration(_with("candidate_difference", b""), contract=APPROVED, inventory_record=INVENTORY),
     "migration.missing:candidate_difference"),
])
def test_malformed_arguments_refuse(call, code):
    assert reason(call) == code


# ------------------------------------------------------------------ semantic controls (consistent forgeries)

def _pop_last(lst):
    lst.pop()


SEMANTIC = [
    ("an approved transition dropped", dict(content=lambda c: _pop_last(c["proposal"]["transitions"])), "lockfile.unapproved_transition"),
    ("a transition without its restoration effect",
     dict(content=lambda c: c["proposal"]["transitions"][0].pop("restoration_effect")), "transition.missing:restoration_effect"),
    ("a blank restoration effect", dict(content=lambda c: c["proposal"]["transitions"][0].update(restoration_effect=" ")),
     "transition.restoration_effect"),
    ("another runtime target", dict(content=lambda c: c["proposal"]["runtime_change"].update(new="4.6.2")), "migration.runtime_change"),
    ("runtime evidence not naming the runtime record",
     dict(content=lambda c: c["proposal"]["runtime_change"].update(evidence="qualified runtime record (unnamed)")),
     "migration.runtime_change_evidence"),
    ("the proposal selects another artifact",
     dict(content=lambda c: c["proposal"]["artifact_selection"]["BH"].update(artifact_sha256="0" * 64)), "migration.proposal_selection_differs"),
    ("the run selected another artifact",
     dict(content=lambda c: c["difference"]["artifact_selection"]["BH"].update(artifact_sha256="0" * 64)), "migration.run_selection_differs"),
    ("the run reported one difference fewer", dict(content=lambda c: _pop_last(c["difference"]["field_differences"])),
     "migration.difference_disagrees"),
    ("the run reported a difference as null instead of absent",
     dict(content=lambda c: [d for d in c["difference"]["field_differences"] if d["new"] == {"absent": True}][0].update(new={"value": None})),
     "migration.difference_disagrees"),
    ("the run reported a removal", dict(content=lambda c: c["difference"].update(removed=["BH"])), "migration.difference_membership"),
    ("the run omitted an addition", dict(content=lambda c: c["difference"]["added"].pop("igraph")), "migration.difference_membership"),
    ("the run's added record differs", dict(content=lambda c: c["difference"]["added"]["igraph"].update(Repository="X")),
     "migration.difference_added_record:igraph"),
    ("an addition's lock label is not its artifact's",
     dict(content=lambda c: c["proposal"]["addition_artifacts"]["igraph"].update(lock_repository_label="RSPM")),
     "migration.addition_repository_label:igraph"),
    ("an addition declares another artifact",
     dict(content=lambda c: c["proposal"]["addition_artifacts"]["igraph"].update(artifact_sha256="0" * 64)),
     "migration.addition_artifact_differs:igraph"),
    ("an addition declares another receipt",
     dict(content=lambda c: c["proposal"]["addition_artifacts"]["qvalue"].update(receipt_sha256="0" * 64)),
     "migration.addition_artifact_differs:qvalue"),
    ("a record not reproduced", dict(content=lambda c: c["proposal"]["record_reproduction"]["records"]["BH"].update(reproduced=False)),
     "migration.reproduction_failed"),
    ("a record missing from the reproduction", dict(content=lambda c: c["proposal"]["record_reproduction"]["records"].pop("BH")),
     "migration.reproduction_coverage"),
    ("a plan version that is not the lock's",
     dict(content=lambda c: [a for a in c["cand_plan"]["additions"] if a["package"] == "igraph"][0].update(version="2.3.4")),
     "migration.selection_version:igraph"),
    ("a runtime-supplied version that is not the lock's",
     dict(content=lambda c: c["replay"]["runtime_supplied"][0].update(version="1.7-6")), "migration.selection_version:Matrix"),
    ("a package selected twice", dict(content=lambda c: c["cand_plan"]["additions"].append(copy.deepcopy(c["cand_plan"]["additions"][0]))),
     "selection.duplicate:igraph"),
    ("a plan row with an undeclared key", dict(content=lambda c: c["replay"]["install"][0].update(extra=1)),
     "selection.replay_row:upstream_binary"),
    ("an unknown route", dict(content=lambda c: c["replay"]["install"][0].update(route="mirror")), "selection.replay_route:'mirror'"),
    ("a lock package without a selected artifact", dict(content=lambda c: c["replay"]["install"].pop(0)), "migration.selection_coverage"),
    ("the replay was of another baseline", dict(content=lambda c: c["replay"].update(lockfile_canonical_sha256="0" * 64)),
     "migration.plans_bind_another_baseline"),
    ("the candidate plan binds another replay", dict(plan_binding=lambda p: p["binding"]["replay"].update(replay_plan_sha256="0" * 64)),
     "migration.candidate_plan_binds_another_replay"),
    ("the candidate plan binds another runtime record",
     dict(plan_binding=lambda p: p["binding"]["replay"].update(runtime_record_sha256="0" * 64)), "migration.runtime_record_binding"),
    ("the run evidence was not inventoried", dict(proposal=lambda p: p["inputs"].update(evidence_zip_sha256="0" * 64)),
     "migration.run_evidence_not_inventoried"),
    ("the plans are not the ones the inventory measured", dict(replay_dump=_dump1, content=lambda c: c["replay"].update(note="x")),
     "migration.inventory_measured_other_plans"),
    ("the regeneration differs in substance", dict(regenerated=lambda d: d.update(status="ADMITTED")), "equivalence.substantive_difference"),
    ("the regeneration names another generator", dict(regenerated=lambda d: d.update(generator_sha256="0" * 64)), "equivalence.new_generator"),
    ("the equivalence record denies equality", dict(equivalence=lambda d: d.update(substantive_content_equal=False)), "equivalence.record_claim"),
    ("the equivalence record binds another proposal",
     dict(equivalence=lambda d: d.update(approved_generator_sha256="0" * 64)), "equivalence.record_binding"),
    ("an unknown proposal schema", dict(proposal=lambda p: p.update(schema="gvc.lock-transition-proposal/2")), "migration.proposal_schema"),
    ("a replay plan whose label no longer describes its body",
     dict(stale_labels=True, content=lambda c: c["replay"]["install"][0].update(sha256="0" * 64)), "migration.replay_plan_body_digest"),
    ("a candidate plan whose label no longer describes its body",
     dict(stale_labels=True, content=lambda c: c["cand_plan"]["additions"][0]["artifact"].update(sha256="0" * 64)),
     "migration.candidate_plan_body_digest"),
]


@pytest.mark.parametrize("label, hooks, code", SEMANTIC, ids=[s[0] for s in SEMANTIC])
def test_each_semantic_check_refuses_a_consistent_forgery(label, hooks, code):
    hooks = dict(hooks)
    ev, contract = forge(**hooks)
    assert reason(lambda: admit_lockfile_migration(ev, contract=contract, inventory_record=INVENTORY)) == code


def test_another_inventory_entry_for_the_run_evidence_is_refused():
    other = dataclasses.replace(APPROVED, candidate_evidence_entry="run-inventory-20261008:candidate_lock_20261008T012225Z")
    assert reason(lambda: admit_lockfile_migration(evidence(), contract=other, inventory_record=INVENTORY)) == "migration.run_evidence_not_inventoried"


def test_a_runtime_supplied_disagreement_with_the_inventory_is_refused():
    doc = json.loads(INVENTORY.render())
    doc["runtime_supplied"][0]["runtime_record_sha256"] = "1" * 64
    altered = ArtifactInventoryRecord.parse((json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii"))
    assert reason(lambda: admit_lockfile_migration(evidence(), contract=APPROVED, inventory_record=altered)) == "migration.runtime_supplied_disagrees"


def test_a_duplicate_json_key_in_the_proposal_is_refused():
    raw = PRESERVED["approved_proposal"].replace(b'\n "schema":', b'\n "schema": "x",\n "schema":', 1)
    assert raw != PRESERVED["approved_proposal"]
    regen = json.loads(PRESERVED["regenerated_proposal"])
    contract = dataclasses.replace(APPROVED, proposal_sha256=sha(raw))
    with pytest.raises(AdmissionError) as exc:
        admit_lockfile_migration(_with("approved_proposal", raw), contract=contract, inventory_record=INVENTORY)
    assert str(exc.value).startswith("duplicate_json_key:") and regen["schema"] == "gvc.lock-transition-proposal/1"


# ------------------------------------------------------------------ the selection owner directly

def test_the_derived_selection_is_one_artifact_per_locked_package():
    selection, versions = derive_artifact_selection(json.loads(PRESERVED["replay_plan"]), json.loads(PRESERVED["candidate_plan"]))
    lock = json.loads(canonical_lf(PRESERVED["candidate_lockfile"]))
    assert set(selection) == set(versions) == set(lock["Packages"])
    assert all(lock["Packages"][p]["Version"] == v for p, v in versions.items())
    assert selection["qvalue"]["route"] == "local_build" and selection["qvalue"]["receipt_sha256"] is not None
    assert selection["Matrix"] == {"artifact_sha256": None, "kind": None, "receipt_sha256": None, "route": "runtime"}
    assert selection["renv"]["route"] == "bootstrap"


def test_the_contract_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        APPROVED.proposal_sha256 = "0" * 64       # type: ignore[misc]
    assert type(APPROVED) is ApprovedMigration
