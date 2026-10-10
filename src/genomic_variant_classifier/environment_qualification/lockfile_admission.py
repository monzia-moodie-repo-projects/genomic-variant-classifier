"""Admission of the approved lockfile migration (owner rulings 2026-10-08c, 2026-10-08d, 2026-10-08e, 2026-10-08f).

APPROVAL IS NOT ADMISSION. Ruling 2026-10-08e approved one exact proposal (APPROVED.proposal_sha256: 90 field transitions in 13
existing packages, 15 additions, R 4.6.0 -> 4.6.1, nothing else) as the intended migration; ruling 2026-10-08f kept it and ordered
its admission against the actual baseline and candidate, preserving the regeneration evidence. Admission therefore requires the
baseline lockfile, the candidate lockfile, the selected artifacts (the two sealed installation plans) and the proposal to AGREE:

  identities      every input has the digest the approval and the proposal bind -- exact bytes and canonical LF text kept as two
                  declared domains, never compared with each other
  equivalence     the regenerated proposal differs from the approved one ONLY in generator_sha256 (ruling 2026-10-08f section 5:
                  one explained exception, never a general provenance exemption); the approved bytes remain the ones admitted
  transition      admission.admit_lock_transition: exactly the approved runtime change, the reviewed additions at their versions,
                  no removal or version change, and field differences EQUAL to the approved transitions (each stating its
                  restoration effect)
  selection       the proposal's artifact selection EQUALS the selection derived here from the sealed plans (one artifact per locked
                  package, by content: artifact and receipt digests, never a file name), and the candidate run's own difference
                  report agrees with the repository's exact field difference
  provenance      each addition's lock record carries the repository label of ITS selected artifact (ruling 2026-10-08d: the
                  transition admission is tied to the artifact plan for new-package provenance)
  run evidence    the candidate lockfile came from the admitted candidate run: its evidence archive is the one the committed
                  artifact-inventory record found in the store (content match), and the plans bind the same runtime record

The authority on what renv records for an installed artifact stays renv 1.2.3 running in the qualified R 4.6.1 (the candidate
run): the proposal's independent Python re-implementation is CORROBORATION, reported but never promoted to a second authority
(ruling 2026-10-08e section 4). Nothing here authorizes installation: the isolated replay re-verifies every byte it consumes.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass

from genomic_variant_classifier.environment_qualification.admission import (
    ABSENT, EvidenceState, admit_lock_transition, exact_field_diff, plan_digest)
from genomic_variant_classifier.environment_qualification.r_runtime import (
    RUNTIME_BASELINE, RUNTIME_TARGET, AdmissionError, canonical, require, strict_json)

logger = logging.getLogger(__name__)

__all__ = ["ApprovedMigration", "APPROVED", "MigrationEvidence", "SCHEMA", "canonical_lf", "derive_artifact_selection",
           "verify_proposal_equivalence", "admit_lockfile_migration"]

SCHEMA = "gvc.lockfile-migration-admission/1"


@dataclass(frozen=True)
class ApprovedMigration:
    """The ACCEPTANCE CONTRACT: digests fixed by the owner's rulings, never read from the evidence being admitted."""

    proposal_sha256: str                    # ruling 2026-10-08e: the approved proposal (exact bytes)
    approved_generator_sha256: str          # the generator that produced it
    regenerated_proposal_sha256: str        # ruling 2026-10-08f section 5: the regeneration, corroboration only
    regenerating_generator_sha256: str
    equivalence_sha256: str                 # the equivalence record (schema 2) that compared them
    replay_plan_file_sha256: str            # the sealed replay plan v3 (file bytes)
    candidate_plan_file_sha256: str         # the sealed candidate plan v2 (file bytes)
    candidate_evidence_entry: str           # the artifact-inventory requirement naming the candidate run's evidence archive


APPROVED = ApprovedMigration(
    proposal_sha256="f21ac4bca99dabd28f2201fec4e27115c3ad4cc8791050f7d7ddbaf0dee645b9",
    approved_generator_sha256="2cd777bb611b36599a7e2ae3930ca138dc42d16db5a17116dfa1f12e7e2c5580",
    regenerated_proposal_sha256="d4910f60c315ca647acd4f4a30a4fb57af12c3e995e6ddc01515dfe31eebb593",
    regenerating_generator_sha256="652ff6806ec5adb567ab374b1593bd1a29524ba0a06ec1e66991935adaa11f52",
    equivalence_sha256="0f7b21584e3e728a9cb60a9c029f3f6bbbd6649b0a00c0a81eb49f0b0ab3d4c0",
    replay_plan_file_sha256="c578eb10572c3aac927a447b97600860f9cd38118653c83ff74f0aa291d3d50e",
    candidate_plan_file_sha256="cc1251f87270b87dee22cd241a9555ee45f9d6ecc5643454d9cad227fd0ef2af",
    candidate_evidence_entry="run-inventory-20261008:candidate_lock_v2_20261008T183528Z",
)


@dataclass(frozen=True)
class MigrationEvidence:
    """The exact bytes admitted together. Each is preserved verbatim in the migration record."""

    baseline_lock: bytes
    candidate_lock: bytes
    proposal: bytes
    regenerated_proposal: bytes
    equivalence: bytes
    replay_plan: bytes
    candidate_plan: bytes
    candidate_difference: bytes


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_lf(raw: bytes) -> bytes:
    """The canonical LF text domain: CRLF -> LF; a lone CR is refused (it has no canonical form)."""
    require(type(raw) is bytes and len(raw) > 0, "migration.empty_bytes")
    require(b"\r" not in raw.replace(b"\r\n", b""), "migration.lone_cr")
    return raw.replace(b"\r\n", b"\n")


def _json(raw: bytes, what: str):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise AdmissionError("migration.encoding:" + what)
    require(not text.startswith("﻿"), "migration.byte_order_mark:" + what)
    try:
        doc = strict_json(text)
    except ValueError as exc:                       # json.JSONDecodeError is a ValueError; AdmissionError passes through unchanged
        if isinstance(exc, AdmissionError):
            raise
        raise AdmissionError("migration.json:" + what)
    require(type(doc) is dict, "migration.root_not_object:" + what)
    return doc


def _keys(row, expected: set, reason: str) -> None:
    require(type(row) is dict and set(row) == expected, reason)


def _hex(value) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def derive_artifact_selection(replay_plan: dict, candidate_plan: dict) -> dict:
    """package -> {"route", "kind", "artifact_sha256", "receipt_sha256"} from the two sealed plans, and package -> version.

    The replay plan selects the baseline's artifacts (upstream binaries and local builds), the renv bootstrap source and the three
    runtime-supplied packages; the candidate plan selects the additions. Every row's key set is the measured shape of its route; a
    package selected twice is refused. Returns (selection, versions)."""
    selection, versions = {}, {}

    def add(package, version, entry):
        require(type(package) is str and package != "" and type(version) is str and version != "", "selection.identity")
        require(package not in selection, "selection.duplicate:" + package)
        selection[package], versions[package] = entry, version

    install = replay_plan.get("install")
    require(type(install) is list and len(install) > 0, "selection.replay_install")
    for row in install:
        route = row.get("route") if type(row) is dict else None
        if route == "upstream_binary":
            _keys(row, {"file_name", "kind", "package", "route", "sha256", "version"}, "selection.replay_row:upstream_binary")
            require(row["kind"] == "windows_binary" and _hex(row["sha256"]), "selection.replay_row:upstream_binary")
            add(row["package"], row["version"], {"artifact_sha256": row["sha256"], "kind": "windows_binary", "receipt_sha256": None,
                                                 "route": "upstream_binary"})
        elif route == "local_build":
            _keys(row, {"file_name", "kind", "package", "receipt_sha256", "route", "sha256", "version"}, "selection.replay_row:local_build")
            require(row["kind"] == "local_binary" and _hex(row["sha256"]) and _hex(row["receipt_sha256"]), "selection.replay_row:local_build")
            add(row["package"], row["version"], {"artifact_sha256": row["sha256"], "kind": "local_binary",
                                                 "receipt_sha256": row["receipt_sha256"], "route": "local_build"})
        else:
            raise AdmissionError("selection.replay_route:" + repr(route))
    bootstrap = replay_plan.get("bootstrap")
    require(type(bootstrap) is list and len(bootstrap) == 1, "selection.bootstrap")
    for row in bootstrap:
        _keys(row, {"file_name", "kind", "package", "sha256", "version"}, "selection.bootstrap_row")
        require(row["kind"] == "source" and _hex(row["sha256"]), "selection.bootstrap_row")
        add(row["package"], row["version"], {"artifact_sha256": row["sha256"], "kind": "source", "receipt_sha256": None,
                                             "route": "bootstrap"})
    supplied = replay_plan.get("runtime_supplied")
    require(type(supplied) is list and len(supplied) > 0, "selection.runtime_supplied")
    for row in supplied:
        _keys(row, {"package", "priority", "version"}, "selection.runtime_row")
        require(row["priority"] == "recommended", "selection.runtime_row")
        add(row["package"], row["version"], {"artifact_sha256": None, "kind": None, "receipt_sha256": None, "route": "runtime"})
    additions = candidate_plan.get("additions")
    require(type(additions) is list and len(additions) > 0, "selection.candidate_additions")
    for row in additions:
        _keys(row, {"artifact", "package", "route", "version"}, "selection.addition_row")
        art = row["artifact"]
        if row["route"] == "upstream_binary":
            _keys(art, {"description_sha256", "file_name", "kind", "provenance", "sha256", "size"}, "selection.addition_artifact:upstream_binary")
            require(art["kind"] == "windows_binary" and _hex(art["sha256"]), "selection.addition_artifact:upstream_binary")
            entry = {"artifact_sha256": art["sha256"], "kind": "windows_binary", "receipt_sha256": None, "route": "upstream_binary"}
        elif row["route"] == "local_build":
            _keys(art, {"build_plan_sha256", "build_summary_sha256", "description_sha256", "file_name", "inspection_record_sha256",
                        "kind", "load_record_sha256", "provenance", "receipt_sha256", "sha256", "size"}, "selection.addition_artifact:local_build")
            require(art["kind"] == "local_binary" and _hex(art["sha256"]) and _hex(art["receipt_sha256"]), "selection.addition_artifact:local_build")
            entry = {"artifact_sha256": art["sha256"], "kind": "local_binary", "receipt_sha256": art["receipt_sha256"], "route": "local_build"}
        else:
            raise AdmissionError("selection.addition_route:" + repr(row["route"]))
        add(row["package"], row["version"], entry)
    return selection, versions


def verify_proposal_equivalence(approved: bytes, regenerated: bytes, equivalence: bytes, contract: ApprovedMigration) -> dict:
    """Ruling 2026-10-08f section 5 (owner reference), with a TYPE-PRESERVING comparison: Python's == makes 1 == True and
    1 == 1.0, so payloads are compared as canonical JSON. The ONE permitted difference is generator_sha256."""
    require(_sha(approved) == contract.proposal_sha256, "equivalence.approved_digest")
    require(_sha(regenerated) == contract.regenerated_proposal_sha256, "equivalence.regenerated_digest")
    require(_sha(equivalence) == contract.equivalence_sha256, "equivalence.record_digest")
    old, new = _json(approved, "approved_proposal"), _json(regenerated, "regenerated_proposal")
    require(old.get("generator_sha256") == contract.approved_generator_sha256, "equivalence.old_generator")
    require(new.get("generator_sha256") == contract.regenerating_generator_sha256, "equivalence.new_generator")
    strip = (lambda d: {k: v for k, v in d.items() if k != "generator_sha256"})
    require(canonical(strip(old)) == canonical(strip(new)), "equivalence.substantive_difference")
    record = _json(equivalence, "equivalence_record")
    _keys(record, {"approval", "approved_generator_sha256", "approved_proposal_sha256", "checker_sha256", "comparison",
                   "permitted_difference", "regenerated_proposal_sha256", "regenerating_generator_sha256", "schema",
                   "substantive_content_equal"}, "equivalence.record_shape")
    require(record["schema"] == "gvc.lock-transition-equivalence/2", "equivalence.record_schema")
    require((record["approved_proposal_sha256"], record["regenerated_proposal_sha256"], record["approved_generator_sha256"],
             record["regenerating_generator_sha256"]) == (contract.proposal_sha256, contract.regenerated_proposal_sha256,
                                                          contract.approved_generator_sha256, contract.regenerating_generator_sha256),
            "equivalence.record_binding")
    require(record["permitted_difference"] == "generator_sha256" and record["substantive_content_equal"] is True, "equivalence.record_claim")
    return {"approved_proposal_sha256": contract.proposal_sha256, "regenerated_proposal_sha256": contract.regenerated_proposal_sha256,
            "equivalence_sha256": contract.equivalence_sha256, "permitted_difference": "generator_sha256",
            "substantive_content_equal": True, "recorded_checker_sha256": record["checker_sha256"]}


def _side(value) -> dict:
    return {"absent": True} if value is ABSENT else {"value": value}


def admit_lockfile_migration(evidence: MigrationEvidence, *, contract: ApprovedMigration, inventory_record) -> dict:
    """Admit the migration, or refuse with ONE reason code (AdmissionError). Deterministic: no time, no filesystem access.

    inventory_record: the committed, validated repository_records.artifact_inventory.ArtifactInventoryRecord."""
    from genomic_variant_classifier.repository_records.artifact_inventory import ArtifactInventoryRecord   # owner of that record type
    require(type(evidence) is MigrationEvidence, "migration.evidence_type")
    require(type(contract) is ApprovedMigration, "migration.contract_type")
    require(type(inventory_record) is ArtifactInventoryRecord, "migration.inventory_record_type")
    for name in MigrationEvidence.__dataclass_fields__:
        require(type(getattr(evidence, name)) is bytes and len(getattr(evidence, name)) > 0, "migration.missing:" + name)

    # 1. equivalence of the approved and regenerated proposals; the APPROVED bytes are the ones admitted
    equivalence = verify_proposal_equivalence(evidence.proposal, evidence.regenerated_proposal, evidence.equivalence, contract)
    proposal = _json(evidence.proposal, "proposal")
    require(proposal.get("schema") == "gvc.lock-transition-proposal/1", "migration.proposal_schema")
    inputs = proposal.get("inputs")
    _keys(inputs, {"baseline_canonical_sha256", "candidate_canonical_sha256", "candidate_plan_sha256", "candidate_sha256",
                   "difference_record_sha256", "evidence_zip_sha256", "replay_plan_sha256", "runtime_record_sha256"},
          "migration.proposal_inputs")

    # 2. identities, in their declared domains
    baseline_text, candidate_text = canonical_lf(evidence.baseline_lock), canonical_lf(evidence.candidate_lock)
    require(_sha(baseline_text) == inputs["baseline_canonical_sha256"], "migration.baseline_digest")
    require(_sha(evidence.candidate_lock) == inputs["candidate_sha256"], "migration.candidate_exact_digest")
    require(_sha(candidate_text) == inputs["candidate_canonical_sha256"], "migration.candidate_canonical_digest")
    require(_sha(evidence.candidate_difference) == inputs["difference_record_sha256"], "migration.difference_digest")
    require(_sha(evidence.replay_plan) == contract.replay_plan_file_sha256, "migration.replay_plan_file_digest")
    require(_sha(evidence.candidate_plan) == contract.candidate_plan_file_sha256, "migration.candidate_plan_file_digest")
    replay_plan, candidate_plan = _json(evidence.replay_plan, "replay_plan"), _json(evidence.candidate_plan, "candidate_plan")
    require(replay_plan.get("schema") == "gvc.replay-plan/1" and candidate_plan.get("schema") == "gvc.candidate-plan/1", "migration.plan_schema")
    require(plan_digest(replay_plan) == replay_plan.get("plan_sha256") == inputs["replay_plan_sha256"], "migration.replay_plan_body_digest")
    require(plan_digest(candidate_plan) == candidate_plan.get("plan_sha256") == inputs["candidate_plan_sha256"],
            "migration.candidate_plan_body_digest")
    # The replay was of THIS baseline, and the candidate plan was built on that replay and baseline.
    binding = candidate_plan.get("binding")
    require(type(binding) is dict and type(binding.get("replay")) is dict, "migration.candidate_plan_binding")
    require(replay_plan.get("lockfile_canonical_sha256") == binding.get("lockfile_canonical_sha256") == inputs["baseline_canonical_sha256"],
            "migration.plans_bind_another_baseline")
    require(binding["replay"].get("replay_plan_file_sha256") == contract.replay_plan_file_sha256
            and binding["replay"].get("replay_plan_sha256") == inputs["replay_plan_sha256"], "migration.candidate_plan_binds_another_replay")
    require(binding["replay"].get("runtime_record_sha256") == inputs["runtime_record_sha256"], "migration.runtime_record_binding")

    # 3. the transition itself (admission.admit_lock_transition -- the ONE owner of exact lockfile transitions)
    baseline, candidate = _json(baseline_text, "baseline_lock"), _json(candidate_text, "candidate_lock")
    runtime = proposal.get("runtime_change")
    _keys(runtime, {"evidence", "field", "new", "old"}, "migration.runtime_change_shape")
    require((runtime["field"], runtime["old"], runtime["new"]) == ("R.Version", RUNTIME_BASELINE, RUNTIME_TARGET), "migration.runtime_change")
    require(inputs["runtime_record_sha256"] in runtime["evidence"], "migration.runtime_change_evidence")
    additions, transitions = proposal.get("additions"), proposal.get("transitions")
    require(type(additions) is dict and type(transitions) is list, "migration.proposal_shape")
    transition = admit_lock_transition(baseline, candidate, observed_version=runtime["new"], additions=additions,
                                       approved_transitions=transitions)

    # 4. the selected artifacts: derived here from the sealed plans, equal to the proposal's and the candidate run's
    selection, versions = derive_artifact_selection(replay_plan, candidate_plan)
    require(set(selection) == set(candidate["Packages"]), "migration.selection_coverage")
    for package, version in versions.items():
        require(candidate["Packages"][package]["Version"] == version, "migration.selection_version:" + package)
    require(proposal.get("artifact_selection") == selection, "migration.proposal_selection_differs")
    difference = _json(evidence.candidate_difference, "candidate_difference")
    _keys(difference, {"added", "artifact_selection", "baseline_canonical_sha256", "candidate_sha256", "field_differences", "removed",
                       "schema", "top_level"}, "migration.difference_shape")
    require(difference["schema"] == "gvc.candidate-difference/1", "migration.difference_schema")
    require(difference["artifact_selection"] == selection, "migration.run_selection_differs")
    require((difference["baseline_canonical_sha256"], difference["candidate_sha256"]) == (inputs["baseline_canonical_sha256"],
                                                                                          inputs["candidate_sha256"]), "migration.difference_binding")
    # the run's own report and the repository's exact field difference agree (ABSENT-aware, type-preserving)
    ours = sorted(canonical({"field": f, "new": _side(n), "old": _side(o), "package": p}) for p, f, o, n in exact_field_diff(baseline, candidate))
    theirs = difference["field_differences"]
    require(type(theirs) is list and sorted(canonical(r) for r in theirs) == ours and len(theirs) == len(ours), "migration.difference_disagrees")
    require(difference["removed"] == [] and type(difference["added"]) is dict and set(difference["added"]) == set(additions),
            "migration.difference_membership")
    for name, record in difference["added"].items():
        require(canonical(record) == canonical(candidate["Packages"][name]), "migration.difference_added_record:" + name)

    # 5. new-package provenance tied to the selected artifact (ruling 2026-10-08d)
    plan_additions = {row["package"]: row for row in candidate_plan["additions"]}
    require({p: r["version"] for p, r in plan_additions.items()} == additions, "migration.additions_not_the_plans")
    addition_artifacts = proposal.get("addition_artifacts")
    require(type(addition_artifacts) is dict and set(addition_artifacts) == set(additions), "migration.addition_artifacts_coverage")
    for name, declared in sorted(addition_artifacts.items()):
        _keys(declared, {"artifact_sha256", "description_sha256", "kind", "lock_repository_label", "provenance", "receipt_sha256", "route"},
              "migration.addition_artifact_shape:" + name)
        art = plan_additions[name]["artifact"]
        require((declared["artifact_sha256"], declared["description_sha256"], declared["kind"], declared["provenance"], declared["route"],
                 declared["receipt_sha256"]) == (art["sha256"], art["description_sha256"], art["kind"], art["provenance"],
                                                 plan_additions[name]["route"], art.get("receipt_sha256")),
                "migration.addition_artifact_differs:" + name)
        require(candidate["Packages"][name].get("Repository") == declared["lock_repository_label"], "migration.addition_repository_label:" + name)

    # 6. corroboration (reported, never authority): the independent re-implementation reproduced every candidate record
    reproduction = proposal.get("record_reproduction")
    _keys(reproduction, {"method", "records"}, "migration.reproduction_shape")
    records = reproduction["records"]
    require(type(records) is dict and set(records) == set(candidate["Packages"]), "migration.reproduction_coverage")
    require(all(type(r) is dict and r.get("reproduced") is True for r in records.values()), "migration.reproduction_failed")

    # 7. the candidate lockfile came from the admitted candidate run, whose evidence archive the committed inventory found
    entries = {r.entry_id: r for r in inventory_record.requirements}
    results = {r.entry_id: r for r in inventory_record.results()}
    entry = entries.get(contract.candidate_evidence_entry)
    require(entry is not None and entry.sha256 == inputs["evidence_zip_sha256"], "migration.run_evidence_not_inventoried")
    require(results[contract.candidate_evidence_entry].state is EvidenceState.MATCH, "migration.run_evidence_not_matched")
    # ... and that measurement was taken against THESE plans and THIS runtime record
    measured = {d.label: d.sha256 for d in inventory_record.plan_documents}
    require(measured.get("replay-v3-plan") == contract.replay_plan_file_sha256
            and measured.get("candidate-v2-plan") == contract.candidate_plan_file_sha256, "migration.inventory_measured_other_plans")
    supplied = {(r.package, r.version, r.runtime_record_sha256) for r in inventory_record.runtime_supplied}
    require(supplied == {(p, versions[p], inputs["runtime_record_sha256"]) for p, e in selection.items() if e["route"] == "runtime"},
            "migration.runtime_supplied_disagrees")

    routes = Counter(e["route"] for e in selection.values())
    return {
        "schema": SCHEMA,
        "approved_proposal_sha256": contract.proposal_sha256,
        "equivalence": equivalence,
        "baseline": {"canonical_sha256": inputs["baseline_canonical_sha256"], "exact_sha256": _sha(evidence.baseline_lock),
                     "r_version": baseline["R"]["Version"], "packages": len(baseline["Packages"])},
        "candidate": {"exact_sha256": inputs["candidate_sha256"], "canonical_sha256": inputs["candidate_canonical_sha256"],
                      "r_version": candidate["R"]["Version"], "packages": len(candidate["Packages"])},
        "plans": {"replay_plan_file_sha256": contract.replay_plan_file_sha256, "replay_plan_sha256": inputs["replay_plan_sha256"],
                  "candidate_plan_file_sha256": contract.candidate_plan_file_sha256, "candidate_plan_sha256": inputs["candidate_plan_sha256"]},
        "candidate_difference_sha256": inputs["difference_record_sha256"],
        "runtime_record_sha256": inputs["runtime_record_sha256"],
        "candidate_run_evidence": {"inventory_record_id": inventory_record.record_id.value, "entry_id": contract.candidate_evidence_entry,
                                   "sha256": inputs["evidence_zip_sha256"]},
        "transition": transition,
        "selection": {"packages": len(selection), "routes": dict(sorted(routes.items()))},
        "corroboration": {"method": reproduction["method"], "records_reproduced": len(records),
                          "status": "corroboration only -- renv 1.2.3 in the qualified R (the candidate run) is the authority"},
        "claim": ("ADMITTED: the baseline, the candidate, the selected artifacts and the approved proposal agree. A lockfile admission, "
                  "not an installation authorization and not environment qualification: the isolated replay re-verifies every byte it "
                  "installs, and the method tests run on that library."),
    }
