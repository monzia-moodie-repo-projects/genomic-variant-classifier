"""The maintenance-evidence channel and its measured qualification.

Author: Monzia Moodie
"""
from __future__ import annotations

import copy
import json
import re

import pytest

from genomic_variant_classifier.operations.maintenance_channel import (
    ChannelError, ChannelQualification, back_up_evidence, cleanup_operations,
    enumerate_evidence, evidence_directory, is_retained, policy_digest,
    publish_evidence, read_evidence, require_backup_population,
    restore_evidence, validate_evidence)

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


def _document(operation_id="op-1"):
    return {"schema": "gvc.maintenance-evidence", "schema_version": 1,
            "operation_id": operation_id,
            "operation_kind": "archive_admission",
            "policy_identity": policy_digest(), "plan_sha256": "a" * 64,
            "predecessor_commit": "b" * 40, "candidate_commit": "c" * 40,
            "integrated_commit": "d" * 40,
            "preimage_manifest_sha256": "e" * 64,
            "postimage_manifest_sha256": "f" * 64,
            "approved_entries_sha256": "0" * 64,
            "validation_evidence_sha256": "1" * 64,
            "admitted_records": 88, "preserved_records": 18,
            "resulting_records": 106, "completion_route": "normal",
            "recorded_at_utc": "2026-09-08T00:00:00Z"}


def test_evidence_is_published_and_read_back(tmp_path):
    operation = tmp_path / "op-1"
    operation.mkdir()
    digest = publish_evidence(operation, _document())
    assert _SHA256.fullmatch(digest)
    assert read_evidence(operation)["operation_id"] == "op-1"
    assert is_retained(operation)


def test_a_second_publication_is_refused_not_replaced(tmp_path):
    operation = tmp_path / "op-1"
    operation.mkdir()
    publish_evidence(operation, _document())
    before = (evidence_directory(operation)
              / "maintenance_evidence.json").read_bytes()
    with pytest.raises(ChannelError):
        publish_evidence(operation, _document())
    assert (evidence_directory(operation)
            / "maintenance_evidence.json").read_bytes() == before


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema_version=True),
    lambda d: d.update(schema_version=1.0),
    lambda d: d.update(integrated_commit="d" * 7),
    lambda d: d.update(plan_sha256="A" * 64),
    lambda d: d.update(admitted_records=87),
    lambda d: d.update(admitted_records=True),
    lambda d: d.update(admitted_records=-1),
    lambda d: d.update(completion_route="probably fine"),
    lambda d: d.update(operation_kind="installation"),
    lambda d: d.update(schema="unrelated"),
    lambda d: d.update(extra=1),
    lambda d: d.pop("plan_sha256"),
])
def test_invalid_evidence_is_refused(mutate):
    document = _document()
    mutate(document)
    with pytest.raises(ChannelError):
        validate_evidence(document)


def test_record_counts_must_reconcile():
    document = _document()
    document["resulting_records"] = 105
    with pytest.raises(ChannelError):
        validate_evidence(document)


def test_an_unreadable_record_raises_and_is_preserved(tmp_path):
    operation = tmp_path / "op-damaged"
    evidence_directory(operation).mkdir(parents=True)
    target = evidence_directory(operation) / "maintenance_evidence.json"
    target.write_bytes(b'{"unfinished"')
    with pytest.raises(ChannelError):
        read_evidence(operation)
    assert target.read_bytes() == b'{"unfinished"'


def test_enumeration_reports_the_unreadable(tmp_path):
    good = tmp_path / "op-good"
    good.mkdir()
    publish_evidence(good, _document("op-good"))
    bad = tmp_path / "op-bad"
    evidence_directory(bad).mkdir(parents=True)
    (evidence_directory(bad) / "maintenance_evidence.json").write_bytes(b"{")
    rows = {r["operation_id"]: r for r in enumerate_evidence(tmp_path)}
    assert rows["op-good"]["state"] == "readable"
    assert rows["op-bad"]["state"] == "unreadable"


def test_an_absent_root_is_empty_and_a_file_root_is_damage(tmp_path):
    """Absence and damage are different answers."""
    assert enumerate_evidence(tmp_path / "never-existed") == []
    marker = tmp_path / "a-file"
    marker.write_bytes(b"x")
    with pytest.raises(ChannelError):
        enumerate_evidence(marker)


import contextlib


@contextlib.contextmanager
def _exclusion(name):
    """A stand-in for the operation coordination cleanup must acquire."""
    yield name


def test_cleanup_refuses_an_operation_bearing_evidence(tmp_path):
    retained = tmp_path / "op-retained"
    retained.mkdir()
    publish_evidence(retained, _document("op-retained"))
    plain = tmp_path / "op-plain"
    plain.mkdir()
    report = cleanup_operations(
        tmp_path, remove_resolved={"op-retained", "op-plain"},
        exclusion_for=_exclusion)
    assert report["removed"] == ["op-plain"]
    assert "op-retained" in report["refused"]
    assert read_evidence(retained) is not None


def test_a_missing_marker_beside_evidence_is_damage_not_permission(tmp_path):
    """MEASURED 2026-09-08: the previous cleanup treated an absent marker as
    permission to delete, even with the evidence still there."""
    damaged = tmp_path / "op-damaged"
    damaged.mkdir()
    publish_evidence(damaged, _document("op-damaged"))
    (evidence_directory(damaged) / ".retained").unlink()
    report = cleanup_operations(tmp_path, remove_resolved={"op-damaged"},
                                exclusion_for=_exclusion)
    assert report["removed"] == []
    assert "damage" in report["refused"]["op-damaged"]
    assert read_evidence(damaged) is not None


def test_cleanup_without_coordination_refuses_everything(tmp_path):
    """An uncoordinated delete is the failure mode this function prevents, so
    it refuses rather than proceeding."""
    plain = tmp_path / "op-plain"
    plain.mkdir()
    report = cleanup_operations(tmp_path, remove_resolved={"op-plain"},
                                exclusion_for=None)
    assert report["removed"] == []
    assert "no coordination mechanism" in report["refused"]["op-plain"]
    assert plain.is_dir()


def test_cleanup_acquires_the_exclusion_for_each_operation(tmp_path):
    acquired = []

    @contextlib.contextmanager
    def recording(name):
        acquired.append(name)
        yield name

    for name in ("op-a", "op-b"):
        (tmp_path / name).mkdir()
    cleanup_operations(tmp_path, remove_resolved={"op-a", "op-b"},
                       exclusion_for=recording)
    assert sorted(acquired) == ["op-a", "op-b"]


def test_testing_this_helper_is_not_a_claim_about_real_cleanup():
    """Demonstrating that the REPOSITORY's transaction cleanup and cache
    eviction exclude this subtree are separate claims about separate code,
    and neither is established here."""
    assert cleanup_operations.__doc__ is not None
    assert "separate claims about separate code" in cleanup_operations.__doc__


def test_a_backup_carries_the_dependencies_needed_to_interpret_it(tmp_path):
    operation = tmp_path / "op-1"
    operation.mkdir()
    publish_evidence(operation, _document())
    (operation / "specification.json").write_bytes(b'{"spec": true}\n')
    (operation / "completion_receipt.json").write_bytes(b'{"receipt": true}\n')
    manifest = back_up_evidence(operation, tmp_path / "backup")
    assert "manifest_sha256" in manifest
    assert "specification.json" in manifest["files"]
    assert "completion_receipt.json" in manifest["files"]


def test_a_restore_round_trip_verifies_every_file(tmp_path):
    operation = tmp_path / "op-1"
    operation.mkdir()
    publish_evidence(operation, _document())
    (operation / "specification.json").write_bytes(b'{"spec": true}\n')
    (operation / "completion_receipt.json").write_bytes(b'{"receipt": true}\n')
    manifest = back_up_evidence(operation, tmp_path / "backup")
    import shutil
    shutil.rmtree(operation)
    restore_evidence(tmp_path / "backup", operation,
                     expected_manifest_sha256=manifest["manifest_sha256"])
    assert read_evidence(operation)["operation_id"] == "op-1"


def test_a_backup_never_replaces_an_existing_destination(tmp_path):
    operation = tmp_path / "op-1"
    operation.mkdir()
    publish_evidence(operation, _document())
    existing = tmp_path / "backup"
    existing.mkdir()
    with pytest.raises(ChannelError):
        back_up_evidence(operation, existing)


def test_a_backup_with_no_evidence_is_refused(tmp_path):
    empty = tmp_path / "op-empty"
    empty.mkdir()
    with pytest.raises(ChannelError):
        back_up_evidence(empty, tmp_path / "backup")


def test_a_qualification_is_NOT_a_capability():
    """MEASURED 2026-09-08: a caller built one claiming full qualification,
    including power-loss durability, in a single expression. An earlier
    docstring claimed callers could not; a frozen dataclass never prevented
    it. The value means some run recorded these outcomes -- a consumer must
    bind the transcript digest and check the platform and scope."""
    forged = ChannelQualification(
        policy_sha256="0" * 64, implementation_sha256="0" * 64,
        test_evidence_sha256="0" * 64, platform="linux",
        scope="a single temporary workspace",
        process_crash_recovery_verified=True, cleanup_exclusion_verified=True,
        restoration_verified=True, power_loss_durability_verified=True,
        unverified_reasons=())
    assert forged.power_loss_durability_verified is True
    assert forged.requires_operator_review() == ()


def test_a_qualification_names_its_platform_and_scope():
    """A Linux result does not qualify Windows, and a backup beside its
    original in one workspace does not establish storage-device loss."""
    recorded = ChannelQualification(
        policy_sha256="0" * 64, implementation_sha256="0" * 64,
        test_evidence_sha256="0" * 64, platform="linux",
        scope="a single temporary workspace; NOT a separate storage device",
        process_crash_recovery_verified=True, cleanup_exclusion_verified=True,
        restoration_verified=True, power_loss_durability_verified=False,
        unverified_reasons=("power_loss_durability_verified",))
    assert "power_loss_durability_verified" in recorded.requires_operator_review()
    assert "NOT a separate storage device" in recorded.scope


def test_a_backup_manifest_naming_a_parent_escape_is_refused():
    """MEASURED 2026-09-08: '../source.txt' was accepted from an unvalidated
    manifest."""
    with pytest.raises(ChannelError):
        require_backup_population({"../source.txt": {}})


def test_a_backup_missing_an_interpreting_dependency_is_refused(tmp_path):
    operation = tmp_path / "op-1"
    operation.mkdir()
    publish_evidence(operation, _document())
    with pytest.raises(ChannelError):
        back_up_evidence(operation, tmp_path / "backup")


def test_a_restore_refuses_a_wrong_manifest_digest(tmp_path):
    operation = tmp_path / "op-1"
    operation.mkdir()
    publish_evidence(operation, _document())
    (operation / "specification.json").write_bytes(b'{"spec": true}\n')
    (operation / "completion_receipt.json").write_bytes(b'{"receipt": true}\n')
    back_up_evidence(operation, tmp_path / "backup")
    import shutil
    shutil.rmtree(operation)
    with pytest.raises(ChannelError):
        restore_evidence(tmp_path / "backup", operation,
                         expected_manifest_sha256="0" * 64)


def test_a_restore_verifies_all_bytes_before_publishing_any(tmp_path):
    operation = tmp_path / "op-1"
    operation.mkdir()
    publish_evidence(operation, _document())
    (operation / "specification.json").write_bytes(b'{"spec": true}\n')
    (operation / "completion_receipt.json").write_bytes(b'{"receipt": true}\n')
    manifest = back_up_evidence(operation, tmp_path / "backup")
    (tmp_path / "backup" / "specification.json").write_bytes(b"tampered\n")
    target = tmp_path / "op-restored"
    with pytest.raises(ChannelError):
        restore_evidence(tmp_path / "backup", target,
                         expected_manifest_sha256=manifest["manifest_sha256"])
    assert not target.exists()
    assert not list(tmp_path.glob(".restore-*.pending"))


def test_a_restore_never_replaces_an_existing_operation(tmp_path):
    operation = tmp_path / "op-1"
    operation.mkdir()
    publish_evidence(operation, _document())
    (operation / "specification.json").write_bytes(b'{"spec": true}\n')
    (operation / "completion_receipt.json").write_bytes(b'{"receipt": true}\n')
    manifest = back_up_evidence(operation, tmp_path / "backup")
    with pytest.raises(ChannelError):
        restore_evidence(tmp_path / "backup", operation,
                         expected_manifest_sha256=manifest["manifest_sha256"])


def test_a_directory_at_the_evidence_path_is_damage_not_absence(tmp_path):
    """`if not final.is_file(): return None` reported a DIRECTORY as absent --
    the category error unit S was installed to eliminate."""
    operation = tmp_path / "op-1"
    evidence_directory(operation).mkdir(parents=True)
    (evidence_directory(operation) / "maintenance_evidence.json").mkdir()
    with pytest.raises(ChannelError):
        read_evidence(operation)


def test_an_absent_record_is_absence(tmp_path):
    operation = tmp_path / "op-1"
    evidence_directory(operation).mkdir(parents=True)
    assert read_evidence(operation) is None
