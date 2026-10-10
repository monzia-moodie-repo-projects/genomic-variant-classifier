"""The COMMITTED lockfile migration record and the renv.lock succession it declares (ADR-0004 AUTHORITY-SUCCESSION-1).

The repository holds itself to the migration: exactly one record, its manifest in its deterministic rendering, every preserved
artifact present with its exact bytes and nothing else beside them, the admission RE-DERIVED from those preserved bytes by the
admission policy equal to the admission the manifest states (the behavioural gate), and the live renv.lock holding exactly the
admitted successor -- its canonical LF text -- while the predecessor survives verbatim in the record.

Author: Monzia Moodie
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from genomic_variant_classifier.environment_qualification.lockfile_admission import (
    APPROVED, MigrationEvidence, admit_lockfile_migration, canonical_lf)
from genomic_variant_classifier.repository_records.artifact_inventory import scan_records
from genomic_variant_classifier.repository_records.classification import ProvenanceRelation
from genomic_variant_classifier.repository_records.lockfile_migration import (
    PARTS, LockfileMigrationError, LockfileMigrationManifest, family_root)
from genomic_variant_classifier.repository_records.path_budget import budget_violations
from genomic_variant_classifier.repository_records.roles import RecordsOntologyError

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT.joinpath(*family_root().parts)
#: The committed record by id -> SHA-256 of its manifest's exact bytes (pinned here; a new migration is a NEW record added here).
PINNED = {"REC-722981980c854434ac9e3e7ec86956fe": "6cc2f4888ba29081e7c1761cacaa17424123b976ba140b41c05cb1367876c7f8"}


def _manifest_path(record_id: str) -> Path:
    return FAMILY / record_id / "manifest.json"


def _load(record_id: str = "REC-722981980c854434ac9e3e7ec86956fe") -> LockfileMigrationManifest:
    return LockfileMigrationManifest.parse(_manifest_path(record_id).read_bytes())


def test_the_family_holds_exactly_the_pinned_records():
    assert sorted(p.name for p in FAMILY.iterdir()) == sorted(PINNED)
    for record_id in PINNED:
        assert sorted(p.name for p in (FAMILY / record_id).iterdir()) == ["artifacts", "manifest.json"]


@pytest.mark.parametrize("record_id", sorted(PINNED))
def test_the_manifest_is_exactly_the_reviewed_bytes_and_round_trips(record_id):
    raw = _manifest_path(record_id).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINNED[record_id]
    manifest = LockfileMigrationManifest.parse(raw)
    assert manifest.render() == raw and manifest.record_id.value == record_id


def test_every_preserved_artifact_is_present_exact_and_alone():
    parts = _load().read_preserved(ROOT)
    assert sorted(parts) == sorted(PARTS)


def test_the_admission_rederived_from_the_preserved_bytes_is_the_recorded_admission():
    """The behavioural gate of the succession: the policy, run NOW on the preserved bytes, reproduces what the record states."""
    manifest = _load()
    p = manifest.read_preserved(ROOT)
    inventory = [r for r in scan_records(ROOT) if r.record_id.value == manifest.admission["candidate_run_evidence"]["inventory_record_id"]]
    assert len(inventory) == 1
    evidence = MigrationEvidence(baseline_lock=p["baseline_lockfile"], candidate_lock=p["candidate_lockfile"], proposal=p["approved_proposal"],
                                 regenerated_proposal=p["regenerated_proposal"], equivalence=p["equivalence_record"],
                                 replay_plan=p["replay_plan"], candidate_plan=p["candidate_plan"], candidate_difference=p["candidate_difference"])
    assert admit_lockfile_migration(evidence, contract=APPROVED, inventory_record=inventory[0]) == manifest.admission
    assert manifest.admission["approved_proposal_sha256"] == APPROVED.proposal_sha256


def test_the_live_lockfile_is_exactly_the_admitted_successor():
    manifest = _load()
    live = (ROOT / "renv.lock").read_bytes()
    assert b"\r" not in live, "renv.lock must be the canonical LF text (the record keeps the CRLF original)"
    assert hashlib.sha256(live).hexdigest() == manifest.succession.successor_canonical_sha256
    assert live == canonical_lf(manifest.read_preserved(ROOT)["candidate_lockfile"])
    lock = json.loads(live)
    assert (lock["R"]["Version"], lock["Bioconductor"]["Version"], len(lock["Packages"])) == ("4.6.1", "3.23", 102)


def test_the_predecessor_survives_verbatim_and_is_no_longer_live():
    manifest = _load()
    baseline = manifest.read_preserved(ROOT)["baseline_lockfile"]
    assert hashlib.sha256(canonical_lf(baseline)).hexdigest() == manifest.succession.predecessor_canonical_sha256
    assert baseline != (ROOT / "renv.lock").read_bytes()
    entry = [f for f in manifest.files if f.part == "baseline_lockfile"][0]
    assert entry.disposition.provenance == (ProvenanceRelation.SUPERSEDED_AUTHORITY,)


def test_the_preserved_candidate_keeps_its_original_line_endings_in_git():
    """`*.lock text eol=lf` would rewrite the CRLF candidate on commit; MIGRATION-ARTIFACTS-PRESERVED-1 must unset text for it."""
    rel = [f for f in _load().files if f.part == "candidate_lockfile"][0].identity.instance.canonical_path
    out = subprocess.run(["git", "check-attr", "text", "--", rel], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and out.stdout.strip() == rel + ": text: unset", out.stdout + out.stderr
    assert b"\r\n" in (ROOT / rel).read_bytes()


def test_the_artifacts_are_flat_and_every_path_is_within_the_repository_path_budget():
    """MEASURED 2026-10-10: one directory per part made the longest artifact path 169 characters -- 260 in the owner's %TEMP%
    clone -- and the installer's apply failed on Windows. Flat, the longest is 148."""
    manifest = _load()
    assert len(set(PARTS.values())) == len(PARTS)
    paths = [f.identity.instance.canonical_path for f in manifest.files] + [manifest.root.as_posix() + "/manifest.json"]
    assert all(budget_violations(p) == [] for p in paths), [budget_violations(p) for p in paths]
    assert max(len(p) for p in paths) == 148
    assert all(PurePosixPath(f.identity.instance.canonical_path).parent == manifest.root / "artifacts" for f in manifest.files)
    assert [p for p in (ROOT.joinpath(*manifest.root.parts) / "artifacts").iterdir() if not p.is_file()] == []


def test_a_shared_basename_is_refused_when_the_owner_is_loaded():
    """The flat layout is unambiguous only while the eight basenames are distinct; the owner refuses to load otherwise. Executed on
    the owner's own source with one basename duplicated, as a fresh module in the same package."""
    import types

    import genomic_variant_classifier.repository_records.lockfile_migration as owner
    source = Path(owner.__file__).read_text(encoding="utf-8")
    original = '"candidate_plan": "candidate_plan.json",'
    assert source.count(original) == 1
    module = types.ModuleType("lockfile_migration_duplicated_basename")
    module.__package__ = "genomic_variant_classifier.repository_records"
    code = compile(source.replace(original, '"candidate_plan": "replay_plan.json",'), owner.__file__, "exec")
    with pytest.raises(RecordsOntologyError) as exc:
        exec(code, module.__dict__)
    assert "share a basename" in str(exc.value)


def test_no_record_file_is_ignored():
    """NUL-separated, with a sentinel that MUST be reported, so a non-discriminating instrument fails instead of passing vacuously."""
    manifest = _load()
    paths = [manifest.root.as_posix() + "/manifest.json"] + [f.identity.instance.canonical_path for f in manifest.files]
    sentinel = "probe_not_evidence.log"
    out = subprocess.run(["git", "check-ignore", "--no-index", "-z", "--stdin"],
                         input=("\0".join(paths + [sentinel]) + "\0").encode("utf-8"), capture_output=True, cwd=ROOT)
    assert out.returncode in (0, 1), out.stderr.decode("utf-8", "replace")
    assert [x for x in out.stdout.decode("utf-8").split("\0") if x] == [sentinel]


# ------------------------------------------------------------------ the typed owner refuses (on copies)

def _copy_record(tmp_path) -> Path:
    src = FAMILY / sorted(PINNED)[0]
    dst = tmp_path.joinpath(*family_root().parts) / src.name
    shutil.copytree(src, dst)
    return dst


def _owner_error(call) -> str:
    with pytest.raises(LockfileMigrationError) as exc:
        call()
    return str(exc.value)


def test_a_changed_artifact_byte_is_refused(tmp_path):
    dst = _copy_record(tmp_path)
    target = dst / "artifacts" / PARTS["approved_proposal"]
    target.write_bytes(target.read_bytes() + b" ")
    assert "approved_proposal: size differs" == _owner_error(lambda: _load().read_preserved(tmp_path))


def test_an_extra_file_beside_the_artifacts_is_refused(tmp_path):
    dst = _copy_record(tmp_path)
    (dst / "artifacts" / "stray.json").write_bytes(b"{}\n")
    assert _owner_error(lambda: _load().read_preserved(tmp_path)).startswith("inventory: on disk but not indexed")


def test_a_directory_inside_the_flat_artifacts_is_refused(tmp_path):
    dst = _copy_record(tmp_path)
    (dst / "artifacts" / "regenerated_proposal").mkdir()
    assert _owner_error(lambda: _load().read_preserved(tmp_path)).endswith("not a regular file (artifacts/ is flat)")


def test_a_missing_artifact_is_refused(tmp_path):
    dst = _copy_record(tmp_path)
    (dst / "artifacts" / PARTS["equivalence_record"]).unlink()
    assert "indexed but not on disk" in _owner_error(lambda: _load().read_preserved(tmp_path))


def _mutated_manifest(mutate) -> bytes:
    doc = json.loads(_manifest_path(sorted(PINNED)[0]).read_bytes())
    mutate(doc)
    return (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["files"].pop(0), "files: parts must be exactly"),
    (lambda d: d["files"][0].update(canonical_path=d["files"][0]["canonical_path"].replace("/artifacts/", "/other/")), "must be records/"),
    (lambda d: d["succession"].update(successor_canonical_sha256=d["succession"]["predecessor_canonical_sha256"]), "succession: successor equals"),
    (lambda d: d["succession"].update(successor_canonical_sha256="0" * 64), "succession: successor is not the admitted candidate"),
    (lambda d: d["succession"].update(path="renv/renv.lock"), "succession: path"),
    (lambda d: d["admission"]["candidate"].update(exact_sha256="0" * 64), "candidate_lockfile: not the admitted candidate bytes"),
    (lambda d: d["admission"].update(schema="gvc.lockfile-migration-admission/2"), "admission: schema"),
    (lambda d: d.update(extra=1), "manifest: undeclared key(s)"),
    (lambda d: d["files"][0].update(retention="transient_diagnostic"), "retention"),
    (lambda d: d["files"][0].update(disclosure="restricted_verbatim"), "RESTRICTED_VERBATIM may not be ADMITTED_VERBATIM"),
    (lambda d: d["files"][0].update(disclosure="secret"), "unrecognised vocabulary term"),
])
def test_the_manifest_owner_refuses(mutate, message):
    with pytest.raises(Exception) as exc:
        LockfileMigrationManifest.parse(_mutated_manifest(mutate))
    assert message in str(exc.value)


def test_a_float_or_a_duplicate_key_is_refused():
    raw = _manifest_path(sorted(PINNED)[0]).read_bytes()
    assert "non-integer number" in _owner_error(lambda: LockfileMigrationManifest.parse(raw.replace(b'"schema_version": 1', b'"schema_version": 1.0')))
    duplicated = raw.replace(b'  "schema_version": 1,', b'  "schema_version": 1,\n  "schema_version": 1,', 1)
    assert duplicated != raw and "duplicate key" in _owner_error(lambda: LockfileMigrationManifest.parse(duplicated))


def test_a_non_canonical_rendering_is_refused():
    raw = _manifest_path(sorted(PINNED)[0]).read_bytes()
    assert "round-trip differs" in _owner_error(lambda: LockfileMigrationManifest.parse(raw.replace(b"\n", b"\n ", 1)))


def test_the_record_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _load().as_of = "x"           # type: ignore[misc]
