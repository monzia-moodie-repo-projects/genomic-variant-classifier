"""The artifact-inventory verification record (owner rulings 2026-10-08c, 2026-10-08e, 2026-10-08f): requirement, content and location are three
entities; each location is observed once; every requirement's result is DERIVED; counts are of requirements, content objects and
locations (never redundancy); construction is validation; rendering is deterministic; parsing is strict and round-trips; nothing
undisclosable enters a public record; the current index is a pure projection; readiness is judged against the PLAN, not the record, and
names every piece of code that determined it; the store root is accepted only as a plain canonical path (no link, junction or reparse
point from the anchor down); one filesystem identity cannot hold two contents.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import types
from pathlib import Path

import pytest

from genomic_variant_classifier.environment_qualification import admission
from genomic_variant_classifier.environment_qualification.admission import EvidenceState, artifact_readiness, readiness_decision
from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError
from genomic_variant_classifier.paths.runtime_paths import PROJECT_NAME, PROJECT_SENTINELS, resolve_runtime_paths
from genomic_variant_classifier.repository_records import artifact_inventory as ai
from genomic_variant_classifier.repository_records.identity import RecordId
from genomic_variant_classifier.repository_records.roles import ArtifactRole, role_for_path

ROOT = Path(__file__).resolve().parents[2]
R1, R2, R3 = ("REC-" + c * 32 for c in "abc")
DA, DB, DX, DV, DF = ("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)     # content A, content B, other bytes, volume, file
P, C = ai.ArtifactPurpose, ai.LocationCondition
S = "gvc-artifacts"


def req(entry_id="acq:A", sha=DA, size=10, purposes=(P.ACQUIRED,), frm="acquisition", hint=(S, "accepted/A.zip"), kind="windows_binary",
        package="A", version="1"):
    return ai.Requirement(entry_id, kind, package, version, sha, size, purposes, frm, hint)


def loc(path="accepted/A.zip", cond=C.READ, sha=DA, size=10, file_id=DF, vol=DV, store=S):
    if cond is not C.READ:
        return ai.LocationObservation(store, path, cond, None, None, None, None)
    return ai.LocationObservation(store, path, cond, sha, size, file_id, vol)


def search(sha=DA, size=10, frm="acquisition", scope="same_size_files", read=1, unreadable=0):
    return ai.ContentSearch(sha, size, frm, scope, read, unreadable)


def collection(hashed=2):
    return ai.Collection(100, hashed, hashed, 100, "size and modification time identical before and after each hash; the whole census "
                         "repeated before finalization", "no artifact writer ran during the measurement (operator statement)")


def record(**over):
    fields = dict(record_id=RecordId(R1), previous_record_id=None, measurement_started_at="2026-10-08T19:30:00Z",
                  measurement_completed_at="2026-10-08T19:31:00Z", scope="every artifact named by the listed plans", verifier_sha256=DX,
                  stores=(S,), plan_documents=(ai.PlanDocument("acquisition", DB), ai.PlanDocument("replay-plan", DX)),
                  requirements=(req(), req("replay:B", sha=DB, size=None, purposes=(P.SELECTED_UPSTREAM_BINARY,), frm="replay-plan",
                                           hint=(S, "accepted/B.zip"), package="B")),
                  locations=(loc(), loc("archive/A.zip", file_id="f" * 64), loc("accepted/B.zip", C.ABSENT)),
                  searches=(search(read=2), search(DB, None, None, "every_file", read=100)),
                  collection=collection(), runtime_supplied=(ai.RuntimeSupplied("Matrix", "1.7-5", "recommended", DX),),
                  gaps=(ai.KnownGap("fixtures_20261007T022427Z", "its evidence bundle was never retained; no digest exists"),))
    fields.update(over)
    return ai.ArtifactInventoryRecord(**fields)


def reason(fn):
    with pytest.raises(ai.ArtifactInventoryError) as exc:
        fn()
    return str(exc.value)


def results(r):
    return {x.entry_id: x for x in r.results()}


# ------------------------------------------------------------------------------------------------------------------ construction
def test_a_valid_record_renders_deterministically_and_round_trips():
    r = record()
    raw = r.render()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n") and raw == r.render()
    assert ai.ArtifactInventoryRecord.parse(raw) == r and ai.ArtifactInventoryRecord.parse(raw).render() == raw


def test_placement_follows_the_existing_role_function_and_the_record_id():
    p = str(record().canonical_path)
    assert p == "records/verification/environment-qualification/artifact-inventory/{}.json".format(R1)
    assert role_for_path(p) is ArtifactRole.VERIFICATION_RESULT


def test_order_of_construction_does_not_matter():
    r = record()
    shuffled = record(requirements=tuple(reversed(r.requirements)), locations=tuple(reversed(r.locations)),
                      searches=tuple(reversed(r.searches)), plan_documents=tuple(reversed(r.plan_documents)))
    assert shuffled == r and shuffled.render() == r.render()
    assert req(purposes=(P.BUILD_SOURCE, P.ACQUIRED)) == req(purposes=(P.ACQUIRED, P.BUILD_SOURCE))


def test_states_are_the_admission_layers_vocabulary():
    assert {s.value for s in EvidenceState} == {"match", "mismatch", "unavailable", "invalid_evidence"}
    assert all(type(x.state) is EvidenceState for x in record().results())


# ------------------------------------------------------------------------------------------------------------------ derived semantics
def test_results_are_derived_from_locations_with_the_hint_condition_reported_separately():
    res = results(record())
    a, b = res["acq:A"], res["replay:B"]
    assert (a.state, a.reason, a.hint_condition, a.matching_locations) == (EvidenceState.MATCH, "matched", "matched",
                                                                            ((S, "accepted/A.zip"), (S, "archive/A.zip")))
    assert (b.state, b.reason, b.hint_condition, b.search_complete) == (EvidenceState.UNAVAILABLE, "not_found", "absent", True)


def test_a_damaged_hint_with_a_valid_copy_elsewhere_is_satisfied_and_the_damage_is_structured():
    r = record(locations=(loc(sha=DX), loc("archive/A.zip", file_id="f" * 64), loc("accepted/B.zip", C.ABSENT)))
    a = results(r)["acq:A"]
    assert (a.state, a.hint_condition, a.matching_locations) == (EvidenceState.MATCH, "digest_mismatch", ((S, "archive/A.zip"),))


def test_a_zero_byte_archive_at_the_hint_is_a_recorded_mismatch():
    empty = hashlib.sha256(b"").hexdigest()
    r = record(locations=(loc(sha=empty, size=0), loc("accepted/B.zip", C.ABSENT)), searches=(search(read=1), search(DB, None, None, "every_file")))
    a = results(r)["acq:A"]
    assert (a.state, a.reason, a.hint_condition) == (EvidenceState.MISMATCH, "hint_holds_other_content", "digest_mismatch")


def test_an_incomplete_search_never_becomes_a_claim_of_absence():
    r = record(searches=(search(read=2), search(DB, None, None, "every_file", read=99, unreadable=1)))
    b = results(r)["replay:B"]
    assert (b.state, b.reason, b.search_complete) == (EvidenceState.UNAVAILABLE, "search_incomplete", False)
    assert r.counts()["incomplete_searches"] == 1


@pytest.mark.parametrize("cond, expected", [(C.NOT_REGULAR_FILE, "hint_not_regular_file"), (C.UNREADABLE, "hint_unreadable")])
def test_an_unreadable_or_irregular_hint_is_invalid_evidence(cond, expected):
    r = record(locations=(loc(cond=cond), loc("accepted/B.zip", C.ABSENT)), searches=(search(read=0, unreadable=1), search(DB, None, None, "every_file")))
    a = results(r)["acq:A"]
    assert (a.state, a.reason) == (EvidenceState.INVALID, expected)


# ------------------------------------------------------------------------------------------------------------------ counting properties
def test_counts_are_of_requirements_content_and_locations_never_redundancy():
    c = record().counts()
    assert (c["requirements"], c["expected_content_objects"], c["matched_content_objects"]) == (2, 2, 1)
    assert (c["verified_matching_locations"], c["additional_matching_locations"]) == (2, 1)
    assert (c["distinct_filesystem_files_among_matching_locations"], c["distinct_volume_identities_among_matching_locations"]) == (2, 1)
    assert "physical" not in json.dumps(c)
    assert c["requirement_states"] == {"match": 1, "mismatch": 0, "unavailable": 1, "invalid_evidence": 0}
    assert "replicas" not in json.dumps(c)


def test_PROPERTY_requirement_multiplicity_a_second_reference_to_identical_content_adds_no_content_or_location():
    base = record()
    more = record(requirements=base.requirements + (req("replay:A", purposes=(P.SELECTED_UPSTREAM_BINARY,), frm="replay-plan"),))
    a, b = base.counts(), more.counts()
    assert b["requirements"] == a["requirements"] + 1
    for k in ("expected_content_objects", "matched_content_objects", "verified_matching_locations", "additional_matching_locations"):
        assert a[k] == b[k], k


def test_a_hard_link_is_counted_as_one_filesystem_file():
    r = record(locations=(loc(), loc("archive/A.zip"), loc("accepted/B.zip", C.ABSENT)))    # same file identity: a hard link
    c = r.counts()
    assert (c["verified_matching_locations"], c["distinct_filesystem_files_among_matching_locations"]) == (2, 1)


def test_one_file_identifier_on_two_volumes_is_two_filesystem_files():
    """Identity is the PAIR (volume, file) within the store: an opaque file identifier need not embed its volume."""
    r = record(locations=(loc(), loc("archive/A.zip", vol="9" * 64), loc("accepted/B.zip", C.ABSENT)))
    c = r.counts()
    assert (c["distinct_filesystem_files_among_matching_locations"], c["distinct_volume_identities_among_matching_locations"]) == (2, 2)


def test_PROPERTY_one_filesystem_identity_cannot_hold_two_contents():
    """Ruling 2026-10-08f section 3: a hard link observed with different bytes is a consistency failure, not two files."""
    other = hashlib.sha256(b"other").hexdigest()
    build = lambda: record(requirements=record().requirements + (req("acq:O", sha=other, size=10, hint=None, package="O"),),   # noqa: E731
                           locations=(loc(), loc("archive/O.zip", sha=other), loc("accepted/B.zip", C.ABSENT)),
                           searches=record().searches + (search(other),))
    assert "inventory.file_identity_content_conflict" in reason(build)
    # the same content under one identity (a hard link) and other content under ANOTHER identity are both fine
    record(locations=(loc(), loc("archive/A.zip"), loc("accepted/B.zip", C.ABSENT)))


# ------------------------------------------------------------------------------------------------------------------ refusals
def _with(i, **kw):
    def build():
        locs = list(record().locations)
        o = locs[i]
        fields = dict(store=o.store, path=o.path, condition=o.condition, sha256=o.sha256, size_bytes=o.size_bytes,
                      file_identity=o.file_identity, volume_identity=o.volume_identity)
        fields.update(kw)
        locs[i] = ai.LocationObservation(**fields)
        return record(locations=tuple(locs))
    return build


@pytest.mark.parametrize("build, fragment", [
    (lambda: record(locations=record().locations + (loc(sha=DB),)), "observed twice"),                      # one location, two identities
    (lambda: record(requirements=record().requirements + (req("x:A", size=11),)), "digest_size_conflict"),
    (_with(2, size_bytes=11), "differ in size"),                                              # archive/A.zip (sorted index 2)
    (_with(0, size_bytes=11), "differ in size"),
    (lambda: record(locations=record().locations + (loc("stray/Z.zip", sha=DX, file_id="9" * 64),)), "neither a hint nor a holder"),
    (lambda: record(locations=(loc(), loc("archive/A.zip", file_id="f" * 64))), "was not observed"),      # B's hint never observed
    (lambda: record(searches=record().searches[:1]), "coverage differs"),
    (lambda: record(searches=record().searches + (search(DX),)), "coverage differs"),
    (lambda: record(searches=(search(size=11), record().searches[1])), "must use, and attribute"),
    (lambda: record(searches=(search(frm=None), record().searches[1])), "must use, and attribute"),
    (lambda: record(searches=(record().searches[0], search(DB, 5, None, "every_file"))), "must be the found content's size"),
    (lambda: ai.ContentSearch(DA, 10, "acquisition", "every_file", 1, 0), "bound size implies"),
    (lambda: ai.ContentSearch(DA, None, None, "same_size_files", 1, 0), "the scope must be every_file"),
    (lambda: ai.ContentSearch(DA, None, "acquisition", "every_file", 1, 0), "names a document but no size"),
    (lambda: ai.ContentSearch(DA, 10, None, "same_size_files", -1, 0), "non-negative"),
    (lambda: loc(size=-1), "observed size >= 0"),
    (lambda: loc(size=True), "observed size >= 0"),
    (lambda: ai.LocationObservation(S, "a/b", C.ABSENT, DA, None, None, None), "records no content"),
    (lambda: ai.LocationObservation(S, "a/b", "read", DA, 1, DF, DV), "must be a LocationCondition"),
    (lambda: req(size=0), "expected size > 0"),
    (lambda: req(size=1.0), "expected size > 0"),
    (lambda: req(purposes=()), "purposes"),
    (lambda: req(purposes=("acquired",)), "purposes"),
    (lambda: req(kind="zip"), "kind"),
    (lambda: req(sha="A" * 64), "lowercase SHA-256"),
    (lambda: record(requirements=(req(), req())), "duplicate entry_id"),
    (lambda: record(requirements=(req(frm="nowhere"),) + record().requirements[1:]), "names no plan document"),
    (lambda: record(requirements=(req(hint=("other-store", "a/b")),) + record().requirements[1:]), "undeclared store"),
    (lambda: record(previous_record_id=RecordId(R1)), "cannot follow itself"),
    (lambda: record(measurement_completed_at="2026-10-08T19:29:59Z"), "completed before it started"),
    (lambda: record(measurement_started_at="2026-13-01T00:00:00Z"), "not a real UTC time"),
    (lambda: record(runtime_supplied=(ai.RuntimeSupplied("A", "1", "recommended", DX),)), "also has a content requirement"),
    (lambda: record(gaps=record().gaps * 2), "duplicate name"),
    (lambda: record(requirements=(), locations=(), searches=()), "must be non-empty"),
    (lambda: record(stores=("GVC Artifacts",)), "neutral store identifier"),
    (lambda: record(collection=ai.Collection(1, 2, 2, 1, "x", "y")), "more files hashed than indexed"),
    (lambda: record(collection=ai.Collection(9, 2, 1, 9, "x", "y")), "re-checked before finalization"),
    (lambda: record(collection=ai.Collection(100, 2, 2, 2, "x", "y")), "repeated census must compare every indexed file"),
    (lambda: record(collection=ai.Collection(100, 2, 2, 99, "x", "y")), "repeated census must compare every indexed file"),
    (lambda: record(collection=collection(hashed=1)), "fewer files hashed than read locations"),
])
def test_construction_refuses(build, fragment):
    assert fragment in reason(build)


@pytest.mark.parametrize("value", [
    "C:\\Users\\monzi\\GVC_artifacts\\x.zip", "C:/Users/monzi/x.zip", "found under /home/runner/x", "../outside", "~/x",
    "see https://example.org/signed?token=1", "Users/monzi/x", "accepted\\source\\x",
])
def test_nothing_undisclosable_enters_a_path_the_scope_or_a_gap(value):
    assert "not disclosable" in reason(lambda: record(scope=value))
    assert "not disclosable" in reason(lambda: ai.KnownGap("g", value))
    with pytest.raises(ai.ArtifactInventoryError):
        req(hint=(S, value))


@pytest.mark.parametrize("path", ["a/./b", "a//b", "/a", "a/", "."])
def test_locations_are_normalised_relative_posix_paths(path):
    with pytest.raises(ai.ArtifactInventoryError):
        loc(path)


# ------------------------------------------------------------------------------------------------------------------ strict parsing
def _doc():
    return json.loads(record().render())


def _render(doc):
    return (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


@pytest.mark.parametrize("mutate, fragment", [
    (lambda d: d.__setitem__("schema_version", True), "schema"),
    (lambda d: d.__setitem__("retention", "supersedable_snapshot"), "retention"),
    (lambda d: d.__setitem__("extra", 1), "undeclared key"),
    (lambda d: d.pop("gaps"), "missing"),
    (lambda d: d["counts"].__setitem__("requirements", 99), "round-trip"),
    (lambda d: d["results"][0].__setitem__("state", "unavailable"), "round-trip"),            # results are DERIVED, never authored
    (lambda d: d["searches"][0].__setitem__("complete", False), "round-trip"),
    (lambda d: d["locations"][0].__setitem__("condition", "gone"), "unrecognised condition"),
    (lambda d: d["requirements"][0].__setitem__("purposes", ["gift"]), "unrecognised purpose"),
    (lambda d: d["requirements"].reverse(), "round-trip"),
])
def test_parse_refuses(mutate, fragment):
    d = _doc()
    mutate(d)
    assert fragment in reason(lambda: ai.ArtifactInventoryRecord.parse(_render(d)))


@pytest.mark.parametrize("raw, fragment", [
    (b"\xef\xbb\xbf{}", "byte-order mark"), (b'{"a": 1, "a": 2}', "duplicate key"), (b'{"x": 1.5}', "non-integer number"),
    (b'{"x": NaN}', "non-integer number"), (b"", "empty"), (b"{", "invalid JSON"),
])
def test_parse_is_strict(raw, fragment):
    assert fragment in reason(lambda: ai.ArtifactInventoryRecord.parse(raw))


# ------------------------------------------------------------------------------------------------------------------ the derived index
def _chain():
    return record(), record(record_id=RecordId(R2), previous_record_id=RecordId(R1), measurement_started_at="2026-10-09T08:00:00Z",
                            measurement_completed_at="2026-10-09T08:05:00Z")


def test_the_index_is_a_pure_projection_naming_the_current_record():
    first, second = _chain()
    assert ai.render_index([second, first]) == ai.render_index([first, second])
    body = json.loads(ai.render_index([first, second]))
    assert body["current"] == R2 and body["derived"] is True and [r["record_id"] for r in body["records"]] == [R1, R2]
    assert json.loads(ai.render_index([]))["current"] is None


@pytest.mark.parametrize("records, fragment", [
    (lambda f, s: [s], "predecessor is missing"),
    (lambda f, s: [f, s, record(record_id=RecordId(R3), previous_record_id=RecordId(R1))], "fork"),
    (lambda f, s: [f, record(record_id=RecordId(R2))], "exactly one first record"),
    (lambda f, s: [f, f], "duplicate record id"),
])
def test_the_index_refuses_an_ambiguous_chain(records, fragment):
    first, second = _chain()
    assert fragment in reason(lambda: ai.render_index(records(first, second)))


def _repo_with(records, names=None):
    root = Path(tempfile.mkdtemp())
    family = root.joinpath(*ai.family_root().parts)
    family.mkdir(parents=True)
    for i, r in enumerate(records):
        (family / (names or {}).get(i, r.record_id.value + ".json")).write_bytes(r.render())
    return root, family


def test_scan_reads_every_record_and_the_index_equals_its_projection():
    first, second = _chain()
    root, family = _repo_with([first, second])
    (family / "index.json").write_bytes(ai.render_index(ai.scan_records(root)))
    assert ai.scan_records(root) == (first, second)


def test_scan_refuses_a_record_filed_under_another_name_and_stray_files():
    root, _ = _repo_with([record()], names={0: R2 + ".json"})
    assert "differs from its record id" in reason(lambda: ai.scan_records(root))
    root, family = _repo_with([record()])
    (family / "notes.txt").write_text("x")
    assert "not a record file" in reason(lambda: ai.scan_records(root))


def test_the_committed_family_if_any_is_valid_and_its_index_is_the_projection():
    """NON-VACUOUS once records exist; until the first record is committed the directory must not exist (ADR-0004)."""
    family = ROOT.joinpath(*ai.family_root().parts)
    if not family.exists():
        return
    records = ai.scan_records(ROOT)
    assert records, "the family directory exists but holds no record"
    assert (family / "index.json").read_bytes() == ai.render_index(records)


# ------------------------------------------------------------------------------------------------------------------ local binding
def _bindings(tmp: Path, stores) -> Path:
    p = tmp / "artifact_stores.json"
    p.write_text(json.dumps({"schema": ai.BINDINGS_SCHEMA, "schema_version": 1, "stores": stores}))
    return p


def test_bindings_resolve_inside_the_store():
    tmp = Path(tempfile.mkdtemp()).resolve()
    (tmp / "store" / "accepted").mkdir(parents=True)
    b = ai.load_store_bindings(_bindings(tmp, {S: str(tmp / "store")}))
    assert ai.resolve_location(b, S, "accepted/x.zip") == (tmp / "store").resolve() / "accepted" / "x.zip"
    assert "no local binding" in reason(lambda: ai.resolve_location(b, "other-store", "a"))


def test_a_junction_in_the_path_is_refused_before_anything_is_read(monkeypatch):
    """Junctions cannot be created on this platform, so the predicate is SIMULATED for one directory; the Windows test below makes
    a real one. Either way the refusal names the junction and nothing beneath it is read."""
    tmp = Path(tempfile.mkdtemp()).resolve()
    (tmp / "store" / "jn").mkdir(parents=True)
    b = ai.load_store_bindings(_bindings(tmp, {S: str(tmp / "store")}))
    target = (tmp / "store" / "jn").resolve()
    original = Path.is_junction if hasattr(Path, "is_junction") else None
    monkeypatch.setattr(Path, "is_junction", lambda self: self.resolve() == target or (original(self) if original else False), raising=False)
    assert "inventory.location_component_redirected" in reason(lambda: ai.resolve_location(b, S, "jn/x.zip"))
    assert ai.is_link_or_junction(tmp / "store" / "jn") and ai.is_redirected(tmp / "store" / "jn")


@pytest.mark.skipif(os.name != "nt", reason="directory junctions exist only on Windows")
def test_a_real_windows_junction_is_refused():
    import subprocess
    tmp = Path(tempfile.mkdtemp()).resolve()
    (tmp / "store").mkdir()
    (tmp / "outside").mkdir()
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(tmp / "store" / "jn"), str(tmp / "outside")], capture_output=True, text=True)
    assert made.returncode == 0, made.stdout + made.stderr
    b = ai.load_store_bindings(_bindings(tmp, {S: str(tmp / "store")}))
    assert ai.is_link_or_junction(tmp / "store" / "jn")
    assert "inventory.location_component_redirected" in reason(lambda: ai.resolve_location(b, S, "jn/x.zip"))


def test_a_symbolic_link_in_the_path_is_refused():
    tmp = Path(tempfile.mkdtemp()).resolve()
    (tmp / "store").mkdir()
    (tmp / "outside").mkdir()
    try:
        os.symlink(tmp / "outside", tmp / "store" / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links need a privilege this account lacks (the junction tests cover the boundary)")
    b = ai.load_store_bindings(_bindings(tmp, {S: str(tmp / "store")}))
    assert "inventory.location_component_redirected" in reason(lambda: ai.resolve_location(b, S, "link/x.zip"))


# --------------------------------------------------------------------------- the root chain (ruling 2026-10-08f section 2)
def _symlink_or_skip(target: Path, link: Path):
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links need a privilege this account lacks (the junction tests cover the boundary)")


def test_a_redirected_ANCESTOR_of_the_bound_root_is_refused_before_resolution(monkeypatch):
    """The owner's counterexample: bound root parent-alias/store, parent-alias a link to another directory, store ordinary.
    Previously resolve_location ACCEPTED it and returned the resolved target. Refused now, and nothing is resolved."""
    tmp = Path(tempfile.mkdtemp()).resolve()
    (tmp / "real" / "store").mkdir(parents=True)
    _symlink_or_skip(tmp / "real", tmp / "alias")
    b = {S: tmp / "alias" / "store"}
    resolved = []
    real_resolve = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: resolved.append(self) or real_resolve(self, strict=strict))
    assert "inventory.root_component_redirected" in reason(lambda: ai.resolve_location(b, S, "x.zip"))
    assert "inventory.root_component_redirected" in reason(lambda: ai.checked_store_root(tmp / "alias" / "store"))
    assert resolved == []                                      # refused BEFORE any resolution
    assert ai.checked_store_root(tmp / "real" / "store") == tmp / "real" / "store"


def test_a_reparse_point_ancestor_is_refused_by_its_attribute_alone(monkeypatch):
    """The Windows branch SIMULATED (a stat result carrying FILE_ATTRIBUTE_REPARSE_POINT, e.g. a cloud placeholder, which neither
    is_symlink nor is_junction reports); the Windows-only tests below exercise it on a real junction."""
    tmp = Path(tempfile.mkdtemp()).resolve()
    (tmp / "cloud" / "store").mkdir(parents=True)
    real_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        observed = real_lstat(path, *args, **kwargs)
        if Path(path) == tmp / "cloud":
            # a faithful stat-like value: Windows' ntpath.isjunction reads st_reparse_tag (a cloud tag, not a mount point)
            return types.SimpleNamespace(st_mode=observed.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
                                         st_reparse_tag=0x9000601A)
        return observed
    monkeypatch.setattr(os, "lstat", lstat)
    assert not ai.is_link_or_junction(tmp / "cloud")
    assert ai.is_redirected(tmp / "cloud")
    assert "inventory.root_component_redirected" in reason(lambda: ai.checked_store_root(tmp / "cloud" / "store"))
    (tmp / "plain" / "cloud").mkdir(parents=True)
    monkeypatch.setattr(os, "lstat", lambda path, *a, **k: lstat(tmp / "cloud" if Path(path) == tmp / "plain" / "cloud" else path, *a, **k))
    assert "inventory.location_component_redirected" in reason(lambda: ai.resolve_location({S: tmp / "plain"}, S, "cloud/x.zip"))


@pytest.mark.parametrize("attributes, mode, expected", [
    (stat.FILE_ATTRIBUTE_REPARSE_POINT, stat.S_IFDIR, True), (stat.FILE_ATTRIBUTE_REPARSE_POINT | stat.FILE_ATTRIBUTE_DIRECTORY, stat.S_IFDIR, True),
    (stat.FILE_ATTRIBUTE_DIRECTORY, stat.S_IFDIR, False), (0, stat.S_IFLNK, True), (0, stat.S_IFREG, False), (None, stat.S_IFDIR, False),
])
def test_the_redirection_predicate(attributes, mode, expected):
    observed = types.SimpleNamespace(st_mode=mode) if attributes is None else types.SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
    assert ai._redirected(observed) is expected


def test_the_root_must_be_a_plain_canonical_absolute_directory(monkeypatch):
    tmp = Path(tempfile.mkdtemp()).resolve()
    (tmp / "store").mkdir()
    (tmp / "file").write_bytes(b"x")
    assert "inventory.root_not_plain_absolute_path" in reason(lambda: ai.checked_store_root(Path("relative/store")))
    assert "inventory.root_not_plain_absolute_path" in reason(lambda: ai.checked_store_root(tmp / "store" / ".." / "store"))
    assert "inventory.root_component_unreadable" in reason(lambda: ai.checked_store_root(tmp / "missing" / "store"))
    assert "inventory.root_component_not_directory" in reason(lambda: ai.checked_store_root(tmp / "file"))
    real_resolve = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: tmp / "elsewhere" if self == tmp / "store" else real_resolve(self, strict=strict))
    assert "inventory.root_not_canonical" in reason(lambda: ai.checked_store_root(tmp / "store"))     # e.g. an 8.3 or substituted alias


def test_an_uninspectable_component_is_refused_never_read_as_plain(monkeypatch):
    tmp = Path(tempfile.mkdtemp()).resolve()
    (tmp / "store").mkdir()
    real_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        if Path(path).name == "locked":
            raise PermissionError("denied")
        return real_lstat(path, *args, **kwargs)
    monkeypatch.setattr(os, "lstat", lstat)
    assert "inventory.component_unreadable" in reason(lambda: ai.resolve_location({S: tmp / "store"}, S, "locked/x.zip"))
    assert ai.is_redirected(tmp / "store" / "absent") is False                 # nothing there: absence is observed separately


def _windows_junction(link: Path, target: Path):
    import subprocess
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
    assert made.returncode == 0, made.stdout + made.stderr


@pytest.mark.skipif(os.name != "nt", reason="directory junctions exist only on Windows")
def test_a_real_windows_junction_ANCESTOR_is_refused_by_the_reparse_attribute(monkeypatch):
    """REAL Windows: a junction above the bound root carries FILE_ATTRIBUTE_REPARSE_POINT; refused even when the junction-specific
    predicate is disabled, so the attribute branch itself is exercised (ruling 2026-10-08f: the Windows branch needs a Windows test)."""
    tmp = Path(tempfile.mkdtemp()).resolve()
    (tmp / "real" / "store").mkdir(parents=True)
    _windows_junction(tmp / "alias", tmp / "real")
    assert os.lstat(tmp / "alias").st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    assert "inventory.root_component_redirected" in reason(lambda: ai.resolve_location({S: tmp / "alias" / "store"}, S, "x.zip"))
    monkeypatch.setattr(ai, "is_link_or_junction", lambda path: False)
    assert "inventory.root_component_redirected" in reason(lambda: ai.checked_store_root(tmp / "alias" / "store"))
    (tmp / "plain").mkdir()
    _windows_junction(tmp / "plain" / "jn", tmp / "real")
    assert "inventory.location_component_redirected" in reason(lambda: ai.resolve_location({S: tmp / "plain"}, S, "jn/x.zip"))


@pytest.mark.parametrize("stores, fragment", [
    ({S: "relative/root"}, "an absolute root"), ({}, "non-empty object"), ({"Bad Id": "/abs"}, "neutral store identifier"),
])
def test_bindings_refuse(stores, fragment):
    assert fragment in reason(lambda: ai.load_store_bindings(_bindings(Path(tempfile.mkdtemp()), stores)))


def test_the_binding_file_lives_outside_every_checkout():
    root = Path(tempfile.mkdtemp())
    for sentinel in PROJECT_SENTINELS:                       # the runtime-paths module's OWN sentinel list, not a copy
        target = root / sentinel
        target.parent.mkdir(parents=True, exist_ok=True)
        if sentinel.endswith(".toml"):
            target.write_text('[project]\nname = "{}"\nversion = "0.1.0"\n'.format(PROJECT_NAME), encoding="utf-8")
        else:
            target.mkdir(exist_ok=True)
    cache = Path(tempfile.mkdtemp())
    paths = resolve_runtime_paths(project_root=root, cache_root=cache, environ={})
    assert paths.artifact_store_bindings == cache.resolve() / "artifact_stores.json"
    assert root.resolve() not in paths.artifact_store_bindings.parents
    assert paths.describe()["artifact_store_bindings"] == str(paths.artifact_store_bindings)
    assert root.resolve() not in resolve_runtime_paths(project_root=root, environ={}).artifact_store_bindings.parents


# ------------------------------------------------------------------------------------------------------------------ readiness (admission)
REQUIRED = (("acq:A", DA),)
TREE = "1" * 40


def _lf(module) -> str:
    """Independent of the code under test: the canonical LF text digest of a module's file."""
    return hashlib.sha256(Path(module.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def impl(**over):
    modules = [{"path": "src/genomic_variant_classifier/environment_qualification/admission.py", "sha256": _lf(admission)},
               {"path": "src/genomic_variant_classifier/repository_records/artifact_inventory.py", "sha256": _lf(ai)}]
    fields = {"repository_tree": TREE, "collector_sha256": DX, "loaded_repository_modules": modules}      # record().verifier_sha256 == DX
    fields.update(over)
    return fields


def decide(r=None, required=REQUIRED, **kw):
    r = r or record()
    args = dict(plan_documents={"replay-plan": DX}, evaluated_at="2026-10-08T19:32:00Z", implementation=impl())
    args.update(kw)
    return readiness_decision(r, r.render(), required, **args)


def test_readiness_is_judged_against_the_plan_and_bound():
    r = record()
    out = artifact_readiness(r, REQUIRED)
    assert out == {"artifact_inputs_ready": True, "requirements": [{"entry_id": "acq:A", "ready": True, "reason": "matched"}]}
    d = decide(r)
    assert d["artifact_inputs_ready"] is True and d["record_sha256"] == hashlib.sha256(r.render()).hexdigest()
    assert d["record_id"] == R1 and d["plan_documents"] == {"replay-plan": DX}
    assert "policy_code_sha256" not in d                    # superseded by the implementation block (ruling 2026-10-08f section 4)


def test_the_decision_names_every_piece_of_code_that_determined_it_with_declared_domains():
    i = decide()["implementation"]
    assert (i["repository_tree"], i["collector_sha256"]) == (TREE, DX)
    assert (i["record_owner_sha256"], i["admission_sha256"]) == (_lf(ai), _lf(admission))
    assert i["digest_domains"] == {"repository_tree": "git_tree_object_id", "collector_sha256": "exact_bytes_sha256",
                                   "record_owner_sha256": "canonical_lf_text_sha256", "admission_sha256": "canonical_lf_text_sha256",
                                   "loaded_repository_modules": "canonical_lf_text_sha256"}
    assert [m["path"] for m in i["loaded_repository_modules"]] == sorted(m["path"] for m in impl()["loaded_repository_modules"])


def test_the_decision_is_historical_readiness_never_an_authorization_of_later_use():
    claim = decide()["claim"]
    assert claim.startswith("HISTORICAL readiness") and "installation admission must recheck the bytes it is about to consume" in claim


def test_canonical_text_digests_agree_across_line_endings_and_refuse_a_lone_cr():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "lf.py").write_bytes(b"a\nb\n")
    (tmp / "crlf.py").write_bytes(b"a\r\nb\r\n")
    (tmp / "cr.py").write_bytes(b"a\rb\n")
    assert admission.canonical_text_sha256(tmp / "lf.py") == admission.canonical_text_sha256(tmp / "crlf.py") == hashlib.sha256(b"a\nb\n").hexdigest()
    with pytest.raises(AdmissionError):
        admission.canonical_text_sha256(tmp / "cr.py")


def _wrong_module_digest():
    modules = impl()["loaded_repository_modules"]
    return [modules[0], dict(modules[1], sha256="0" * 64)]


@pytest.mark.parametrize("implementation, code", [
    (impl(collector_sha256="0" * 64), "readiness.collector_not_the_records_verifier"),
    (impl(collector_sha256="xyz"), "readiness.collector_digest_invalid"),
    (impl(repository_tree="HEAD"), "readiness.repository_tree_invalid"),
    (impl(loaded_repository_modules=impl()["loaded_repository_modules"][:1]), "readiness.interpreting_module_not_verified"),
    (impl(loaded_repository_modules=_wrong_module_digest()), "readiness.interpreting_module_not_verified"),
    (impl(loaded_repository_modules=list(reversed(impl()["loaded_repository_modules"]))), "readiness.loaded_modules_invalid"),
    (impl(loaded_repository_modules=impl()["loaded_repository_modules"] * 2), "readiness.loaded_modules_invalid"),
    (impl(loaded_repository_modules=[]), "readiness.loaded_modules_invalid"),
    (impl(loaded_repository_modules=[{"path": "C:\\x.py", "sha256": DA}]), "readiness.loaded_modules_invalid"),
    (dict(impl(), extra=1), "readiness.implementation_shape"),
    ({k: v for k, v in impl().items() if k != "repository_tree"}, "readiness.implementation_shape"),
])
def test_the_implementation_binding_refuses(implementation, code):
    with pytest.raises(AdmissionError) as exc:
        decide(implementation=implementation)
    assert code in str(exc.value)


def test_only_a_typed_inventory_record_is_decided():
    class Lookalike:
        render = staticmethod(lambda: b"x")
    with pytest.raises(AdmissionError) as exc:
        readiness_decision(Lookalike(), b"x", REQUIRED, plan_documents={"p": DX}, evaluated_at="2026-10-08T19:32:00Z", implementation=impl())
    assert "readiness.record_type" in str(exc.value)


def test_PROPERTY_historical_honesty_a_missing_unrequired_bundle_leaves_readiness_unchanged():
    r = record()                                             # replay:B is unavailable, but the plan here requires only acq:A
    assert artifact_readiness(r, REQUIRED)["artifact_inputs_ready"] is True
    assert results(r)["replay:B"].state is EvidenceState.UNAVAILABLE


@pytest.mark.parametrize("required, expected", [
    ((("replay:B", DB),), "content_not_matched:not_found"),
    ((("acq:A", DB),), "expected_digest_not_plan_digest"),                     # PROPERTY: requirement independence
    ((("acq:Z", DA),), "requirement_not_recorded"),
])
def test_readiness_refuses_an_unmatched_misdeclared_or_missing_input(required, expected):
    out = artifact_readiness(record(), required)
    assert out["artifact_inputs_ready"] is False and out["requirements"][0]["reason"] == expected


@pytest.mark.parametrize("required, code", [
    ((), "readiness.empty_or_untyped_requirements"),                           # PROPERTY: non-vacuity
    ([("acq:A", DA)], "readiness.empty_or_untyped_requirements"),
    ((("acq:A", DA), ("acq:A", DA)), "readiness.duplicate_requirement"),
    ((("acq:A", "ab"),), "readiness.invalid_requirement"),
])
def test_readiness_refuses_malformed_requirements(required, code):
    with pytest.raises(AdmissionError) as exc:
        artifact_readiness(record(), required)
    assert code in str(exc.value)


def test_a_decision_cannot_be_bound_to_other_bytes():
    r = record()
    with pytest.raises(AdmissionError) as exc:
        readiness_decision(r, r.render() + b" ", REQUIRED, plan_documents={"replay-plan": DX}, evaluated_at="2026-10-08T19:32:00Z",
                           implementation=impl())
    assert "readiness.record_bytes_not_this_record" in str(exc.value)
    with pytest.raises(AdmissionError):
        decide(r, plan_documents={})
