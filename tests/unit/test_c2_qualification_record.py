"""The C2 qualification record, verified OFFLINE from the checkout -- no Downloads, no cache, no GitHub (owner ruling 2026-10-02b).

Stage 1: inventory + exact bytes against the typed manifest, with the required cases from the CONTRACT. Stage 2: the semantic
replay -- archived receipts bound to each round's RECORDED checker identity (independent of the receipt and stable across future
checker changes), subjects from GitHub's own run records, deliveries reconciled, journals classified by the production reader.
Plumbing: no record file may be silently ignored or normalised -- both failures occurred in this project.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pytest

from genomic_variant_classifier.repository_records import qualification_manifest as q
from genomic_variant_classifier.repository_records.roles import RecordsOntologyError
from genomic_variant_classifier.source_monitor import c2_github as gh
from genomic_variant_classifier.source_monitor import c2_protocol as c2
from genomic_variant_classifier.source_monitor import deployment as dep

ROOT = Path(__file__).resolve().parents[2]
_FOUND = sorted((ROOT / "records" / "verification" / q.SUBDIRECTORY).glob("REC-*/manifest.json"))


@pytest.fixture(scope="module")
def manifest_path():
    assert len(_FOUND) == 1, "expected exactly one C2 qualification record, found {}".format([p.parent.name for p in _FOUND])
    return _FOUND[0]


@pytest.fixture(scope="module")
def manifest(manifest_path):
    return q.QualificationManifest.parse(manifest_path.read_bytes())


def _files(manifest, round_id, suffix=""):
    return [ROOT.joinpath(*PurePosixPath(f.identity.instance.canonical_path).parts) for f in manifest.files
            if f.round_id == round_id and f.identity.instance.canonical_path.endswith(suffix)]


# ---------------------------------------------------------------- stage 1 and plumbing

def test_stage_one_every_file_is_present_and_exact(manifest):
    assert manifest.verify(ROOT, required_cases=q.REQUIRED_CASES) == len(manifest.files) >= 24
    assert {r.round_id for r in manifest.rounds} == {"round-1", "round-2"}
    assert manifest.deviation == q.DEVIATION_STATEMENT


def test_no_record_file_is_ignored_or_normalised(manifest, manifest_path):
    """Measured twice in this project: `*.log` silently kept evidence out of a commit. And a normalised artifact is no
    longer evidence of anything."""
    paths = [f.identity.instance.canonical_path for f in manifest.files]
    # NUL-separated BYTES (-z): MEASURED 2026-10-02, CRLF-terminated --stdin input (what text mode sends on Windows) makes git
    # report NOTHING ignored -- the carriage return joins the path -- so a text-mode check passes VACUOUSLY there. The sentinel
    # MUST be reported: a guard on the guard, so a non-discriminating instrument fails instead of passing on any platform.
    sentinel = "probe_not_evidence.log"
    out = subprocess.run(["git", "check-ignore", "--no-index", "-z", "--stdin"],
                         input=("\0".join(paths + [sentinel]) + "\0").encode("utf-8"), capture_output=True, cwd=ROOT)
    assert out.returncode in (0, 1), out.stderr.decode("utf-8", "replace")
    reported = [x for x in out.stdout.decode("utf-8").split("\0") if x]
    assert reported == [sentinel], "record files ignored by .gitignore (or the instrument did not discriminate): {}".format(reported)
    attrs = subprocess.run(["git", "check-attr", "text", "--"] + paths + [manifest_path.relative_to(ROOT).as_posix()],
                           capture_output=True, text=True, cwd=ROOT, check=True).stdout.splitlines()
    by_path = {line.split(": text: ")[0]: line.split(": text: ")[1] for line in attrs}
    assert all(by_path[p] == "unset" for p in paths), {p: v for p, v in by_path.items() if p in paths and v != "unset"}
    assert by_path[manifest_path.relative_to(ROOT).as_posix()] == "set"


# ---------------------------------------------------------------- negative controls

def _mutate(manifest_path, change):
    doc = json.loads(manifest_path.read_bytes())
    change(doc)
    return (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


@pytest.mark.parametrize("change", [
    lambda d: d["files"][0].__setitem__("size_bytes", float(d["files"][0]["size_bytes"])),
    lambda d: d["files"][0].__setitem__("size_bytes", True),
    lambda d: d["files"][0].__setitem__("extra", 1),
    lambda d: d.__setitem__("deviation", d["deviation"].replace("process deviation", "deviation")),
    lambda d: d.__setitem__("cases", [c for c in d["cases"] if not (c["round"] == "round-2" and c["name"] == "normal")]),
    lambda d: d["cases"][0].__setitem__("name", "made-up-case"),
    lambda d: d["files"][0].__setitem__("canonical_path", "records/audits/x.zip"),
    lambda d: d["rounds"][0].__setitem__("checker_identities", []),
    lambda d: d.__setitem__("schema_version", 2),
], ids=["float-size", "bool-size", "undeclared-key", "deviation-altered", "missing-case", "foreign-case", "path-outside-root",
        "no-identities", "unknown-version"])
def test_the_manifest_refuses_every_malformation(manifest_path, change):
    with pytest.raises(RecordsOntologyError):
        q.QualificationManifest.parse(_mutate(manifest_path, change))


def test_the_manifest_refuses_duplicate_keys_and_non_canonical_rendering(manifest_path):
    raw = manifest_path.read_bytes()
    with pytest.raises(RecordsOntologyError):
        q.QualificationManifest.parse(raw.replace(b'{\n  "as_of"', b'{\n  "as_of": "2000-01-01T00:00:00Z",\n  "as_of"', 1))
    with pytest.raises(RecordsOntologyError):
        q.QualificationManifest.parse(json.dumps(json.loads(raw), sort_keys=True).encode("ascii") + b"\n")


@pytest.mark.parametrize("cases", [frozenset(), frozenset({"normal"}), set(q.REQUIRED_CASES)])
def test_the_required_cases_come_from_the_contract_never_the_manifest(manifest, cases):
    with pytest.raises(RecordsOntologyError):
        manifest.verify(ROOT, required_cases=cases)


def test_stage_one_detects_a_tampered_byte_and_a_stray_file(manifest, tmp_path):
    rel = manifest.root.as_posix()
    shutil.copytree(ROOT / rel, tmp_path / rel)
    target = next(tmp_path.joinpath(*PurePosixPath(f.identity.instance.canonical_path).parts) for f in manifest.files)
    original = target.read_bytes()
    target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(RecordsOntologyError, match="digest differs"):
        manifest.verify(tmp_path, required_cases=q.REQUIRED_CASES)
    target.write_bytes(original)
    (target.parent / "stray.txt").write_bytes(b"x")
    with pytest.raises(RecordsOntologyError, match="inventory"):
        manifest.verify(tmp_path, required_cases=q.REQUIRED_CASES)


# ---------------------------------------------------------------- stage 2: the semantic replay

def _zip_member(blob, suffix):
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        hits = [n for n in zf.namelist() if n.endswith(suffix)]
        assert len(hits) == 1, (suffix, hits)
        return zf.read(hits[0])


def _members(package):
    with zipfile.ZipFile(package) as zf:
        return {n: zf.read(n) for n in zf.namelist()}


def test_stage_two_round_one_receipts_bind_to_the_recorded_identities(manifest):
    identities = set(next(r for r in manifest.rounds if r.round_id == "round-1").checker_identities)
    receipts = [p.read_bytes() for p in _files(manifest, "round-1", "_receipt.json")]
    for package in _files(manifest, "round-1", ".zip"):
        receipts += [blob for name, blob in _members(package).items() if name.endswith("/receipt.json")]
    assert len(receipts) == 7
    for raw in receipts:
        env = json.loads(raw)
        c2.validate_payload(env["payload"])
        assert c2.digest("gvc.receipt/v1", env["payload"]) == env["receipt_sha256"]
        assert (env["payload"]["checker"]["code_manifest_sha256"], env["payload"]["checker"]["policy_sha256"]) in identities


def test_stage_two_round_two_reconciles_every_exercise(manifest):
    rnd = next(r for r in manifest.rounds if r.round_id == "round-2")
    (code, policy), = rnd.checker_identities
    deployment = dep.parse(next(p for p in _files(manifest, "round-2", "qualification_deployment_enabled.json")).read_bytes())
    dest = c2.Destination(deployment.repository_id, deployment.issue_id, deployment.issue_number)
    keys, seen = {}, set()
    for package in _files(manifest, "round-2", ".zip"):
        m = _members(package)
        top = next(iter(m)).split("/")[0]
        ex = json.loads(m[top + "/exercise.json"].decode("utf-8-sig"))
        comments = [c for page in json.loads(m[top + "/issue1_comments.json"]) for c in page]
        verifier = {n: b for n, b in m.items() if n.startswith(top + "/verifier/")}
        jobs = lambda n: [j for page in json.loads(verifier["{}/verifier/attempt-{}/jobs.json".format(top, n)]) for j in page["jobs"]]
        publish_log = lambda n: verifier["{}/verifier/attempt-{}/job-{}.log".format(top, n, next(
            j["id"] for j in jobs(n) if j["name"] == gh.PUBLISH_JOB))].decode("utf-8")
        if ex["mode"] == "scenario":
            run = json.loads(m[top + "/source/run.json"])
            verdict = next(b for n, b in verifier.items() if n.endswith("-source-monitor-verdict-attempt-1.zip"))
            raw = _zip_member(verdict, "receipt.json")
            p = json.loads(raw)["payload"]
            subject = {"repository": deployment.repository, "repository_id": deployment.repository_id,
                       "workflow_id": deployment.source_workflow_id, "workflow_path": deployment.source_workflow_path,
                       "run_id": run["id"], "run_number": run["run_number"], "attempt": run["run_attempt"], "commit": run["head_sha"]}
            checker = {"commit": ex["qualification_head"], "code_manifest_sha256": code, "policy_sha256": policy}
            finished = datetime.strptime(p["evaluation"]["finished_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            c2.open_receipt(raw, c2.Bindings(subject, checker, int(ex["verifier_run"]), 1), finished)
            key = c2.delivery_id(p, dest)
            outcome = json.loads(_zip_member(next(b for n, b in verifier.items() if n.endswith("-c2-delivery-outcome-attempt-1.zip")),
                                             "outcome.json"))
            mine = [c for c in comments if "<!-- gvc:c2:v1:{} -->".format(key) in c["body"]]
            assert outcome["delivery_id"] == key and outcome["post_issued"] is True and len(mine) == 1
            assert mine[0]["body"] == c2.render_comment(p, dest) and mine[0]["user"]["id"] == deployment.trusted_author_id
            assert gh.read_journal(publish_log(1), key) is True
            keys[ex["scenario"]] = (key, ex["verifier_run"])
        seen.add(ex["mode"])
    assert set(keys) == {"normal", "claims-disagree", "acquisition-limit"} and seen == {"scenario", "rerun", "manual"}
    key, verifier_run = keys["normal"]
    for package in _files(manifest, "round-2", ".zip"):
        m = _members(package)
        top = next(iter(m)).split("/")[0]
        ex = json.loads(m[top + "/exercise.json"].decode("utf-8-sig"))
        if ex["mode"] == "scenario":
            continue
        verifier = {n: b for n, b in m.items() if n.startswith(top + "/verifier/")}
        attempt = 2 if ex["mode"] == "rerun" else 1
        jobs = [j for page in json.loads(verifier["{}/verifier/attempt-{}/jobs.json".format(top, attempt)]) for j in page["jobs"]]
        log = verifier["{}/verifier/attempt-{}/job-{}.log".format(top, attempt, next(j["id"] for j in jobs if j["name"] == gh.PUBLISH_JOB))]
        outcome = json.loads(_zip_member(next(b for n, b in verifier.items()
                                              if n.endswith("-c2-delivery-outcome-attempt-{}.zip".format(attempt))), "outcome.json"))
        assert outcome["post_issued"] is False and log.count(b"C2-ATTEMPT") == 0
        if ex["mode"] == "rerun":
            assert ex["verifier_run"] == verifier_run and outcome["reason"] == "matching_comment" and outcome["delivery_id"] == key
            assert gh.read_journal(log.decode("utf-8"), key) is False
        else:
            assert (outcome["action"], outcome["reason"], outcome["delivery_id"]) == ("preview", "manual_verification", "")


def test_the_padded_report_is_the_original_plus_spaces(manifest):
    package = next(p for p in _files(manifest, "round-2", ".zip") if "acquisition-limit" in p.name)
    m = _members(package)
    source = {n: b for n, b in m.items() if "/source/" in n}
    original = _zip_member(next(b for n, b in source.items() if n.endswith("-qualification-original-report.zip")), "report.json")
    selected_archive = next(b for n, b in source.items() if n.endswith("-source-monitor-report.zip"))
    selected = _zip_member(selected_archive, "report.json")
    assert selected.startswith(original) and set(selected[len(original):]) == {0x20} and len(selected_archive) > 1024 * 1024
    assert hashlib.sha256(selected).hexdigest() != hashlib.sha256(original).hexdigest()
