"""The source-monitor run verifier (change C1, 2026-09-27) -- the owner's acceptance list, case by case.

Every negative case starts from the REAL run #8 (GitHub's own run record, artifact listing and report archive,
and the run commit's exact blobs, all preserved under tests/fixtures/source_monitor_runs/) and changes ONE
thing. Cases that change the report repack the archive and update GitHub's digest and size consistently, so
each test isolates the failure it names instead of tripping the digest check first.

Acceptance (approval-control README, section C): wrong repository / workflow / branch / event, stale policy,
wrong run attempt, duplicate artifacts, digest mismatch, invalid archive members, malformed report, absent
target, orphan target, iterator inputs, stale evidence, producer/witness disagreement -- and a valid exit-1
result verifies SUCCESSFULLY with an unresolved review item.

Author: Monzia Moodie
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from genomic_variant_classifier.source_monitor import interpretation_contract as ic
from genomic_variant_classifier.source_monitor import report_verifier as rv
from genomic_variant_classifier.source_monitor.run_monitor import REQUIRED_TARGETS

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "source_monitor_runs"
RUN_ID = 36300779115
NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)


def _json(name):
    with open(FIX / name, encoding="utf-8") as fh:
        return json.load(fh)


RUN, ARTIFACTS = _json("run8_attempt1.json"), _json("run8_artifacts.json")   # the ATTEMPT record
ARCHIVE = (FIX / "run8_source-monitor-report.zip").read_bytes()
_BLOBS = _json("run8_commit_blobs.json")


def read_blob(commit, path):
    """The run commit's preserved blobs. A path the commit LACKS is the typed BlobAbsent (the epoch descriptor did not
    exist at 8e7d762, so run 8 is an epoch-1 run); an unknown COMMIT is a refusal, never absence."""
    if commit != _BLOBS["commit"]:
        raise KeyError("no blobs preserved for {}".format(commit))
    if path not in _BLOBS["blobs"]:
        raise ic.BlobAbsent("{}:{}".format(commit, path))
    return base64.b64decode(_BLOBS["blobs"][path]["base64"])


APPROVAL_PATH = "docs/approvals/APPROVAL_2026-09-24_gnomad-4.1.1.json"


def _bound(policy, commit, reader):
    """The interpretation of `commit` under `policy`, reconstructed from the reader's bytes (approval bytes included)."""
    return ic.reconstruct(policy, lambda path: reader(commit, path), approved_record_bytes=reader(commit, APPROVAL_PATH),
                          approved_release="4.1.1")


#: Run 8's OWN version-1 interpretation (an ADMITTED legacy commit) -- "today's policy" for tests not about policy change.
CURRENT_V1 = _bound(ic.select_policy(_BLOBS["commit"], read_blob), _BLOBS["commit"], read_blob)
_DEFAULT = object()


def _report():
    with zipfile.ZipFile(io.BytesIO(ARCHIVE)) as zf:
        return json.loads(zf.read("report.json"))


REPORT = _report()


def _repack(report=None, raw=None, members=None, arts=None):
    """A consistent archive + artifact listing: GitHub's digest and size follow the new bytes. `arts` is the listing to
    start from (default: run 8's), e.g. one whose head_sha was rewritten for a synthetic commit."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in (members or [("report.json", raw if raw is not None else json.dumps(report).encode())]):
            zf.writestr(name, data)
    archive = buf.getvalue()
    arts = copy.deepcopy(ARTIFACTS if arts is None else arts)
    arts["artifacts"][0].update(digest="sha256:" + hashlib.sha256(archive).hexdigest(), size_in_bytes=len(archive))
    return archive, arts


def _verify(archive=ARCHIVE, run=RUN, artifacts=ARTIFACTS, run_id=RUN_ID, attempt=1, latest=1, now=NOW, current=_DEFAULT,
            reader=None):
    return rv.verify(archive, run, artifacts, run_id=run_id, run_attempt=attempt, latest_attempt=latest,
                     read_blob=reader or read_blob, now=now, current=CURRENT_V1 if current is _DEFAULT else current,
                     required_targets=REQUIRED_TARGETS)


def _problems(v, flag):
    return " | ".join(v.problems[flag])


def test_the_fixture_is_the_real_run_8_artifact():
    assert "sha256:" + hashlib.sha256(ARCHIVE).hexdigest() == ARTIFACTS["artifacts"][0]["digest"]
    assert RUN["run_attempt"] == 1 and _json("run8.json")["run_attempt"] == 1   # the run was never rerun
    assert RUN["id"] == RUN_ID and RUN["head_sha"] == _BLOBS["commit"]


def test_a_valid_exit_1_run_verifies_SUCCESSFULLY_with_an_unresolved_review_item():
    """Do not restore the earlier 'verification requires zero findings' mistake (ruling 2026-09-25)."""
    v = _verify()
    assert v.verified and all(v.flags.values()), v.as_document()["problems"]
    assert v.flags["review_required"] is True
    assert v.review_items == ['gnomad-public-releases: release prefix "release/4.1.2/" is newer than the approved 4.1.1']


# ------------------------------------------------------------------ execution
@pytest.mark.parametrize("path, value, fragment", [
    (("repository", "full_name"), "someone/fork", "repository is 'someone/fork'"),
    (("head_repository", "full_name"), "someone/fork", "head_repository is 'someone/fork'"),
    (("path",), ".github/workflows/other.yml", "run path"),
    (("head_branch",), "dev", "run head_branch"),
    (("event",), "pull_request", "run event 'pull_request'"),
    (("status",), "in_progress", "run status"),
], ids=["wrong-repository", "fork", "wrong-workflow", "wrong-branch", "wrong-event", "not-completed"])
def test_wrong_run_identity_is_refused(path, value, fragment):
    run = copy.deepcopy(RUN)
    node = run
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    v = _verify(run=run)
    assert not v.flags["execution_authenticated"] and not v.verified
    assert fragment in _problems(v, "execution_authenticated")


def test_the_wrong_run_id_is_refused():
    v = _verify(run_id=RUN_ID + 1)
    assert not v.flags["execution_authenticated"] and "run id" in _problems(v, "execution_authenticated")


def test_the_wrong_run_attempt_is_refused():
    v = _verify(attempt=2)                       # attempt 2 of a run whose latest attempt is 1
    assert "is not an attempt of this run" in _problems(v, "execution_authenticated")
    v = _verify(attempt=2, latest=2)             # a real attempt 2, but verified against attempt 1's record
    assert "the attempt record is for attempt 1, not 2" in _problems(v, "execution_authenticated")


def test_an_undeclared_attempt_on_a_rerun_run_is_ambiguous():
    """MEASURED 2026-09-27: an attempt record's run_attempt is that attempt's own number, so the ATTEMPT-1
    record of a later-rerun run still says 1. Ambiguity must come from the RUN record's latest attempt."""
    v = _verify(latest=2)
    assert "ambiguous" in _problems(v, "execution_authenticated")


def test_a_declared_identity_must_match_gitHubs_records():
    report = copy.deepcopy(REPORT)
    report["github_run"] = {"repository": rv.EXPECTED_REPOSITORY, "run_id": str(RUN_ID), "run_attempt": "2",
                            "sha": RUN["head_sha"], "workflow_ref": None}
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    assert "declares run_attempt '2'" in _problems(v, "execution_authenticated")
    report["github_run"]["run_attempt"] = "1"
    archive, arts = _repack(report)
    assert _verify(archive=archive, artifacts=arts).flags["execution_authenticated"] is True


def test_duplicate_artifacts_are_an_ambiguous_selection():
    arts = copy.deepcopy(ARTIFACTS)
    arts["artifacts"].append(copy.deepcopy(arts["artifacts"][0]))
    arts["total_count"] = 2
    v = _verify(artifacts=arts)
    # Refused BEFORE any archive is read: exactly the ambiguity, nothing replayed, no review item, every flag false.
    # (A mutation that picked the first artifact still ended "not verified" through check_execution's own check --
    # this test also pins that NO later layer ran on an ambiguous selection.)
    assert v.problems["execution_authenticated"] == [
        "expected exactly ONE 'source-monitor-report' artifact, found 2 -- ambiguous selection is refused"]
    assert v.review_items == [] and not any(v.flags.values())


@pytest.mark.parametrize("change, fragment", [
    ({"expired": True}, "expired"),
    ({"workflow_run": {"id": 1, "head_sha": RUN["head_sha"]}}, "belongs to run 1"),
    ({"created_at": "2026-09-27T07:40:26Z"}, "outside the attempt's window"),
])
def test_an_artifact_not_of_this_attempt_is_refused(change, fragment):
    arts = copy.deepcopy(ARTIFACTS)
    arts["artifacts"][0].update(change)
    assert fragment in _problems(_verify(artifacts=arts), "execution_authenticated")


def test_an_incomplete_artifact_listing_is_refused():
    arts = dict(copy.deepcopy(ARTIFACTS), total_count=2)
    assert "listing is incomplete" in _problems(_verify(artifacts=arts), "execution_authenticated")


def test_iterator_inputs_are_refused_not_consumed():
    arts = dict(copy.deepcopy(ARTIFACTS))
    arts["artifacts"] = (a for a in ARTIFACTS["artifacts"])
    v = _verify(artifacts=arts)
    assert not v.verified and "must be a dict and a list" in _problems(v, "execution_authenticated")


# ------------------------------------------------------------------ archive and report
def test_a_digest_mismatch_is_a_refusal_not_a_warning():
    tampered = bytearray(ARCHIVE)
    tampered[-30] ^= 0x01
    v = _verify(archive=bytes(tampered))
    assert not v.verified and "differs from GitHub's" in _problems(v, "execution_authenticated")


@pytest.mark.parametrize("members, fragment", [
    ([("report.json", b"{}"), ("extra.txt", b"x")], "exactly 'report.json'"),
    ([("../report.json", b"{}")], "exactly 'report.json'"),
    ([("reports/", b"")], "exactly 'report.json'"),
], ids=["extra-member", "path-traversal", "directory"])
def test_invalid_archive_members_are_refused(members, fragment):
    archive, arts = _repack(members=members)
    assert fragment in _problems(_verify(archive=archive, artifacts=arts), "execution_authenticated")


@pytest.mark.parametrize("raw, fragment", [
    (b"not json", "could not be admitted"),
    (b'{"schema": 1, "schema": 2}', "duplicate JSON key"),
    (b'{"x": NaN}', "non-JSON numeric constant"),
    (b"\xef\xbb\xbf{}", "byte-order mark"),
    (json.dumps(dict(REPORT, schema="other")).encode(), "unsupported report schema"),
    (json.dumps(dict(REPORT, exit_code=3)).encode(), "exit_code"),
], ids=["not-json", "duplicate-key", "nan", "bom", "wrong-schema", "bad-exit-code"])
def test_a_malformed_report_is_refused(raw, fragment):
    archive, arts = _repack(raw=raw)
    v = _verify(archive=archive, artifacts=arts)
    assert not v.verified and fragment in _problems(v, "execution_authenticated")


@pytest.mark.parametrize("mutate, fragment", [
    (lambda r: r.update(results=[]), "absent: ['gnomad-public-releases']"),
    (lambda r: r["results"].append(dict(r["results"][0], target="orphan-target")), "orphan: ['orphan-target']"),
    (lambda r: r["results"].append(copy.deepcopy(r["results"][0])), "duplicate targets"),
], ids=["absent-target", "orphan-target", "duplicate-target"])
def test_absent_orphan_and_duplicate_targets_are_refused(mutate, fragment):
    report = copy.deepcopy(REPORT)
    mutate(report)
    archive, arts = _repack(report)
    assert fragment in _problems(_verify(archive=archive, artifacts=arts), "execution_authenticated")


# ------------------------------------------------------------------ configuration
def test_a_part_changed_without_its_fingerprint_is_refused():
    report = copy.deepcopy(REPORT)
    report["interpretation"]["parts"]["adapter_code"] = "0" * 64
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    assert not v.flags["configuration_bound"]
    assert v.problems["configuration_bound"] == [
        "the report's interpretation does not bind to {}: reported parts differ from the independent reconstruction in "
        "['adapter_code']".format(RUN["head_sha"])]   # whole-document binding names the PART, not just a bad aggregate


@pytest.mark.parametrize("part", ["approval", "release_rules", "request_plan", "adapter_code", "verifier_code",
                                  "environment_lock"])
def test_a_SELF_CONSISTENT_part_substitution_is_refused(part):
    """DEFECT REPRODUCED 2026-09-28 (release_rules, request_plan): a part changed WITH a recomputed fingerprint was
    accepted, because only some parts were reconstructed. Now every part is rebuilt independently."""
    report = copy.deepcopy(REPORT)
    report["interpretation"]["parts"][part] = "e" * 64
    report["interpretation"]["fingerprint"] = ic.fingerprint(report["interpretation"]["parts"], version=1)
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    assert not v.flags["configuration_bound"] and not v.verified
    assert v.problems["configuration_bound"] == [
        "the report's interpretation does not bind to {}: reported parts differ from the independent reconstruction "
        "in ['{}']".format(RUN["head_sha"], part)]


def test_a_run_whose_commit_cannot_be_read_is_not_bound():
    run = dict(RUN, head_sha="0" * 40)
    v = _verify(run=run)
    assert not v.flags["configuration_bound"]


# ------------------------------------------------------------------ observation and claims
def test_a_producer_claim_without_a_witness_is_a_disagreement():
    report = copy.deepcopy(REPORT)
    report["results"][0]["findings"].append('release prefix "release/9.9.9/" is newer than the approved 4.1.1')
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    newer = rv.ReviewItem("gnomad-public-releases", "newer", "release/4.1.2/")
    claimed = sorted(Counter([newer, rv.ReviewItem("gnomad-public-releases", "newer", "release/9.9.9/")]).items())
    assert not v.flags["claims_reconciled"] and v.problems["claims_reconciled"] == [
        "the producer's review claims {} differ from the replay's {} as a multiset".format(claimed, [(newer, 1)])]


def test_a_qualification_that_the_replay_does_not_reproduce_is_refused():
    report = copy.deepcopy(REPORT)
    report["qualification"]["gnomad-public-releases"]["unsupported_names"] = ["release/5.0.0rc1/"]
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    assert not v.flags["claims_reconciled"] and v.problems["claims_reconciled"] == [
        "gnomad-public-releases: the report's qualification differs from the replay (value or JSON type) in "
        "['unsupported_names']"]


def test_missing_captures_make_the_observation_incomplete():
    report = copy.deepcopy(REPORT)
    report["results"][0]["captures"] = []
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    assert not v.flags["observation_complete"] and not v.verified


# ------------------------------------------------------------------ current obligation
def test_stale_evidence_is_valid_history_but_not_todays_obligation():
    now = NOW + timedelta(days=9)
    v = _verify(now=now)
    assert v.verified and not v.flags["current_monitoring_obligation_satisfied"]
    assert v.problems["current_monitoring_obligation_satisfied"] == [       # age from the ATTEMPT start (conservative)
        "the attempt started {} ago; the monitoring obligation needs one within {}".format(
            now - rv._time(RUN["run_started_at"]), rv.MAX_AGE)]


def test_a_stale_policy_is_valid_history_but_not_todays_obligation():
    current = ic.Bound(1, None, tuple(sorted(dict(CURRENT_V1.parts, verifier_code="1" * 64).items())))
    v = _verify(current=current)
    assert v.verified and not v.flags["current_monitoring_obligation_satisfied"]
    assert v.problems["current_monitoring_obligation_satisfied"] == [
        "this run's interpretation (version 1, policy None) is not today's (version 1, policy None); parts differ in "
        "['verifier_code']"]


def test_the_verdict_document_carries_the_six_flags_exactly():
    doc = _verify().as_document()
    assert list(doc["flags"]) == list(rv.FLAGS) == [
        "execution_authenticated", "configuration_bound", "observation_complete", "claims_reconciled",
        "review_required", "current_monitoring_obligation_satisfied"]


# ------------------------------------------------------------------ the command-line wrapper (offline)
def _cli():
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("verify_monitor_run", root / "scripts" / "verify_monitor_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _serve(archive=ARCHIVE):
    api = "https://api.github.com/repos/" + rv.EXPECTED_REPOSITORY
    pages = {api + "/actions/runs/{}".format(RUN_ID): (FIX / "run8.json").read_bytes(),
             api + "/actions/runs/{}/attempts/1".format(RUN_ID): (FIX / "run8_attempt1.json").read_bytes(),
             api + "/actions/runs/{}/artifacts?per_page=100".format(RUN_ID): (FIX / "run8_artifacts.json").read_bytes(),
             api + "/actions/artifacts/{}/zip".format(ARTIFACTS["artifacts"][0]["id"]): archive}
    return lambda url, limit: pages[url]


def test_the_cli_verifies_the_real_run_offline_and_exits_0(tmp_path):
    out = tmp_path / "verdict.json"
    code = _cli().main(["--run-id", str(RUN_ID), "--run-attempt", "1", "--verdict", str(out)], fetch=_serve(),
                       read_blob=read_blob, now=NOW, current=CURRENT_V1)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and doc["verified"] and all(doc["flags"].values())


def test_the_cli_exits_2_on_a_tampered_archive(tmp_path):
    tampered = bytearray(ARCHIVE)
    tampered[-30] ^= 0x01
    out = tmp_path / "verdict.json"
    code = _cli().main(["--run-id", str(RUN_ID), "--run-attempt", "1", "--verdict", str(out)], fetch=_serve(bytes(tampered)),
                       read_blob=read_blob, now=NOW, current=CURRENT_V1)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert code == 2 and not doc["verified"] and "differs from GitHub's" in " ".join(doc["problems"]["execution_authenticated"])


def test_the_token_is_attached_unredirected_and_only_over_https(monkeypatch):
    """MEASURED 2026-09-27: urllib forwards add_header headers to a redirected host (GitHub's artifact download
    redirects to a storage host) but not add_unredirected_header ones. The token must never leave api.github.com."""
    cli, seen = _cli(), {}

    class Resp(io.BytesIO):
        def geturl(self):
            return "https://storage.example/blob"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout):
        seen["carried_by_a_redirect"] = dict(req.headers)             # urllib copies ONLY these to a redirect
        seen["unredirected"] = dict(req.unredirected_hdrs)
        return Resp(b"ok")
    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    assert cli.http_fetch("https://api.github.com/x", "SECRET", 10) == b"ok"
    assert seen["unredirected"].get("Authorization") == "Bearer SECRET"
    assert not any(k.lower() == "authorization" for k in seen["carried_by_a_redirect"])
    with pytest.raises(ValueError, match="non-HTTPS"):
        cli.http_fetch("http://api.github.com/x", "SECRET", 10)
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda req, timeout: Resp(b"x" * 11))
    with pytest.raises(ValueError, match="exceeds"):
        cli.http_fetch("https://api.github.com/x", "SECRET", 10)


def test_an_absent_report_artifact_is_named_as_absence_not_ambiguity():
    """MEASURED 2026-09-27: preview run #2 against CI run 36320627013 said "found 0 -- ambiguous selection is refused"."""
    arts = dict(copy.deepcopy(ARTIFACTS), artifacts=[], total_count=0)
    v = _verify(artifacts=arts)
    assert v.problems["execution_authenticated"] == [
        "no 'source-monitor-report' artifact exists for this run -- the report is absent"]
    assert v.review_items == [] and not any(v.flags.values())


def test_the_cli_writes_its_summary_only_where_it_is_told(tmp_path):
    """The summary path is INJECTED; main() never reads $GITHUB_STEP_SUMMARY itself (CI run #896)."""
    out, summary = tmp_path / "verdict.json", tmp_path / "summary.md"
    _cli().main(["--run-id", str(RUN_ID), "--run-attempt", "1", "--verdict", str(out)], fetch=_serve(), read_blob=read_blob,
                now=NOW, current=CURRENT_V1, step_summary=str(summary))
    text = summary.read_text(encoding="utf-8")
    assert "Run 36300779115 attempt 1: **VERIFIED**" in text
    assert all("| `{}` | True |".format(flag) in text for flag in rv.FLAGS)


# ------------------------------------------------------------------ reproduced defects (owner's probes, 2026-09-28)
# Each case below was REPRODUCED against main 6f37d9b three ways (the owner's run, an independent reproduction, the
# owner's probe script re-run here) and every reason is PINNED: a wrong reason is a defect, not a pass.
@pytest.mark.parametrize("code, expected", [(0, 1), (2, 1)])
def test_an_inconsistent_exit_code_never_suppresses_review(code, expected):
    report = copy.deepcopy(REPORT)
    report["exit_code"] = code
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    assert v.flags["review_required"] is True                      # derived from the replay, never the exit code
    assert v.review_items == ['gnomad-public-releases: release prefix "release/4.1.2/" is newer than the approved 4.1.1']
    assert not v.flags["claims_reconciled"] and not v.verified
    assert not v.flags["current_monitoring_obligation_satisfied"]
    assert "the report's exit code {} conflicts with the trusted replay, which requires {}".format(code, expected) in \
        v.problems["claims_reconciled"]


def test_a_boolean_report_schema_version_is_refused():
    archive, arts = _repack(raw=json.dumps(dict(REPORT, schema_version=True)).encode())
    v = _verify(archive=archive, artifacts=arts)
    assert v.problems["execution_authenticated"] == [
        "the report could not be admitted: ValueError: unsupported report schema 'gvc.monitor-run-report' vTrue"]


def test_an_empty_current_reconstruction_cannot_satisfy_today():
    v = _verify(current=ic.Bound(1, None, ()))
    assert v.verified and not v.flags["current_monitoring_obligation_satisfied"]
    assert v.problems["current_monitoring_obligation_satisfied"] == [
        "this run's interpretation (version 1, policy None) is not today's (version 1, policy None); parts differ in "
        "{}".format(sorted(CURRENT_V1.parts))]


def test_a_missing_current_reconstruction_cannot_satisfy_today():
    v = _verify(current=None)
    assert v.verified and v.problems["current_monitoring_obligation_satisfied"] == [
        "the run's and today's interpretations must both be reconstructed to compare them"]


def test_a_verifier_clock_before_the_run_cannot_satisfy_today():
    now = NOW - timedelta(days=30)
    v = _verify(now=now)
    assert v.verified and not v.flags["current_monitoring_obligation_satisfied"]
    assert v.problems["current_monitoring_obligation_satisfied"] == [
        "the verifier clock {} precedes the attempt's end {} beyond the {} allowance".format(
            now, rv._time(RUN["updated_at"]), rv.SKEW)]


def test_a_naive_verifier_clock_is_refused():
    v = _verify(now=NOW.replace(tzinfo=None))
    assert v.problems["current_monitoring_obligation_satisfied"] == [
        "freshness could not be established: the verification time must be timezone-aware"]


# ------------------------------------------------------------------ interpretation policies (version 2, review revision 3)
_ROOT = Path(__file__).resolve().parents[2]
_POLICY_BYTES = (_ROOT / "configs/source_monitor_interpretation.json").read_bytes()
#: A SYNTHETIC non-legacy commit. 8e7d762 (run 8) is an admitted LEGACY commit, where a policy file is refused as a
#: contradiction, so every version-2 case runs on this commit with the attempt and artifact records rewritten consistently.
V2C = "5" * 40


def _v2_reader(extra=None):
    """V2C's tree: the run-8 blobs plus the committed policy file and orchestrator (this checkout); `extra` overrides or
    removes (None) paths, or supplies an Exception to raise."""
    files = {"configs/source_monitor_interpretation.json": _POLICY_BYTES,
             "src/genomic_variant_classifier/source_monitor/run_monitor.py":
                 (_ROOT / "src/genomic_variant_classifier/source_monitor/run_monitor.py").read_bytes()}
    files.update(extra or {})

    def read(commit, path):
        if commit != V2C:
            return read_blob(commit, path)
        if path in files:
            if files[path] is None:
                raise ic.BlobAbsent("{}:{}".format(commit, path))
            if isinstance(files[path], Exception):
                raise files[path]
            return files[path]
        return read_blob(_BLOBS["commit"], path)
    return read


def _v2_records():
    run = dict(copy.deepcopy(RUN), head_sha=V2C)
    arts = copy.deepcopy(ARTIFACTS)
    arts["artifacts"][0]["workflow_run"]["head_sha"] = V2C
    return run, arts


def _v2_verify(report, reader, current=_DEFAULT):
    run, arts = _v2_records()
    archive, arts = _repack(report, arts=arts)
    return _verify(archive=archive, run=run, artifacts=arts, reader=reader, current=current)


V2_BOUND = _bound(ic.parse_policy(_POLICY_BYTES), V2C, _v2_reader())


def test_a_version_2_commit_REFUSES_a_valid_version_1_report():
    """Downgrade protection: a perfectly self-consistent version-1 report cannot pass under a version-2 policy."""
    v = _v2_verify(REPORT, _v2_reader())
    assert not v.flags["configuration_bound"] and not v.verified
    assert v.problems["configuration_bound"] == [
        "the report's interpretation does not bind to {}: the interpretation fields ['fingerprint', 'parts'] differ from "
        "the version-2 contract ['contract_sha256', 'fingerprint', 'parts', 'schema_version']".format(V2C)]


def test_a_version_2_report_binds_under_its_policy_and_is_current_only_against_version_2():
    report = dict(copy.deepcopy(REPORT), interpretation=V2_BOUND.as_document())
    v = _v2_verify(report, _v2_reader())
    assert v.flags["configuration_bound"] and v.verified
    assert v.problems["current_monitoring_obligation_satisfied"] == [
        "this run's interpretation (version 2, policy {}) is not today's (version 1, policy None); parts differ in "
        "['orchestrator_code']".format(V2_BOUND.contract_sha256)]
    v = _v2_verify(report, _v2_reader(), current=V2_BOUND)
    assert all(v.flags.values()), v.as_document()["problems"]


def test_a_version_2_report_with_a_changed_orchestrator_does_not_bind():
    report = dict(copy.deepcopy(REPORT), interpretation=V2_BOUND.as_document())   # produced by the REAL orchestrator
    v = _v2_verify(report, _v2_reader({"src/genomic_variant_classifier/source_monitor/run_monitor.py": b"# changed\n"}))
    assert v.problems["configuration_bound"] == [
        "the report's interpretation does not bind to {}: reported parts differ from the independent reconstruction "
        "in ['orchestrator_code']".format(V2C)]


def test_a_changed_policy_file_changes_the_contract_digest_and_does_not_bind():
    """The version-2 fingerprint binds the committed policy FILE: even a reformatting (same rules and plan) is refused."""
    reformatted = json.dumps(json.loads(_POLICY_BYTES), indent=4, sort_keys=True).encode("ascii") + b"\n"
    assert reformatted != _POLICY_BYTES
    report = dict(copy.deepcopy(REPORT), interpretation=V2_BOUND.as_document())
    v = _v2_verify(report, _v2_reader({"configs/source_monitor_interpretation.json": reformatted}))
    assert v.problems["configuration_bound"] == [
        "the report's interpretation does not bind to {}: the interpretation differs from the independent "
        "reconstruction in ['contract_sha256', 'fingerprint']".format(V2C)]


def test_an_unreadable_policy_file_is_a_REFUSAL_never_a_downgrade():
    reader = _v2_reader({"configs/source_monitor_interpretation.json": PermissionError("simulated unreadable blob")})
    v = _v2_verify(REPORT, reader)
    assert v.problems["configuration_bound"] == [
        "the interpretation at {} could not be reconstructed: PermissionError: simulated unreadable blob".format(V2C)]


def test_a_non_legacy_commit_without_a_policy_file_is_REFUSED():
    v = _v2_verify(REPORT, _v2_reader({"configs/source_monitor_interpretation.json": None}))
    assert v.problems["configuration_bound"] == [
        "the interpretation at {} could not be reconstructed: ContractError: configs/source_monitor_interpretation.json "
        "is absent at {}, which is outside the admitted legacy history".format(V2C, V2C)]


def test_a_legacy_commit_WITH_a_policy_file_is_REFUSED():
    def reader(commit, path):
        if commit == _BLOBS["commit"] and path == "configs/source_monitor_interpretation.json":
            return _POLICY_BYTES
        return read_blob(commit, path)
    v = _verify(reader=reader)
    assert v.problems["configuration_bound"] == [
        "the interpretation at {0} could not be reconstructed: ContractError: legacy history contradicts the "
        "authenticated tree: configs/source_monitor_interpretation.json is present at {0}".format(RUN["head_sha"])]


def test_a_boolean_interpretation_version_is_refused_in_the_version_1_epoch():
    report = copy.deepcopy(REPORT)
    report["interpretation"]["schema_version"] = True
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    assert v.problems["configuration_bound"] == [
        "the report's interpretation does not bind to {}: a version-1 interpretation's schema_version must be an integer "
        "in [1, 1] (never a boolean), got True".format(RUN["head_sha"])]


# ------------------------------------------------------------------ review revision 3: defects reproduced 2026-09-28
# Built EXACTLY as the owner's probe_current.py builds them; on main 6f37d9b AND on the first v2 rebuild each gave
# verified=True with six true flags (measured 2026-09-29). Every reason is pinned.
def _probe(name):
    from genomic_variant_classifier.source_monitor.request_verifier import qualify
    r = copy.deepcopy(REPORT)
    target = r["results"][0]["target"]
    if name == "integrity_findings":
        r["results"][0]["captures"][0]["response_sha256"] = "0" * 64
        r["qualification"][target] = qualify(target, r["results"][0]["captures"]).as_document()
    elif name == "boolean_nested_count":
        r["qualification"][target]["captures_examined"] = True
    else:
        r["qualification"][target]["eligible_for_existence_claim"] = 1
    archive, arts = _repack(r)
    return _verify(archive=archive, artifacts=arts)


def test_a_replay_integrity_finding_BLOCKS_even_when_the_report_carries_it_faithfully():
    """'Report agrees with replay' does not imply 'replayed evidence is valid'. The genuine 4.1.2 witness remains a
    review item (authenticated, bound evidence), but the observation is not complete and the exit code must be 2."""
    v = _probe("integrity_findings")
    assert not v.verified and not v.flags["observation_complete"] and not v.flags["claims_reconciled"]
    assert v.problems["observation_complete"] == [
        "gnomad-public-releases: blocking validation finding evidence.integrity_mismatch: declared digest "
        "0000000000000000 does not match sha256(retained body) 577c27b247c50be2"]
    assert v.problems["claims_reconciled"] == ["the report's exit code 1 conflicts with the trusted replay, which requires 2"]
    assert v.flags["review_required"] and v.review_items == [
        'gnomad-public-releases: release prefix "release/4.1.2/" is newer than the approved 4.1.1']


@pytest.mark.parametrize("name, field", [("boolean_nested_count", "captures_examined"),
                                         ("integer_nested_flag", "eligible_for_existence_claim")])
def test_a_NESTED_boolean_integer_substitution_is_refused(name, field):
    """Python's == equates True with 1 even inside dicts; the qualification is compared with strict_equal."""
    v = _probe(name)
    assert not v.verified and v.flags["observation_complete"] and not v.flags["claims_reconciled"]
    assert v.problems["claims_reconciled"] == [
        "gnomad-public-releases: the report's qualification differs from the replay (value or JSON type) in "
        "['{}']".format(field)]


def test_review_is_required_only_for_authenticated_AND_bound_evidence():
    """Review revision 3: a finding derived from evidence that does not bind to its configuration is not a trusted review
    item. The items are still derived and reported, but review_required stays false."""
    report = copy.deepcopy(REPORT)
    report["interpretation"]["parts"]["adapter_code"] = "e" * 64
    report["interpretation"]["fingerprint"] = ic.fingerprint(report["interpretation"]["parts"], version=1)
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    assert not v.flags["configuration_bound"] and v.flags["review_required"] is False
    assert v.review_items == ['gnomad-public-releases: release prefix "release/4.1.2/" is newer than the approved 4.1.1']


def test_a_policy_outside_the_replay_handler_is_UNSUPPORTED_never_replayed_under_todays_constants():
    doc = json.loads(_POLICY_BYTES)
    doc["request_plan"]["approved_baseline"] = "4.1.2"
    with pytest.raises(ic.ContractError) as exc:
        rv.replay_handler(ic.parse_policy(json.dumps(doc).encode("ascii")))
    assert str(exc.value) == ("no reviewed replay handler implements this policy (semantics 'gnomad-release-listing-v1', "
                              "baseline '4.1.2')")


def test_an_inconsistent_attempt_chronology_cannot_satisfy_today():
    arts = copy.deepcopy(ARTIFACTS)
    arts["artifacts"][0]["created_at"] = "2026-09-27T06:40:30Z"            # one second AFTER the attempt ended
    archive, arts = _repack(copy.deepcopy(REPORT), arts=arts)
    v = _verify(archive=archive, artifacts=arts)
    assert v.problems["current_monitoring_obligation_satisfied"] == [
        "the attempt chronology is inconsistent: created 2026-09-27 06:40:16+00:00, started 2026-09-27 06:40:16+00:00, "
        "artifact 2026-09-27 06:40:30+00:00, ended 2026-09-27 06:40:29+00:00"]
    assert not v.flags["execution_authenticated"]        # check_execution refuses it independently (outside the window)
