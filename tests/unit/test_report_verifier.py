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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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
    if commit != _BLOBS["commit"]:
        raise KeyError("no blobs preserved for {}".format(commit))
    return base64.b64decode(_BLOBS["blobs"][path]["base64"])


def _report():
    with zipfile.ZipFile(io.BytesIO(ARCHIVE)) as zf:
        return json.loads(zf.read("report.json"))


REPORT = _report()


def _repack(report=None, raw=None, members=None):
    """A consistent archive + artifact listing: GitHub's digest and size follow the new bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in (members or [("report.json", raw if raw is not None else json.dumps(report).encode())]):
            zf.writestr(name, data)
    archive = buf.getvalue()
    arts = copy.deepcopy(ARTIFACTS)
    arts["artifacts"][0].update(digest="sha256:" + hashlib.sha256(archive).hexdigest(), size_in_bytes=len(archive))
    return archive, arts


def _verify(archive=ARCHIVE, run=RUN, artifacts=ARTIFACTS, run_id=RUN_ID, attempt=1, latest=1, now=NOW, current=None):
    return rv.verify(archive, run, artifacts, run_id=run_id, run_attempt=attempt, latest_attempt=latest,
                     read_blob=read_blob, now=now,
                     current_parts=REPORT["interpretation"]["parts"] if current is None else current,
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
def test_a_part_that_is_not_the_runs_file_is_refused():
    report = copy.deepcopy(REPORT)
    report["interpretation"]["parts"]["adapter_code"] = "0" * 64
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    assert not v.flags["configuration_bound"] and "adapter_code part" in _problems(v, "configuration_bound")
    assert "does not recompute" in _problems(v, "configuration_bound")


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
    assert not v.flags["claims_reconciled"] and "no exact independent witness" in _problems(v, "claims_reconciled")


def test_a_qualification_that_the_replay_does_not_reproduce_is_refused():
    report = copy.deepcopy(REPORT)
    report["qualification"]["gnomad-public-releases"]["unsupported_names"] = ["release/5.0.0rc1/"]
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    assert not v.flags["claims_reconciled"] and "differs from the replay in ['unsupported_names']" in _problems(v, "claims_reconciled")


def test_missing_captures_make_the_observation_incomplete():
    report = copy.deepcopy(REPORT)
    report["results"][0]["captures"] = []
    archive, arts = _repack(report)
    v = _verify(archive=archive, artifacts=arts)
    assert not v.flags["observation_complete"] and not v.verified


# ------------------------------------------------------------------ current obligation
def test_stale_evidence_is_valid_history_but_not_todays_obligation():
    v = _verify(now=NOW + timedelta(days=9))
    assert v.verified and not v.flags["current_monitoring_obligation_satisfied"]
    assert "old; the monitoring obligation" in _problems(v, "current_monitoring_obligation_satisfied")


def test_a_stale_policy_is_valid_history_but_not_todays_obligation():
    current = dict(REPORT["interpretation"]["parts"], verifier_code="1" * 64)
    v = _verify(current=current)
    assert v.verified and not v.flags["current_monitoring_obligation_satisfied"]
    assert "verifier_code of this run" in _problems(v, "current_monitoring_obligation_satisfied")


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
                       read_blob=read_blob, now=NOW, current_parts=REPORT["interpretation"]["parts"])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and doc["verified"] and all(doc["flags"].values())


def test_the_cli_exits_2_on_a_tampered_archive(tmp_path):
    tampered = bytearray(ARCHIVE)
    tampered[-30] ^= 0x01
    out = tmp_path / "verdict.json"
    code = _cli().main(["--run-id", str(RUN_ID), "--run-attempt", "1", "--verdict", str(out)], fetch=_serve(bytes(tampered)),
                       read_blob=read_blob, now=NOW, current_parts=REPORT["interpretation"]["parts"])
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
                now=NOW, current_parts=REPORT["interpretation"]["parts"], step_summary=str(summary))
    text = summary.read_text(encoding="utf-8")
    assert "Run 36300779115 attempt 1: **VERIFIED**" in text
    assert all("| `{}` | True |".format(flag) in text for flag in rv.FLAGS)
