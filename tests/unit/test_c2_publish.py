"""scripts/publish_monitor_receipt.py -- the ONE issue writer (C2, owner ruling 2026-09-29), end to end against a fake
GitHub for the real repository. The checker receipt is produced by the REAL checker script (offline, run-8 fixtures) with
TODAY's reconstruction, so the publisher's independently recomputed bindings must match it exactly.

Author: Monzia Moodie
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

from genomic_variant_classifier.source_monitor import c2_github as gh
from genomic_variant_classifier.source_monitor import c2_protocol as c2
from genomic_variant_classifier.source_monitor import deployment as dep
from genomic_variant_classifier.source_monitor import report_verifier as rv
from tests.unit.test_report_verifier import FIX, NOW, RUN_ID, _cli, _serve, read_blob

ROOT = Path(__file__).resolve().parents[2]
SHA, CURRENT, WORKFLOW_ID = "a" * 40, 900, 555
DEP = dep.load(ROOT)
BASE = gh.API_ROOT + "/repos/" + DEP.repository
ISSUE = BASE + "/issues/27"


def _publisher():
    spec = importlib.util.spec_from_file_location("publish_monitor_receipt", ROOT / "scripts" / "publish_monitor_receipt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PUB = _publisher()
PIN = PUB.pinned(DEP)


@pytest.fixture(scope="module")
def receipt_b64(tmp_path_factory):
    d = tmp_path_factory.mktemp("receipt")
    _cli().main(["--run-id", str(RUN_ID), "--run-attempt", "1", "--verdict", str(d / "v.json"), "--receipt", str(d / "r.json"),
                 "--checker-commit", SHA, "--evaluation-run-id", str(CURRENT), "--evaluation-attempt", "1"],
                fetch=_serve(), read_blob=read_blob, now=NOW, current=rv.current_reconstruction(ROOT))
    return base64.b64encode((d / "r.json").read_bytes()).decode("ascii")


class Fake:
    def __init__(self, runs=(), extra=None, post="created"):
        self.post, self.comments, self.calls = post, [], []
        self.routes = {
            ("GET", BASE + "/actions/runs/{}/attempts/1".format(RUN_ID)): (FIX / "run8_attempt1.json").read_bytes(),
            ("GET", BASE): json.dumps({"id": PIN.repository_id, "full_name": PIN.repository}).encode(),
            ("GET", BASE + "/issues?labels=source-monitor-alert&state=open&per_page=100"):
                json.dumps([{"id": PIN.issue_id, "number": 27}]).encode(),
            ("GET", ISSUE): json.dumps({"id": PIN.issue_id, "number": 27, "state": "open", "repository_url": BASE,
                                        "labels": [{"name": "source-monitor-alert"}]}).encode(),
            ("GET", BASE + "/actions/runs/{}".format(CURRENT)): json.dumps({"id": CURRENT, "workflow_id": WORKFLOW_ID}).encode(),
            ("GET", BASE + "/actions/workflows/{}/runs?per_page=100".format(WORKFLOW_ID)):
                json.dumps({"total_count": len(runs), "workflow_runs": list(runs)}).encode()}
        self.routes.update(extra or {})

    def __call__(self, method, url, body=None, *, allow_redirect=False):
        self.calls.append((method, url))
        if method not in ("GET", "POST"):
            return 200, {}, b"{}"          # RECORDED and answered, so the tests' own assertions catch any other write
        if method == "POST":
            row = {"id": 5000, "user": {"id": PIN.author_id}, "body": json.loads(body)["body"], "issue_url": ISSUE}
            if self.post == "created":
                self.comments.append(row)
                return 201, {}, json.dumps(row).encode()
            return 502, {}, b"bad gateway"
        if url.startswith(ISSUE + "/comments"):
            return 200, {}, json.dumps(self.comments).encode()
        answer = self.routes[(method, url)]
        return (200, {}, answer) if isinstance(answer, bytes) else answer

    @property
    def posts(self):
        return sum(1 for m, _ in self.calls if m == "POST")


def publish(tmp_path, fake, receipt, event="workflow_run"):
    outcome = tmp_path / "outcome.json"
    code = PUB.main(["--event", event, "--source-run-id", str(RUN_ID), "--source-attempt", "1", "--current-run-id",
                     str(CURRENT), "--current-attempt", "1", "--workflow-sha", SHA, "--outcome", str(outcome)],
                    request=fake, receipt_b64=receipt, clock=lambda: NOW)
    assert outcome.is_file(), "NO outcome record was written"          # absence is a stated failure, never a crash
    return code, json.loads(outcome.read_text(encoding="ascii"))


def _payload(receipt):
    subject = gh.subject_from_attempt(json.loads((FIX / "run8_attempt1.json").read_text(encoding="utf-8")))
    return c2.open_receipt(base64.b64decode(receipt), c2.Bindings(subject, PUB.expected_checker(SHA, DEP), CURRENT, 1), NOW)


DEST = c2.Destination(1151261021, 5600463137, 27)


def test_a_checker_receipt_is_published_with_exactly_one_post(tmp_path, capsys, receipt_b64):
    fake = Fake()
    code, record = publish(tmp_path, fake, receipt_b64)
    key = c2.delivery_id(_payload(receipt_b64), DEST)
    assert (code, fake.posts) == (0, 1) and "RESULT acknowledged created" in capsys.readouterr().out
    assert record == {"schema": gh.OUTCOME_SCHEMA, "schema_version": 1, "delivery_id": key, "post_issued": True,
                      "action": "acknowledged", "reason": "created"}
    assert "<!-- gvc:c2:v1:{} -->".format(key) in fake.comments[0]["body"]


def test_without_a_checker_receipt_the_coordinator_reports_verification_unavailable(tmp_path, capsys):
    fake = Fake()
    try:
        code, record = publish(tmp_path, fake, "")
    except c2.Refusal as exc:
        pytest.fail("the coordinator's own receipt was refused: {}".format(exc.code))
    out = capsys.readouterr().out
    assert (code, fake.posts, record["post_issued"]) == (0, 1, True)
    assert "receipt: issuer coordinator event kind verification_unavailable" in out
    assert '"issuer_role": "coordinator"' in fake.comments[0]["body"]


def test_a_manual_verifier_run_is_a_preview_and_still_writes_its_record(tmp_path, capsys, receipt_b64):
    fake = Fake()
    code, record = publish(tmp_path, fake, receipt_b64, event="workflow_dispatch")
    assert (code, fake.posts, record["post_issued"], record["delivery_id"], record["action"]) == (0, 0, False, "", "preview")
    assert [u for _, u in fake.calls] == [BASE + "/actions/runs/{}/attempts/1".format(RUN_ID)]   # no destination, no history


def test_a_refused_receipt_still_writes_a_no_post_record(tmp_path, receipt_b64):
    doc = json.loads(base64.b64decode(receipt_b64))
    doc["payload"]["checker"]["commit"] = "b" * 40
    doc["receipt_sha256"] = c2.digest("gvc.receipt/v1", doc["payload"])        # a recomputed checksum cannot rebind
    tampered = base64.b64encode(c2.canonical(doc)).decode("ascii")
    with pytest.raises(c2.Refusal) as exc:
        publish(tmp_path, Fake(), tampered)
    assert exc.value.code == "binding.checker"
    assert (tmp_path / "outcome.json").is_file(), "NO outcome record was written"
    record = json.loads((tmp_path / "outcome.json").read_text(encoding="ascii"))
    assert (record["post_issued"], record["delivery_id"]) == (False, "")


def _outcome_zip(delivery_id, post_issued):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("outcome.json", json.dumps({"schema": gh.OUTCOME_SCHEMA, "schema_version": 1, "delivery_id": delivery_id,
                                                "post_issued": post_issued, "action": "x", "reason": "y"}))
    return buf.getvalue()


def _prior_run(record_zip):
    runs = [{"id": 800, "run_attempt": 1, "display_title": PUB.run_name(RUN_ID, 1), "workflow_id": WORKFLOW_ID}]
    extra = {("GET", BASE + "/actions/runs/800/attempts/1/jobs?per_page=100"): json.dumps({"total_count": 1, "jobs": [
                 {"name": gh.PUBLISH_JOB, "steps": [{"name": gh.DELIVERY_STEP, "status": "completed", "conclusion": "success"}]}]}).encode(),
             ("GET", BASE + "/actions/runs/800/artifacts?per_page=100"): json.dumps({"total_count": 1, "artifacts": [
                 {"id": 81, "name": gh.OUTCOME_ARTIFACT_PREFIX + "1", "expired": False}]}).encode(),
             ("GET", BASE + "/actions/artifacts/81/zip"): record_zip}
    return runs, extra


def test_a_prior_dispatch_of_this_delivery_is_unknown_never_a_second_post(tmp_path, capsys, receipt_b64):
    runs, extra = _prior_run(_outcome_zip(c2.delivery_id(_payload(receipt_b64), DEST), True))
    fake = Fake(runs=runs, extra=extra)
    code, record = publish(tmp_path, fake, receipt_b64)
    assert (code, fake.posts, record["post_issued"], record["reason"]) == (1, 0, False, "prior_dispatch_unresolved")


def test_a_prior_previews_record_does_not_block_a_later_delivery(tmp_path, receipt_b64):
    """REGRESSION (found designing these tests): a preview once wrote NO record, so its started delivery step read as
    UNKNOWN and would have blocked that source run's delivery forever."""
    runs, extra = _prior_run(_outcome_zip("", False))
    fake = Fake(runs=runs, extra=extra)
    code, record = publish(tmp_path, fake, receipt_b64)
    assert (code, fake.posts, record["reason"]) == (0, 1, "created")


def test_a_post_lost_before_commit_is_unknown_and_recorded_as_issued(tmp_path, receipt_b64):
    fake = Fake(post="lost")
    code, record = publish(tmp_path, fake, receipt_b64)
    assert (code, fake.posts, record["post_issued"], record["reason"]) == (1, 1, True, "post_outcome_unknown")


def test_identifiers_are_validated_before_any_request(tmp_path):
    fake = Fake()
    with pytest.raises(SystemExit) as exc:
        PUB.main(["--event", "workflow_run", "--source-run-id", "0", "--source-attempt", "1", "--current-run-id", "1",
                  "--current-attempt", "1", "--workflow-sha", SHA, "--outcome", str(tmp_path / "o.json")],
                 request=fake, receipt_b64="")
    assert exc.value.code == 2 and fake.calls == []


def test_importing_the_publisher_reads_no_workflow_variable():
    source = (ROOT / "scripts" / "publish_monitor_receipt.py").read_text(encoding="utf-8")
    body, main_block = source.split('if __name__ == "__main__":')
    assert "os.environ" not in body and "GITHUB_" not in body.replace("GITHUB_*", "")
    assert main_block.count("os.environ") == 2


@pytest.mark.parametrize("receipt_kind", ["checker", "coordinator"])
def test_the_writer_never_closes_edits_or_deletes_anything(tmp_path, receipt_b64, receipt_kind):
    """The legacy alert's protection P16 (2026-09-17: an auto-close without reading the report), carried BEHAVIOURALLY:
    across a delivery, every request is a GET except AT MOST ONE POST, and that POST creates a comment on the pinned issue
    -- no PATCH (close / edit), no PUT, no DELETE, no other POST target."""
    fake = Fake()
    publish(tmp_path, fake, receipt_b64 if receipt_kind == "checker" else "")
    methods = {m for m, _ in fake.calls}
    assert methods <= {"GET", "POST"}
    assert [u for m, u in fake.calls if m == "POST"] == [ISSUE + "/comments"]



def test_checker_unavailable_receipt_reaches_writer(tmp_path, capsys):
    """REGRESSION (owner ruling 2026-10-01; defect reproduced on 78488f8): an evidence-collection failure must survive the
    writer's INDEPENDENT binding. The REAL checker's unavailable receipt goes UNCHANGED to the REAL publisher."""
    served = _serve()

    def fail_artifact_listing(url, limit):
        if "/artifacts" in url:
            raise OSError("qualification: artifact listing unavailable")
        return served(url, limit)

    receipt_path = tmp_path / "unavailable-receipt.json"
    checker_exit = _cli().main(["--run-id", str(RUN_ID), "--run-attempt", "1", "--verdict", str(tmp_path / "verdict.json"),
                                "--receipt", str(receipt_path), "--checker-commit", SHA, "--evaluation-run-id", str(CURRENT),
                                "--evaluation-attempt", "1"],
                               fetch=fail_artifact_listing, read_blob=read_blob, now=NOW, current=rv.current_reconstruction(ROOT))
    assert checker_exit == 2 and receipt_path.is_file()
    encoded = base64.b64encode(receipt_path.read_bytes()).decode("ascii")
    try:
        payload = _payload(encoded)                              # PUB.expected_checker(SHA, DEP): independent of the receipt
    except c2.Refusal as exc:
        pytest.fail("the writer refuses the checker's own unavailable receipt: {}".format(exc.code))
    assert payload["issuer_role"] == "checker"
    assert payload["decision"] == {"status": "unavailable", "verified": None, "flags": None, "reviews": [],
                                   "reasons": [{"code": "checker.unavailable", "target": ""}]}
    fake = Fake()
    publisher_exit, outcome = publish(tmp_path, fake, encoded)
    assert (publisher_exit, fake.posts) == (0, 1)
    assert (outcome["action"], outcome["reason"], outcome["post_issued"]) == ("acknowledged", "created", True)
    assert fake.comments[0]["body"] == c2.render_comment(payload, DEST)
    assert outcome["delivery_id"] == c2.delivery_id(payload, DEST)
    publisher_exit, repeated = publish(tmp_path, fake, encoded)          # acknowledgement against the persisted comment
    assert (publisher_exit, fake.posts, repeated["action"], repeated["post_issued"]) == (0, 1, "acknowledged", False)


def test_publication_disabled_validates_but_never_posts(tmp_path, capsys):
    """The commissioning state (owner ruling 2026-10-01, step 2): the reviewed code runs, validates the receipt, and
    writes its outcome record -- but no destination lookup, no history, no POST. BOTH jobs read the SAME checked-out
    configuration, so the checker produces its receipt under the same (disabled) deployment: the deployment digest is part
    of the bound policy, and a receipt from another deployment would correctly fail binding."""
    doc = json.loads((ROOT / dep.CONFIG_PATH).read_text(encoding="utf-8"))
    doc["publication_enabled"] = False
    root = tmp_path / "trusted"
    (root / "configs").mkdir(parents=True)
    (root / dep.CONFIG_PATH).write_bytes(json.dumps(doc).encode("ascii"))
    _cli().main(["--run-id", str(RUN_ID), "--run-attempt", "1", "--verdict", str(tmp_path / "v.json"), "--receipt",
                 str(tmp_path / "r.json"), "--checker-commit", SHA, "--evaluation-run-id", str(CURRENT), "--evaluation-attempt", "1"],
                fetch=_serve(), read_blob=read_blob, now=NOW, current=rv.current_reconstruction(ROOT), deployment_root=root)
    receipt_b64 = base64.b64encode((tmp_path / "r.json").read_bytes()).decode("ascii")
    fake = Fake()
    outcome = tmp_path / "outcome.json"
    code = PUB.main(["--event", "workflow_run", "--source-run-id", str(RUN_ID), "--source-attempt", "1", "--current-run-id",
                     str(CURRENT), "--current-attempt", "1", "--workflow-sha", SHA, "--outcome", str(outcome)],
                    request=fake, receipt_b64=receipt_b64, clock=lambda: NOW, deployment_root=root)
    record = json.loads(outcome.read_text(encoding="ascii"))
    assert (code, fake.posts, record["action"], record["reason"], record["post_issued"]) == (0, 0, "preview", "publication_disabled", False)
    assert [u for _, u in fake.calls] == [BASE + "/actions/runs/{}/attempts/1".format(RUN_ID)]


def test_a_deployment_refusal_still_writes_a_no_post_record_the_history_reads_as_no_dispatch(tmp_path):
    """2026-10-01 (found by qualification, live run 36846920781): the deployment was loaded OUTSIDE the try, so a refusal
    -- a DEFINITE no-attempt -- left NO outcome record, and every later attempt for that source run read UNKNOWN."""
    doc = json.loads((ROOT / dep.CONFIG_PATH).read_text(encoding="utf-8"))
    doc["trusted_author_id"] = None
    root = tmp_path / "trusted"
    (root / "configs").mkdir(parents=True)
    (root / dep.CONFIG_PATH).write_bytes(json.dumps(doc).encode("ascii"))
    outcome, fake = tmp_path / "outcome.json", Fake()
    with pytest.raises(dep.DeploymentError) as exc:
        PUB.main(["--event", "workflow_run", "--source-run-id", str(RUN_ID), "--source-attempt", "1", "--current-run-id",
                  str(CURRENT), "--current-attempt", "1", "--workflow-sha", SHA, "--outcome", str(outcome)],
                 request=fake, receipt_b64="", clock=lambda: NOW, deployment_root=root)
    assert exc.value.code == "deployment.unresolved" and fake.calls == []
    assert outcome.is_file(), "NO outcome record was written"
    record = json.loads(outcome.read_text(encoding="ascii"))
    assert (record["post_issued"], record["delivery_id"]) == (False, "")
    import io as _io
    import zipfile as _zipfile
    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("outcome.json", outcome.read_bytes())
    assert gh._read_outcome(buf.getvalue(), "k" * 64) is False      # the REAL reader: no POST for any delivery
