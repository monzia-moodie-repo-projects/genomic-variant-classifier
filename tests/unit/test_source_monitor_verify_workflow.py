"""The verify-and-publish workflow's safety contract (C1 2026-09-27; C2 owner rulings 2026-09-29/30).

Least privilege PER JOB (workflow permissions are {} and ONLY the publish job can write issues); BOTH jobs run the
trusted workflow commit and verify HEAD; untrusted values reach shells only through the environment; the names the
publisher's history depends on are tied to the code's own constants, so a rename on either side fails here.

Author: Monzia Moodie
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

from genomic_variant_classifier.source_monitor import c2_github as gh

_ROOT = Path(__file__).resolve().parents[2]
_PATH = _ROOT / ".github" / "workflows" / "source_monitor_verify.yml"


@pytest.fixture(scope="module")
def raw():
    return _PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def wf(raw):
    return yaml.safe_load(raw)


def _steps(wf, job):
    return wf["jobs"][job]["steps"]


def _step(wf, job, name):
    found = [s for s in _steps(wf, job) if s.get("name") == name]
    assert len(found) == 1, (job, name)
    return found[0]


def test_permissions_are_granted_per_job_and_only_publish_writes_issues(wf):
    assert wf["permissions"] == {}
    assert wf["jobs"]["verify"]["permissions"] == {"contents": "read", "actions": "read"}
    assert wf["jobs"]["publish"]["permissions"] == {"contents": "read", "actions": "read", "issues": "write"}
    assert set(wf["jobs"]) == {"verify", "publish"}


def test_the_trigger_is_the_source_monitor_workflow_on_main(wf):
    on = wf.get("on", wf.get(True))
    assert on["workflow_run"] == {"workflows": ["source-monitor"], "types": ["completed"]}
    assert set(on["workflow_dispatch"]["inputs"]) == {"run_id", "run_attempt"}
    gate = "github.event_name == 'workflow_dispatch' || github.event.workflow_run.head_branch == 'main'"
    assert wf["jobs"]["verify"]["if"] == gate and gate in wf["jobs"]["publish"]["if"]


@pytest.mark.parametrize("job", ["verify", "publish"])
def test_both_jobs_check_out_and_verify_the_workflow_commit(wf, job):
    checkout = next(s for s in _steps(wf, job) if s.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"] == {"ref": "${{ github.workflow_sha }}", "persist-credentials": False}
    check = _step(wf, job, "Verify the checkout is the workflow commit")
    assert check["env"] == {"WORKFLOW_SHA": "${{ github.workflow_sha }}"}
    assert '[ "$(git rev-parse HEAD)" = "$WORKFLOW_SHA" ]' in check["run"]
    assert _steps(wf, job).index(checkout) + 1 == _steps(wf, job).index(check)     # verified IMMEDIATELY after checkout


@pytest.mark.parametrize("job", ["verify", "publish"])
def test_untrusted_values_reach_the_shell_only_through_env(wf, job):
    """An expression interpolated into a run: script is shell code; inputs, commit, branch and job outputs are untrusted."""
    untrusted = re.compile(r"\$\{\{[^}]*(inputs|head_sha|head_branch|workflow_run|github\.event|needs\.)[^}]*\}\}")
    for step in _steps(wf, job):
        assert not untrusted.search(step.get("run", "")), step.get("name")


def test_the_run_commit_and_the_source_identifiers_are_validated_before_use(wf):
    step = _step(wf, "verify", "Identify and fetch the run's commit")["run"]
    assert step.index("^[0-9a-f]{40}$") < step.index("git fetch")
    assert "^[0-9]{1,20}$" in step and "set -euo pipefail" in step
    deliver = _step(wf, "publish", gh.DELIVERY_STEP)["run"]
    assert deliver.index("^[0-9]{1,20}$") < deliver.index("python scripts/publish_monitor_receipt.py")


def test_the_receipt_is_written_before_the_exit_code_and_emitted_by_a_separate_always_step(wf):
    verify = _step(wf, "verify", "Verify the run and write the receipt")
    assert "--receipt receipt.json" in verify["run"] and "--checker-commit" in verify["run"]
    emit = _step(wf, "verify", "Hand the receipt to the publish job")
    assert emit.get("id") == "emit" and emit.get("if") == "always()"          # absence is a failed assertion, never a crash
    assert wf["jobs"]["verify"]["outputs"] == {"receipt_b64": "${{ steps.emit.outputs.receipt_b64 }}"}
    deliver = _step(wf, "publish", gh.DELIVERY_STEP)
    assert deliver["env"]["RECEIPT_B64"] == "${{ needs.verify.outputs.receipt_b64 }}"


def test_publish_runs_after_a_failed_verification_but_never_after_cancellation(wf):
    job = wf["jobs"]["publish"]
    assert job["needs"] == "verify"
    assert job["if"].startswith("${{ !cancelled() && needs.verify.result != 'cancelled' && (")


def test_the_one_writer_queues_instead_of_replacing(wf):
    assert wf["jobs"]["publish"]["concurrency"] == {"group": "source-monitor-alert", "queue": "max", "cancel-in-progress": False}
    assert wf["concurrency"]["group"] != "source-monitor-alert"          # never acquired twice (enclosing + job)


def test_the_names_the_history_depends_on_are_the_codes_own(wf, raw):
    """dispatch_history finds prior attempts by the run-name, the publish job's name, the delivery step's name and the
    outcome artifact's prefix: each must equal the code's constant, or history silently finds nothing."""
    spec = importlib.util.spec_from_file_location("publish", _ROOT / "scripts" / "publish_monitor_receipt.py")
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)
    rendered = wf["run-name"].replace("${{ github.event.workflow_run.id || inputs.run_id }}", "123").replace(
        "${{ github.event.workflow_run.run_attempt || inputs.run_attempt }}", "4")
    assert rendered == pub.run_name(123, 4) == "verify 123/4"
    assert wf["jobs"]["publish"]["name"] == gh.PUBLISH_JOB
    assert _step(wf, "publish", gh.DELIVERY_STEP)
    upload = _step(wf, "publish", "Upload the delivery outcome")
    assert upload["if"] == "always()" and upload["with"]["name"] == gh.OUTCOME_ARTIFACT_PREFIX + "${{ github.run_attempt }}"
    assert upload["with"]["path"] == "outcome.json"


def test_every_artifact_name_is_unique_per_attempt(wf):
    """Artifact names are unique within a run and a re-run is an attempt of the SAME run (a second upload: HTTP 409)."""
    names = [s["with"]["name"] for j in ("verify", "publish") for s in _steps(wf, j) if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert len(names) == 2 and all(n.endswith("-attempt-${{ github.run_attempt }}") for n in names)
