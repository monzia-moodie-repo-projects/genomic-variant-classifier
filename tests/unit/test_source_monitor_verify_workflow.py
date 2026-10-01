"""The verify-and-publish workflow's safety contract (C1 2026-09-27; C2 owner rulings 2026-09-29/30).

Least privilege PER JOB (workflow permissions are {} and ONLY the publish job can write issues); BOTH jobs run the
trusted workflow commit and verify HEAD; untrusted values reach shells only through the environment; the names the
publisher's history depends on are tied to the code's own constants, so a rename on either side fails here.

Author: Monzia Moodie
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
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


POSIX = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None,
                           reason="executes a Linux-runner shell block; needs a POSIX bash (CI runs it). On Windows, `bash` is the "
                                  "bash.exe LAUNCHER (measured 2026-10-01: Bash/0x80110474), so shutil.which alone would not skip")


@POSIX
def test_the_run_commit_is_fetched_with_an_environment_scoped_credential_never_in_argv(wf, tmp_path):
    """2026-10-01 (found by qualification): a PRIVATE repository refuses an anonymous fetch ("could not read Username").
    The REAL step script runs under bash with a FAKE git that records its argv and environment: the credential must
    reach git ONLY through its environment, never a command line, never persisted; prompting is disabled."""
    script = _step(wf, "verify", "Identify and fetch the run's commit")["run"]
    record = tmp_path / "git_calls.txt"
    fake = tmp_path / "bin" / "git"
    fake.parent.mkdir()
    fake.write_text('#!/usr/bin/env bash\n{ printf "ARGV"; printf " %q" "$@"; printf "\\n"; '
                    'printf "PROMPT=%s KEY=%s VALUE=%s\\n" "$GIT_TERMINAL_PROMPT" "$GIT_CONFIG_KEY_0" "$GIT_CONFIG_VALUE_0"; } '
                    '>> "$RECORD"\nexit 0\n', encoding="ascii")
    fake.chmod(0o755)
    token = "ghs_QUALIFICATIONTESTTOKEN0123"
    # A MINIMAL environment, deliberately (unlike the {**os.environ} convention): an inherited GIT_CONFIG_* would contaminate
    # the very mechanism under test.
    env = {"PATH": "{}{}{}".format(fake.parent, os.pathsep, os.environ["PATH"]), "RECORD": str(record), "GH_TOKEN": token,
           "EVENT_NAME": "workflow_run", "EVENT_RUN_ID": "36846886992", "EVENT_RUN_ATTEMPT": "1",
           "EVENT_HEAD_SHA": "f9ad6542b08d1c1c2306374daaf53b354698202b", "INPUT_RUN_ID": "", "INPUT_RUN_ATTEMPT": "",
           "GITHUB_OUTPUT": str(tmp_path / "out.txt"), "GITHUB_REPOSITORY": "o/r"}
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    calls = record.read_text(encoding="ascii").splitlines()
    fetch = [i for i, line in enumerate(calls) if line.startswith("ARGV fetch")]
    assert len(fetch) == 1
    argv, environ = calls[fetch[0]], calls[fetch[0] + 1]
    assert argv == "ARGV fetch --no-tags --depth=1 origin f9ad6542b08d1c1c2306374daaf53b354698202b"
    import base64
    basic = base64.b64encode("x-access-token:{}".format(token).encode()).decode()
    assert environ == "PROMPT=0 KEY=http.https://github.com/.extraheader VALUE=AUTHORIZATION: basic {}".format(basic)
    assert token not in "\n".join(line for line in calls if line.startswith("ARGV")) and basic not in argv
    checkout = next(s for s in _steps(wf, "verify") if s.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["persist-credentials"] is False          # never persisted
