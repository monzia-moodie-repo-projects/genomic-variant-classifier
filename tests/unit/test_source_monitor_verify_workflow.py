"""The preview verification workflow's safety contract (change C1, 2026-09-27; owner rulings 2026-09-25/26).

Least privilege, trusted code, untrusted values only through the environment, and NO issue writing while the
existing alert remains the single production writer.

Author: Monzia Moodie
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_PATH = _ROOT / ".github" / "workflows" / "source_monitor_verify.yml"


@pytest.fixture(scope="module")
def raw():
    return _PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def wf(raw):
    return yaml.safe_load(raw)


def test_permissions_are_exactly_read_contents_and_read_actions(wf, raw):
    assert wf["permissions"] == {"contents": "read", "actions": "read"}
    assert all("permissions" not in job for job in wf["jobs"].values())      # no job widens them
    assert "write" not in raw.split("permissions:", 1)[1].split("concurrency:", 1)[0]


def test_it_writes_no_issue_while_the_alert_is_the_single_writer(raw):
    """Preview: source_monitor_alert.yml stays the only production issue writer until C2."""
    for forbidden in ("issues:", "createComment", "issues.create", "github-script", "gh issue"):
        assert forbidden not in raw, forbidden


def test_the_trigger_mirrors_the_alerts(wf):
    on = wf.get("on", wf.get(True))
    with open(_ROOT / ".github/workflows/source_monitor_alert.yml", encoding="utf-8") as fh:
        alert = yaml.safe_load(fh)
    assert on["workflow_run"] == alert.get("on", alert.get(True))["workflow_run"]
    assert wf["jobs"]["verify"]["if"] == "github.event_name == 'workflow_dispatch' || github.event.workflow_run.head_branch == 'main'"


def test_it_checks_out_trusted_code_without_persisting_credentials(wf):
    checkout = next(s for s in wf["jobs"]["verify"]["steps"] if s.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"] == {"ref": "${{ github.event.repository.default_branch }}", "persist-credentials": False}


def test_untrusted_values_reach_the_shell_only_through_env(wf):
    """An expression interpolated into a run: script is shell code; inputs, commit and branch names are untrusted."""
    untrusted = re.compile(r"\$\{\{[^}]*(inputs|head_sha|head_branch|workflow_run|github\.event)[^}]*\}\}")
    for step in wf["jobs"]["verify"]["steps"]:
        assert not untrusted.search(step.get("run", "")), step.get("name")


def test_the_run_commit_is_validated_before_it_is_fetched(wf):
    step = next(s for s in wf["jobs"]["verify"]["steps"] if s.get("id") == "run")["run"]
    assert step.index("^[0-9a-f]{40}$") < step.index("git fetch")
    assert "^[0-9]{1,20}$" in step and "set -euo pipefail" in step
