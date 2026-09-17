"""Behavioral regression for source_monitor_alert.yml's embedded script --
executes the actual production code, never a hand-copied duplicate.

WHY THIS FILE EXISTS
=====================
MEASURED 2026-09-17, from a fifth external ruling reviewing WD: WD's own
regression tests (test_source_monitor_alert_workflow.py) assert that
particular strings occur in the workflow's YAML source. They do not
execute the rendering behavior. The functional verification that actually
proved WD, AF and NC correct -- extracting the embedded script and running
it in Node against controlled doubles -- was done by hand, repeatedly,
across several units this session, and never became a repeatable test.
This file is that harness, made permanent.

"That harness must consume the same script or module used by the
workflow. A separately copied rendering implementation would recreate the
fixture-divergence problem." alert_behavior_harness.extract_script() reads
.github/workflows/source_monitor_alert.yml directly, at test-collection
time, every run -- there is no second copy of the rendering logic here to
drift from the first.

WHAT THIS FILE CANNOT DO
=========================
It cannot prove GitHub will deliver the event, or that actions/download-
artifact retrieves the right run's artifact in practice -- see
test_source_monitor_alert_workflow.py's own docstring for that boundary.
It CAN prove that, given the report a real download would have produced,
the script renders the correct body and attempts (or does not attempt)
the correct GitHub API calls.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from .alert_behavior_harness import rendered_body, run_alert_script

_NODE_MISSING = shutil.which("node") is None
pytestmark = pytest.mark.skipif(
    _NODE_MISSING,
    reason="node is not on PATH; this harness executes the workflow's own "
           "embedded JavaScript and cannot substitute a Python "
           "reimplementation without recreating the fixture-divergence "
           "problem it exists to close")

_WORKFLOW_RUN_CONTEXT = {
    "eventName": "workflow_run", "sha": "deadbeef",
    "serverUrl": "https://github.com", "repo": {"owner": "o", "repo": "r"},
    "payload": {"workflow_run": {
        "conclusion": "failure", "head_sha": "a1b2c3d4",
        "html_url": "https://github.com/o/r/actions/runs/1", "run_number": 1}},
}


def test_the_rulings_own_acceptance_case(tmp_path):
    """The exact scenario and assertions specified by the ruling itself:
    a qualified witness must appear; an unsupported producer claim
    alongside it must not; dry-run must attempt no production write."""
    report = {
        "exit_code": 2,
        "results": [{"target": "gnomad-public-releases",
                     "reason": "evidence.qualification_unestablished",
                     "findings": ["release 9.9.9 is newer than the approved 4.1"]}],
        "unqualified": ["gnomad-public-releases"],
        "qualification": {"gnomad-public-releases":
                          {"positive_witnesses": ["4.1.1"]}},
    }
    result = run_alert_script(
        tmp_path=tmp_path, env={"DRY_RUN": "true", "SIMULATE_EXIT_CODE": "none"},
        context=_WORKFLOW_RUN_CONTEXT, report=report)
    body = rendered_body(result)
    assert "### Scientific review" in body
    assert "4.1.1" in body
    assert "9.9.9" not in body
    assert result["production_writes"] == []


def test_a_real_run_with_an_open_issue_attempts_exactly_one_comment(tmp_path):
    """The harness's own non-vacuity proof: dry-run's empty
    production_writes must not be trivially always empty. A genuine
    (non-dry-run) success with an open issue must attempt exactly one
    createComment, matching AF's suspend-auto-close fix precisely."""
    result = run_alert_script(
        tmp_path=tmp_path, env={"DRY_RUN": "false", "SIMULATE_EXIT_CODE": "none"},
        context={**_WORKFLOW_RUN_CONTEXT,
                 "payload": {"workflow_run": {"conclusion": "success",
                            "head_sha": "a1b2c3d4",
                            "html_url": "https://github.com/o/r/actions/runs/1",
                            "run_number": 1}}},
        open_issues=[{"number": 17}])
    writes = result["production_writes"]
    assert len(writes) == 1
    assert writes[0]["op"] == "createComment"
    assert writes[0]["args"]["issue_number"] == 17
    assert "does NOT close" in writes[0]["args"]["body"]


def test_no_witness_means_no_scientific_review_section(tmp_path):
    """The negative half of the ruling's own table: an empty witness
    array must produce no Scientific review section at all, not an
    empty one."""
    report = {
        "exit_code": 2,
        "results": [{"target": "x", "reason": "evidence.qualification_unestablished"}],
        "unqualified": ["x"],
        "qualification": {"x": {"positive_witnesses": []}},
    }
    result = run_alert_script(
        tmp_path=tmp_path, env={"DRY_RUN": "true", "SIMULATE_EXIT_CODE": "none"},
        context=_WORKFLOW_RUN_CONTEXT, report=report)
    assert "### Scientific review" not in rendered_body(result)


def test_a_malformed_qualification_block_does_not_crash(tmp_path):
    """Behavioral confirmation of the same defensive handling
    test_source_monitor_alert_workflow.py checks structurally: execution
    completes and the section is correctly absent, not merely that the
    right substring exists somewhere in the source."""
    report = {"exit_code": 2, "results": [{"target": "x", "reason": "y"}],
              "unqualified": ["x"], "qualification": "not an object at all"}
    result = run_alert_script(
        tmp_path=tmp_path, env={"DRY_RUN": "true", "SIMULATE_EXIT_CODE": "none"},
        context=_WORKFLOW_RUN_CONTEXT, report=report)
    assert "### Scientific review" not in rendered_body(result)


@pytest.mark.xfail(
    reason="MEASURED 2026-09-17: a witness value that is itself an "
           "object, rather than a string, renders as the literal text "
           "'[object Object]'. Confirmed by the ruling that reviewed WD, "
           "and reproduced directly here. This is a known, deliberately "
           "deferred limitation -- 'individual witness values are not "
           "validated' -- to be closed by the single report-contract "
           "work item (P2), not a standalone patch. This test documents "
           "the limitation as a live behavioral fact rather than letting "
           "it go unrecorded; it is expected to start passing once P2 "
           "lands, at which point xfail should be removed, not widened.",
    strict=False)
def test_a_non_string_witness_value_does_not_render_as_object_object(tmp_path):
    report = {"exit_code": 2, "results": [{"target": "x", "reason": "y"}],
              "unqualified": ["x"],
              "qualification": {"x": {"positive_witnesses": [{"a": 1}]}}}
    result = run_alert_script(
        tmp_path=tmp_path, env={"DRY_RUN": "true", "SIMULATE_EXIT_CODE": "none"},
        context=_WORKFLOW_RUN_CONTEXT, report=report)
    assert "[object Object]" not in rendered_body(result)
