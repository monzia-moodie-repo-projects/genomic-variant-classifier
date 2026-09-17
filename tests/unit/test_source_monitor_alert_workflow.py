"""The source-monitor alert workflow, checked without waiting for a real
failure -- or for either of its two DIFFERENT kinds of failure.

WHY THIS FILE EXISTS
====================
source_monitor.yml's job conclusion is deliberately tied to run_monitor.py's
exit code, unfiltered -- see that workflow's own comment: "the job's
conclusion still comes from the step above". On 2026-09-16 that meant a
genuine finding -- release 4.1.1 newer than the approved 4.1 -- surfaced as
nothing more than a red X in the Actions tab: the same "the check fired;
there was no reader" shape ci_failure_alert.yml exists to close for CI,
confirmed by reading that workflow and its own test completely before this
one was written.

THE ONE THING THIS FILE CHECKS THAT test_ci_failure_alert_workflow.py NEVER
NEEDED TO
------------------------------------------------------------------------
CI has one failure shape. source-monitor has two -- exit code 1 (a genuine
finding, review required) and exit code 2 (the monitor could not qualify its
evidence) -- and GitHub's own success/failure conclusion cannot tell them
apart. Distinguishing them requires the uploaded report.json, which
workflow_dispatch has no real run to fetch: a naive port of
ci_failure_alert.yml's dispatch would only ever exercise the "no report
found" fallback. simulate_exit_code exists so both branches can be exercised
on demand, and this file pins that they are.

WHAT THIS FILE CANNOT DO
------------------------
It cannot prove GitHub will deliver the event, or that actions/download-
artifact actually retrieves the right run's artifact in practice. It checks
the contract this repository controls: the trigger, the branch filter, the
permissions, the idempotence, and that the exit-code distinction and its
simulation path are genuinely wired, not merely described in a comment.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml   # HARD dependency, matching test_ci_failure_alert_workflow.py's
               # own note: pinned in requirements.txt, imported at runtime by
               # two source modules. DELIBERATELY not pytest.importorskip --
               # a module-level importorskip collapses every test in this
               # file into ONE skip entry when the package is absent, which
               # is exactly how the graph-neural-network branch went
               # untested for 508 CI runs (tests/EXPECTED_SUITE_SIZE, roadmap
               # 6.17). PyYAML missing is a broken environment, not an
               # optional extra.

WORKFLOW = (pathlib.Path(__file__).resolve().parents[2]
            / ".github" / "workflows" / "source_monitor_alert.yml")


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert WORKFLOW.is_file(), f"{WORKFLOW} is missing"
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _triggers(workflow: dict) -> dict:
    """PyYAML parses the bare key `on` as the BOOLEAN True, because YAML 1.1
    treats on/off/yes/no as booleans. Reading workflow["on"] therefore raises
    KeyError on a perfectly valid file -- the same trap
    test_ci_failure_alert_workflow.py already names."""
    return workflow[True] if True in workflow else workflow["on"]


# --------------------------------------------------------------------------- #
# 1. the trigger
# --------------------------------------------------------------------------- #
def test_the_workflow_file_exists(workflow):
    assert workflow["name"] == "Source monitor alert"


def test_it_fires_on_completion_of_the_source_monitor_workflow(workflow):
    run = _triggers(workflow)["workflow_run"]
    assert run["workflows"] == ["source-monitor"], (
        "must name the source-monitor workflow's DISPLAY name exactly -- "
        "confirmed 2026-09-16 as 'source-monitor', hyphenated, distinct "
        "from the underscored filename. A mismatch here silently never "
        "triggers, the same failure mode this file's sibling test guards "
        "against for CI.")
    assert "completed" in run["types"]


def test_it_can_be_exercised_by_hand(workflow):
    """A guard that has never fired is a guard nobody knows works."""
    assert "workflow_dispatch" in _triggers(workflow)


def test_the_manual_exercise_defaults_to_dry_run(workflow):
    """Exercising the alert must not create noise, or nobody will exercise it."""
    inputs = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert inputs["dry_run"]["default"] == "true"


def test_the_manual_exercise_can_simulate_either_kind_of_failure(workflow):
    """The one thing ci_failure_alert.yml never needed: CI has a single
    failure shape, source-monitor has two, and workflow_dispatch has no real
    run to download a report from. Without this, only the "no report found"
    fallback could ever be exercised by hand."""
    inputs = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert inputs["simulate_exit_code"]["default"] == "none"
    assert set(inputs["simulate_exit_code"]["options"]) == {"none", "1", "2"}


# --------------------------------------------------------------------------- #
# 2. scope and permissions
# --------------------------------------------------------------------------- #
def test_it_only_acts_on_main(raw):
    """A red dev branch is not a red release."""
    assert "github.event.workflow_run.head_branch == 'main'" in raw


def test_a_manual_dispatch_bypasses_the_branch_filter(raw):
    assert "github.event_name == 'workflow_dispatch' ||" in raw


def test_it_requests_exactly_the_permissions_it_needs(workflow):
    """contents: read and issues: write, matching ci_failure_alert.yml,
    PLUS actions: read -- the one permission that workflow needs and this
    one does not, because only this one downloads a cross-run artifact."""
    assert workflow["permissions"] == {
        "contents": "read", "issues": "write", "actions": "read"}


def test_it_serialises_so_two_results_cannot_race(workflow):
    assert workflow["concurrency"]["group"] == "source-monitor-alert"
    assert workflow["concurrency"]["cancel-in-progress"] is False


# --------------------------------------------------------------------------- #
# 3. the cross-run artifact download -- the one thing genuinely new here
# --------------------------------------------------------------------------- #
def test_it_downloads_from_the_triggering_run_not_its_own(raw):
    """run-id + github-token are what make a CROSS-RUN download possible at
    all. Without them, actions/download-artifact can only ever see this
    alert workflow's OWN artifacts -- which do not exist -- never the
    triggering run's."""
    assert "run-id: ${{ github.event.workflow_run.id }}" in raw
    assert "github-token: ${{ secrets.GITHUB_TOKEN }}" in raw


def test_the_download_step_only_runs_for_a_real_triggering_run(raw):
    """workflow_dispatch has no workflow_run.id to download from at all;
    attempting the download unconditionally would fail every manual
    exercise of this workflow for a reason unrelated to what is being
    tested."""
    assert "if: github.event_name == 'workflow_run'" in raw


def test_a_failed_download_does_not_crash_the_alert(raw):
    """The report might genuinely be missing -- an artifact expired, or a
    run failed before the upload step ever ran. The alert must still say
    SOMETHING rather than crash with no issue opened at all."""
    assert "continue-on-error: true" in raw


# --------------------------------------------------------------------------- #
# 4. the exit-code distinction -- what this workflow exists to add
# --------------------------------------------------------------------------- #
def test_exit_code_one_is_read_as_review_required(raw):
    assert "exitCode === 1" in raw
    assert "REVIEW REQUIRED" in raw


def test_exit_code_two_is_read_as_the_monitor_could_not_qualify_evidence(raw):
    assert "exitCode === 2" in raw
    assert "MONITOR COULD NOT QUALIFY EVIDENCE" in raw


def test_an_unrecognised_exit_code_says_so_rather_than_guessing(raw):
    """A THIRD exit code this alert has never seen must not be silently
    forced into one of the two known shapes -- that would be a diagnosis
    this alert has no basis for making."""
    assert "does not yet recognise that value" in raw


def test_a_missing_report_says_so_rather_than_inventing_a_kind(raw):
    assert "report.json could not be found or read" in raw


def test_the_simulation_inputs_exercise_the_same_branches_a_real_run_would(raw):
    """simulate_exit_code must feed the SAME exitCode variable a real
    downloaded report would populate -- not a separate, parallel code path
    that could drift from what real failures actually exercise."""
    assert "if (simulate === '1')" in raw
    assert "if (simulate === '2')" in raw
    assert "exit_code: 1" in raw
    assert "exit_code: 2" in raw


# --------------------------------------------------------------------------- #
# 5. behaviour that keeps the issue list honest
# --------------------------------------------------------------------------- #
def test_it_reuses_an_open_issue_instead_of_opening_another(raw):
    assert "listForRepo" in raw
    assert "createComment" in raw


def test_it_comments_but_does_not_auto_close_when_source_monitor_goes_clean(raw):
    """MEASURED 2026-09-17, from an external ruling's own execution of this
    exact script: the success branch closed an open issue WITHOUT ever
    reading the report. GitHub's own conclusion and the report's exit_code
    are two different signals; nothing here cross-checked them -- so a
    workflow-level 'success' with a missing or contradictory report could
    silently close a genuine review item. Per the ruling's own recommended
    immediate, safe change, automatic closure is suspended: a clean
    conclusion now comments for visibility only. state: 'closed' must not
    appear anywhere in this file any longer."""
    assert "conclusion === 'success'" in raw
    assert "state: 'closed'" not in raw
    assert "does NOT close the issue automatically" in raw


def test_simulation_cannot_write_to_a_real_issue_even_with_dry_run_false(raw):
    """MEASURED 2026-09-17: simulate_exit_code and dry_run were independent
    inputs. Dispatching with simulate_exit_code=1 and dry_run=false posted
    FABRICATED content to a real production issue -- confirmed by executing
    this script's logic with a mocked issues.create and observing it get
    called with the simulated body. Simulation must now force preview mode
    in code, not merely by a default a caller can override."""
    assert "const dryRun = simulate !== 'none' ||" in raw


def test_a_non_array_results_field_does_not_crash_the_script(raw):
    """MEASURED 2026-09-17: `for (const r of (report.results || []))` threw
    an uncaught TypeError when report.results was a non-array truthy value
    -- confirmed by executing this exact line with such a value. An
    uncaught exception here means NO comment, NO issue update, and a failed
    job -- silently losing the alert for exactly the malformed-report case
    most in need of one."""
    assert "Array.isArray(report.results)" in raw
    assert "Array.isArray(report.unqualified)" in raw
    assert "malformedShape" in raw


def test_a_null_element_inside_results_does_not_crash_the_script(raw):
    """MEASURED 2026-09-17, from a second external ruling's own execution
    of this exact script AFTER the first four fixes: report.results:
    [null] still threw an uncaught TypeError. The outer-array check
    validated that results WAS an array; it never validated what was
    INSIDE it. Confirmed directly, then fixed: every element is now
    checked for being a non-null object before anything reads its fields."""
    assert "r === null || typeof r !== 'object'" in raw


def test_a_non_array_findings_on_one_result_does_not_crash_the_script(raw):
    """MEASURED 2026-09-17, the second half of the same gap: a result
    object with findings: {a: 1} -- a well-formed element containing a
    malformed findings field -- also threw an uncaught TypeError,
    confirmed directly by executing that exact shape."""
    assert "r.findings !== undefined && !Array.isArray(r.findings)" in raw


def test_exit_two_surfaces_an_independently_qualified_witness_separately(raw):
    """MEASURED 2026-09-17, a fourth external ruling: HC preserves a
    genuine witness on an INCOMPLETE (exit 2) result, but nothing here
    ever rendered it -- confirmed by running this exact branch against
    HC's own decisive scenario and finding the witness text nowhere in
    the output. Read from report.qualification[target].
    positive_witnesses, never from results[].findings, which can carry
    an unsupported producer claim kept deliberately for diagnosis."""
    assert "scientificReview" in raw
    assert "positive_witnesses" in raw
    assert "### Scientific review" in raw


def test_the_scientific_review_section_is_absent_when_there_is_no_witness(raw):
    """A section that always appears, even empty, would be as misleading
    as one that never appears -- confirmed the generated body omits it
    entirely when positive_witnesses is empty for every target."""
    assert "scientificLines.length" in raw


def test_a_cancelled_run_changes_nothing(raw):
    assert "neither success nor failure" in raw


def test_a_manual_dispatch_exercises_the_failure_branch(raw):
    """An alert tested only on its success branch is untested. With no
    workflow_run payload the script must treat the situation as a FAILURE."""
    assert "const conclusion = run ? run.conclusion : 'failure';" in raw


# --------------------------------------------------------------------------- #
# 6. the alert says what to check first
# --------------------------------------------------------------------------- #
def test_the_issue_body_explains_why_a_red_run_might_not_mean_broken(raw):
    assert "GitHub's own" in raw
    assert "success/failure conclusion cannot tell them apart" in raw


def test_the_workflow_records_why_it_exists(raw):
    """The motivating incident, in the file, so a future reader does not
    have to find this test to learn why the workflow was added."""
    assert "4.1.1" in raw
    assert "there was no" in raw.lower() or "NO READER" in raw or "no reader" in raw
