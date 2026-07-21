"""The CI failure alert workflow, checked without waiting for a real failure.

WHY THIS FILE EXISTS
====================
On 2026-07-21 at 12:58 the suite-size ratchet did exactly what it was built to
do. Commit d7c4d35 conflated two changes, left `tests/EXPECTED_SUITE_SIZE` at
2417 while the tree collected 2446, and Continuous Integration run #562 aborted
at COLLECTION time with `pytest.UsageError` -- exit code 4, three minutes
fourteen seconds against the usual fourteen, because no test ever ran.

main stayed red for roughly two hours and nobody looked.

The detection worked perfectly. The notification loop did not exist.

A guard that has never fired is a guard nobody knows works, which is the exact
failure this repository keeps finding: `assert_data_usable` was well tested and
called from nowhere. So the alert workflow is not merely added -- its wiring is
PARSED AND PINNED here, and it carries a `workflow_dispatch` trigger so the
whole path can be exercised on demand.

A NOTE ON THE YAML IMPORT
-------------------------
PyYAML is imported DIRECTLY, not via `pytest.importorskip`. A module-level
importorskip collapses every test here into a single skip entry if the package
is absent, and the suite-size ratchet would see 14 tests vanish with no
explanation. PyYAML is pinned in requirements.txt and imported at runtime by two
source modules; its absence is a broken environment and must fail loudly.

WHAT THIS FILE CANNOT DO
------------------------
It cannot prove GitHub will deliver the event. It checks the contract this
repository controls: the trigger, the branch filter, the permissions, the
idempotence, and the fact that a manual dispatch exercises the FAILURE branch
rather than the trivially-safe success branch.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml   # HARD dependency: pyyaml==6.0.3 in requirements.txt, and imported
               # at runtime by data/pipeline.py and utils/helpers.py.
               #
               # DELIBERATELY *NOT* pytest.importorskip. A module-level
               # importorskip collapses every test in the file into ONE skip
               # entry when the package is absent -- which is exactly how the
               # graph-neural-network branch went untested for 508 Continuous
               # Integration runs (see the note in tests/EXPECTED_SUITE_SIZE and
               # roadmap 6.17). PyYAML missing is a broken environment, not an
               # optional extra, and it must fail loudly rather than quietly
               # subtract 15 tests from the ratchet's view of the suite.

WORKFLOW = (pathlib.Path(__file__).resolve().parents[2]
            / ".github" / "workflows" / "ci_failure_alert.yml")


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
    KeyError on a perfectly valid file -- a trap worth naming rather than
    rediscovering."""
    return workflow[True] if True in workflow else workflow["on"]


# --------------------------------------------------------------------------- #
# 1. the trigger
# --------------------------------------------------------------------------- #
def test_the_workflow_file_exists(workflow):
    assert workflow["name"] == "CI failure alert"


def test_it_fires_on_completion_of_the_ci_workflow(workflow):
    run = _triggers(workflow)["workflow_run"]
    assert run["workflows"] == ["CI"], (
        "must name the CI workflow exactly; a renamed or mistyped workflow "
        "silently never triggers")
    assert "completed" in run["types"]


def test_it_can_be_exercised_by_hand(workflow):
    """A guard that has never fired is a guard nobody knows works."""
    assert "workflow_dispatch" in _triggers(workflow)


def test_the_manual_exercise_defaults_to_dry_run(workflow):
    """Exercising the alert must not create noise, or nobody will exercise it."""
    inputs = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert inputs["dry_run"]["default"] == "true"


# --------------------------------------------------------------------------- #
# 2. scope and permissions
# --------------------------------------------------------------------------- #
def test_it_only_acts_on_main(raw):
    """A red dev branch is not a red release."""
    assert "github.event.workflow_run.head_branch == 'main'" in raw


def test_a_manual_dispatch_bypasses_the_branch_filter(raw):
    """Otherwise the path could never be exercised from anywhere but a real
    main failure -- which is the situation this workflow exists to avoid."""
    assert "github.event_name == 'workflow_dispatch' ||" in raw


def test_it_requests_exactly_the_permissions_it_needs(workflow):
    """contents: read and issues: write. Anything broader is an unnecessary
    token scope on a workflow that runs automatically."""
    assert workflow["permissions"] == {"contents": "read", "issues": "write"}


def test_it_serialises_so_two_results_cannot_race(workflow):
    """Two CI results landing together must not open two issues."""
    assert workflow["concurrency"]["group"] == "ci-failure-alert"
    assert workflow["concurrency"]["cancel-in-progress"] is False


# --------------------------------------------------------------------------- #
# 3. behaviour that keeps the issue list honest
# --------------------------------------------------------------------------- #
def test_it_reuses_an_open_issue_instead_of_opening_another(raw):
    """A pile of identical alerts is noise, and noise is unread."""
    assert "listForRepo" in raw
    assert "createComment" in raw


def test_it_closes_the_issue_when_ci_goes_green(raw):
    """An alert that only ever opens leaves a list that no longer describes
    reality."""
    assert "state: 'closed'" in raw
    assert "conclusion === 'success'" in raw


def test_a_cancelled_run_changes_nothing(raw):
    """'The run did not finish' is not 'main is broken', and it is not 'main is
    fine' either. Reporting it as either is how a monitoring system lies -- the
    same lesson the drift monitor learned when UNKNOWN was reported as none."""
    assert "neither success nor failure" in raw


def test_a_manual_dispatch_exercises_the_failure_branch(raw):
    """An alert tested only on its success branch is untested. With no
    workflow_run payload the script must treat the situation as a FAILURE."""
    assert "const conclusion = run ? run.conclusion : 'failure';" in raw


# --------------------------------------------------------------------------- #
# 4. the alert says what to check first
# --------------------------------------------------------------------------- #
def test_the_issue_body_names_the_exit_code_that_means_the_ratchet(raw):
    """Exit code 4 is pytest.UsageError, and on this repository that almost
    always means the suite-size ratchet. Saying so turns a red tick into a
    diagnosis."""
    assert "code 4" in raw
    assert "SUITE-SIZE RATCHET" in raw


def test_the_issue_body_forbids_lowering_the_expected_count(raw):
    """If the count went DOWN, tests have vanished and the reason matters more
    than the colour."""
    assert "Do NOT lower the expected number" in raw


def test_the_workflow_records_why_it_exists(raw):
    """The motivating incident, in the file, so a future reader does not have to
    find this test to learn why the workflow was added."""
    assert "d7c4d35" in raw
    assert "nobody looked" in raw.lower() or "NOBODY LOOKED" in raw
