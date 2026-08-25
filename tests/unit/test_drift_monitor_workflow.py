"""The monthly workflow may not pretend to perform a drift assessment.

Created 2026-08-24.

WHAT THIS GUARDS
----------------
Until 2026-08-24 the `feature-drift` job invoked
`scripts/run_drift_monitor.py` with `--features-only` and NO `--new-data` and no
`--new-clinvar`. MEASURED against `run_drift_monitor.py:313-351`, that
combination takes the else branch and returns `EXIT_NOT_CHECKED`. The monthly
cron therefore ran a monitor that MUST return 4, by construction, every month.

The workflow's own comment named the earlier form of the same defect: "THIS STEP
HAD NEVER ONCE RUN A DRIFT COMPUTATION", and before its 2026-07-13 repair it
emitted `drift_level=none` -- a clean bill of health for a check that never
happened, which also meant the notify job, gated on `!= 'none'`, "HAS NEVER RUN.
NOT ONCE."

The 2026-07-13 repair made the outcome honest. This unit removes the pretence.
These cases keep both properties.

PARSED, NEVER GREPPED
---------------------
MEASURED 2026-08-24: `drift_monitor.yml` contained EIGHT occurrences of
`python scripts/run_drift_monitor.py` and only THREE `python scripts/*.py`
invocations were executed at all -- the others were `echo` guidance, a shell
comment, and a YAML comment. A test that greps would judge all eight; text
occurrence is not semantic execution.

So every assertion below walks the parsed document. `yaml` is a hard dependency
already: pyyaml==6.0.3 is in requirements.txt and
`tests/unit/test_ci_failure_alert_workflow.py` and
`tests/unit/test_workflow_action_pins.py` both import it at module scope.

Author: Monzia Moodie
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "drift_monitor.yml"
READINESS_ADAPTER = "scripts/report_drift_readiness.py"
ASSESSMENT_SCRIPT = "scripts/run_drift_monitor.py"
FEATURE_DRIFT_JOB = "feature-drift"

#: The standing, recorded capability gap. A NEW not-checked reason is an
#: incident; this one is a fact already written down.
KNOWN_GAP = "candidate_discovery_not_implemented"

#: Values the workflow must never author for itself. Each is projected from one
#: record by the readiness adapter; a workflow that assigns one becomes a second
#: author of semantic state, which is how `feature_drift_checked=false` beside
#: `drift_level=none` became constructible.
ADAPTER_OWNED = (
    "readiness_status",
    "feature_drift_checked",
    "not_checked_reason",
    "drift_level",
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{WORKFLOW} did not parse to a mapping"
    return document


def _executed_commands(step: dict) -> tuple:
    """Lines of a `run:` block that actually execute.

    An `echo`-ed command is guidance printed to a log. A `#` line is a comment.
    Neither runs, and counting them is the defect this helper exists to avoid.
    """
    run = step.get("run")
    if not run:
        return ()

    executed = []
    for line in run.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # A COMPOUND COMMAND HIDES BEHIND ITS FIRST WORD. An earlier draft
        # skipped any line BEGINNING with `echo`, so
        # `echo "x"; drift_level=none >> "$GITHUB_OUTPUT"` was classified as
        # guidance and everything after the semicolon became invisible. That
        # is the same "text occurrence is not semantic execution" error as
        # counting echoed guidance -- in the opposite direction, under-counting
        # instead of over-counting. Found by sabotage: the mutation reported
        # NOTHING FAILED, and the guard was fine.
        for segment in re.split(r";|&&|\|\|", stripped):
            segment = segment.strip()
            if not segment or segment.startswith("#"):
                continue
            # An `echo` SEGMENT prints; it does not run what it prints.
            if re.match(r"echo\b|echo[\"']", segment):
                continue
            executed.append(segment)
    return tuple(executed)


def _job_steps(workflow: dict, job: str) -> list:
    assert job in workflow["jobs"], (
        f"job {job!r} is absent; the workflow has {sorted(workflow['jobs'])}")
    return workflow["jobs"][job].get("steps", [])


# ---------------------------------------------------------------------------
# 1. THE MONTHLY JOB DOES NOT PERFORM AN ASSESSMENT
# ---------------------------------------------------------------------------

def test_the_monthly_job_does_not_invoke_the_drift_monitor(workflow):
    """The invocation that could not produce a verdict is gone.

    Not weakened, not made conditional -- absent. A step that invokes an
    assessment it cannot complete is the thing being removed.
    """
    offenders = []
    for step in _job_steps(workflow, FEATURE_DRIFT_JOB):
        for command in _executed_commands(step):
            if ASSESSMENT_SCRIPT in command:
                offenders.append((step.get("name"), command))

    assert not offenders, (
        f"the {FEATURE_DRIFT_JOB} job executes {ASSESSMENT_SCRIPT}:\n"
        + "\n".join(f"  {name}: {cmd}" for name, cmd in offenders)
        + "\n\nWithout --new-data or --new-clinvar that returns "
          "EXIT_NOT_CHECKED by construction, every month."
    )


def test_the_monthly_job_invokes_the_readiness_adapter(workflow):
    """Removing the invocation is only half the repair.

    A job that reports nothing is as useless as one that reports falsely. The
    readiness command is what makes the absence a stated fact.
    """
    invoked = [
        command
        for step in _job_steps(workflow, FEATURE_DRIFT_JOB)
        for command in _executed_commands(step)
        if READINESS_ADAPTER in command
    ]
    assert invoked, (
        f"no step in {FEATURE_DRIFT_JOB} executes {READINESS_ADAPTER}. The job "
        "would emit no readiness at all."
    )


def test_the_assessment_script_survives_untouched():
    """The monitor is not deleted. It is not invoked HERE.

    `EXIT_NOT_CHECKED = 4` has three negative controls in
    tests/unit/test_invariant_ownership.py and its semantics remain correct for
    the assessment command that will run when a candidate population exists.
    """
    script = REPO / ASSESSMENT_SCRIPT
    assert script.is_file(), (
        f"{ASSESSMENT_SCRIPT} is absent. This unit retires an INVOCATION, "
        "never the assessment itself."
    )
    assert "EXIT_NOT_CHECKED = 4" in script.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. THE WORKFLOW AUTHORS NO SEMANTIC STATE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ADAPTER_OWNED, ids=ADAPTER_OWNED)
def test_the_workflow_does_not_author_a_readiness_field(workflow, field):
    """Every output is projected from one record, never assigned in YAML.

    Otherwise a later "simplification" can delete the command and put the
    literals back, and the four fields become four authors again.
    """
    offenders = []
    for job, definition in workflow["jobs"].items():
        for step in definition.get("steps", []):
            for command in _executed_commands(step):
                if re.search(rf"\b{field}=", command):
                    offenders.append((job, step.get("name"), command))

    assert not offenders, (
        f"{field} is assigned by the workflow itself:\n"
        + "\n".join(f"  {j}/{n}: {c}" for j, n, c in offenders)
        + f"\n\nIt must come from {READINESS_ADAPTER}, which derives all four "
          "from one record."
    )


def test_the_job_declares_the_readiness_outputs(workflow):
    declared = workflow["jobs"][FEATURE_DRIFT_JOB].get("outputs", {})
    for field in ADAPTER_OWNED:
        assert field in declared, (
            f"the {FEATURE_DRIFT_JOB} job does not declare {field!r}; "
            f"it declares {sorted(declared)}"
        )


def test_every_declared_output_has_a_step_that_can_produce_it(workflow):
    """A declared output nothing sets renders as an empty string.

    MEASURED 2026-08-24: `exit_code` remained declared and consumed by the
    notify script after the readiness adapter replaced the step that set it. It
    would have rendered as a blank field in every issue -- silent, and wrong.

    THE PREDICATE IS THE PRODUCING STEP, NOT A LITERAL IN THE YAML. An earlier
    draft of this test searched the workflow text for `<field>=` and therefore
    contradicted `test_the_workflow_does_not_author_a_readiness_field`
    directly: one demanded the literal be absent, the other demanded it be
    present. Both cannot hold, and running them is what showed it.

    An output is produced when the step it names EXISTS and writes to
    $GITHUB_OUTPUT. What it writes there is the adapter's business.
    """
    declared = workflow["jobs"][FEATURE_DRIFT_JOB].get("outputs", {})
    steps = _job_steps(workflow, FEATURE_DRIFT_JOB)
    by_id = {step["id"]: step for step in steps if "id" in step}

    assert declared, f"the {FEATURE_DRIFT_JOB} job declares no outputs"

    for field, expression in declared.items():
        source = re.search(r"steps\.(\w+)\.outputs\.(\w+)", str(expression))
        assert source, (
            f"output {field!r} is not projected from a step: {expression!r}")

        step_id = source.group(1)
        assert step_id in by_id, (
            f"output {field!r} reads steps.{step_id}.outputs."
            f"{source.group(2)}, but no step in {FEATURE_DRIFT_JOB} has "
            f"id {step_id!r}. It would render as an empty string. "
            f"Steps with ids: {sorted(by_id)}"
        )

        commands = _executed_commands(by_id[step_id])
        assert any("GITHUB_OUTPUT" in command for command in commands), (
            f"step {step_id!r} never writes to $GITHUB_OUTPUT, so output "
            f"{field!r} cannot be produced. Its executed commands are "
            f"{list(commands)}."
        )


# ---------------------------------------------------------------------------
# 3. A KNOWN GAP IS NOT A MONTHLY INCIDENT
# ---------------------------------------------------------------------------

def test_the_known_capability_gap_does_not_open_a_monthly_issue(workflow):
    """Replacing a permanent false-green with a permanent noisy-yellow is not
    a repair. Alert fatigue is the next defect."""
    condition = str(workflow["jobs"]["notify"].get("if", ""))
    assert KNOWN_GAP in condition, (
        "the notify condition does not mention the known capability gap. "
        "Since readiness now emits UNKNOWN every run, a condition of "
        "`drift_level != 'none'` files an issue every month."
    )
    assert "not_checked_reason" in condition


def test_a_new_not_checked_reason_still_notifies(workflow):
    """Suppressing the standing gap must not suppress every refusal.

    A reason that is not the recorded one is an incident: something changed.
    """
    condition = str(workflow["jobs"]["notify"].get("if", ""))
    assert "!=" in condition and KNOWN_GAP in condition, (
        "the condition must EXCLUDE the known gap, not exclude all "
        "not-checked states."
    )
    assert "feature_drift_checked" in condition


def test_a_measured_drift_still_notifies(workflow):
    condition = str(workflow["jobs"]["notify"].get("if", ""))
    assert "drift_level != 'none'" in condition.replace('"', "'"), (
        "a measured, non-none drift level no longer triggers notification."
    )


# ---------------------------------------------------------------------------
# 4. THE ALERT BODY RENDERS ITS CAUSE, IT DOES NOT RESTATE ONE
# ---------------------------------------------------------------------------

def _notify_script(workflow: dict) -> str:
    for step in workflow["jobs"]["notify"]["steps"]:
        script = (step.get("with") or {}).get("script")
        if script:
            return script
    raise AssertionError("the notify job has no github-script step")


def _live_lines(script: str) -> tuple:
    return tuple(line for line in script.split("\n")
                 if not line.strip().startswith("//"))


def test_the_alert_body_names_no_deleted_step(workflow):
    """DRIFT-ALERT-BODY-STALE-1.

    The body stated that the ROOT CAUSE was a "Download reference splits from
    Google Drive" step that was "still a PLACEHOLDER". That step was removed;
    the reference profile is committed and verified earlier in this workflow.
    The alert named a cause repaired weeks before, in the one place a human
    reads.
    """
    live = "\n".join(_live_lines(_notify_script(workflow)))
    for stale in ("Google Drive", "real download NOT wired yet",
                  "reference splits were absent"):
        assert stale not in live, (
            f"the alert body still states {stale!r} as a live cause. "
            "Prose that restates a cause goes stale; prose that renders one "
            "cannot."
        )


def test_the_alert_body_reads_no_output_the_job_does_not_declare(workflow):
    declared = set(workflow["jobs"][FEATURE_DRIFT_JOB].get("outputs", {}))
    live = "\n".join(_live_lines(_notify_script(workflow)))

    read = set(re.findall(
        r"needs\.feature-drift\.outputs\.(\w+)", live))
    undeclared = sorted(read - declared)
    assert not undeclared, (
        f"the alert body reads {undeclared}, which the job does not declare. "
        "Each would render as an empty string."
    )


def test_the_alert_body_derives_its_explanation_from_the_typed_reason(workflow):
    live = "\n".join(_live_lines(_notify_script(workflow)))
    assert "NOT_CHECKED_EXPLANATIONS" in live, (
        "the alert body does not render its explanation from the reason code."
    )
    assert KNOWN_GAP in live


def test_severity_is_read_from_the_severity_not_from_an_exit_code(workflow):
    """MEASURED 2026-08-24: the body compared `exitCode >= 3` -- a STRING
    against a number. With an empty string every comparison is false, so every
    measured drift would have been described as "Minor"."""
    live = "\n".join(_live_lines(_notify_script(workflow)))
    assert "exitCode" not in live
    assert "driftLevel === 'severe'" in live
    assert "driftLevel === 'significant'" in live
