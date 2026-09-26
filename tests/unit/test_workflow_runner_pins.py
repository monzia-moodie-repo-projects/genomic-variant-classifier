"""Every workflow job runs on a PINNED runner image (2026-09-26, owner decision).

MEASURED 2026-09-26: 13 of 14 jobs across 7 workflows used `ubuntu-latest`, and GitHub
annotated every run: "The ubuntu-latest label will migrate to Ubuntu 26 beginning October
19, 2026." A floating label changes Python, Git and system libraries under every required
check with no commit in this repository -- unrecorded environment drift that would read as
a regression. All jobs are pinned to `ubuntu-24.04`, the image the checks passed on (one job,
teardown_abort_diagnostic.yml, already was). Moving to Ubuntu 26 is a separate, tested change.

Author: Monzia Moodie
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = sorted((_ROOT / ".github" / "workflows").glob("*.yml"))
PINNED_RUNNER = "ubuntu-24.04"


def _jobs():
    for path in _WORKFLOWS:
        with open(path, encoding="utf-8") as fh:
            workflow = yaml.safe_load(fh)
        for job_id, job in workflow["jobs"].items():
            yield path.name, job_id, job.get("runs-on")


def test_there_are_workflows_and_jobs_to_check():
    """Guard against a vacuous pass: a moved or renamed directory would yield no jobs."""
    assert len(_WORKFLOWS) >= 7 and sum(1 for _ in _jobs()) >= 14


@pytest.mark.parametrize("workflow,job,runs_on", list(_jobs()), ids=lambda v: str(v))
def test_every_job_runs_on_the_pinned_image(workflow, job, runs_on):
    assert runs_on == PINNED_RUNNER, (
        f"{workflow}:{job} runs on {runs_on!r}; pin it to {PINNED_RUNNER!r}. A floating label "
        f"(e.g. ubuntu-latest) changes the environment under every check without a commit.")
