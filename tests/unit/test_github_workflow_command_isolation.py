"""No test may publish to the running GitHub workflow (added 2026-09-27).

MEASURED on CI run #896 (merge of #28): a TEST published a fabricated "NOT VERIFIED" verdict about a real run onto
the pytest jobs' summaries, through $GITHUB_STEP_SUMMARY. tests/conftest.py now removes the five workflow-command
variables around every test. Checking "absent inside a test" alone would be VACUOUS off CI (they are never set
locally), so the parent test launches a CHILD pytest WITH the variables pointing at real files, and requires the
child's probe to find them absent and the files to stay EMPTY.

Author: Monzia Moodie
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_NAMES = ("GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT", "GITHUB_ENV", "GITHUB_PATH", "GITHUB_STATE")
_CHILD = "GVC_WORKFLOW_ISOLATION_CHILD"
#: Captured at MODULE IMPORT (collection time), BEFORE any fixture runs -- what a module publishing at import would see.
_AT_IMPORT = {name: os.environ.get(name) for name in _NAMES}


def test_probe_workflow_command_variables_are_absent_inside_a_test():
    """Runs in EVERY run (no skip): in CI the variables really are set around the suite, so this is a live check;
    inside the child below they point at real files, so it is a live check off CI too."""
    assert _AT_IMPORT == {name: None for name in _NAMES}          # layer B: gone before this module was even imported
    for name in _NAMES:
        assert name not in os.environ, name
        target = os.environ.get(name)          # what a careless publisher would do
        if target:
            Path(target).write_text("LEAKED\n", encoding="utf-8")


def test_no_test_can_publish_to_the_workflow(tmp_path):
    files = {name: tmp_path / name.lower() for name in _NAMES}
    for f in files.values():
        f.write_text("", encoding="utf-8")
    env = dict(os.environ, **{name: str(f) for name, f in files.items()}, **{_CHILD: "1"})
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/unit/test_github_workflow_command_isolation.py::test_probe_workflow_command_variables_are_absent_inside_a_test"],
                       cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "1 passed" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    assert all(f.read_text(encoding="utf-8") == "" for f in files.values())


def test_every_ci_pytest_invocation_removes_the_publication_variables_before_the_interpreter_starts():
    """LAYER C: ci.yml runs each pytest under `env -u` for all five variables (owner ruling 2026-09-28)."""
    import yaml
    root = Path(__file__).resolve().parents[2]
    with open(root / ".github" / "workflows" / "ci.yml", encoding="utf-8") as fh:
        wf = yaml.safe_load(fh)
    invocations = [line for job in wf["jobs"].values() for step in job.get("steps", [])
                   for line in (step.get("run") or "").splitlines() if "pytest tests" in line and not line.lstrip().startswith("#")]
    assert len(invocations) == 2, invocations
    prefix = "env -u GITHUB_STEP_SUMMARY -u GITHUB_OUTPUT -u GITHUB_ENV -u GITHUB_PATH -u GITHUB_STATE"
    runs = [step.get("run") or "" for job in wf["jobs"].values() for step in job.get("steps", [])]
    assert sum(r.count(prefix) for r in runs) == 2
