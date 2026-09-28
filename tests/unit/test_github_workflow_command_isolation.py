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


def test_probe_workflow_command_variables_are_absent_inside_a_test():
    """Runs in EVERY run (no skip): in CI the variables really are set around the suite, so this is a live check;
    inside the child below they point at real files, so it is a live check off CI too."""
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
