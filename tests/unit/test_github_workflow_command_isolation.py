"""No test may publish to the running GitHub workflow (added 2026-09-27).

MEASURED on CI run #896 (merge of #28): a TEST published a fabricated "NOT VERIFIED" verdict about a real run onto
the pytest jobs' summaries, through $GITHUB_STEP_SUMMARY. tests/conftest.py now removes the workflow-command
variables (its _GITHUB_WORKFLOW_COMMAND_FILES -- the ONE source, parsed here, never duplicated) around every test. Checking "absent inside a test" alone would be VACUOUS off CI (they are never set
locally), so the parent test launches a CHILD pytest WITH the variables pointing at real files, and requires the
child's probe to find them absent and the files to stay EMPTY.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _conftest_set():
    """tests/conftest.py's _GITHUB_WORKFLOW_COMMAND_FILES, read by PARSING the file (a conftest is never imported)."""
    tree = ast.parse((_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "_GITHUB_WORKFLOW_COMMAND_FILES" for t in n.targets)]
    assert len(values) == 1, values
    return values[0]


_NAMES = _conftest_set()
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
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/unit/test_github_workflow_command_isolation.py::test_probe_workflow_command_variables_are_absent_inside_a_test"],
                       cwd=_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "1 passed" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    assert all(f.read_text(encoding="utf-8") == "" for f in files.values())


def test_every_ci_pytest_invocation_removes_the_publication_variables_before_the_interpreter_starts():
    """LAYER C: each ci.yml pytest invocation runs under `env -u` for EXACTLY the conftest set (owner ruling 2026-09-28)."""
    import yaml
    with open(_ROOT / ".github" / "workflows" / "ci.yml", encoding="utf-8") as fh:
        wf = yaml.safe_load(fh)
    code = [line for job in wf["jobs"].values() for step in job.get("steps", [])
            for line in (step.get("run") or "").splitlines() if not line.lstrip().startswith("#")]
    assert len([line for line in code if "pytest tests" in line]) == 2
    removals = [set(re.findall(r"-u (\S+)", line)) for line in code if line.lstrip().startswith("env -u ")]
    assert removals == [set(_NAMES), set(_NAMES)], removals


def test_the_removed_set_is_every_documented_publication_file():
    """GitHub's documented WRITABLE workflow-command files, pinned (2026-09-29): dropping one fails here even if every
    copy drifted together. GITHUB_ARTIFACTS_LIST is read-only metadata and is deliberately not removed."""
    assert set(_NAMES) == {"GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT", "GITHUB_ENV", "GITHUB_PATH", "GITHUB_STATE",
                           "GITHUB_ARTIFACTS"}
    assert "GITHUB_ARTIFACTS_LIST" not in _NAMES
