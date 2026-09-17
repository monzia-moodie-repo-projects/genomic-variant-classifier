"""Extracts and executes source_monitor_alert.yml's own embedded script,
via a real Node.js subprocess with mocked github/fs/core interfaces.

WHY THIS EXISTS
===============
MEASURED 2026-09-17, from a fifth external ruling: WD's own regression
tests assert that particular strings occur in the workflow's YAML source.
They do not execute the rendering behavior. The functional verification
that actually proved WD correct -- extracting the script and running it
in Node against controlled doubles -- was done by hand, once, and never
became a repeatable test. This module is that harness, made permanent and
pointed at the file this repository actually ships, not a hand-copied
duplicate of it.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

WORKFLOW = (pathlib.Path(__file__).resolve().parents[2]
            / ".github" / "workflows" / "source_monitor_alert.yml")

NODE = shutil.which("node")

#: A Node.js driver, mocking exactly the four interfaces the extracted
#: script actually uses: fs (real, scoped to a temp cwd), core.info
#: (captured), context (supplied), github.rest.issues.* (mocked and
#: RECORDED, never performed for real).
_DRIVER = r"""
const fs = require('fs');
const path = require('path');

const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
process.chdir(input.cwd);
for (const [k, v] of Object.entries(input.env)) process.env[k] = v;

const infoLines = [];
const core = { info: (m) => infoLines.push(String(m)) };

const productionWrites = [];
const github = { rest: { issues: {
  listForRepo: async () => ({ data: input.openIssues || [] }),
  create: async (args) => { productionWrites.push({ op: 'create', args }); return { data: { number: 1 } }; },
  createComment: async (args) => { productionWrites.push({ op: 'createComment', args }); },
  update: async (args) => { productionWrites.push({ op: 'update', args }); },
} } };

const context = input.context;

async function main() {
  const fn = new Function('require', 'fs', 'core', 'context', 'github',
    '"use strict"; return (async () => {' + input.script + '})()');
  await fn(require, fs, core, context, github);
  process.stdout.write(JSON.stringify({
    info_lines: infoLines, production_writes: productionWrites,
  }));
}

main().catch((e) => {
  process.stderr.write(JSON.stringify({ error: e.message, stack: e.stack }));
  process.exit(1);
});
"""


class ExtractionError(Exception):
    """The embedded script could not be located or extracted."""


def extract_script(workflow_path: pathlib.Path | None = None) -> str:
    """Pull the github-script `script:` block out of the workflow file
    verbatim, de-indented. Reads the ACTUAL file this repository ships;
    a hand-copied duplicate would recreate the fixture-divergence problem
    this harness exists to close.

    workflow_path defaults to the MODULE-LEVEL WORKFLOW global, read at
    CALL time, not bound as a default-parameter value: a default
    parameter is evaluated once, at function-definition time, so
    monkeypatching module.WORKFLOW afterward -- exactly what tests need
    to point this at a fixture file -- would silently have no effect."""
    if workflow_path is None:
        workflow_path = WORKFLOW
    if not workflow_path.is_file():
        raise ExtractionError("{} does not exist".format(workflow_path))
    lines = workflow_path.read_text(encoding="utf-8").split("\n")
    start = end = None
    for i, line in enumerate(lines):
        if line.strip() == "script: |":
            start = i + 1
        if start is not None and line.strip().startswith("core.info(`Opened #"):
            end = i + 2
            break
    if start is None or end is None:
        raise ExtractionError(
            "could not locate the script: | block (or its final line) in "
            "{}".format(workflow_path))
    body = lines[start:end]
    dedented = [l[12:] if l.startswith(" " * 12) else l for l in body]
    return "\n".join(dedented)


def run_alert_script(
    *, tmp_path: pathlib.Path, env: dict, context: dict,
    report: dict | None = None, open_issues: list | None = None,
    workflow_path: pathlib.Path | None = None,
) -> dict:
    """Execute the ACTUAL extracted script in a real Node subprocess.

    report: if given, written to <tmp_path>/source-monitor-report/report.json
    before the script runs, exactly where the download-artifact step would
    have placed it.

    Returns {"info_lines": [...], "production_writes": [...]}.
    production_writes records every attempted issues.create / createComment
    / update call; an empty list is the harness's own proof that dry-run
    genuinely attempted no production mutation, not merely that the test
    author intended it not to.
    """
    if NODE is None:
        raise RuntimeError("node is not on PATH")
    script = extract_script(workflow_path)
    cwd = tmp_path / "run"
    cwd.mkdir(exist_ok=True)
    if report is not None:
        report_dir = cwd / "source-monitor-report"
        report_dir.mkdir(exist_ok=True)
        (report_dir / "report.json").write_text(json.dumps(report))

    driver_path = tmp_path / "driver.js"
    driver_path.write_text(_DRIVER)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({
        "cwd": str(cwd), "env": env, "context": context, "script": script,
        "openIssues": open_issues or [],
    }))

    proc = subprocess.run(
        [NODE, str(driver_path), str(input_path)],
        capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(
            "alert script execution failed: {}".format(proc.stderr))
    return json.loads(proc.stdout)


def rendered_body(result: dict) -> str:
    """The issue body the script would have posted or previewed, joined
    from its own core.info() log lines -- excluding the diagnostic
    event=... line and the open-issue lookup line, matching exactly what
    a human would see in the actual GitHub issue."""
    return "\n".join(
        l for l in result["info_lines"]
        if not l.startswith("event=") and not l.startswith("open ")
        and not l.startswith("[dry run] would"))
