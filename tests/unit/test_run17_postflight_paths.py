"""Exhaustive path-derivation + launcher-drift tests for Run17_Postflight.ps1.

Two independent guarantees:

1. PATH DERIVATION (exhaustive): for each of the 3 configs, invoke the postflight in
   -DryRun mode (no SSH/SCP) and assert all 4 derived paths exactly match the expected
   single-stem derivation. 3 configs x 4 paths = 12 assertions, total coverage of the
   closed config set. Also asserts an invalid -Config is rejected (ValidateSet).

2. LAUNCHER CROSS-CHECK (drift detection): reads the OUTDIR= stem from each of the three
   committed launchers (launch_run17_baseline.sh / _r12only.sh / _r13only.sh) and asserts
   the postflight's $ConfigPaths stems MATCH. If a launcher's OUTDIR ever changes without
   the postflight following, this test goes RED -- mechanical drift detection.

Skips gracefully if pwsh is unavailable (CI runs the launcher cross-check regardless,
which is pure-Python file parsing and needs no PowerShell).
"""
from __future__ import annotations
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

def _find_repo_root(start: Path) -> Path:
    # Ancestor-walk for the dir containing scripts/launch_run17_baseline.sh (mirrors
    # tests/conftest.py: walk ancestors, do not hardcode depth -- survives tests/ moves).
    here = start.resolve()
    for anc in (here.parent, *here.parents):
        if (anc / "scripts" / "launch_run17_baseline.sh").is_file():
            return anc
    raise RuntimeError(f"repo root (dir with scripts/launch_run17_baseline.sh) not found from {start}")


REPO = _find_repo_root(Path(__file__))
SCRIPTS = REPO / "scripts"
POSTFLIGHT = SCRIPTS / "Run17_Postflight.ps1"

CONFIG_STEMS = {
    "both": "run17_baseline",
    "r12only": "run17_r12only",
    "r13only": "run17_r13only",
}

# Map each config to the launcher whose OUTDIR= stem must agree with it.
CONFIG_LAUNCHER = {
    "both": "launch_run17_baseline.sh",
    "r12only": "launch_run17_r12only.sh",
    "r13only": "launch_run17_r13only.sh",
}

PWSH = shutil.which("pwsh") or shutil.which("powershell")


def _expected_paths(stem: str) -> dict[str, str]:
    return {
        "RemoteLog": f"/workspace/{stem}_master.log",
        "RemoteOutputs": f"/workspace/genomic-variant-classifier/outputs/{stem}/full",
        "RemoteReport": f"/workspace/{stem}_report",
        "LocalReport_suffix": f"{stem}_report",  # tail-match (RepoRoot varies)
    }


# ----------------------------------------------------------------------------
# 1. LAUNCHER CROSS-CHECK (pure Python; runs everywhere, no pwsh needed)
# ----------------------------------------------------------------------------
def _launcher_outdir_stem(launcher_path: Path) -> str:
    """Extract the run17_* stem from a launcher's OUTDIR= line.
    OUTDIR="$REPO/outputs/run17_r12only/full" -> 'run17_r12only'."""
    text = launcher_path.read_text(encoding="utf-8")
    m = re.search(r'OUTDIR="\$REPO/outputs/(run17_[a-z0-9_]+)/full"', text)
    assert m, f"no OUTDIR= run17 stem found in {launcher_path.name}"
    return m.group(1)


@pytest.mark.parametrize("config,stem", CONFIG_STEMS.items())
def test_postflight_stem_matches_launcher_outdir(config, stem):
    """The postflight's config->stem must match the launcher's OUTDIR stem (drift guard)."""
    launcher = SCRIPTS / CONFIG_LAUNCHER[config]
    if not launcher.exists():
        pytest.skip(f"{launcher.name} not present")
    launcher_stem = _launcher_outdir_stem(launcher)
    assert launcher_stem == stem, (
        f"DRIFT: postflight maps {config} -> {stem} but "
        f"{launcher.name} OUTDIR stem is {launcher_stem}"
    )


def test_postflight_configpaths_table_is_exactly_the_closed_set():
    """The postflight must declare exactly the 3 known configs (closed whitelist)."""
    text = POSTFLIGHT.read_text(encoding="utf-8")
    # ValidateSet must list exactly both/r12only/r13only
    vs = re.search(r"ValidateSet\(([^)]*)\)", text)
    assert vs, "no ValidateSet on -Config"
    listed = set(re.findall(r"'([a-z0-9]+)'", vs.group(1)))
    assert listed == set(CONFIG_STEMS), f"ValidateSet {listed} != {set(CONFIG_STEMS)}"
    # $ConfigPaths hashtable must map each config to its expected stem
    for config, stem in CONFIG_STEMS.items():
        assert re.search(rf"'{config}'\s*=\s*'{stem}'", text), (
            f"$ConfigPaths missing {config} -> {stem}"
        )


# ----------------------------------------------------------------------------
# 2. EXHAUSTIVE PATH DERIVATION via -DryRun (requires pwsh)
# ----------------------------------------------------------------------------
def _run_dryrun(config: str) -> str:
    cmd = [
        PWSH, "-NoProfile", "-File", str(POSTFLIGHT),
        "-Config", config,
        "-SshHost", "dummy", "-SshPort", "22",
        "-InstanceId", "00000000", "-HourlyRate", "0.5",
        "-DryRun",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    # Surface a real failure instead of letting an empty stdout fail a content assertion
    # with no context (CI #485: a Windows-targeted .ps1 erroring under a non-Windows pwsh
    # produced empty stdout). On Windows (where these tests run) this makes a genuine
    # PowerShell breakage show the actual error.
    if res.returncode != 0 or not res.stdout.strip():
        raise AssertionError(
            f"-DryRun produced no usable output for config={config} "
            f"(exit {res.returncode}). stderr:\n{res.stderr}"
        )
    return res.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Run17_Postflight.ps1 is Windows-targeted; path-derivation is validated on the Windows preflight host (CI/Linux runs the pure-Python launcher cross-check instead)")
@pytest.mark.parametrize("config,stem", CONFIG_STEMS.items())
def test_dryrun_derives_expected_paths(config, stem):
    out = _run_dryrun(config)
    exp = _expected_paths(stem)
    assert f"RemoteLog     = {exp['RemoteLog']}" in out, f"{config}: RemoteLog wrong\n{out}"
    assert f"RemoteOutputs = {exp['RemoteOutputs']}" in out, f"{config}: RemoteOutputs wrong\n{out}"
    assert f"RemoteReport  = {exp['RemoteReport']}" in out, f"{config}: RemoteReport wrong\n{out}"
    assert exp["LocalReport_suffix"] in out, f"{config}: LocalReport suffix missing\n{out}"
    # DryRun must NOT have done any teardown / must exit cleanly
    assert "No SSH/SCP/destroy performed" in out, f"{config}: DryRun didn't short-circuit\n{out}"


@pytest.mark.skipif(sys.platform != "win32", reason="Run17_Postflight.ps1 is Windows-targeted; ValidateSet rejection is validated on the Windows preflight host")
def test_invalid_config_rejected():
    cmd = [
        PWSH, "-NoProfile", "-File", str(POSTFLIGHT),
        "-Config", "bogus",
        "-SshHost", "dummy", "-SshPort", "22",
        "-InstanceId", "0", "-HourlyRate", "0.5", "-DryRun",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    # ValidateSet rejection -> nonzero exit + error mentioning the param
    assert res.returncode != 0, "invalid -Config should be rejected"
