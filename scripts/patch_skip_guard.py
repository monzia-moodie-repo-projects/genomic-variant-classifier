#!/usr/bin/env python3
"""Fix the pwsh-test skip guard in tests/unit/test_run17_postflight_paths.py.

CI #485 (f195d09) surfaced: GitHub's ubuntu runner SHIPS pwsh, so `shutil.which("pwsh")`
finds it, `PWSH` is not None, and `skipif(PWSH is None)` did NOT fire. The Windows-targeted
.ps1 then ran under Linux pwsh, errored to stderr, produced empty stdout, and the 3
test_dryrun_derives_expected_paths content-assertions failed on ''. (test_invalid_config_rejected
"passed" by luck -- it checks returncode != 0, which a Linux failure also satisfies.)

Fix (3 anchored edits):
  1. add `import sys`.
  2. both `@pytest.mark.skipif(PWSH is None, ...)` -> `@pytest.mark.skipif(sys.platform != "win32", ...)`.
     The .ps1 is Windows-targeted (Windows paths, Test-ArtifactPresent dot-source); its path-
     DERIVATION is a Windows behavior, meaningful only on Windows. Linux/CI -> skip. The pure-Python
     launcher cross-check (the drift guard) is unaffected and keeps running in CI.
  3. harden _run_dryrun: capture stderr and, on nonzero exit / empty stdout, raise with the stderr
     so a real Windows breakage shows the PowerShell error instead of an opaque empty-string mismatch.

Anchor-based + idempotent (sentinel: the new skipif text). BOM-free/LF.
Usage: python patch_skip_guard.py <tests/unit/test_run17_postflight_paths.py>
"""
from __future__ import annotations
import sys
from pathlib import Path

# --- anchors (verbatim from the committed file) ---
A_IMPORT = "import shutil\nimport subprocess\nfrom pathlib import Path"
R_IMPORT = "import shutil\nimport subprocess\nimport sys\nfrom pathlib import Path"

A_SKIP1 = '@pytest.mark.skipif(PWSH is None, reason="pwsh/powershell not available")\n@pytest.mark.parametrize("config,stem", CONFIG_STEMS.items())\ndef test_dryrun_derives_expected_paths(config, stem):'
R_SKIP1 = '@pytest.mark.skipif(sys.platform != "win32", reason="Run17_Postflight.ps1 is Windows-targeted; path-derivation is validated on the Windows preflight host (CI/Linux runs the pure-Python launcher cross-check instead)")\n@pytest.mark.parametrize("config,stem", CONFIG_STEMS.items())\ndef test_dryrun_derives_expected_paths(config, stem):'

A_SKIP2 = '@pytest.mark.skipif(PWSH is None, reason="pwsh/powershell not available")\ndef test_invalid_config_rejected():'
R_SKIP2 = '@pytest.mark.skipif(sys.platform != "win32", reason="Run17_Postflight.ps1 is Windows-targeted; ValidateSet rejection is validated on the Windows preflight host")\ndef test_invalid_config_rejected():'

A_RUN = '''def _run_dryrun(config: str) -> str:
    cmd = [
        PWSH, "-NoProfile", "-File", str(POSTFLIGHT),
        "-Config", config,
        "-SshHost", "dummy", "-SshPort", "22",
        "-InstanceId", "00000000", "-HourlyRate", "0.5",
        "-DryRun",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return res.stdout'''
R_RUN = '''def _run_dryrun(config: str) -> str:
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
            f"(exit {res.returncode}). stderr:\\n{res.stderr}"
        )
    return res.stdout'''

SENTINEL = 'skipif(sys.platform != "win32"'


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python patch_skip_guard.py <test_file.py>")
        return 2
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"ERROR: {p} not found")
        return 2
    raw = p.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        print("ERROR: file has a UTF-8 BOM; expected BOM-free.")
        return 2
    text = raw.decode("utf-8")

    if SENTINEL in text:
        print("ALREADY PATCHED (platform-gate sentinel present); no change.")
        return 0

    edits = [("import sys", A_IMPORT, R_IMPORT),
             ("skipif #1 (dryrun)", A_SKIP1, R_SKIP1),
             ("skipif #2 (invalid)", A_SKIP2, R_SKIP2),
             ("_run_dryrun hardening", A_RUN, R_RUN)]
    for name, a, _ in edits:
        n = text.count(a)
        if n != 1:
            print(f"ANCHOR FAILED [{name}]: occurs {n}x (expected 1); no change.")
            return 1

    for _name, a, r in edits:
        text = text.replace(a, r, 1)

    # safety: PWSH symbol still used inside _run_dryrun/test bodies; ensure we didn't strip its def
    if "PWSH = shutil.which" not in text:
        print("SAFETY: PWSH definition vanished; aborting.")
        return 1

    p.with_suffix(p.suffix + ".bak").write_bytes(raw)
    p.write_text(text, encoding="utf-8", newline="")
    print(f"PATCHED: {p} (platform-gate skip + stderr-surfacing _run_dryrun).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
