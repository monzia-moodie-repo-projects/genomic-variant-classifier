#!/usr/bin/env python3
"""
patch_test_launch_run17_bashcheck.py  --  Monzia Moodie

Rewrite ONLY the test_bash_syntax_valid method in tests/unit/test_launch_run17.py to syntax-check
the launcher's CONTENT via `bash -n -c <text>` instead of passing a path to bash. Git-Bash/MSYS on
Windows resolves neither backslash paths (C:\\ -> \\P,\\g read as escapes -> C:Projects...) nor
forward-slash drive paths (C:/... ; it wants /c/... mount form), so every path-based invocation
fails with "No such file or directory". Passing the script text touches no filesystem and is robust
on Windows AND Linux.

Operates on the file already on disk (immune to Downloads filename collisions). Idempotent: if the
method already uses `-c`, it does nothing. EOL/BOM-safe; anchors on the skipif decorator through EOF
(test_bash_syntax_valid is the final block in the file in every shipped version).
"""
from __future__ import annotations
import sys
from pathlib import Path

TARGET = Path("tests/unit/test_launch_run17.py")
ANCHOR = "@pytest.mark.skipif(shutil.which(\"bash\") is None"

NEW_BLOCK = (
    '@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")\n'
    'def test_bash_syntax_valid():\n'
    '    # Syntax-check the CONTENT via `bash -n -c`, never a path. Git-Bash/MSYS on Windows\n'
    '    # resolves neither backslash paths (C:\\\\ -> escape mangling) nor C:/ drive paths\n'
    '    # (it expects /c/... mount form) as argv; passing the script text sidesteps all of it.\n'
    '    content = _LAUNCHER.read_text()\n'
    '    r = subprocess.run(["bash", "-n", "-c", content], capture_output=True, text=True)\n'
    '    assert r.returncode == 0, f"bash -n failed:\\n{r.stderr}"\n'
)


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root)", file=sys.stderr); return 2
    raw = TARGET.read_bytes()
    crlf = raw.count(b"\r\n"); lf = raw.count(b"\n") - crlf
    eol = "\r\n" if crlf >= lf else "\n"
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")

    if '"-c"' in text and "bash" in text:
        print("[skip] test_bash_syntax_valid already uses `bash -n -c` content mode")
        return 0
    n = text.count(ANCHOR)
    if n != 1:
        print(f"ERROR: skipif anchor found {n}x (expected 1); not patching", file=sys.stderr)
        return 3
    head = text[: text.index(ANCHOR)]
    new_text = head + NEW_BLOCK
    if not new_text.endswith("\n"):
        new_text += "\n"
    TARGET.write_bytes(new_text.replace("\n", eol).encode("utf-8"))
    print(f"[patched] test_bash_syntax_valid -> bash -n -c content  (eol={'CRLF' if eol!=chr(10) else 'LF'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
