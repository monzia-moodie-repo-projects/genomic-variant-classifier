#!/usr/bin/env python3
"""Fix Run_Preflight_VM.sh (G2): repoint stale launch_run11_vm.sh references to
launch_run15_baseline.sh -- in particular the final actionable 'Proceed to:'
echo, which otherwise sends the operator to the Run-11 (dirty-cohort, no
--min-review-tier) launcher and silently undoes Path A. Count-guarded,
idempotent, line-ending agnostic."""
from __future__ import annotations
import shutil, sys
from pathlib import Path

OLD = "launch_run11_vm.sh"
NEW = "launch_run15_baseline.sh"

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    data = raw.replace("\r\n", "\n")
    n = data.count(OLD)
    if n == 0:
        print(f"SKIP: no '{OLD}' references in {path} (idempotent)"); return 0
    out = data.replace(OLD, NEW)
    if out.count(OLD) != 0:
        print("ABORT: residual OLD references remain"); return 2
    final = out.replace("\n", nl) if nl == "\r\n" else out
    shutil.copy2(path, path.with_suffix(path.suffix + ".launcherref.bak"))
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"patched {path}: {n} '{OLD}' -> '{NEW}'; endings={'CRLF' if nl==chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "scripts/Run_Preflight_VM.sh"))
