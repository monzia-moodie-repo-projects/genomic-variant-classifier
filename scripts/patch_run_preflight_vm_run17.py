#!/usr/bin/env python3
"""
patch_run_preflight_vm_run17.py  --  Monzia Moodie

scripts/Run_Preflight_VM.sh still points at Run 15: its PRINTED success line says
"Proceed to: bash scripts/launch_run15_baseline.sh" and the failure line names the Run-15 launcher --
active misdirection toward the wrong launcher for a Run-17 provisioning. The env/hardware checks are
run-agnostic and unchanged; this only retargets the launcher references and the header. Exact-string,
idempotent, EOL/BOM-safe. Run from repo root.
"""
from __future__ import annotations
import sys
from pathlib import Path

TARGET = Path("scripts/Run_Preflight_VM.sh")
TOKEN_OLD = "launch_run15_baseline.sh"
TOKEN_NEW = "launch_run17_baseline.sh"
HEADER_OLD = "On-VM ENVIRONMENT/HARDWARE pre-flight for Run 15 (Charter v1.1 gate G2)."
HEADER_NEW = "On-VM ENVIRONMENT/HARDWARE pre-flight for Run 17 (env checks are run-agnostic; Charter v1.1 gate G2)."


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root)", file=sys.stderr); return 2
    raw = TARGET.read_bytes()
    crlf = raw.count(b"\r\n"); lf = raw.count(b"\n") - crlf
    eol = "\r\n" if crlf >= lf else "\n"
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if TOKEN_NEW in text and TOKEN_OLD not in text:
        print("[skip] Run_Preflight_VM.sh already retargeted to launch_run17_baseline.sh"); return 0
    n = text.count(TOKEN_OLD)
    if n == 0:
        print("ERROR: no launch_run15_baseline.sh references found; not patching", file=sys.stderr); return 3
    text = text.replace(TOKEN_OLD, TOKEN_NEW)
    if HEADER_OLD in text:
        text = text.replace(HEADER_OLD, HEADER_NEW)
    data = text.replace("\n", eol).encode("utf-8")
    if had_bom:
        data = b"\xef\xbb\xbf" + data
    TARGET.write_bytes(data)
    print(f"[patched] Run_Preflight_VM.sh: {n} launcher ref(s) -> {TOKEN_NEW}; header retargeted "
          f"(eol={'CRLF' if eol!=chr(10) else 'LF'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
