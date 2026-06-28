#!/usr/bin/env python3
"""Update Run_Preflight_Local.ps1 pytest floors to the measured post-move suite.

Post-move (test_run17_postflight_paths.py relocated scripts/ -> tests/, now collected):
  collected = 1498, local passed = 1491 / 7 skipped, CI passed ~= 1487 (4 pwsh tests skip
  without PowerShell). Floors keyed to the CI case (lower passed) so the gate holds in BOTH
  local (pwsh present) and CI (pwsh absent):
    MinPytest (collected floor) 1485 -> 1496   (measured 1498, 2 headroom)
    minPass   (passed floor)    1480 -> 1485   (CI passed ~1487, 2 headroom; local 1491 >= 1485)

Anchor-based + idempotent (sentinel = the new provenance date). BOM-free/LF preserved.
Usage: python patch_preflight_floor.py <Run_Preflight_Local.ps1>
"""
from __future__ import annotations
import sys
from pathlib import Path

A_MIN = "    [int]$MinPytest       = 1485,"
R_MIN = "    [int]$MinPytest       = 1496,"

A_PASS = "        $minPass = 1480  # Run-17: 1483 passed / 2 skipped / 1485 collected (2026-06-27); 3-test headroom below 1483"
R_PASS = "        $minPass = 1485  # Run-17 post-move: 1498 collected / 1491 passed / 7 skipped local (2026-06-28); CI passes ~1487 (4 pwsh tests skip w/o PowerShell) -> floor keyed to CI case"

SENTINEL = "2026-06-28); CI passes ~1487"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python patch_preflight_floor.py <Run_Preflight_Local.ps1>")
        return 2
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"ERROR: {p} not found")
        return 2
    raw = p.read_bytes()
    bom = raw[:3] == b"\xef\xbb\xbf"
    text = raw.decode("utf-8-sig" if bom else "utf-8")

    if SENTINEL in text:
        print("ALREADY PATCHED (sentinel present); no change.")
        return 0

    problems = []
    for a in (A_MIN, A_PASS):
        n = text.count(a)
        if n != 1:
            problems.append(f"  anchor x{n} (expected 1): {a.strip()[:60]}")
    if problems:
        print("ANCHOR FAILED -- no change:")
        print("\n".join(problems))
        return 1

    text = text.replace(A_MIN, R_MIN, 1).replace(A_PASS, R_PASS, 1)
    p.with_suffix(p.suffix + ".bak").write_bytes(raw)
    # preserve original BOM-ness (preflight files in this repo are BOM-free per convention)
    data = text.encode("utf-8")
    p.write_bytes(data)
    print(f"PATCHED: {p} (MinPytest 1485->1496, minPass 1480->1485). Backup: {p.name}.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
