#!/usr/bin/env python3
"""Correct the preflight floor provenance comment: it wrongly referenced CI.

The $minPass floor gates the PREFLIGHT, which runs the FULL tests/ suite locally
(1498 collected / 1491 passed). CI is a separate job that runs `pytest tests/unit/`
with --maxfail=5 and NO count-floor -- it does not read this value. The old comment
"CI passes ~1487 ... floor keyed to CI case" was misleading on both points. Rewrite to
reference the preflight full-suite run (what the floor actually gates), keeping the
pwsh-skip note (still relevant for any pwsh-less full-suite run).

Floor VALUES unchanged (1496 / 1485). Anchor-based + idempotent. BOM-free/LF.
Usage: python patch_floor_comment.py <Run_Preflight_Local.ps1>
"""
from __future__ import annotations
import sys
from pathlib import Path

ANCHOR = "        $minPass = 1485  # Run-17 post-move: 1498 collected / 1491 passed / 7 skipped local (2026-06-28); CI passes ~1487 (4 pwsh tests skip w/o PowerShell) -> floor keyed to CI case"
REPLACEMENT = "        $minPass = 1485  # Run-17 post-move (2026-06-28): preflight runs FULL tests/ locally = 1498 collected / 1491 passed / 7 skipped; floor gates THIS run. 2-pass headroom below 1491 (lower if any pwsh-dependent test skips on a pwsh-less full-suite host). NB: CI is a separate job (pytest tests/unit/ --maxfail=5, no count-floor) and does not read this value."

SENTINEL = "preflight runs FULL tests/ locally"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python patch_floor_comment.py <Run_Preflight_Local.ps1>")
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
    n = text.count(ANCHOR)
    if n != 1:
        print(f"ANCHOR FAILED: occurs {n}x (expected 1); no change.")
        return 1
    # Guard: must NOT alter the floor VALUE -- both old and new start with the same $minPass = 1485
    if "$minPass = 1485" not in REPLACEMENT:
        print("SAFETY: replacement would change the floor value; aborting.")
        return 1

    new = text.replace(ANCHOR, REPLACEMENT, 1)
    p.with_suffix(p.suffix + ".bak").write_bytes(raw)
    p.write_bytes(new.encode("utf-8"))
    print(f"PATCHED: {p} (floor-comment CI reference corrected; value 1485 unchanged).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
