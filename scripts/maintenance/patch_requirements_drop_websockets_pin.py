#!/usr/bin/env python3
"""Disable the hard websockets==16.0 pin in requirements.txt.

The pin conflicts with the langgraph subtree (langchain -> langgraph ->
langgraph-sdk -> websockets), producing pip ResolutionImpossible on a clean
install. websockets is a transitive-only dependency and should not carry a hard
top-level pin; commenting it out lets the resolver pick a version compatible
with langgraph-sdk (and any other consumer). Empirically validated: the probe
bootstrap dropped this exact pin and the VM env resolved + installed cleanly.

Commenting (not deleting) keeps provenance and is trivially reversible.

One anchor, count-guarded (must match exactly once), backup-first, idempotent,
line-ending preserving. NOTE: requirements.txt is not Python, so there is no
AST check; instead we assert the line count is unchanged (comment-in-place).

Usage:  python patch_requirements_drop_websockets_pin.py [path/to/requirements.txt]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

DEFAULT = "requirements.txt"

OLD = "websockets==16.0"
NEW = (
    "# websockets==16.0  # disabled 2026-06-08: hard pin conflicts with the "
    "langgraph-sdk transitive websockets requirement (pip ResolutionImpossible "
    "on a clean install). websockets is transitive-only; the resolver selects a "
    "compatible version."
)
MARKER = "# disabled 2026-06-08: hard pin conflicts with the langgraph-sdk"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT)
    if not path.exists():
        print(f"FATAL: {path} not found (run from repo root or pass the path).")
        return 2

    raw = path.read_bytes()
    original = raw.decode("utf-8")

    if MARKER in original:
        print("Already patched (marker present); no changes made.")
        return 0

    # Count-guard on the STANDALONE pin line (not a substring, so a comment that
    # merely mentions websockets==16.0 elsewhere does not trip it).
    lines = original.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if ln.rstrip("\r\n") == OLD]
    if len(hits) != 1:
        print(f"FATAL: standalone pin line matched {len(hits)} times (expected exactly 1): {OLD!r}")
        return 3
    idx = hits[0]
    eol = lines[idx][len(lines[idx].rstrip("\r\n")):]  # preserve that line's EOL
    lines[idx] = NEW + eol
    text = "".join(lines)

    # Sanity: same number of lines (comment in place, nothing added/removed).
    if len(text.splitlines()) != len(original.splitlines()):
        print("FATAL: line count changed; aborting.")
        return 5

    backup = path.with_suffix(path.suffix + f".bak_ws_{time.strftime('%Y%m%d_%H%M%S')}")
    backup.write_bytes(raw)
    path.write_bytes(text.encode("utf-8"))
    print(f"Patched {path}")
    print(f"  backup: {backup.name}")
    print("  1 pin commented out; line count unchanged; line endings preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
