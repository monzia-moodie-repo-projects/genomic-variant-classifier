#!/usr/bin/env python3
"""Harden test_run17_postflight_paths.py's repo-root resolution: parents[1] -> ancestor-walk.

Matches tests/conftest.py's own convention (walk ancestors for the dir containing scripts/,
do NOT hardcode depth) so the test survives being located at tests/ root, scripts/, or deeper.
Anchor-based + idempotent: validates the exact parents[1] line occurs once, replaces it with a
_find_repo_root() helper, leaves everything else byte-identical.

Usage: python patch_test_repo_resolution.py <path-to-test_run17_postflight_paths.py>
"""
from __future__ import annotations
import sys
from pathlib import Path

ANCHOR = "REPO = Path(__file__).resolve().parents[1]"
SENTINEL = "_find_repo_root"  # idempotency marker

REPLACEMENT = '''def _find_repo_root(start: Path) -> Path:
    # Ancestor-walk for the dir containing scripts/launch_run17_baseline.sh (mirrors
    # tests/conftest.py: walk ancestors, do not hardcode depth -- survives tests/ moves).
    here = start.resolve()
    for anc in (here.parent, *here.parents):
        if (anc / "scripts" / "launch_run17_baseline.sh").is_file():
            return anc
    raise RuntimeError(f"repo root (dir with scripts/launch_run17_baseline.sh) not found from {start}")


REPO = _find_repo_root(Path(__file__))'''


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python patch_test_repo_resolution.py <test_file.py>")
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
        print("ALREADY PATCHED (sentinel _find_repo_root present); no change.")
        return 0
    n = text.count(ANCHOR)
    if n != 1:
        print(f"ANCHOR FAILED: '{ANCHOR}' occurs {n}x (expected 1); no change.")
        return 1

    new = text.replace(ANCHOR, REPLACEMENT, 1)
    # Backup then write (LF, BOM-free)
    p.with_suffix(p.suffix + ".bak").write_bytes(raw)
    p.write_text(new, encoding="utf-8", newline="")
    print(f"PATCHED: {p} (parents[1] -> ancestor-walk). Backup: {p.name}.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
