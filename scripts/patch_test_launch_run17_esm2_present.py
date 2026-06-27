#!/usr/bin/env python3
r"""patch_test_launch_run17_esm2_present.py

Flip test_esm2_consciously_absent -> test_esm2_uniprot_index_wired in tests/unit/test_launch_run17.py.

This session's launch wiring deliberately ADDED --esm2-uniprot-index (HGVSp parser delivered -> ESM-2 now
carries real signal; launch_run17_baseline.sh L202, verified BB4). The test still asserts ESM-2 is ABSENT,
encoding the superseded decision. Flip it to assert PRESENCE, mirroring how the file's other 'wired'
assertions are written (test_required_flags_present style).

ANCHOR-BASED, IDEMPOTENT, LF.
"""
from __future__ import annotations
import argparse, ast
from pathlib import Path

TARGET = Path("tests/unit/test_launch_run17.py")
MARKER = "def test_esm2_uniprot_index_wired"

OLD = '''def test_esm2_consciously_absent(text):
    arg_lines = [ln for ln in text.splitlines()
                 if "ARGS=" in ln and not ln.strip().startswith("#")]
    assert all("--esm2-uniprot-index" not in ln for ln in arg_lines)'''

NEW = '''def test_esm2_uniprot_index_wired(text):
    # ESM-2 is now deliberately wired (HGVSp parser delivered -> ESM-2 carries real signal).
    # launch_run17_baseline.sh appends --esm2-uniprot-index to ARGS; assert its PRESENCE.
    arg_lines = [ln for ln in text.splitlines()
                 if "ARGS=" in ln and not ln.strip().startswith("#")]
    assert any("--esm2-uniprot-index" in ln for ln in arg_lines), \\
        "--esm2-uniprot-index expected in ARGS (ESM-2 wired this session)"'''


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("OK (idempotent): esm2 test already flipped to presence."); return 0
    c = src.count(OLD)
    if c != 1:
        print(f"FAIL: anchor occurs {c}x (need 1)."); return 3
    if ns.check:
        print("CHECK: anchor found once."); print("RESULT: PASS (check)"); return 0

    backup = TARGET.with_suffix(".py.pre_esm2_present.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")
    new = src.replace(OLD, NEW, 1)
    TARGET.write_text(new, encoding="utf-8", newline="\n")

    after = TARGET.read_text(encoding="utf-8")
    checks = {
        "renamed to _wired": "def test_esm2_uniprot_index_wired(text):" in after,
        "asserts presence (any)": 'assert any("--esm2-uniprot-index" in ln' in after,
        "old absent-assert gone": 'assert all("--esm2-uniprot-index" not in ln' not in after,
        "old name gone": "def test_esm2_consciously_absent" not in after,
    }
    try:
        ast.parse(after); checks["compiles"] = True
    except SyntaxError as e:
        checks["compiles"] = False; print("  SYNTAX ERROR:", e)
    for k, v in checks.items():
        print(f"  {'OK' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
