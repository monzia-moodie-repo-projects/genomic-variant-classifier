#!/usr/bin/env python3
r"""patch_test_dbsnp_path.py

Fix the bare-relative module path in tests/unit/test_dbsnp_pop_preference.py.

BUG: line 2 does spec_from_file_location("b", "build_dbsnp_parquet.py") -- a bare CWD-relative
path. pytest runs from repo root where that file does NOT exist (it is at scripts/), so collection
raises FileNotFoundError and ABORTS THE ENTIRE SUITE (collection error interrupts all tests).

FIX: anchor to repo root via __file__, matching every other test's convention
(_BUILDER = Path(__file__).resolve().parents[2] / "scripts" / "build_dbsnp_parquet.py").

This is the LONE offender: every other test in the suite already anchors to __file__ / a _BUILDER
path; only this one used a bare filename. Verified via grep of module_from_spec/spec_from_file_location.

ANCHOR-BASED, IDEMPOTENT.
"""
from __future__ import annotations
import argparse
from pathlib import Path

TARGET = Path("tests/unit/test_dbsnp_pop_preference.py")

OLD = '''import importlib.util
spec = importlib.util.spec_from_file_location("b", "build_dbsnp_parquet.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)'''

NEW = '''import importlib.util, pathlib
_BUILDER = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_dbsnp_parquet.py"
spec = importlib.util.spec_from_file_location("b", _BUILDER)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)'''

MARKER = 'parents[2] / "scripts" / "build_dbsnp_parquet.py"'


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("OK (idempotent): test already anchors build_dbsnp_parquet.py to repo root."); return 0
    c = src.count(OLD)
    if c != 1:
        print(f"FAIL: anchor occurs {c}x (need 1). The test header may differ from expected."); return 3
    if ns.check:
        print("CHECK: anchor found once."); print("RESULT: PASS (check)"); return 0
    backup = TARGET.with_suffix(".py.pre_dbsnp_path.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")
    TARGET.write_text(src.replace(OLD, NEW, 1), encoding="utf-8", newline="\n")
    after = TARGET.read_text(encoding="utf-8")
    ok = MARKER in after and 'spec_from_file_location("b", _BUILDER)' in after
    # also compile-check
    import ast
    try:
        ast.parse(after); compiles = True
    except SyntaxError as e:
        compiles = False; print("  SYNTAX ERROR:", e)
    print(f"  {'OK' if ok else 'MISSING'}  path anchored to repo root")
    print(f"  {'OK' if compiles else 'FAIL'}  test still compiles")
    print("RESULT:", "PASS" if (ok and compiles) else "FAIL")
    return 0 if (ok and compiles) else 5


if __name__ == "__main__":
    raise SystemExit(main())
