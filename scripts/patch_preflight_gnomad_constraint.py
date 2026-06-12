#!/usr/bin/env python3
"""patch_preflight_gnomad_constraint.py -- add a fifth Run-16 preflight check for the
gnomAD-constraint TSV, the data behind gene_constraint_oe (Run-15 #2 feature).

Without --gnomad-constraint, train.py leaves gnomad_constraint_path=None -> stub mode
-> loeuf constant -> gene_constraint_oe deadzones. This gate fails loud if the TSV is
missing/too small. Edits scripts/preflight_run16_inputs.py (check fn + arg + results)
and appends tests to tests/unit/test_preflight_run16_inputs.py.

Idempotent, py_compile/ast-gated, newline-preserving, ASCII. Author: Monzia Moodie."""
from __future__ import annotations

import ast
import py_compile
import shutil
import sys
from pathlib import Path

PRE = Path("scripts/preflight_run16_inputs.py")
TST = Path("tests/unit/test_preflight_run16_inputs.py")
PRE_MARKER = "def check_gnomad_constraint("
TST_MARKER = "test_gnomad_constraint_present_passes"

CHECK_FN = (
    'def check_gnomad_constraint(path, min_mb=1.0):\n'
    '    if path is None:\n'
    '        return False, "gnomad constraint: FAIL (not supplied -- gene_constraint_oe would deadzone via stub mode)"\n'
    '    p = Path(path)\n'
    '    if not p.exists():\n'
    '        return False, f"gnomad constraint: FAIL (not found: {path} -- stub mode -> gene_constraint_oe deadzones)"\n'
    '    mb = p.stat().st_size / 1e6\n'
    '    ok = mb >= min_mb\n'
    '    return ok, (f"gnomad constraint: {\'PASS\' if ok else \'FAIL\'} "\n'
    '                f"({path}, {mb:.1f} MB, need >= {min_mb} MB)")\n'
)

PRE_EDITS = [
    # 1. insert the check fn before def aggregate
    (
        '    return ok, f"feature count: {\'PASS\' if ok else \'FAIL\'} (EXPECTED_TABULAR_FEATURE_COUNT={C}, want {EXPECTED_COUNT})"\n'
        '\n\ndef aggregate(results):\n',
        '    return ok, f"feature count: {\'PASS\' if ok else \'FAIL\'} (EXPECTED_TABULAR_FEATURE_COUNT={C}, want {EXPECTED_COUNT})"\n'
        '\n\n' + CHECK_FN + '\n\ndef aggregate(results):\n',
    ),
    # 2. add the argparse flag (default = the real .tsv path) before args = ap.parse_args
    (
        '    ap.add_argument("--alphamissense", default=None, help="AlphaMissense scores parquet (required)")\n'
        '    args = ap.parse_args(argv)\n',
        '    ap.add_argument("--alphamissense", default=None, help="AlphaMissense scores parquet (required)")\n'
        '    ap.add_argument("--gnomad-constraint",\n'
        '                    default="data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv",\n'
        '                    help="gnomAD v4.1 constraint TSV (revives gene_constraint_oe via loeuf)")\n'
        '    args = ap.parse_args(argv)\n',
    ),
    # 3. add to the results list
    (
        '        check_exists("alphamissense", args.alphamissense),\n'
        '        check_feature_count(),\n'
        '    ]\n',
        '        check_exists("alphamissense", args.alphamissense),\n'
        '        check_gnomad_constraint(args.gnomad_constraint),\n'
        '        check_feature_count(),\n'
        '    ]\n',
    ),
]

TST_APPEND = '''

def test_gnomad_constraint_present_passes(tmp_path):
    p = tmp_path / "constraint.tsv"
    p.write_bytes(b"x" * 2_000_000)  # 2 MB
    ok, _ = pf.check_gnomad_constraint(str(p))
    assert ok is True


def test_gnomad_constraint_missing_fails():
    ok, _ = pf.check_gnomad_constraint("nope_constraint_12345.tsv")
    assert ok is False


def test_gnomad_constraint_stub_too_small_fails(tmp_path):
    p = tmp_path / "stub.tsv"
    p.write_bytes(b"x" * 1000)  # 1 KB -> below min
    ok, _ = pf.check_gnomad_constraint(str(p))
    assert ok is False


def test_gnomad_constraint_none_fails():
    ok, _ = pf.check_gnomad_constraint(None)
    assert ok is False
'''


def _read(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def _write(p, text):
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def _edit_preflight():
    raw = _read(PRE)
    if PRE_MARKER in raw:
        return "preflight: already patched"
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    for i, (old, new) in enumerate(PRE_EDITS, 1):
        if text.count(old) != 1:
            raise SystemExit(f"ERROR: preflight edit {i}: anchor count {text.count(old)} != 1")
        text = text.replace(old, new, 1)
    shutil.copy2(PRE, PRE.with_suffix(PRE.suffix + ".bak"))
    _write(PRE, text.replace("\n", nl))
    py_compile.compile(str(PRE), doraise=True)
    return "preflight: patched (+check_gnomad_constraint, +arg, +results); py_compile OK"


def _append_tests():
    raw = _read(TST)
    if TST_MARKER in raw:
        return "tests: already appended"
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n").rstrip("\n") + "\n" + TST_APPEND
    ast.parse(text)
    shutil.copy2(TST, TST.with_suffix(TST.suffix + ".bak"))
    _write(TST, text.replace("\n", nl))
    return "tests: appended 4 gnomad-constraint cases; ast OK"


def main():
    if not PRE.exists() or not TST.exists():
        print("ERROR: run from repo root (preflight/test not found).")
        return 2
    print(_edit_preflight())
    print(_append_tests())
    return 0


if __name__ == "__main__":
    sys.exit(main())
