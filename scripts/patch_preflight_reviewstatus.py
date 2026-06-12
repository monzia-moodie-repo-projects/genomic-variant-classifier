#!/usr/bin/env python3
"""patch_preflight_reviewstatus.py -- add a sixth Run-16 preflight check: the cohort
carries a ReviewStatus column.

train.py hardcodes min_review_tier=3 with no CLI override; _load_and_label raises if
the cohort lacks ReviewStatus (tier<5). The smoke caught this after the preflight
passed exit 0 over the same cohort -- this closes that blind spot. Edits
scripts/preflight_run16_inputs.py (check fn + results) and appends tests.

Idempotent, py_compile/ast-gated, newline-preserving, ASCII. Author: Monzia Moodie."""
from __future__ import annotations

import ast
import py_compile
import shutil
import sys
from pathlib import Path

PRE = Path("scripts/preflight_run16_inputs.py")
TST = Path("tests/unit/test_preflight_run16_inputs.py")
PRE_MARKER = "def check_cohort_reviewstatus("
TST_MARKER = "test_reviewstatus_present_passes"

CHECK_FN = (
    'def check_cohort_reviewstatus(clinvar_path):\n'
    '    p = Path(clinvar_path)\n'
    '    if not p.exists():\n'
    '        return False, f"cohort ReviewStatus: FAIL (not found: {clinvar_path})"\n'
    '    try:\n'
    '        import pyarrow.parquet as pq\n'
    '    except ImportError:\n'
    '        return None, "cohort ReviewStatus: ENV (pyarrow not importable)"\n'
    '    cols = set(pq.ParquetFile(p).schema.names)\n'
    '    ok = "ReviewStatus" in cols\n'
    '    return ok, (\n'
    '        "cohort ReviewStatus: " + ("PASS (present)" if ok else\n'
    '         "FAIL (MISSING -- train.py min_review_tier=3 aborts at _load_and_label; "\n'
    '         "run scripts/augment_reviewstatus.py)")\n'
    '    )\n'
)

PRE_EDITS = [
    # 1. insert the check fn just before check_feature_count
    (
        "\n\ndef check_feature_count():\n",
        "\n\n" + CHECK_FN + "\n\ndef check_feature_count():\n",
    ),
    # 2. add to results, right after the cohort ref/alt check
    (
        "    results = [\n        check_cohort_ref_alt(args.clinvar),\n",
        "    results = [\n        check_cohort_ref_alt(args.clinvar),\n"
        "        check_cohort_reviewstatus(args.clinvar),\n",
    ),
]

TST_APPEND = '''

def _write_cohort_rev(tmp_path, with_review):
    cols = {"chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["C"],
            "fasta_seq_ref": [_REF], "fasta_seq_alt": [_ALT]}
    if with_review:
        cols["ReviewStatus"] = ["criteria_provided,_multiple_submitters"]
    name = "rev.parquet" if with_review else "norev.parquet"
    p = tmp_path / name
    pd.DataFrame(cols).to_parquet(p)
    return str(p)


def test_reviewstatus_present_passes(tmp_path):
    ok, _ = pf.check_cohort_reviewstatus(_write_cohort_rev(tmp_path, True))
    assert ok is True


def test_reviewstatus_absent_fails(tmp_path):
    ok, _ = pf.check_cohort_reviewstatus(_write_cohort_rev(tmp_path, False))
    assert ok is False


def test_reviewstatus_missing_file_fails():
    ok, _ = pf.check_cohort_reviewstatus("nope_cohort_98765.parquet")
    assert ok is False
'''


def _read(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def _write(p, text):
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def main() -> int:
    if not PRE.exists() or not TST.exists():
        print("ERROR: run from repo root.")
        return 2
    raw = _read(PRE)
    if PRE_MARKER in raw:
        print("preflight: already patched")
    else:
        nl = "\r\n" if "\r\n" in raw else "\n"
        text = raw.replace("\r\n", "\n")
        for i, (old, new) in enumerate(PRE_EDITS, 1):
            if text.count(old) != 1:
                print(f"ERROR: preflight edit {i}: anchor count {text.count(old)} != 1")
                return 2
            text = text.replace(old, new, 1)
        shutil.copy2(PRE, PRE.with_suffix(PRE.suffix + ".bak"))
        _write(PRE, text.replace("\n", nl))
        py_compile.compile(str(PRE), doraise=True)
        print("preflight: patched (+check_cohort_reviewstatus, +results); py_compile OK")

    traw = _read(TST)
    if TST_MARKER in traw:
        print("tests: already appended")
    else:
        nl = "\r\n" if "\r\n" in traw else "\n"
        body = traw.replace("\r\n", "\n").rstrip("\n") + "\n" + TST_APPEND
        ast.parse(body)
        shutil.copy2(TST, TST.with_suffix(TST.suffix + ".bak"))
        _write(TST, body.replace("\n", nl))
        print("tests: appended 3 ReviewStatus cases; ast OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
