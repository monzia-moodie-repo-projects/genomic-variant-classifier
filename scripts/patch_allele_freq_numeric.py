#!/usr/bin/env python3
"""
patch_allele_freq_numeric.py -- pandas-3 readiness for the allele_freq fillna in
real_data_prep._join_gnomad (the ONLY .fillna site that emits the object-downcast
FutureWarning; proven by pandas3_equivalence_harness.py warnings.json).

CORRECT FIX (v2): cast BOTH operands to numeric BEFORE the fillna, so .fillna never runs
on an object column -- which is what actually triggers the downcast warning. (v1 wrapped the
result in pd.to_numeric, which did not stop the inner .fillna from downcasting first; the
warning merely moved to the inner line. This version eliminates it.)

allele_freq is an allele frequency (always float) and is used only numerically downstream
(.notna/.isna at ~546/554/561; float feature at ~1097). Casting an already-numeric column to
numeric is value-identical -> the equivalence harness proves feature_hash stays 49e98393...
On pandas 2.x: no object .fillna -> no downcast -> no warning. On pandas 3.0: both operands are
float, fillna stays float -> no dtype surprise. Identical on both.

Anchored (not line-numbered; the file drifts). Idempotent (sentinel). .bak backup. Aborts if the
anchor is absent or non-unique. Also reverts a prior v1 application if present.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")

ORIG = '        df["allele_freq"] = df["allele_freq"].fillna(df.get("gnomad_af", float("nan")))'

# v1 (incorrect) block that may already be applied -- revert it back to ORIG first.
V1 = (
    '        # pandas-3 readiness: allele_freq is an allele frequency (always float); make the\n'
    '        # fillna result explicitly numeric so pandas 3.0 does not leave it object-typed\n'
    '        # (and pandas 2.x does not silent-downcast). Value-identical on both; see\n'
    '        # pandas3_equivalence_harness.py (feature_hash unchanged).\n'
    '        df["allele_freq"] = pd.to_numeric(\n'
    '            df["allele_freq"].fillna(df.get("gnomad_af", float("nan"))), errors="coerce"\n'
    '        )'
)

NEW = (
    '        # pandas-3 readiness: allele_freq is an allele frequency (always float). Cast BOTH\n'
    '        # operands to numeric BEFORE the fillna so it never runs on an object column (which\n'
    '        # is what triggers the pandas object-downcast: silent in 2.x, removed in 3.0).\n'
    '        # Value-identical on both pandas 2.x and 3.x; see pandas3_equivalence_harness.py\n'
    '        # (feature_hash unchanged).\n'
    '        _af = pd.to_numeric(df["allele_freq"], errors="coerce")\n'
    '        _gf = pd.to_numeric(\n'
    '            df.get("gnomad_af", pd.Series(float("nan"), index=df.index)), errors="coerce"\n'
    '        )\n'
    '        df["allele_freq"] = _af.fillna(_gf)'
)

SENTINEL = "Cast BOTH\n        # operands to numeric BEFORE the fillna"


def main() -> int:
    if not TARGET.exists():
        print(f"[FAIL] {TARGET} not found (run from repo root)")
        return 2
    text = TARGET.read_text(encoding="utf-8")

    if SENTINEL in text:
        print("[idempotent] correct (v2) patch already applied; no change.")
        return 0

    bak = TARGET.with_suffix(TARGET.suffix + ".bak")

    # If v1 is applied, revert it to ORIG first so we patch from a clean base.
    if V1 in text:
        print("[revert] found prior v1 patch; reverting to original line before applying v2.")
        text = text.replace(V1, ORIG)

    n = text.count(ORIG)
    if n == 0:
        print("[FAIL] anchor (original allele_freq fillna line) not found after any v1 revert.")
        print("       Expected exactly:")
        print("       " + ORIG.strip())
        return 3
    if n > 1:
        print(f"[FAIL] anchor found {n} times -- expected exactly 1. Aborting.")
        return 4

    shutil.copy2(TARGET, bak)
    new_text = text.replace(ORIG, NEW)
    TARGET.write_text(new_text, encoding="utf-8")

    after = TARGET.read_text(encoding="utf-8")
    import ast
    try:
        ast.parse(after)
    except SyntaxError as e:
        shutil.copy2(bak, TARGET)
        print(f"[FAIL] post-patch syntax error ({e}); restored from .bak.")
        return 5

    ok = (SENTINEL in after) and ("_af.fillna(_gf)" in after) and (after.count(ORIG) == 0)
    print(f"[ok] v2 patched + compiles; sentinel present: {ok}")
    print(f"[ok] backup at {bak} (remove before committing)")
    return 0 if ok else 6


if __name__ == "__main__":
    sys.exit(main())
