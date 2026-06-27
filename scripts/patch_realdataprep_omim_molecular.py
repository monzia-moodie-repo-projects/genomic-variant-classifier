#!/usr/bin/env python3
r"""patch_realdataprep_omim_molecular.py

Add the omim_n_diseases_molecular block to the TRAINING feature builder in
real_data_prep.py (mirror of the inference builder in variant_ensemble.py).
Without this, the column is live in inference but dead in training -> silent
train/inference feature drift.

Anchor verified against read 14c. ANCHOR-BASED, IDEMPOTENT.
"""
from __future__ import annotations
import argparse, py_compile
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")
MARKER = 'feats["omim_n_diseases_molecular"]'

OLD = '''        feats["omim_n_diseases"] = (
            df.get("omim_n_diseases", pd.Series([0] * len(df), index=df.index))
            .fillna(0)
            .astype(int)
        )
        feats["omim_is_autosomal_dominant"] = ('''

NEW = '''        feats["omim_n_diseases"] = (
            df.get("omim_n_diseases", pd.Series([0] * len(df), index=df.index))
            .fillna(0)
            .astype(int)
        )
        feats["omim_n_diseases_molecular"] = (
            df.get("omim_n_diseases_molecular", pd.Series([0] * len(df), index=df.index))
            .fillna(0)
            .astype(int)
        )
        feats["omim_is_autosomal_dominant"] = ('''


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("OK (idempotent): training builder already has molecular block."); return 0
    c = src.count(OLD)
    if c != 1:
        print(f"FAIL: anchor occurs {c}x (need 1)."); return 3
    if ns.check:
        print("CHECK: training-builder anchor found once."); print("RESULT: PASS (check)"); return 0
    patched = src.replace(OLD, NEW, 1)
    backup = TARGET.with_suffix(".py.pre_omim_molecular.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")
    TARGET.write_text(patched, encoding="utf-8", newline="\n")
    after = TARGET.read_text(encoding="utf-8")
    ok = MARKER in after
    print(f"  {'OK' if ok else 'MISSING'}  training molecular block")
    try:
        py_compile.compile(str(TARGET), doraise=True); print("  OK  real_data_prep.py compiles")
    except py_compile.PyCompileError as exc:
        print(f"  FAIL compile: {exc}"); ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
