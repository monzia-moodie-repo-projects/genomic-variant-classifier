#!/usr/bin/env python3
r"""patch_tabular_omim_molecular.py

Register omim_n_diseases_molecular as a tabular feature (the (3)-confirmed-molecular-basis
OMIM disease count). Touches models/variant_ensemble.py in three places:
  1. EXPECTED_TABULAR_FEATURE_COUNT: 87 -> 88
  2. TABULAR_FEATURES list: insert "omim_n_diseases_molecular" after "omim_n_diseases";
     bump the group comment "Gene-disease annotation (3)" -> "(4)"
  3. engineer_features inference builder: add the feats["omim_n_diseases_molecular"] block
     after feats["omim_n_diseases"]

Anchors verified against the live file (reads 14b, 15a, 15b). ANCHOR-BASED, IDEMPOTENT.
"""
from __future__ import annotations
import argparse
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/models/variant_ensemble.py")

COUNT_ANCHOR = "EXPECTED_TABULAR_FEATURE_COUNT = 87\n"
COUNT_INSERT = "EXPECTED_TABULAR_FEATURE_COUNT = 88\n"

LIST_ANCHOR = (
    "    # Gene-disease annotation (3)\n"
    '    "omim_n_diseases",\n'
    '    "omim_is_autosomal_dominant",\n'
)
LIST_INSERT = (
    "    # Gene-disease annotation (4)\n"
    '    "omim_n_diseases",\n'
    '    "omim_n_diseases_molecular",\n'
    '    "omim_is_autosomal_dominant",\n'
)

BUILDER_ANCHOR = (
    '    feats["omim_n_diseases"] = (\n'
    '        df.get("omim_n_diseases", pd.Series([0] * len(df), index=df.index))\n'
    '        .fillna(0)\n'
    '        .astype(int)\n'
    '    )\n'
    '    feats["omim_is_autosomal_dominant"] = (\n'
)
BUILDER_INSERT = (
    '    feats["omim_n_diseases"] = (\n'
    '        df.get("omim_n_diseases", pd.Series([0] * len(df), index=df.index))\n'
    '        .fillna(0)\n'
    '        .astype(int)\n'
    '    )\n'
    '    feats["omim_n_diseases_molecular"] = (\n'
    '        df.get("omim_n_diseases_molecular", pd.Series([0] * len(df), index=df.index))\n'
    '        .fillna(0)\n'
    '        .astype(int)\n'
    '    )\n'
    '    feats["omim_is_autosomal_dominant"] = (\n'
)

MARKER = '"omim_n_diseases_molecular",'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src and "EXPECTED_TABULAR_FEATURE_COUNT = 88" in src:
        print("OK (idempotent): already registered."); return 0
    anchors = {"count": (COUNT_ANCHOR, src.count(COUNT_ANCHOR)),
               "list": (LIST_ANCHOR, src.count(LIST_ANCHOR)),
               "builder": (BUILDER_ANCHOR, src.count(BUILDER_ANCHOR))}
    ok = True
    for name, (_, c) in anchors.items():
        if c != 1:
            print(f"FAIL: anchor '{name}' occurs {c}x (need 1)."); ok = False
    if not ok: return 3
    if ns.check:
        print("CHECK: all 3 anchors found once."); print("RESULT: PASS (check)"); return 0
    patched = (src.replace(COUNT_ANCHOR, COUNT_INSERT, 1)
                  .replace(LIST_ANCHOR, LIST_INSERT, 1)
                  .replace(BUILDER_ANCHOR, BUILDER_INSERT, 1))
    backup = TARGET.with_suffix(".py.pre_omim_molecular.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")
    TARGET.write_text(patched, encoding="utf-8", newline="")
    after = TARGET.read_text(encoding="utf-8")
    checks = [("count=88", "EXPECTED_TABULAR_FEATURE_COUNT = 88" in after),
              ("list entry", '    "omim_n_diseases_molecular",\n' in after),
              ("group (4)", "# Gene-disease annotation (4)" in after),
              ("builder block", 'feats["omim_n_diseases_molecular"]' in after)]
    allok = True
    for label, present in checks:
        print(f"  {'OK' if present else 'MISSING'}  {label}"); allok &= present
    import py_compile
    try:
        py_compile.compile(str(TARGET), doraise=True); print("  OK  variant_ensemble.py compiles")
    except py_compile.PyCompileError as exc:
        print(f"  FAIL  compile: {exc}"); allok = False
    print("RESULT:", "PASS" if allok else "FAIL")
    return 0 if allok else 5


if __name__ == "__main__":
    raise SystemExit(main())
