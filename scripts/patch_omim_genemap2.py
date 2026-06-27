#!/usr/bin/env python3
r"""patch_omim_genemap2.py

FIX a confirmed silent-zero: omim_is_autosomal_dominant is 0 for ALL variants.

DataPrepPipeline builds the OMIM connector as:
    omim = OMIMConnector(mim2gene_path=ac.omim_path)          # genemap2_path NOT passed
OMIMConnector._parse_mim2gene hard-codes omim_is_autosomal_dominant=0 (mim2gene has no
inheritance field); the REAL AD flag only comes from _parse_genemap2_autosomal_dominant,
which runs ONLY if genemap2_path is set. So the AD column is dead despite genemap2.txt
being on disk. (The step-8 log only reports omim_n_diseases, so the dead column is invisible.)

This patch (src/genomic_variant_classifier/data/real_data_prep.py):
  1. adds AnnotationConfig field:  omim_genemap2_path: Optional[Path] = None   (after omim_path)
  2. threads it into the connector: OMIMConnector(mim2gene_path=ac.omim_path,
                                                  genemap2_path=ac.omim_genemap2_path)
  3. extends the step-8 log to also report omim_is_autosomal_dominant>0 count
     (so the column can never silently die again).

Anchors verified against the live file (reads 7a + 7b). ANCHOR-BASED, IDEMPOTENT.

  python scripts/patch_omim_genemap2.py --check
  python scripts/patch_omim_genemap2.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")

# 1. dataclass field (anchor: the omim_path line, verified 7a)
FIELD_ANCHOR = "    omim_path: Optional[Path] = None\n"
FIELD_INSERT = (
    "    omim_path: Optional[Path] = None\n"
    "    omim_genemap2_path: Optional[Path] = None  # OMIM genemap2.txt -> real omim_is_autosomal_dominant (else hard-0)\n"
)

# 2. connector construction (anchor verified 7b)
CALL_ANCHOR = "        omim = OMIMConnector(mim2gene_path=ac.omim_path)\n"
CALL_INSERT = (
    "        omim = OMIMConnector(\n"
    "            mim2gene_path=ac.omim_path,\n"
    "            genemap2_path=ac.omim_genemap2_path,\n"
    "        )\n"
)

# 3. extend the step-8 log to also surface the AD count (anchor verified 7b)
LOG_ANCHOR = (
    '        logger.info(\n'
    '            "Score annotation 8/17 (OMIM): %d variants with omim_n_diseases > 0.",\n'
    '            int(\n'
    '                (\n'
    '                    df.get("omim_n_diseases", pd.Series([0] * len(df), index=df.index))\n'
    '                    > 0\n'
    '                ).sum()\n'
    '            ),\n'
    '        )\n'
)
LOG_INSERT = (
    '        logger.info(\n'
    '            "Score annotation 8/17 (OMIM): %d variants with omim_n_diseases > 0, "\n'
    '            "%d with omim_is_autosomal_dominant > 0.",\n'
    '            int(\n'
    '                (\n'
    '                    df.get("omim_n_diseases", pd.Series([0] * len(df), index=df.index))\n'
    '                    > 0\n'
    '                ).sum()\n'
    '            ),\n'
    '            int(\n'
    '                (\n'
    '                    df.get("omim_is_autosomal_dominant", pd.Series([0] * len(df), index=df.index))\n'
    '                    > 0\n'
    '                ).sum()\n'
    '            ),\n'
    '        )\n'
)

MARKER = "omim_genemap2_path: Optional[Path] = None  # OMIM genemap2.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()

    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found.")
        return 2
    src = TARGET.read_text(encoding="utf-8")

    if MARKER in src:
        print("OK (idempotent): omim_genemap2_path already wired.")
        return 0

    anchors = {
        "field (omim_path)": (FIELD_ANCHOR, src.count(FIELD_ANCHOR)),
        "connector call": (CALL_ANCHOR, src.count(CALL_ANCHOR)),
        "step-8 log": (LOG_ANCHOR, src.count(LOG_ANCHOR)),
    }
    ok_anchor = True
    for name, (_, cnt) in anchors.items():
        if cnt != 1:
            print(f"FAIL: anchor '{name}' occurs {cnt}x (need 1).")
            ok_anchor = False
    if not ok_anchor:
        return 3

    if ns.check:
        print("CHECK: all three anchors found exactly once (field, connector call, step-8 log).")
        print("RESULT: PASS (check)")
        return 0

    patched = (src
               .replace(FIELD_ANCHOR, FIELD_INSERT, 1)
               .replace(CALL_ANCHOR, CALL_INSERT, 1)
               .replace(LOG_ANCHOR, LOG_INSERT, 1))

    backup = TARGET.with_suffix(".py.pre_omim_genemap2.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="")
        print(f"OK: backup -> {backup}")
    TARGET.write_text(patched, encoding="utf-8", newline="")

    after = TARGET.read_text(encoding="utf-8")
    checks = [
        ("field added", "omim_genemap2_path: Optional[Path] = None  # OMIM genemap2.txt" in after),
        ("genemap2_path threaded", "genemap2_path=ac.omim_genemap2_path," in after),
        ("mim2gene still passed", "mim2gene_path=ac.omim_path," in after),
        ("log reports AD count", "with omim_is_autosomal_dominant > 0." in after),
    ]
    ok = True
    for label, present in checks:
        print(f"  {'OK' if present else 'MISSING'}  {label}")
        ok &= present
    import py_compile
    try:
        py_compile.compile(str(TARGET), doraise=True)
        print("  OK  real_data_prep.py compiles")
    except py_compile.PyCompileError as exc:
        print(f"  FAIL  compile: {exc}")
        ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
