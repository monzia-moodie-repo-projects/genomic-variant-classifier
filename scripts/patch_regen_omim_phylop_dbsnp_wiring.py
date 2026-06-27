#!/usr/bin/env python3
r"""patch_regen_omim_phylop_dbsnp_wiring.py

regen_splits_local.py claims (docstring) to be an "EXACT mirror of
run_phase2_eval.main()'s AnnotationConfig wiring", but it is MISSING three Run-17
paths that run_phase2_eval DOES wire: omim_path, phylop_path, dbsnp_path. As a
result, any prep/coverage check run through regen shows OMIM, PhyloP, and dbSNP as
silent-zero even when their data + connectors are correct -- defeating regen's whole
purpose as the cheap pre-GPU gate.

This patch adds, mirroring run_phase2_eval exactly:
  1. three argparse args: --omim-path, --phylop-path, --dbsnp-path
  2. three AnnotationConfig kwargs: omim_path=, phylop_path=, dbsnp_path=

Anchors verified against the live file (reads 5c + 6a). ANCHOR-BASED, IDEMPOTENT.

  python scripts/patch_regen_omim_phylop_dbsnp_wiring.py --check
  python scripts/patch_regen_omim_phylop_dbsnp_wiring.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

TARGET = Path("scripts/regen_splits_local.py")

# --- argparse anchor: the --clingen-path block (verified 6a). Add 3 args AFTER it. ---
ARG_ANCHOR = (
    '    p.add_argument(\n'
    '        "--clingen-path",\n'
    '        default=None,\n'
    '        help="ClinGen Gene-Disease Validity CSV; when omitted, clingen_validity_score defaults to 0.",\n'
    '    )\n'
)
ARG_INSERT = ARG_ANCHOR + (
    '    p.add_argument("--omim-path", default=None,\n'
    '                   help="OMIM mim2gene file; when omitted, omim_* default to 0.")\n'
    '    p.add_argument("--phylop-path", default=None,\n'
    '                   help="PhyloP conservation source (.bw/.parquet); when omitted, phylop_score=0.0.")\n'
    '    p.add_argument("--dbsnp-path", default=None,\n'
    '                   help="dbSNP allele-frequency parquet; when omitted, dbsnp_af=0.0.")\n'
)

# --- AnnotationConfig anchor: the clingen_path line (verified 5c). Add 3 kwargs BEFORE eve_path. ---
CFG_ANCHOR = (
    "        clingen_path=Path(args.clingen_path) if args.clingen_path else None,\n"
    "        rnaseq_path=Path(args.rnaseq_path) if args.rnaseq_path else None,\n"
    "        finngen_path=Path(args.finngen_path) if args.finngen_path else None,\n"
    "        eve_path=Path(args.eve_path) if args.eve_path else None,\n"
)
CFG_INSERT = (
    "        clingen_path=Path(args.clingen_path) if args.clingen_path else None,\n"
    "        rnaseq_path=Path(args.rnaseq_path) if args.rnaseq_path else None,\n"
    "        finngen_path=Path(args.finngen_path) if args.finngen_path else None,\n"
    "        # Run 17 wiring parity with run_phase2_eval (omim/phylop/dbsnp were missing)\n"
    "        omim_path=Path(args.omim_path) if args.omim_path else None,\n"
    "        phylop_path=Path(args.phylop_path) if args.phylop_path else None,\n"
    "        dbsnp_path=Path(args.dbsnp_path) if args.dbsnp_path else None,\n"
    "        eve_path=Path(args.eve_path) if args.eve_path else None,\n"
)

MARKER = "Run 17 wiring parity with run_phase2_eval"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()

    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found.")
        return 2
    src = TARGET.read_text(encoding="utf-8")

    if MARKER in src:
        print("OK (idempotent): regen already has omim/phylop/dbsnp wiring.")
        return 0

    n_arg = src.count(ARG_ANCHOR)
    n_cfg = src.count(CFG_ANCHOR)
    if n_arg != 1:
        print(f"FAIL: argparse anchor (--clingen-path block) occurs {n_arg}x (need 1).")
        return 3
    if n_cfg != 1:
        print(f"FAIL: AnnotationConfig anchor (clingen->...->eve_path) occurs {n_cfg}x (need 1).")
        return 3

    if ns.check:
        print("CHECK: both anchors found exactly once.")
        print("RESULT: PASS (check)")
        return 0

    patched = src.replace(ARG_ANCHOR, ARG_INSERT, 1).replace(CFG_ANCHOR, CFG_INSERT, 1)

    backup = TARGET.with_suffix(".py.pre_opd_wiring.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="")
        print(f"OK: backup -> {backup}")
    TARGET.write_text(patched, encoding="utf-8", newline="")

    after = TARGET.read_text(encoding="utf-8")
    checks = [
        ("--omim-path arg", '"--omim-path"' in after),
        ("--phylop-path arg", '"--phylop-path"' in after),
        ("--dbsnp-path arg", '"--dbsnp-path"' in after),
        ("omim_path kwarg", "omim_path=Path(args.omim_path)" in after),
        ("phylop_path kwarg", "phylop_path=Path(args.phylop_path)" in after),
        ("dbsnp_path kwarg", "dbsnp_path=Path(args.dbsnp_path)" in after),
    ]
    ok = True
    for label, present in checks:
        print(f"  {'OK' if present else 'MISSING'}  {label}")
        ok &= present
    import py_compile
    try:
        py_compile.compile(str(TARGET), doraise=True)
        print("  OK  regen_splits_local.py compiles")
    except py_compile.PyCompileError as exc:
        print(f"  FAIL  compile: {exc}")
        ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
