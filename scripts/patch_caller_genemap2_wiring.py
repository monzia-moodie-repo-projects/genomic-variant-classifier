#!/usr/bin/env python3
r"""patch_caller_genemap2_wiring.py

CRITICAL: wire --omim-genemap2-path through BOTH launch callers so the OMIM
connector actually receives genemap2.txt. After the connector rewrite, genemap2 is
the SOLE source for omim_n_diseases / omim_n_diseases_molecular / omim_is_autosomal_dominant.
Without this, omim_genemap2_path stays None at launch -> connector's genemap2-None
guard fires -> ALL THREE OMIM columns silent-zero across the full cohort.

Two files, two edits each (argparse arg + AnnotationConfig kwarg):
  scripts/run_phase2_eval.py   (production launch path)
  scripts/regen_splits_local.py (prep-only mirror)

Anchors verified against reads 20a/20b/20c. ANCHOR-BASED, IDEMPOTENT, CRLF-safe.
"""
from __future__ import annotations
import argparse, py_compile
from pathlib import Path

EVAL = Path("scripts/run_phase2_eval.py")
REGEN = Path("scripts/regen_splits_local.py")

# ---- run_phase2_eval.py ----
EVAL_ARG_OLD = '''    p.add_argument(
        "--omim-path",'''
EVAL_ARG_NEW = '''    p.add_argument(
        "--omim-genemap2-path",
        default=None,
        help="OMIM genemap2.txt (data/external/omim/genemap2.txt). REQUIRED for "
        "omim_n_diseases/omim_n_diseases_molecular/omim_is_autosomal_dominant; when "
        "omitted all three OMIM columns default to 0 (silent stub).",
    )
    p.add_argument(
        "--omim-path",'''

EVAL_CFG_OLD = '''            # Run 17 annotation wiring (see --omim-path/--phylop-path/--dbsnp-path/--eve-path)
            omim_path=Path(args.omim_path) if args.omim_path else None,'''
EVAL_CFG_NEW = '''            # Run 17 annotation wiring (see --omim-path/--phylop-path/--dbsnp-path/--eve-path)
            omim_path=Path(args.omim_path) if args.omim_path else None,
            omim_genemap2_path=Path(args.omim_genemap2_path) if args.omim_genemap2_path else None,'''

# ---- regen_splits_local.py ----
REGEN_ARG_OLD = '''    p.add_argument("--omim-path", default=None,
                   help="OMIM mim2gene file; when omitted, omim_* default to 0.")'''
REGEN_ARG_NEW = '''    p.add_argument("--omim-path", default=None,
                   help="OMIM mim2gene file; when omitted, omim_* default to 0.")
    p.add_argument("--omim-genemap2-path", default=None,
                   help="OMIM genemap2.txt; REQUIRED for omim_n_diseases/"
                        "omim_n_diseases_molecular/omim_is_autosomal_dominant "
                        "(when omitted, all three default to 0).")'''

REGEN_CFG_OLD = '''        # Run 17 wiring parity with run_phase2_eval (omim/phylop/dbsnp were missing)
        omim_path=Path(args.omim_path) if args.omim_path else None,'''
REGEN_CFG_NEW = '''        # Run 17 wiring parity with run_phase2_eval (omim/phylop/dbsnp were missing)
        omim_path=Path(args.omim_path) if args.omim_path else None,
        omim_genemap2_path=Path(args.omim_genemap2_path) if args.omim_genemap2_path else None,'''

EDITS = [
    (EVAL,  "eval argparse",   EVAL_ARG_OLD,  EVAL_ARG_NEW),
    (EVAL,  "eval config",     EVAL_CFG_OLD,  EVAL_CFG_NEW),
    (REGEN, "regen argparse",  REGEN_ARG_OLD, REGEN_ARG_NEW),
    (REGEN, "regen config",    REGEN_CFG_OLD, REGEN_CFG_NEW),
]
MARKER = "--omim-genemap2-path"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    for p in (EVAL, REGEN):
        if not p.exists():
            print(f"FAIL: {p} not found."); return 2

    # idempotency: if both files already have the genemap2 arg threaded, stop.
    if MARKER in EVAL.read_text(encoding="utf-8") and MARKER in REGEN.read_text(encoding="utf-8") \
       and "omim_genemap2_path=Path(args.omim_genemap2_path)" in EVAL.read_text(encoding="utf-8") \
       and "omim_genemap2_path=Path(args.omim_genemap2_path)" in REGEN.read_text(encoding="utf-8"):
        print("OK (idempotent): both callers already wire --omim-genemap2-path."); return 0

    # anchor verification
    ok = True
    srcs = {EVAL: EVAL.read_text(encoding="utf-8"), REGEN: REGEN.read_text(encoding="utf-8")}
    for path, label, old, _new in EDITS:
        c = srcs[path].count(old)
        if c != 1:
            print(f"FAIL: anchor '{label}' in {path.name} occurs {c}x (need 1)."); ok = False
    if not ok:
        return 3
    if ns.check:
        print("CHECK: all 4 caller anchors found exactly once."); print("RESULT: PASS (check)"); return 0

    # apply, grouped per file
    for path in (EVAL, REGEN):
        src = path.read_text(encoding="utf-8")
        backup = path.with_suffix(".py.pre_genemap2_wiring.bak")
        if not backup.exists():
            backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")
        for p2, _label, old, new in EDITS:
            if p2 == path:
                src = src.replace(old, new, 1)
        path.write_text(src, encoding="utf-8", newline="\n")

    # post-checks
    allok = True
    for path in (EVAL, REGEN):
        after = path.read_text(encoding="utf-8")
        arg_ok = MARKER in after
        cfg_ok = "omim_genemap2_path=Path(args.omim_genemap2_path)" in after
        print(f"  {'OK' if arg_ok else 'MISSING'}  {path.name}: --omim-genemap2-path arg")
        print(f"  {'OK' if cfg_ok else 'MISSING'}  {path.name}: omim_genemap2_path kwarg")
        allok &= arg_ok and cfg_ok
        try:
            py_compile.compile(str(path), doraise=True); print(f"  OK  {path.name} compiles")
        except py_compile.PyCompileError as exc:
            print(f"  FAIL compile {path.name}: {exc}"); allok = False
    print("RESULT:", "PASS" if allok else "FAIL")
    return 0 if allok else 5


if __name__ == "__main__":
    raise SystemExit(main())
