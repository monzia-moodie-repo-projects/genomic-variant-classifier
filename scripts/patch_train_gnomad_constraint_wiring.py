#!/usr/bin/env python3
"""patch_train_gnomad_constraint_wiring.py -- wire the gnomAD-constraint TSV into
scripts/train.py.

train.py builds AnnotationConfig without gnomad_constraint_path, so it stays None ->
GnomADConstraintConnector(tsv_path=None) runs in STUB MODE -> loeuf is a constant ->
gene_constraint_oe (Run-15 #2 feature) silently deadzones in a train.py-driven Run 16.
This adds --gnomad-constraint, threads gnomad_constraint_path into AnnotationConfig,
and records it in annotation_sources -- the same wiring ecd0474 added for ESM-2.

Idempotent, py_compile-gated, newline-preserving, ASCII. Author: Monzia Moodie."""
from __future__ import annotations

import py_compile
import shutil
import sys
from pathlib import Path

TARGET = Path("scripts/train.py")
MARKER = "gnomad_constraint_path=Path(args.gnomad_constraint)"

EDITS = [
    # 1. argparse flag -- insert a --gnomad-constraint block before --skip-svm
    (
        '    p.add_argument(\n        "--skip-svm",\n',
        '    p.add_argument(\n'
        '        "--gnomad-constraint",\n'
        '        default=None,\n'
        '        metavar="PATH",\n'
        '        help=(\n'
        '            "gnomAD v4.1 gene-constraint TSV (gnomad.v4.1.constraint_metrics.tsv.bgz). "\n'
        '            "Adds pli_score, loeuf, syn_z, mis_z; loeuf revives gene_constraint_oe "\n'
        '            "(Run-15 #2 feature). Default: None (stub mode -> constant defaults -> the "\n'
        '            "feature silently deadzones)."\n'
        '        ),\n'
        '    )\n'
        '    p.add_argument(\n        "--skip-svm",\n',
    ),
    # 2. thread into AnnotationConfig, after dbnsfp_path
    (
        "        dbnsfp_path=Path(args.dbnsfp_path) if args.dbnsfp_path else None,\n"
        "        esm2_model_name=args.esm2_model,\n",
        "        dbnsfp_path=Path(args.dbnsfp_path) if args.dbnsfp_path else None,\n"
        "        gnomad_constraint_path=Path(args.gnomad_constraint) if args.gnomad_constraint else None,\n"
        "        esm2_model_name=args.esm2_model,\n",
    ),
    # 3. record in annotation_sources provenance, after dbnsfp
    (
        '                    "dbnsfp": str(args.dbnsfp_path) if args.dbnsfp_path else None,\n'
        '                    "esm2_model": args.esm2_model,\n',
        '                    "dbnsfp": str(args.dbnsfp_path) if args.dbnsfp_path else None,\n'
        '                    "gnomad_constraint": str(args.gnomad_constraint) if args.gnomad_constraint else None,\n'
        '                    "esm2_model": args.esm2_model,\n',
    ),
]


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root).")
        return 2
    with TARGET.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()
    if MARKER in raw:
        print("already patched (marker present); no change.")
        return 0
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    for i, (old, new) in enumerate(EDITS, 1):
        if text.count(old) != 1:
            print(f"ERROR: edit {i}: expected exactly 1 anchor match, found {text.count(old)} -- not patching.")
            return 2
        text = text.replace(old, new, 1)

    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    with TARGET.open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", nl))
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, TARGET)
        print(f"ERROR: py_compile failed, reverted: {exc}")
        return 2
    print(f"patched {TARGET} (backup at {bak.name}); py_compile OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
