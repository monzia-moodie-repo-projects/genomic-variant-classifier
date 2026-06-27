#!/usr/bin/env python3
"""patch_run_phase2_eval_annotation_flags.py

Add the four MISSING annotation CLI flags to scripts/run_phase2_eval.py and
thread each into the existing AnnotationConfig(...) construction, so the
already-present AnnotationConfig fields (omim_path / phylop_path / dbsnp_path /
eve_path) become reachable from the command line instead of permanently
defaulting to None -> stub.

WHY (verified against the live tree, 2026-06-25):
  - AnnotationConfig (real_data_prep.py) ALREADY declares omim_path, phylop_path,
    dbsnp_path, eve_path (and esm2_uniprot_index_path).
  - _annotate_scores() ALREADY builds the connectors from those fields:
        PhyloPConnector(phylop_file=ac.phylop_path)   (step 2,  ~line 693)
        OMIMConnector(mim2gene_path=ac.omim_path)     (step 8,  ~line 804)
        DbSNPConnector(parquet_path=ac.dbsnp_path)    (step 10, ~line 837)
        EVEConnector(eve_path=ac.eve_path)            (step 11)
  - BUT scripts/run_phase2_eval.py never declared --omim-path / --phylop-path /
    --dbsnp-path / --eve-path and never threaded them into AnnotationConfig(...).
    So those four connectors take their silent-stub branch on every run.
    (--esm2-uniprot-index already exists and is already threaded; clingen,
    dbnsfp, lovd, reactome, rnaseq, finngen, gtex are already wired.)

This is the SAME class of fix as the documented Run-10 wiring fix (lovd/dbnsfp/
reactome were silent-zero because run_phase2_eval never passed them).

The patch is ANCHOR-BASED and IDEMPOTENT: it locates unique existing lines and
inserts relative to them. It refuses to run twice (detects its own markers) and
verifies argparse + AnnotationConfig parity after editing. Pure stdlib.

Usage:
    python scripts/patch_run_phase2_eval_annotation_flags.py            # apply
    python scripts/patch_run_phase2_eval_annotation_flags.py --check    # dry-run/report only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path("scripts/run_phase2_eval.py")

# --- argparse block: inserted immediately AFTER the --finngen-path add_argument.
# The unique anchor is the closing of the finngen help string + ")" that ends that
# add_argument call. We anchor on the distinctive finngen help tail.
ARGPARSE_ANCHOR = '        "default to 0/0/1 (Run 9 bug).",\n    )\n'

ARGPARSE_INSERT = '''\
    # --- Run 17 annotation wiring (omim/phylop/dbsnp/eve): these AnnotationConfig
    # fields + their connectors already exist; run_phase2_eval simply never exposed
    # a CLI flag, so they took the silent-stub branch on every run (same class as the
    # Run-10 lovd/dbnsfp/reactome fix). HGVSp parser (delivered) populates the
    # protein_pos/wt_aa/mut_aa that EVE/ESM-2 key on, so EVE now carries real signal.
    p.add_argument(
        "--omim-path",
        default=None,
        help="OMIM mim2gene/genemap2 file (data/external/omim/...). When omitted, "
        "omim_n_diseases/omim_is_autosomal_dominant default to 0 (silent stub).",
    )
    p.add_argument(
        "--phylop-path",
        default=None,
        help="PhyloP conservation source (data/external/phylop/...). When omitted, "
        "phylop_score defaults to 0.0 (silent stub).",
    )
    p.add_argument(
        "--dbsnp-path",
        default=None,
        help="dbSNP allele-frequency parquet (data/external/dbsnp/...). When "
        "omitted, dbsnp_af defaults to 0.0 (silent stub).",
    )
    p.add_argument(
        "--eve-path",
        default=None,
        help="EVE scores: directory of per-protein CSVs (data/external/eve/) or a "
        "merged parquet. Matched by gene_symbol + one-letter aa_change derived from "
        "the HGVSp-parsed protein_change. When omitted, eve_score defaults to 0.5 "
        "(silent stub). Covers missense substitutions only.",
    )
'''

# --- AnnotationConfig(...) threading: inserted immediately AFTER the existing
# finngen_path threading line (the LAST kwarg before the closing paren).
THREAD_ANCHOR = '            finngen_path=Path(args.finngen_path) if args.finngen_path else None,\n'

THREAD_INSERT = '''\
            # Run 17 annotation wiring (see --omim-path/--phylop-path/--dbsnp-path/--eve-path)
            omim_path=Path(args.omim_path) if args.omim_path else None,
            phylop_path=Path(args.phylop_path) if args.phylop_path else None,
            dbsnp_path=Path(args.dbsnp_path) if args.dbsnp_path else None,
            eve_path=Path(args.eve_path) if args.eve_path else None,
'''

MARKER = "Run 17 annotation wiring"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Report what would change; do not write.")
    ns = ap.parse_args()

    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found (run from repo root).")
        return 2

    src = TARGET.read_text(encoding="utf-8")

    # Idempotency: already patched?
    if MARKER in src:
        print("OK (idempotent): patch markers already present; nothing to do.")
        # still run the parity check so re-runs are informative
        return _verify(src)

    problems = []
    if ARGPARSE_ANCHOR not in src:
        problems.append("argparse anchor (finngen help tail + close paren) NOT found")
    if THREAD_ANCHOR not in src:
        problems.append("AnnotationConfig finngen_path threading anchor NOT found")
    # Each anchor must be unique (insert exactly once).
    if src.count(ARGPARSE_ANCHOR) != 1:
        problems.append(f"argparse anchor occurs {src.count(ARGPARSE_ANCHOR)}x (need exactly 1)")
    if src.count(THREAD_ANCHOR) != 1:
        problems.append(f"threading anchor occurs {src.count(THREAD_ANCHOR)}x (need exactly 1)")
    if problems:
        print("FAIL: cannot safely anchor the patch:")
        for p in problems:
            print(f"  - {p}")
        print("The source may have drifted from what this patch expects. "
              "Re-read the argparse + AnnotationConfig blocks and adjust the anchors.")
        return 3

    patched = src.replace(ARGPARSE_ANCHOR, ARGPARSE_ANCHOR + ARGPARSE_INSERT, 1)
    patched = patched.replace(THREAD_ANCHOR, THREAD_ANCHOR + THREAD_INSERT, 1)

    if ns.check:
        print("CHECK: anchors found, patch would apply cleanly.")
        print(f"  + 4 add_argument calls after --finngen-path")
        print(f"  + 4 threading kwargs after finngen_path=")
        return _verify(patched)

    # Backup then write (UTF-8, LF preserved by Python's text mode on write).
    backup = TARGET.with_suffix(TARGET.suffix + ".pre_annotation_wiring.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8")
        print(f"OK: backup written -> {backup}")
    TARGET.write_text(patched, encoding="utf-8")
    print(f"OK: patched {TARGET}")
    return _verify(patched)


def _verify(text: str) -> int:
    """Post-check: every new flag must appear in argparse AND be threaded."""
    flags = ["--omim-path", "--phylop-path", "--dbsnp-path", "--eve-path"]
    threads = ["omim_path=", "phylop_path=", "dbsnp_path=", "eve_path="]
    ok = True
    for f in flags:
        present = f in text
        print(f"  argparse {f:<14} {'OK' if present else 'MISSING'}")
        ok &= present
    for t in threads:
        present = t in text
        print(f"  thread   {t:<14} {'OK' if present else 'MISSING'}")
        ok &= present
    # Syntactic sanity: the file must still compile.
    try:
        compile(text, str(TARGET), "exec")
        print("  py-compile     OK")
    except SyntaxError as e:
        print(f"  py-compile     FAIL: {e}")
        ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
