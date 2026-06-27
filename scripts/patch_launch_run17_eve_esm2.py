#!/usr/bin/env python3
"""patch_launch_run17_eve_esm2.py

Wire the SIX newly-reachable annotation sources (EVE, ESM-2/UniProt, OMIM,
PhyloP, dbSNP, ClinGen) into scripts/launch_run17_baseline.sh, replacing the
obsolete "intentionally stubbed pending HGVSp parser" NOTE with real wiring.

WHY SIX, not five (verified against the live tree 2026-06-25):
  - eve / omim / phylop / dbsnp: AnnotationConfig fields + connectors exist;
    run_phase2_eval never exposed a CLI flag (fixed by the companion patcher
    patch_run_phase2_eval_annotation_flags.py).
  - esm2: --esm2-uniprot-index already exists + threads (line 279); only the
    LAUNCH script never passed it.
  - clingen: --clingen-path ALREADY exists (line 114) + threads (line 291), but
    the launch script never passed it either -> clingen_validity_score silently 0
    in Run 17 despite 2 ClinGen files on Drive. Caught on re-audit.

The HGVSp parser (src/.../data/hgvsp_parser.py) is delivered, tested (45 passed),
and wired into real_data_prep, populating protein_pos/wt_aa/mut_aa -> EVE/ESM-2
now carry REAL signal, not the known-zero the old NOTE warned about.

Each source hard-fails if its configured file/dir is missing on the VM (no silent
stub), mirroring the FinnGen pattern. Globs are FORMAT-AWARE and SELF-DOCUMENTING:
each echoes the exact file it picked, so a wrong pick is loud in the log, never
silent. The per-connector coverage log lines in _annotate_scores (PhyloP 2/17,
OMIM 8/17, dbSNP 10/17) then prove activation at smoke time.

Backs up to .preeve.bak. Anchor-based + idempotent. Run from repo root.

  python scripts/patch_launch_run17_eve_esm2.py            # apply
  python scripts/patch_launch_run17_eve_esm2.py --check    # report only
"""
from __future__ import annotations

import argparse
from pathlib import Path

TARGET = Path("scripts/launch_run17_baseline.sh")
MARKER = "Run 17 EVE/ESM-2 wiring"

WIRE_ANCHOR = (
    'if [ -f "$FINNGEN_FILE" ]; then ARGS="$ARGS --finngen-path $FINNGEN_FILE"; '
    'echo "==> FinnGen wired: $FINNGEN_FILE" | tee -a "$LOG"; '
    'else echo "==> ABORT: FinnGen file missing: $FINNGEN_FILE" | tee -a "$LOG"; exit 7; fi\n'
)

WIRE_INSERT = '''
# --- Run 17 EVE/ESM-2 wiring (HGVSp parser delivered -> EVE/ESM-2 now carry REAL
#     signal). Plus omim/phylop/dbsnp/clingen, whose CLI flags exist but the launch
#     script never passed (silent-zero). Hard-fail if a configured source is missing
#     on the VM; each echoes the exact file picked (a wrong pick is LOUD, not silent).
# EVE: directory of per-protein CSVs (gene_symbol + HGVSp-derived aa_change).
EVE_DIR="$DATA/external/eve"
if [ -d "$EVE_DIR" ] && [ -n "$(ls -A "$EVE_DIR" 2>/dev/null)" ]; then
    ARGS="$ARGS --eve-path $EVE_DIR"; echo "==> EVE wired: $EVE_DIR ($(ls "$EVE_DIR" | wc -l) files)" | tee -a "$LOG"
else
    echo "==> ABORT: EVE dir missing/empty: $EVE_DIR (stage it to the VM)" | tee -a "$LOG"; exit 8
fi
# ESM-2 UniProt sequence index (offline; else slow live REST per gene).
UNIPROT_INDEX="$DATA/external/uniprot/uniprot_human_reviewed.parquet"
if [ -f "$UNIPROT_INDEX" ]; then
    ARGS="$ARGS --esm2-uniprot-index $UNIPROT_INDEX"; echo "==> ESM-2 UniProt index wired: $UNIPROT_INDEX" | tee -a "$LOG"
else
    echo "==> ABORT: UniProt index missing: $UNIPROT_INDEX" | tee -a "$LOG"; exit 8
fi
# OMIM: prefer a mim2gene file (OMIMConnector(mim2gene_path=...)); else first file.
OMIM_FILE="$(ls "$DATA"/external/omim/*mim2gene* 2>/dev/null | head -n1 || true)"
if [ -z "$OMIM_FILE" ]; then OMIM_FILE="$(ls "$DATA"/external/omim/* 2>/dev/null | grep -v -i 'readme\\|checksum\\|md5' | head -n1 || true)"; fi
if [ -n "$OMIM_FILE" ] && [ -f "$OMIM_FILE" ]; then
    ARGS="$ARGS --omim-path $OMIM_FILE"; echo "==> OMIM wired: $OMIM_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: OMIM file missing under $DATA/external/omim/" | tee -a "$LOG"; exit 8
fi
# PhyloP: single source file.
PHYLOP_FILE="$(ls "$DATA"/external/phylop/* 2>/dev/null | grep -v -i 'readme\\|checksum\\|md5' | head -n1 || true)"
if [ -n "$PHYLOP_FILE" ] && [ -f "$PHYLOP_FILE" ]; then
    ARGS="$ARGS --phylop-path $PHYLOP_FILE"; echo "==> PhyloP wired: $PHYLOP_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: PhyloP file missing under $DATA/external/phylop/" | tee -a "$LOG"; exit 8
fi
# dbSNP: DbSNPConnector(parquet_path=...) wants a parquet.
DBSNP_FILE="$(ls "$DATA"/external/dbsnp/*.parquet 2>/dev/null | head -n1 || true)"
if [ -n "$DBSNP_FILE" ] && [ -f "$DBSNP_FILE" ]; then
    ARGS="$ARGS --dbsnp-path $DBSNP_FILE"; echo "==> dbSNP wired: $DBSNP_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: dbSNP parquet missing under $DATA/external/dbsnp/" | tee -a "$LOG"; exit 8
fi
# ClinGen: Gene-Disease Validity CSV (flag existed but launch never passed it -> silent 0).
CLINGEN_FILE="$(ls "$DATA"/external/clingen/*.csv 2>/dev/null | head -n1 || true)"
if [ -n "$CLINGEN_FILE" ] && [ -f "$CLINGEN_FILE" ]; then
    ARGS="$ARGS --clingen-path $CLINGEN_FILE"; echo "==> ClinGen wired: $CLINGEN_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: ClinGen CSV missing under $DATA/external/clingen/" | tee -a "$LOG"; exit 8
fi
# end Run 17 EVE/ESM-2 wiring
'''

NOTE_OLD = (
    'echo "==> NOTE: --esm2-uniprot-index intentionally absent '
    '(ESM-2/EVE stubbed pending HGVSp parser)" | tee -a "$LOG"\n'
)
NOTE_NEW = (
    'echo "==> NOTE: ESM-2/EVE ACTIVE (HGVSp parser delivered; '
    'protein_pos/wt_aa/mut_aa populated for missense)" | tee -a "$LOG"\n'
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()

    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found (run from repo root).")
        return 2

    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("OK (idempotent): EVE/ESM-2 wiring already present.")
        return 0

    problems = []
    if WIRE_ANCHOR not in src:
        problems.append("FinnGen wiring anchor NOT found (launch script drifted)")
    elif src.count(WIRE_ANCHOR) != 1:
        problems.append(f"FinnGen anchor occurs {src.count(WIRE_ANCHOR)}x (need 1)")
    if NOTE_OLD not in src:
        problems.append("stale stub NOTE line NOT found (already changed?)")
    if problems:
        print("FAIL: cannot safely anchor:")
        for p in problems:
            print(f"  - {p}")
        return 3

    patched = src.replace(WIRE_ANCHOR, WIRE_ANCHOR + WIRE_INSERT, 1)
    patched = patched.replace(NOTE_OLD, NOTE_NEW, 1)

    if ns.check:
        print("CHECK: anchors found; would insert EVE/ESM-2/omim/phylop/dbsnp/clingen "
              "wiring after FinnGen and replace the stub NOTE.")
        return 0

    backup = TARGET.with_suffix(TARGET.suffix + ".preeve.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8")
        print(f"OK: backup -> {backup}")
    TARGET.write_text(patched, encoding="utf-8")
    print(f"OK: patched {TARGET}")

    ok = True
    for needle in ["--eve-path", "--esm2-uniprot-index", "--omim-path",
                   "--phylop-path", "--dbsnp-path", "--clingen-path"]:
        present = needle in patched
        print(f"  wired {needle:<22} {'OK' if present else 'MISSING'}")
        ok &= present
    stub_gone = "intentionally absent" not in patched
    print(f"  stale stub NOTE removed   {'OK' if stub_gone else 'STILL PRESENT'}")
    ok &= stub_gone
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
