#!/usr/bin/env python3
"""apply_finngen_wiring.py -- Author: Monzia Moodie

Idempotent, self-verifying patch: wire FinnGen into scripts/launch_run17_baseline.sh
so Run 17 stops silently zeroing finngen_af_fin/nfsee/enrichment. run_phase2_eval.py
already maps --finngen-path into AnnotationConfig; the launcher simply never passed it.

Adds, right after the `--kg` ARGS line:
    FINNGEN_FILE="$DATA/external/finngen/finnge_R12_annotated_variants_v1.gz"  # registry typo 'finnge'
    if [ -f "$FINNGEN_FILE" ]; then ARGS="$ARGS --finngen-path $FINNGEN_FILE"; \
        echo "==> FinnGen wired: $FINNGEN_FILE"|tee -a "$LOG"; \
    else echo "==> ABORT: FinnGen file missing: $FINNGEN_FILE"|tee -a "$LOG"; exit 7; fi

Fail-LOUD (exit 7) if the file is missing -- FinnGen is now a required, wired source,
so its absence must abort the run, never silently zero the columns. Backup +
verify-or-rollback.
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path

ANCHOR = 'ARGS="$ARGS --kg $KG_PARQUET"'
MARKER = "--finngen-path $FINNGEN_FILE"
BLOCK = (
    'FINNGEN_FILE="$DATA/external/finngen/finnge_R12_annotated_variants_v1.gz"  # registry typo \'finnge\'\n'
    'if [ -f "$FINNGEN_FILE" ]; then ARGS="$ARGS --finngen-path $FINNGEN_FILE"; '
    'echo "==> FinnGen wired: $FINNGEN_FILE" | tee -a "$LOG"; '
    'else echo "==> ABORT: FinnGen file missing: $FINNGEN_FILE" | tee -a "$LOG"; exit 7; fi\n'
)


def main(argv=None) -> int:
    repo = Path(argv[0]) if argv else Path(".")
    f = repo / "scripts" / "launch_run17_baseline.sh"
    if not f.exists():
        print(f"NOT FOUND: {f}")
        return 2
    text = f.read_text(encoding="utf-8")
    if MARKER in text:
        print("already wired (idempotent no-op)")
        return 0
    if ANCHOR not in text:
        print(f"ANCHOR not found: {ANCHOR!r} -- launcher structure changed; aborting")
        return 3
    bak = f.with_suffix(".sh.prefinngen.bak")
    shutil.copy2(f, bak)
    new = text.replace(ANCHOR, ANCHOR + "\n" + BLOCK, 1)
    f.write_text(new, encoding="utf-8")
    # verify
    chk = f.read_text(encoding="utf-8")
    if MARKER not in chk or chk.count(MARKER) != 1:
        shutil.copy2(bak, f)
        print("VERIFY FAILED -- rolled back from backup")
        return 4
    print(f"OK: FinnGen wired into {f} (backup {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
