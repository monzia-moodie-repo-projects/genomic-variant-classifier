#!/usr/bin/env python3
"""patch_launch_run17_eve_entry_map.py

Wire EVE's entry-name -> HGNC map in the launch script by pointing --eve-entry-map
at the SAME $UNIPROT_INDEX already used for --esm2-uniprot-index. The flags are
INDEPENDENT in code; the launch script is the single definition point where they
share a value, so there is no drift and no hidden cross-flag coupling.

Two edits (idempotent, LF-safe -- bash will NOT tolerate CRLF):
  1. After the ESM-2 wiring line (ARGS="$ARGS --esm2-uniprot-index $UNIPROT_INDEX"; ...)
     append a line wiring --eve-entry-map to the same $UNIPROT_INDEX, inside the
     same `if [ -f "$UNIPROT_INDEX" ]` block so it shares the abort-on-missing guard.
  2. Fix the stale header comment (line ~18) that still claims
     "--esm2-uniprot-index intentionally NOT wired: ESM-2/EVE stay stubbed until the
     HGVSp parser" -- both are now wired and the HGVSp parser is delivered.

  python scripts/patch_launch_run17_eve_entry_map.py            # apply
  python scripts/patch_launch_run17_eve_entry_map.py --check    # report only
"""
from __future__ import annotations

import argparse
from pathlib import Path

TARGET = Path("scripts/launch_run17_baseline.sh")
MARKER = "--eve-entry-map $UNIPROT_INDEX"

# 1. ESM-2 wiring line -> append the EVE entry-map line right after it.
ESM2_ANCHOR = (
    '    ARGS="$ARGS --esm2-uniprot-index $UNIPROT_INDEX"; '
    'echo "==> ESM-2 UniProt index wired: $UNIPROT_INDEX" | tee -a "$LOG"\n'
)
ESM2_INSERT = (
    '    ARGS="$ARGS --esm2-uniprot-index $UNIPROT_INDEX"; '
    'echo "==> ESM-2 UniProt index wired: $UNIPROT_INDEX" | tee -a "$LOG"\n'
    '    ARGS="$ARGS --eve-entry-map $UNIPROT_INDEX"; '
    'echo "==> EVE entry-name map wired: $UNIPROT_INDEX (resolves 1433G_HUMAN -> YWHAG)" '
    '| tee -a "$LOG"\n'
)

# 2. Stale header comment fix.
STALE_ANCHOR = (
    "#   - --esm2-uniprot-index intentionally NOT wired: ESM-2/EVE stay stubbed until the HGVSp parser\n"
)
STALE_INSERT = (
    "#   - --esm2-uniprot-index AND --eve-entry-map both wired to $UNIPROT_INDEX (HGVSp parser delivered);\n"
    "#     EVE resolves per-protein entry-name filenames (1433G_HUMAN) to HGNC (YWHAG) via the index.\n"
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
        print("OK (idempotent): --eve-entry-map already wired in launch script.")
        return 0

    # The ESM-2 wiring edit is required. The stale-comment edit is best-effort
    # (the comment text may have drifted); we warn but do not fail if it's absent.
    n_esm2 = src.count(ESM2_ANCHOR)
    if n_esm2 != 1:
        print(f"FAIL: ESM-2 wiring anchor occurs {n_esm2}x (need exactly 1).")
        print("  Expected line:")
        print("  " + ESM2_ANCHOR.strip())
        return 3

    patched = src.replace(ESM2_ANCHOR, ESM2_INSERT, 1)

    n_stale = src.count(STALE_ANCHOR)
    if n_stale == 1:
        patched = patched.replace(STALE_ANCHOR, STALE_INSERT, 1)
        stale_done = True
    else:
        stale_done = False
        print(f"NOTE: stale-comment anchor occurs {n_stale}x (expected 1); "
              "skipping comment fix (non-fatal). Review line ~18 manually.")

    if ns.check:
        print("CHECK: ESM-2 wiring anchor found; would append --eve-entry-map line.")
        print(f"  stale-comment fix: {'will apply' if stale_done else 'SKIPPED (anchor not unique)'}")
        return 0

    backup = TARGET.with_suffix(TARGET.suffix + ".pre_eve_entry_map.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="\n")
        print(f"OK: backup -> {backup}")

    # LF-only write; bash cannot tolerate CRLF.
    TARGET.write_text(patched, encoding="utf-8", newline="\n")
    if b"\r\n" in TARGET.read_bytes():
        print("FAIL: CRLF detected in written file (bash would break).")
        return 5
    print(f"OK: patched {TARGET}")

    ok = True
    present = MARKER in patched
    print(f"  {'OK' if present else 'MISSING'}  --eve-entry-map $UNIPROT_INDEX line")
    ok &= present
    print(f"  {'OK' if stale_done else 'SKIPPED'}  stale line-18 comment fixed")
    # CR byte audit (defensive, beyond the guard above)
    crlf = b"\r\n" in TARGET.read_bytes()
    print(f"  {'OK' if not crlf else 'FAIL'}  no CRLF")
    ok &= not crlf
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
