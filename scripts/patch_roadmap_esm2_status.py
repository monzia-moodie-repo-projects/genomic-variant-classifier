#!/usr/bin/env python3
"""patch_roadmap_esm2_status.py -- in-place ESM-2 status correction (middle path).

Updates the two live status fields in docs/ROADMAP.md that the 2026-06-10
changelog superseded (Section 4B ESM-2 row; Section 5 HGVSp-parser item), so the
living sections are internally truthful. Changelog/archive are untouched.

Count-guarded (aborts if an anchor is not found exactly once -> paste the line
and re-anchor), backup-first, idempotent. Author: Monzia Moodie.
"""
from __future__ import annotations

import datetime as _dt
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROADMAP = REPO / "docs/ROADMAP.md"

EDITS = [
    ('| ESM-2 | esm2_delta_norm | local model+index | MECHANICALLY ACTIVE; '
     'coverage ~3,451 only -> **gated on HGVSp parser** |',
     '| ESM-2 | esm2_delta_norm | local model+index | code-FIXED 2026-06-10: '
     'the ~3,451 cap was a STALE protein-coord index (gate 34e125a; local ceiling '
     '96.6%); gene-resolution hardened (Phase 0); realizes ~2.4M scores after the '
     'Run 16 coord-sync; LLR + 650M/ESM C migration in progress |',
     'code-FIXED 2026-06-10',
     '4B ESM-2 status row'),

    ('- **HGVSp parser (highest leverage, empirically confirmed by Run 15):** '
     'populate protein_pos/wt_aa/mut_aa across the cohort to lift ESM-2 (and EVE) '
     'coverage from ~3,451 to ~1M. Until then ESM-2 carries no real full-run signal.',
     '- **ESM-2 coverage (RESOLVED 2026-06-10):** the ~3,451 cap was a stale '
     'AlphaMissense protein-coord index on the training box, not an HGVSp-parser '
     'gap (protein_pos/wt_aa/mut_aa are populated by step 10b; hgvsp_parser.py / '
     'protein_coords.py already exist). Coverage gate shipped (34e125a; local '
     'ceiling 96.6%); Run 16 prereq is an operational coord-index sync. '
     'Method/model migration to LLR + ESM-2 650M -> ESM C 600M now in progress '
     '(Phase 1).',
     'ESM-2 coverage (RESOLVED 2026-06-10)',
     '5 HGVSp-parser item'),
]


def main() -> int:
    if not ROADMAP.exists():
        print(f"ABORT: missing {ROADMAP}")
        return 2
    text = ROADMAP.read_text(encoding="utf-8")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(ROADMAP, f"{ROADMAP}.bak_{ts}")
    for old, new, marker, label in EDITS:
        if marker in text:
            print(f"  skip (already applied): {label}")
            continue
        n = text.count(old)
        if n != 1:
            print(f"ABORT: anchor for '{label}' found {n}x (expected 1); no changes written")
            return 3
        text = text.replace(old, new, 1)
        print(f"  ok: {label}")
    ROADMAP.write_text(text, encoding="utf-8")
    print(f"DONE. (backup -> ROADMAP.md.bak_{ts})  Regenerate .docx after.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
