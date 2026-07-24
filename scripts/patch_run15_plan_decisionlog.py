#!/usr/bin/env python3
"""Append a dated decision-log correction to docs/runs/RUN_15_PLAN.md recording
that (1) B.D6 is superseded (cnn_1d ENABLED via --seq-windows, NOT --skip-cnn) and
(2) the G1/G2 preflight scripts now exist (superseding PM11a). Append-only, byte-safe
(never decodes existing content, so pre-existing mojibake is untouched), idempotent."""
from __future__ import annotations
import sys
from pathlib import Path

MARKER = b"2026-06-03 -- launch-surface reconciliation"

ENTRY_TEXT = (
    "- **2026-06-03 -- launch-surface reconciliation + two stale-plan corrections** "
    "(docs-only, no code change). "
    "(1) B.D6 SUPERSEDED: the current honest baseline ENABLES cnn_1d as the sequence CNN "
    "via --seq-windows (data/processed/clinvar_grch38_clean_seq.parquet), with run_phase2_eval "
    "aborting if window coverage < 99.5%; it does NOT pass --skip-cnn. This reverses B.D6 "
    "(2026-05-27), which predates the _CNN1DWrapper closure fix (INCIDENT_2026-05-24) and the "
    "seq-window wiring. Authoritative sources now agree: scripts/launch_run15_baseline.sh (ARGS "
    "include --seq-windows, no --skip-cnn) and scripts/preflight_run15_baseline.py (prints "
    "'launch must use ... and NOT --skip-cnn'). B.D6's 'tabular CNN over the 78-dim vector' "
    "description is the stale misconception; run_phase2_eval's --seq-windows/--skip-cnn help "
    "correctly calls cnn_1d the sequence branch. "
    "(2) PM11a SUPERSEDED on preflight existence: PM11a recorded that Run_Preflight_Local.ps1 "
    "(G1) and Run_Preflight_VM.sh (G2) did not exist and had to be built. Both now exist on disk "
    "(Test-Path True, 2026-06-03), alongside scripts/preflight_run15_baseline.py (focused GO/NO-GO: "
    "clean-cohort 0-null/0-dup, _assert_clean_cohort present exactly once, STRING present, "
    "cohort-guard test present), Run15_Postflight.ps1, and Vastai_Destroy_Confirmed.ps1. Phase-F "
    "gates G1/G2 are therefore satisfiable; the launch surface is complete. Working tree clean, "
    "HEAD 1a477a3 pushed to origin/main."
)


def main(path_str: str) -> int:
    path = Path(path_str)
    if not path.exists():
        print(f"ABORT: missing {path}"); return 2
    data = path.read_bytes()
    if MARKER in data:
        print(f"SKIP: {path} already has the 2026-06-03 reconciliation entry (idempotent)"); return 0
    nl = b"\r\n" if b"\r\n" in data else b"\n"
    entry = ENTRY_TEXT.encode("utf-8")  # single logical bullet line (no embedded newlines)
    sep = b"" if data.endswith(nl) else nl
    path.write_bytes(data + sep + entry + nl)
    _eol_label = "CRLF" if nl == b'\r\n' else "LF"
    print(f"appended {len(sep + entry + nl)} bytes to {path}; endings={_eol_label}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "docs/runs/RUN_15_PLAN.md"))
