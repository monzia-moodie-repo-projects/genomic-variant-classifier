#!/usr/bin/env python3
"""patch_launch_contract_watchitems.py -- append two Run-16b watch-items (blend-weight
uniformity, ensemble calibration) into Sec.6 of docs/launch/LAUNCH_CONTRACT_run16.md,
recorded before launch.

Count-guarded (anchor must appear exactly once), idempotent (no-op if already added),
backup-first, CRLF/LF-preserving, ASCII-only. Run from repo root. Author: Monzia Moodie.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("docs/launch/LAUNCH_CONTRACT_run16.md")
ANCHOR = "- real_data_prep.py:501 FutureWarning (gnomAD fillna downcast) -- tech debt."
MARKER = "- BLEND WEIGHTS: the Nelder-Mead blend"
NEW = (
    "- BLEND WEIGHTS: the Nelder-Mead blend returned exactly uniform 0.0769 (=1/13) across\n"
    "  all 13 models, tying the LR stacker (delta -0.0000). Smoke-scale objective is ~flat,\n"
    "  so this may be legitimate, and the deployed combiner is the LR stacker, not the\n"
    "  blend. At full scale, confirm the weights DIVERGE from uniform; still-exact 0.0769\n"
    "  means the blend-weight search is a silent no-op to investigate.\n"
    "- CALIBRATION: ensemble ECE 0.0711 but MCE 0.5340 (one badly-miscalibrated bin); with\n"
    "  kan Brier 0.2223 this is the soft spot. Ranking (what the stacker consumes) is\n"
    "  unaffected, but add an isotonic/Platt pass + per-bin reliability to the metrics\n"
    "  glossary at full scale.\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found (run from repo root).")
        return 1
    raw = TARGET.open("r", encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")

    if MARKER in text:
        print("Already added (BLEND WEIGHTS item present). No-op.")
        return 0
    n = text.count(ANCHOR)
    if n != 1:
        print(f"ABORT: expected exactly 1 anchor line, found {n}. Manual review required.")
        return 1

    eol = text.find("\n", text.find(ANCHOR))
    insert_at = eol + 1  # right after the anchor line's newline
    new_text = text[:insert_at] + NEW + text[insert_at:]

    backup = TARGET.with_suffix(TARGET.suffix + ".pre_watchitems.bak")
    backup.write_bytes(TARGET.read_bytes())
    TARGET.open("w", encoding="utf-8", newline="").write(new_text.replace("\n", nl))
    print(f"OK: added 2 watch-items to Sec.6; backup {backup.name}")
    print(f"  Sec.6 bullets now: {new_text.count(chr(10) + '- ', new_text.find('## 6.'), new_text.find('## 7.'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
