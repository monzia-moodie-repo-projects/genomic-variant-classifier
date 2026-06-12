#!/usr/bin/env python3
"""patch_rdp_maxentscan_delta.py -- register maxentscan_delta in real_data_prep.py:
add to _engineer_features and to the RNA-off default-fill tuple. Idempotent,
backup-first, py_compile-gated, ASCII. Author: Monzia Moodie."""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")
MARKER = "maxentscan_delta"

EDITS = [
    (
        '        # RNA splice-context features (Phase 6.1)\n'
        '        feats["maxentscan_score"] = (\n'
        '            df.get("maxentscan_score", pd.Series([0.0] * len(df), index=df.index))\n'
        '            .fillna(0.0)\n'
        '            .astype(float)\n'
        '        )\n'
        '        feats["dist_to_splice_site"] = (\n',
        '        # RNA splice-context features (Phase 6.1)\n'
        '        feats["maxentscan_score"] = (\n'
        '            df.get("maxentscan_score", pd.Series([0.0] * len(df), index=df.index))\n'
        '            .fillna(0.0)\n'
        '            .astype(float)\n'
        '        )\n'
        '        feats["maxentscan_delta"] = (\n'
        '            df.get("maxentscan_delta", pd.Series([0.0] * len(df), index=df.index))\n'
        '            .fillna(0.0)\n'
        '            .astype(float)\n'
        '        )\n'
        '        feats["dist_to_splice_site"] = (\n',
        "engineer_features add",
    ),
    (
        '            for col, val in [\n'
        '                ("maxentscan_score", 0.0),\n'
        '                ("dist_to_splice_site", 50),\n',
        '            for col, val in [\n'
        '                ("maxentscan_score", 0.0),\n'
        '                ("maxentscan_delta", 0.0),\n'
        '                ("dist_to_splice_site", 50),\n',
        "default-fill tuple add",
    ),
]

def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found"); return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if MARKER in text:
        print("already applied; no change."); return 0
    for old, new, label in EDITS:
        c = text.count(old)
        if c != 1:
            print(f"ABORT: [{label}] anchor found {c} times (expected 1); no change."); return 1
        text = text.replace(old, new, 1)
    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(text.replace("\n", nl), encoding="utf-8", newline="")
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, TARGET); print(f"ABORT: py_compile failed, restored:\n{exc}"); return 1
    print(f"OK: real_data_prep {len(EDITS)} edits applied; backup at {bak}"); return 0

if __name__ == "__main__":
    sys.exit(main())
