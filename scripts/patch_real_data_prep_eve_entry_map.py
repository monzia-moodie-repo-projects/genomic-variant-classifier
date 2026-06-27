#!/usr/bin/env python3
"""patch_real_data_prep_eve_entry_map.py

Thread the EVE entry-name -> HGNC map path through AnnotationConfig into the
EVEConnector, so EVE resolves its per-protein filenames (UniProt entry names,
e.g. 1433G_HUMAN) to the cohort's HGNC symbols (YWHAG) instead of silently
keying on the entry-name prefix (eve_score stuck at 0.5).

Two anchor-based edits (idempotent, LF-safe):
  1. AnnotationConfig: add `eve_entry_map_path: Optional[Path] = None` right
     after the existing `eve_path` field.
  2. _annotate_scores step 11: thread `entry_map_path=ac.eve_entry_map_path`
     into `EVEConnector(eve_path=ac.eve_path)`.

  python scripts/patch_real_data_prep_eve_entry_map.py            # apply
  python scripts/patch_real_data_prep_eve_entry_map.py --check    # report only
"""
from __future__ import annotations

import argparse
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")
MARKER = "eve_entry_map_path"

FIELD_ANCHOR = "    eve_path: Optional[Path] = None\n    hgmd_path: Optional[Path] = None\n"
FIELD_INSERT = (
    "    eve_path: Optional[Path] = None\n"
    "    eve_entry_map_path: Optional[Path] = None  # UniProt index parquet (entry_name col) for EVE entry-name -> HGNC\n"
    "    hgmd_path: Optional[Path] = None\n"
)

CTOR_ANCHOR = "        eve = EVEConnector(eve_path=ac.eve_path)\n"
CTOR_INSERT = "        eve = EVEConnector(eve_path=ac.eve_path, entry_map_path=ac.eve_entry_map_path)\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()

    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found (run from repo root).")
        return 2

    src = TARGET.read_text(encoding="utf-8")

    if MARKER in src and "entry_map_path=ac.eve_entry_map_path" in src:
        print("OK (idempotent): eve_entry_map_path already wired in real_data_prep.py.")
        return 0

    problems = []
    for name, anc in [("AnnotationConfig field", FIELD_ANCHOR), ("EVEConnector ctor", CTOR_ANCHOR)]:
        n = src.count(anc)
        if n != 1:
            problems.append(f"{name}: anchor occurs {n}x (need exactly 1)")
    if problems:
        print("FAIL: cannot safely anchor:")
        for p in problems:
            print(f"  - {p}")
        return 3

    patched = src.replace(FIELD_ANCHOR, FIELD_INSERT, 1).replace(CTOR_ANCHOR, CTOR_INSERT, 1)

    if ns.check:
        print("CHECK: both anchors found; would add field + thread entry_map_path.")
        return 0

    backup = TARGET.with_suffix(TARGET.suffix + ".pre_eve_entry_map.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="\n")
        print(f"OK: backup -> {backup}")
    TARGET.write_text(patched, encoding="utf-8", newline="\n")
    if b"\r\n" in TARGET.read_bytes():
        print("FAIL: CRLF detected in written file.")
        return 5
    print(f"OK: patched {TARGET}")

    ok = True
    for needle in ["eve_entry_map_path: Optional[Path] = None",
                   "EVEConnector(eve_path=ac.eve_path, entry_map_path=ac.eve_entry_map_path)"]:
        present = needle in patched
        print(f"  {'OK' if present else 'MISSING'}  {needle[:60]}")
        ok &= present
    try:
        compile(patched, str(TARGET), "exec")
        print("  py-compile OK")
    except SyntaxError as e:
        print(f"  py-compile FAIL: {e}")
        ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
