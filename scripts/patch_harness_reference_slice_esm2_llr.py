#!/usr/bin/env python3
"""patch_harness_reference_slice_esm2_llr.py -- populate esm2_llr in the harness
reference slice (Phase 1 wiring follow-through).

esm2_llr is a LIVE feature (engineer_features reads df.get("esm2_llr", ...)), so
per the harness convention it must be supplied a non-zero value in
build_reference_slice() -- NOT added to KNOWN_ZERO_DEFAULT (that set is dead
connectors only; cf. the clingen_validity_score note). Unlike esm2_delta_norm (a
norm, >=0), esm2_llr is SIGNED, so it gets a signed range spanning the real
distribution (hotspots ~ -9..-11, benign ~ -6, occasional positive).

Count-guarded, backup-first, idempotent, py_compile-gated. Author: Monzia Moodie.
"""
from __future__ import annotations

import datetime as _dt
import py_compile
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py"

ANCHOR = '"esm2_delta_norm": rng.uniform(0.1, 5, n),'
NEW = (
    '"esm2_delta_norm": rng.uniform(0.1, 5, n),\n'
    '        "esm2_llr": rng.uniform(-12, 4, n),  '
    '# SIGNED (neg=damaging); live feature, NOT allowlisted'
)
MARKER = '"esm2_llr"'


def main() -> int:
    if not HARNESS.exists():
        print(f"ABORT: missing {HARNESS}")
        return 2
    text = HARNESS.read_text(encoding="utf-8")
    if MARKER in text:
        print("  skip (already applied): esm2_llr in build_reference_slice")
        return 0
    n = text.count(ANCHOR)
    if n != 1:
        print(f"ABORT: anchor found {n}x (expected 1); nothing written")
        return 3
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(HARNESS, f"{HARNESS}.bak_{ts}")
    text = text.replace(ANCHOR, NEW, 1)
    HARNESS.write_text(text, encoding="utf-8")
    try:
        py_compile.compile(str(HARNESS), doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"ABORT: py_compile failed: {exc}")
        return 4
    print(f"  ok: esm2_llr populated in build_reference_slice (signed range -12..4)")
    print(f"py_compile clean: correctness_harness.py  (backup -> correctness_harness.py.bak_{ts})")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
