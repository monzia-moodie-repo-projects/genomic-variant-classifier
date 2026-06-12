#!/usr/bin/env python3
"""patch_correctness_harness_maxentscan_delta.py -- keep the correctness-harness
reference slice in sync with the feature set: generate maxentscan_delta with
non-zero synthetic values so stage 5 does not flag it as a silent-zero. This is
the correct fix (NOT adding it to KNOWN_ZERO_DEFAULT, which would mask a real
future regression). Idempotent, backup-first, py_compile-gated, ASCII.
Author: Monzia Moodie."""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py")
MARKER = "maxentscan_delta"

# Substring anchor (robust to leading indent and multi-entry-per-line packing).
OLD = '"maxentscan_score": rng.uniform(-5, 12, n),'
NEW = ('"maxentscan_score": rng.uniform(-5, 12, n), '
       '"maxentscan_delta": rng.uniform(-10, 10, n),')

def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found"); return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if MARKER in text:
        print("already applied (maxentscan_delta present); no change."); return 0
    c = text.count(OLD)
    if c != 1:
        print(f"ABORT: anchor found {c} times (expected 1); no change."); return 1
    text = text.replace(OLD, NEW, 1)
    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(text.replace("\n", nl), encoding="utf-8", newline="")
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, TARGET); print(f"ABORT: py_compile failed, restored:\n{exc}"); return 1
    print(f"OK: correctness_harness reference slice updated; backup at {bak}"); return 0

if __name__ == "__main__":
    sys.exit(main())
