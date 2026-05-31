#!/usr/bin/env python3
"""Make test_cohort_guard.py's duplicate assertion wording-agnostic.

The cohort guard's duplicate message changed to "duplicate variant identity" when it gained
locus-derived-key support (duplicates are flagged whether keyed on variant_id or chrom:pos:ref:alt).
This relaxes the existing test's match from "duplicate variant_id" to the stable substring
"duplicate variant" so it verifies the behaviour without coupling to exact wording.
Idempotent; newline/BOM-safe.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("tests/unit/test_cohort_guard.py")
OLD = 'match="duplicate variant_id"'
NEW = 'match="duplicate variant"'


def main() -> int:
    if not TARGET.exists():
        print(f"NOT FOUND: {TARGET}")
        return 2
    raw = TARGET.read_bytes()
    text = raw.decode("utf-8")
    nl = "\r\n" if b"\r\n" in raw else "\n"
    work = text.replace("\r\n", "\n")
    if OLD not in work and NEW in work:
        print("already patched (idempotent no-op)")
        return 0
    n = work.count(OLD)
    if n != 1:
        print(f"anchor count={n} (expected 1); NO changes made.")
        return 3
    work = work.replace(OLD, NEW, 1)
    TARGET.write_bytes(work.replace("\n", nl).encode("utf-8"))
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
