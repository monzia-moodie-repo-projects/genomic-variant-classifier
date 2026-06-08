#!/usr/bin/env python3
"""
patch_add_expected_feature_count.py
===================================
Insert the single source of truth `EXPECTED_TABULAR_FEATURE_COUNT = 79`
immediately before `TABULAR_FEATURES = [` in
src/genomic_variant_classifier/models/variant_ensemble.py.

This is the ONE count literal a human bumps per feature; every other count
guard references it (test_feature_count_contract, test_splice, test_api), and
the import-time assert in api/pipeline.py is removed in favour of the test.

Guarded (anchor count must == 1), .bak, AST-verified, idempotent, BOM-free.
Run from the repo root.
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/models/variant_ensemble.py")
ANCHOR = "TABULAR_FEATURES = ["
BLOCK = (
    "# Single source of truth for the per-variant tabular feature count.\n"
    "# Bump by +/-1 whenever you add or remove an entry in TABULAR_FEATURES below.\n"
    "# Enforced by tests/unit/test_feature_count_contract.py against both the list\n"
    "# length and INFERENCE_FEATURE_COLUMNS; that test is the deliberate-bump tripwire.\n"
    "EXPECTED_TABULAR_FEATURE_COUNT = 79\n\n"
    "TABULAR_FEATURES = ["
)


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root."); sys.exit(2)
    original = TARGET.read_text(encoding="utf-8")

    if "EXPECTED_TABULAR_FEATURE_COUNT" in original:
        print("  SKIP  EXPECTED_TABULAR_FEATURE_COUNT already present. No changes.")
        return

    n = original.count(ANCHOR)
    if n != 1:
        print(f"  ABORT: anchor '{ANCHOR}' found {n}x (expected 1). Manual review.")
        sys.exit(2)

    text = original.replace(ANCHOR, BLOCK, 1)
    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"ABORT: patched file fails AST parse: {exc}"); sys.exit(3)

    # Sanity: the constant value must match the actual list length at runtime.
    # (We can't import here without heavy deps, so this is checked by the new test;
    #  we just confirm the literal we inserted is the intended 79.)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{stamp}")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print("  OK    inserted EXPECTED_TABULAR_FEATURE_COUNT = 79 before TABULAR_FEATURES")
    print(f"PATCHED {TARGET}  (backup {backup.name})")
    print("AST parse: OK")


if __name__ == "__main__":
    main()
