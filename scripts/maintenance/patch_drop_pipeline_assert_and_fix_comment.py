#!/usr/bin/env python3
"""
patch_drop_pipeline_assert_and_fix_comment.py
=============================================
In src/genomic_variant_classifier/api/pipeline.py:
  A. Remove the import-time `assert len(INFERENCE_FEATURE_COLUMNS) == 79` block
     (4 lines). A feature edit must never be able to crash the API on import;
     the length contract now lives in tests/unit/test_feature_count_contract.py.
     The assert is replaced by an explanatory comment (line 41's derivation stays).
  B. De-number the stale docstring comment "derive the 64 INFERENCE_FEATURE_COLUMNS"
     -> "derive the INFERENCE_FEATURE_COLUMNS" so it can't go stale again.

Each edit is independently guarded (exact-text match, count == 1) and idempotent.
.bak, AST-verified, BOM-free. Run from the repo root.
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/api/pipeline.py")

ASSERT_BLOCK = (
    "assert len(INFERENCE_FEATURE_COLUMNS) == 79, (\n"
    '    f"INFERENCE_FEATURE_COLUMNS has {len(INFERENCE_FEATURE_COLUMNS)} entries; "\n'
    '    "expected 79.  Update TABULAR_FEATURES in src/genomic_variant_classifier/models/variant_ensemble.py."\n'
    ")"
)
ASSERT_REPLACEMENT = (
    "# INFERENCE_FEATURE_COLUMNS is derived from TABULAR_FEATURES above; its length\n"
    "# is enforced by tests/unit/test_feature_count_contract.py rather than asserted\n"
    "# at import time (a feature edit must not crash the API on import)."
)

COMMENT_OLD = "derive the 64 INFERENCE_FEATURE_COLUMNS"
COMMENT_NEW = "derive the INFERENCE_FEATURE_COLUMNS"


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root."); sys.exit(2)
    original = TARGET.read_text(encoding="utf-8")
    text = original
    changed = False

    # ---- Edit A: drop the assert block ----
    if ASSERT_BLOCK in text:
        if text.count(ASSERT_BLOCK) != 1:
            print(f"  ABORT: assert block found {text.count(ASSERT_BLOCK)}x (expected 1)."); sys.exit(2)
        text = text.replace(ASSERT_BLOCK, ASSERT_REPLACEMENT, 1)
        changed = True
        print("  OK    removed import-time assert block")
    elif ASSERT_REPLACEMENT.splitlines()[0] in text:
        print("  SKIP  assert already removed")
    else:
        print("  ABORT: assert block not found and replacement comment absent -- drift."); sys.exit(2)

    # ---- Edit B: de-number the comment ----
    c = text.count(COMMENT_OLD)
    if c == 1:
        text = text.replace(COMMENT_OLD, COMMENT_NEW, 1)
        changed = True
        print("  OK    de-numbered line-108 comment (64 -> none)")
    elif COMMENT_NEW in text:
        print("  SKIP  comment already de-numbered")
    else:
        print(f"  WARN  comment anchor found {c}x; left unchanged (cosmetic).")

    if not changed:
        print("  Nothing to do (already applied)."); return

    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"ABORT: patched file fails AST parse: {exc}"); sys.exit(3)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{stamp}")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print(f"PATCHED {TARGET}  (backup {backup.name})")
    print("AST parse: OK")


if __name__ == "__main__":
    main()
