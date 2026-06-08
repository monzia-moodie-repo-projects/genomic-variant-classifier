#!/usr/bin/env python3
"""
patch_centralize_test_splice_count.py
=====================================
Repoint tests/unit/test_splice_ai_promotion.py's length test at the single
source of truth EXPECTED_TABULAR_FEATURE_COUNT instead of a hardcoded 79, so it
never needs a per-feature edit again.

Edits (guarded, count == 1 unless noted):
  A. add EXPECTED_TABULAR_FEATURE_COUNT to the variant_ensemble import block
  B. assertion `== 79,` -> `== EXPECTED_TABULAR_FEATURE_COUNT,`
  C. message `Expected 79 TABULAR_FEATURES` -> `Expected {EXPECTED_TABULAR_FEATURE_COUNT} TABULAR_FEATURES`
  D. (best-effort) docstring de-hardcode

.bak, AST-verified, idempotent, BOM-free. Run from the repo root.
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("tests/unit/test_splice_ai_promotion.py")

IMPORT_OLD = (
    "from genomic_variant_classifier.models.variant_ensemble import (\n"
    "    TABULAR_FEATURES,\n"
)
IMPORT_NEW = (
    "from genomic_variant_classifier.models.variant_ensemble import (\n"
    "    TABULAR_FEATURES,\n"
    "    EXPECTED_TABULAR_FEATURE_COUNT,\n"
)

ASSERT_OLD = "    assert len(TABULAR_FEATURES) == 79, ("
ASSERT_NEW = "    assert len(TABULAR_FEATURES) == EXPECTED_TABULAR_FEATURE_COUNT, ("

MSG_OLD = "Expected 79 TABULAR_FEATURES"
MSG_NEW = "Expected {EXPECTED_TABULAR_FEATURE_COUNT} TABULAR_FEATURES"

DOC_OLD = '    """TABULAR_FEATURES must have exactly 79 entries (70 existing + 3 FinnGen)."""'
DOC_NEW = '    """TABULAR_FEATURES length must equal EXPECTED_TABULAR_FEATURE_COUNT (single source of truth)."""'


def _guard(text: str, old: str, label: str, mandatory: bool = True) -> int:
    n = text.count(old)
    if n != 1:
        if mandatory:
            print(f"  ABORT: {label} anchor found {n}x (expected 1)."); sys.exit(2)
        print(f"  WARN  {label} found {n}x; skipped (best-effort).")
    return n


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root."); sys.exit(2)
    original = TARGET.read_text(encoding="utf-8")

    if "EXPECTED_TABULAR_FEATURE_COUNT" in original:
        print("  SKIP  test_splice already references EXPECTED_TABULAR_FEATURE_COUNT. No changes.")
        return

    text = original
    _guard(text, IMPORT_OLD, "import block"); text = text.replace(IMPORT_OLD, IMPORT_NEW, 1); print("  OK    added EXPECTED import")
    _guard(text, ASSERT_OLD, "length assertion"); text = text.replace(ASSERT_OLD, ASSERT_NEW, 1); print("  OK    repointed assertion")
    _guard(text, MSG_OLD, "assertion message"); text = text.replace(MSG_OLD, MSG_NEW, 1); print("  OK    repointed message")
    if _guard(text, DOC_OLD, "docstring", mandatory=False) == 1:
        text = text.replace(DOC_OLD, DOC_NEW, 1); print("  OK    de-hardcoded docstring")

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
