#!/usr/bin/env python3
"""
patch_centralize_test_api_count.py
==================================
Repoint tests/unit/test_api.py at the single source of truth
EXPECTED_TABULAR_FEATURE_COUNT instead of the hardcoded 79:
  A. add a module-level import after `from pathlib import Path`
  B. _make_pipeline mock literal      n_features = 79          -> EXPECTED_TABULAR_FEATURE_COUNT
  C. assertion  body["n_features"] == 79                       -> == EXPECTED_TABULAR_FEATURE_COUNT
  D. assertion  len(body["feature_names"]) == 79               -> == EXPECTED_TABULAR_FEATURE_COUNT

After this, adding a feature + bumping the one constant updates the API test
automatically; no more per-feature edits here.

Whitespace-tolerant regex for B/C/D (each count == 1); exact-text import anchor
(count == 1). .bak, AST-verified, idempotent, BOM-free. Run from the repo root.
"""
from __future__ import annotations

import ast
import datetime as _dt
import re
import sys
from pathlib import Path

TARGET = Path("tests/unit/test_api.py")

IMPORT_ANCHOR = "from pathlib import Path"
IMPORT_NEW = (
    "from pathlib import Path\n"
    "from genomic_variant_classifier.models.variant_ensemble import EXPECTED_TABULAR_FEATURE_COUNT"
)

RE_LITERAL = re.compile(r"(\bn_features\s*=\s*)79\b")
RE_ASSERT_NF = re.compile(r'(body\["n_features"\]\s*==\s*)79\b')
RE_ASSERT_FN = re.compile(r'(len\(body\["feature_names"\]\)\s*==\s*)79\b')
REPL = r"\g<1>EXPECTED_TABULAR_FEATURE_COUNT"


def _count(rx, text):
    return len(rx.findall(text))


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root."); sys.exit(2)
    original = TARGET.read_text(encoding="utf-8")

    if "EXPECTED_TABULAR_FEATURE_COUNT" in original:
        print("  SKIP  test_api already references EXPECTED_TABULAR_FEATURE_COUNT. No changes.")
        return

    # Guards
    ia = original.count(IMPORT_ANCHOR)
    if ia != 1:
        print(f"  ABORT: import anchor '{IMPORT_ANCHOR}' found {ia}x (expected 1)."); sys.exit(2)
    a, b, c = _count(RE_LITERAL, original), _count(RE_ASSERT_NF, original), _count(RE_ASSERT_FN, original)
    print(f"  counts: literal(n_features=79)={a}  assert(n_features==79)={b}  assert(feature_names==79)={c}")
    if not (a == 1 and b == 1 and c == 1):
        print("  ABORT: expected exactly one of each 79-anchor. Manual review."); sys.exit(2)

    text = original.replace(IMPORT_ANCHOR, IMPORT_NEW, 1)
    text = RE_LITERAL.sub(REPL, text)
    text = RE_ASSERT_NF.sub(REPL, text)
    text = RE_ASSERT_FN.sub(REPL, text)

    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"ABORT: patched file fails AST parse: {exc}"); sys.exit(3)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{stamp}")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print("  OK    added import + repointed literal and 2 assertions to EXPECTED_TABULAR_FEATURE_COUNT")
    print(f"PATCHED {TARGET}  (backup {backup.name})")
    print("AST parse: OK")


if __name__ == "__main__":
    main()
