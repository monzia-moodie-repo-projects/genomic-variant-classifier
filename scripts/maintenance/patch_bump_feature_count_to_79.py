#!/usr/bin/env python3
"""
patch_bump_feature_count_to_79.py
=================================
Bump the two hardcoded feature-count tripwires from 78 -> 79 after Reactome
added reactome_pathway_count to TABULAR_FEATURES (the third coordinated edit -
the KNOWN_ZERO_DEFAULT allowlist in tests/unit/test_correctness_harness.py - is
handled by a separate patcher once that file's exact set literal is in hand).

Edits (each guarded count==1; .bak per file; AST-verified; idempotent):
  A. src/genomic_variant_classifier/api/pipeline.py
       - assert len(INFERENCE_FEATURE_COLUMNS) == 78  ->  == 79
       - stale message  "expected 74."  ->  "expected 79."
       - stale comment  "the exact 55 columns"  ->  "the exact 79 columns"
  B. tests/unit/test_splice_ai_promotion.py
       - assert len(TABULAR_FEATURES) == 78  ->  == 79
       - stale message  "Expected 74 TABULAR_FEATURES"  ->  "Expected 79 ..."
       - stale docstring "exactly 78 entries" -> "exactly 79 entries"

All anchors are indentation-independent substrings, so they match whether the
code sits at module level or inside a function. Run from the repo root.

NOTE on design (flagged, not auto-fixed): these are magic-number tripwires
duplicated across prod + tests. Each new feature (COSMIC/TCGA/KEGG next) will
trip them again. Recommend centralising the expected count into one constant;
this patcher only restores green for the Reactome addition.
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

PIPELINE = Path("src/genomic_variant_classifier/api/pipeline.py")
SPLICE_TEST = Path("tests/unit/test_splice_ai_promotion.py")


def _sub_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n == 0:
        print(f"  ABORT {label}: anchor not found.")
        sys.exit(2)
    if n > 1:
        print(f"  ABORT {label}: anchor found {n}x (expected 1).")
        sys.exit(2)
    print(f"  OK    {label}")
    return text.replace(old, new, 1)


def _patch_file(path: Path, edits: list[tuple[str, str, str]], done_marker: str) -> bool:
    if not path.exists():
        print(f"  ABORT: {path} not found. Run from the repo root.")
        sys.exit(2)
    original = path.read_text(encoding="utf-8")
    if done_marker in original:
        print(f"  SKIP  {path.name}: already at 79. No changes.")
        return False
    text = original
    for old, new, label in edits:
        text = _sub_once(text, old, new, f"{path.name}: {label}")
    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"  ABORT: {path} fails AST parse after edit: {exc}")
        sys.exit(3)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{stamp}")
    backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"  PATCHED {path}  (backup {backup.name})")
    return True


def main() -> None:
    any_changed = False
    any_changed |= _patch_file(
        PIPELINE,
        [
            ("len(INFERENCE_FEATURE_COLUMNS) == 78",
             "len(INFERENCE_FEATURE_COLUMNS) == 79", "assert 78->79"),
            ("expected 74.  Update TABULAR_FEATURES",
             "expected 79.  Update TABULAR_FEATURES", "message 74->79"),
            ("the exact 55 columns", "the exact 79 columns", "comment 55->79"),
        ],
        done_marker="len(INFERENCE_FEATURE_COLUMNS) == 79",
    )
    any_changed |= _patch_file(
        SPLICE_TEST,
        [
            ("len(TABULAR_FEATURES) == 78",
             "len(TABULAR_FEATURES) == 79", "assert 78->79"),
            ("Expected 74 TABULAR_FEATURES",
             "Expected 79 TABULAR_FEATURES", "message 74->79"),
            ("exactly 78 entries", "exactly 79 entries", "docstring 78->79"),
        ],
        done_marker="len(TABULAR_FEATURES) == 79",
    )
    print("AST parse: OK" if any_changed else "Nothing to do (already at 79).")


if __name__ == "__main__":
    main()
