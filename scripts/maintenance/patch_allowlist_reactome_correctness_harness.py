#!/usr/bin/env python3
"""
patch_allowlist_reactome_correctness_harness.py
================================================
Add `reactome_pathway_count` to the KNOWN_ZERO_DEFAULT allowlist in
src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py.

Why: reactome_pathway_count is wired into the feature contract but stub-zero
until data/external/reactome_gene_pathways.parquet exists (reactome_path=None),
exactly like dbsnp_af / eve_score / the other dead connectors already in this
set. The correctness harness stage-5 zero-audit correctly flags it; the fix is
to record it as a KNOWN (intentional) zero, not to silence the audit globally.

Edits (guarded; .bak; AST-verified; idempotent):
  1. MANDATORY: insert "reactome_pathway_count" as the first entry of the
     frozenset literal (anchor: the unique assignment line). Set-literal entry
     order/indent is cosmetic, so this is safe regardless of the other entries.
  2. BEST-EFFORT: bump the "(21 columns" comment to "(22 columns" if present
     exactly once (warn, don't abort, if the count differs).

Run from the repo root. Re-runnable (skips if the entry is already present).
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py")

ANCHOR = "KNOWN_ZERO_DEFAULT: frozenset[str] = frozenset({"
INSERT = (
    "KNOWN_ZERO_DEFAULT: frozenset[str] = frozenset({\n"
    '    "reactome_pathway_count",  # Phase D: stub-zero until reactome parquet built'
)

COMMENT_OLD = "(21 columns"
COMMENT_NEW = "(22 columns"


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root.")
        sys.exit(2)
    original = TARGET.read_text(encoding="utf-8")
    if "reactome_pathway_count" in original:
        print("  SKIP  reactome_pathway_count already in KNOWN_ZERO_DEFAULT. No changes.")
        return

    n = original.count(ANCHOR)
    if n == 0:
        print("  ABORT: KNOWN_ZERO_DEFAULT frozenset anchor not found "
              "(file drifted from expected definition).")
        sys.exit(2)
    if n > 1:
        print(f"  ABORT: anchor found {n}x (expected 1). Manual review.")
        sys.exit(2)
    text = original.replace(ANCHOR, INSERT, 1)
    print("  OK    inserted reactome_pathway_count into frozenset")

    c = text.count(COMMENT_OLD)
    if c == 1:
        text = text.replace(COMMENT_OLD, COMMENT_NEW, 1)
        print("  OK    bumped comment 21 -> 22 columns")
    else:
        print(f"  WARN  '(21 columns' found {c}x; left comment unchanged (cosmetic).")

    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"ABORT: patched file fails AST parse: {exc}")
        sys.exit(3)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{stamp}")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print(f"PATCHED {TARGET}  (backup {backup.name})")
    print("AST parse: OK")


if __name__ == "__main__":
    main()
