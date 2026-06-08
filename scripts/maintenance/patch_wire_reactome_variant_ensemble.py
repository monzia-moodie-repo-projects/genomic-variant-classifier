#!/usr/bin/env python3
"""
patch_wire_reactome_variant_ensemble.py
=======================================
Lockstep half #1: wire `reactome_pathway_count` into
src/genomic_variant_classifier/models/variant_ensemble.py.

Two edits (each guarded count==1; .bak backup; AST-verified; idempotent):
  1. Append "reactome_pathway_count" to TABULAR_FEATURES (last entry).
  2. Emit feats["reactome_pathway_count"] in engineer_features(), as the LAST
     feature before `return feats.reset_index(drop=True)`.

This MUST be applied together with patch_wire_reactome_real_data_prep.py so the
two feature builders stay column-for-column identical (same set AND order).

Run from the repo root. Safe to re-run (detects the marker, no-ops).
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/models/variant_ensemble.py")

MARKER = 'feats["reactome_pathway_count"]'

OLD_TF = (
    "    # gnomAD v4.1 constraint (4)\n"
    '    "pli_score",\n'
    '    "loeuf",\n'
    '    "syn_z",\n'
    '    "mis_z",\n'
    "]"
)
NEW_TF = (
    "    # gnomAD v4.1 constraint (4)\n"
    '    "pli_score",\n'
    '    "loeuf",\n'
    '    "syn_z",\n'
    '    "mis_z",\n'
    "    # Reactome pathway membership (1) - Phase D\n"
    '    "reactome_pathway_count",\n'
    "]"
)

OLD_EF = (
    '    feats["mis_z"] = (\n'
    '        df.get("mis_z", pd.Series([0.0] * len(df), index=df.index))\n'
    "        .fillna(0.0)\n"
    "        .astype(float)\n"
    "    )\n"
    "\n"
    "    return feats.reset_index(drop=True)"
)
NEW_EF = (
    '    feats["mis_z"] = (\n'
    '        df.get("mis_z", pd.Series([0.0] * len(df), index=df.index))\n'
    "        .fillna(0.0)\n"
    "        .astype(float)\n"
    "    )\n"
    "\n"
    "    # Reactome pathway membership (1) - Phase D\n"
    '    feats["reactome_pathway_count"] = (\n'
    '        df.get("reactome_pathway_count", pd.Series([0] * len(df), index=df.index))\n'
    "        .fillna(0)\n"
    "        .astype(int)\n"
    "        .clip(lower=0)\n"
    "    )\n"
    "\n"
    "    return feats.reset_index(drop=True)"
)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n == 0:
        print(f"  ABORT {label}: anchor not found (file drifted from expected).")
        sys.exit(2)
    if n > 1:
        print(f"  ABORT {label}: anchor found {n}x (expected 1). Manual review.")
        sys.exit(2)
    print(f"  OK    {label}: 1 replacement")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root.")
        sys.exit(2)
    original = TARGET.read_text(encoding="utf-8")
    if MARKER in original:
        print("  SKIP  already wired (reactome_pathway_count present). No changes.")
        return
    text = _replace_once(original, OLD_TF, NEW_TF, "TABULAR_FEATURES")
    text = _replace_once(text, OLD_EF, NEW_EF, "engineer_features")
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
