#!/usr/bin/env python3
"""
patch_wire_reactome_real_data_prep.py
=====================================
Lockstep half #2: wire Reactome into
src/genomic_variant_classifier/data/real_data_prep.py.

Three edits (each guarded count==1; .bak backup; AST-verified; idempotent):
  1. AnnotationConfig: add `reactome_path: Optional[Path] = None`.
  2. _annotate_scores: add step 18 (ReactomeConnector) after the gnomAD-constraint
     step, before `return df` (inline import, matching the per-step convention).
  3. _engineer_features: emit feats["reactome_pathway_count"] as the LAST feature,
     immediately after the mis_z block and before the trailing n_nan fill — so the
     column lands at the same position as in variant_ensemble.engineer_features.

Apply together with patch_wire_reactome_variant_ensemble.py so the two feature
builders stay column-for-column identical. Run from the repo root. Re-runnable.
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")

MARKER = 'feats["reactome_pathway_count"]'

# --- edit 1: AnnotationConfig field -----------------------------------------
OLD_CFG = "    gnomad_constraint_path: Optional[Path] = None  # Phase 3C: gnomAD constraint TSV"
NEW_CFG = (
    "    gnomad_constraint_path: Optional[Path] = None  # Phase 3C: gnomAD constraint TSV\n"
    "    reactome_path: Optional[Path] = None  # Phase D: Reactome gene pathway-count parquet"
)

# --- edit 2: _annotate_scores step 18 ---------------------------------------
OLD_STEP = (
    "        constraint = GnomADConstraintConnector(tsv_path=ac.gnomad_constraint_path)\n"
    "        df = constraint.annotate_dataframe(df)\n"
    "        logger.info(\n"
    '            "Score annotation 17/17 (gnomAD constraint): %d genes with pLI > 0.",\n'
    "            int(\n"
    "                (\n"
    '                    df.get("pli_score", pd.Series([0.0] * len(df), index=df.index)) > 0\n'
    "                ).sum()\n"
    "            ),\n"
    "        )\n"
    "\n"
    "        return df"
)
NEW_STEP = (
    "        constraint = GnomADConstraintConnector(tsv_path=ac.gnomad_constraint_path)\n"
    "        df = constraint.annotate_dataframe(df)\n"
    "        logger.info(\n"
    '            "Score annotation 17/17 (gnomAD constraint): %d genes with pLI > 0.",\n'
    "            int(\n"
    "                (\n"
    '                    df.get("pli_score", pd.Series([0.0] * len(df), index=df.index)) > 0\n'
    "                ).sum()\n"
    "            ),\n"
    "        )\n"
    "\n"
    "        # 18. Reactome gene pathway count (Phase D)\n"
    "        from genomic_variant_classifier.data.reactome import ReactomeConnector\n"
    "\n"
    "        reactome = ReactomeConnector(pathway_path=ac.reactome_path)\n"
    "        df = reactome.annotate_dataframe(df)\n"
    "        logger.info(\n"
    '            "Score annotation 18/18 (Reactome): %d variants with reactome_pathway_count > 0.",\n'
    "            int(\n"
    "                (\n"
    "                    df.get(\n"
    '                        "reactome_pathway_count",\n'
    "                        pd.Series([0] * len(df), index=df.index),\n"
    "                    )\n"
    "                    > 0\n"
    "                ).sum()\n"
    "            ),\n"
    "        )\n"
    "\n"
    "        return df"
)

# --- edit 3: _engineer_features block (last feature, before n_nan fill) ------
OLD_EF = (
    '        feats["mis_z"] = (\n'
    '            df.get("mis_z", pd.Series([0.0] * len(df), index=df.index))\n'
    "            .fillna(0.0)\n"
    "            .astype(float)\n"
    "        )\n"
    "\n"
    "        n_nan = feats.isnull().sum().sum()"
)
NEW_EF = (
    '        feats["mis_z"] = (\n'
    '            df.get("mis_z", pd.Series([0.0] * len(df), index=df.index))\n'
    "            .fillna(0.0)\n"
    "            .astype(float)\n"
    "        )\n"
    "\n"
    "        # Reactome pathway membership (1) - Phase D\n"
    '        feats["reactome_pathway_count"] = (\n'
    '            df.get("reactome_pathway_count", pd.Series([0] * len(df), index=df.index))\n'
    "            .fillna(0)\n"
    "            .astype(int)\n"
    "            .clip(lower=0)\n"
    "        )\n"
    "\n"
    "        n_nan = feats.isnull().sum().sum()"
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
    if MARKER in original or "reactome_path" in original:
        print("  SKIP  already wired (reactome marker present). No changes.")
        return
    text = _replace_once(original, OLD_CFG, NEW_CFG, "AnnotationConfig.reactome_path")
    text = _replace_once(text, OLD_STEP, NEW_STEP, "_annotate_scores step 18")
    text = _replace_once(text, OLD_EF, NEW_EF, "_engineer_features block")
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
