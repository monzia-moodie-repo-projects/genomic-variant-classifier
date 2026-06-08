#!/usr/bin/env python3
"""
patch_align_clingen_cast_to_float.py
====================================
Align the clingen_validity_score cast in
src/genomic_variant_classifier/data/real_data_prep.py from `.astype(int)` to
`.astype(float)`, matching the inference-path builder
(variant_ensemble.engineer_features already casts float).

Why: real_data_prep is the TRAINING feature builder; variant_ensemble is the
INFERENCE builder (imported by api/pipeline.py). The two must produce identical
columns AND dtypes. ClinGen currently maps validity to integers 0-5, so this
changes no values today (0 -> 0.0), but it removes a train/serve dtype
divergence and future-proofs against a fractional ClinGen score (which int
truncation would silently lose while inference keeps it).

Anchored on the UNIQUE clingen block (not bare `.astype(int)`, which appears
many times for genuinely-integer binary features that must stay int).
Guarded (count == 1), .bak, AST-verified, idempotent, BOM-free.
Run from the repo root.
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")

# The clingen assignment block, ending in the int cast. Match the whole block so
# we cannot accidentally touch any other `.astype(int)` in the file.
OLD = (
    '        feats["clingen_validity_score"] = (\n'
    '            df.get("clingen_validity_score", pd.Series([0] * len(df), index=df.index))\n'
    "            .fillna(0)\n"
    "            .astype(int)\n"
    "        )"
)
NEW = (
    '        feats["clingen_validity_score"] = (\n'
    '            df.get("clingen_validity_score", pd.Series([0] * len(df), index=df.index))\n'
    "            .fillna(0)\n"
    "            .astype(float)  # match inference builder (variant_ensemble); int truncated a future fractional score\n"
    "        )"
)


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root."); sys.exit(2)
    original = TARGET.read_text(encoding="utf-8")

    if NEW in original or ".astype(float)  # match inference builder" in original:
        print("  SKIP  clingen cast already float. No changes.")
        return

    n = original.count(OLD)
    if n == 0:
        print("  ABORT: clingen int-cast block not found (file drifted from expected)."); sys.exit(2)
    if n != 1:
        print(f"  ABORT: clingen block found {n}x (expected 1). Manual review."); sys.exit(2)

    text = original.replace(OLD, NEW, 1)
    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"ABORT: patched file fails AST parse: {exc}"); sys.exit(3)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{stamp}")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print("  OK    clingen_validity_score cast int -> float (inference-aligned)")
    print(f"PATCHED {TARGET}  (backup {backup.name})")
    print("AST parse: OK")


if __name__ == "__main__":
    main()
