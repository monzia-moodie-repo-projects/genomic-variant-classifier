"""
scripts/maintenance/patch_constraint_oe_from_loeuf.py
=====================================================
Repoint the dead `gene_constraint_oe` feature at the gnomAD `loeuf` column.

ROOT CAUSE (verified): the gnomAD-constraint connector emits loeuf/pli_score/
syn_z/mis_z but never a `gene_constraint_oe` column, so the engineer_features
`df.get("gene_constraint_oe", <const 1.0>)` always fell to its constant default
-> gene_constraint_oe constant, gene_is_constrained = (1.0 < 0.35) = always 0.
LOEUF *is* the LoF observed/expected upper-bound fraction, so loeuf is the
correct source. This patch makes the fallback read loeuf.

Touches BOTH engineer_features implementations (they must stay in lockstep):
  src/genomic_variant_classifier/data/real_data_prep.py
  src/genomic_variant_classifier/models/variant_ensemble.py

Count-guarded (exactly 1 hit/file), backup-first, idempotent, py_compile-gated.
This CHANGES feature values by design (not equivalence-preserving).
"""
from __future__ import annotations
import datetime as _dt
import py_compile
import sys
from pathlib import Path

_ANCHOR = '"gene_constraint_oe", pd.Series([1.0] * len(df), index=df.index)'
_REPLACE = '"gene_constraint_oe", df.get("loeuf", pd.Series([1.0] * len(df), index=df.index))'

_TARGETS = [
    Path("src/genomic_variant_classifier/data/real_data_prep.py"),
    Path("src/genomic_variant_classifier/models/variant_ensemble.py"),
]


def _patch_one(path: Path) -> str:
    if not path.exists():
        return f"MISSING  {path}"
    text = path.read_text(encoding="utf-8")
    if _REPLACE in text:
        return f"SKIP     {path} (already patched)"
    n = text.count(_ANCHOR)
    if n != 1:
        return f"ABORT    {path}: expected exactly 1 anchor, found {n} (no change)"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_{stamp}")
    bak.write_bytes(path.read_bytes())
    path.write_text(text.replace(_ANCHOR, _REPLACE), encoding="utf-8")
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        path.write_bytes(bak.read_bytes())  # rollback
        return f"ROLLBACK {path}: py_compile failed ({e}); restored from {bak.name}"
    return f"PATCHED  {path} (backup {bak.name})"


def main() -> int:
    rc = 0
    for t in _TARGETS:
        msg = _patch_one(t)
        print(msg)
        if msg.startswith("ABORT") or msg.startswith("ROLLBACK") or msg.startswith("MISSING"):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
