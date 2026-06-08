#!/usr/bin/env python3
"""
patch_ablate_gnn_use_full_graph.py
==================================
Fix the GNN non-degeneracy/timing probe in scripts/ablate_gnn.py so it mirrors
the production scorer.

Bug: ablate_gnn builds `df` = X_train + gene_symbol (from meta_train) but NO
`variant_id`. `GNNScorer.from_trainer` only builds its vid->gene map when BOTH
`variant_id` and `gene_symbol` are present, so with variant_id absent the map is
empty -> every gene resolves to the 0.5 default -> the reported `gnn_score_std`
is ~0 BY CONSTRUCTION, even when the GNN trained fine. That makes the mandated
2-epoch probe report a dead GNN regardless of reality.

Fix: use `GNNScorer.from_full_graph(trainer, full)` (the inductive, gene-keyed
scorer that production uses at run_phase2_eval.py:441). It keys on
`dataset.node_genes` (populated by build_pyg_dataset), so `.score_dataframe(df)`
maps real, varying per-gene scores onto df's gene_symbol.

Single substring substitution (indentation-independent), count == 1, .bak,
AST-verified, idempotent, BOM-free. Run from the repo root.
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("scripts/ablate_gnn.py")
OLD = "GNNScorer.from_trainer(trainer, full, df).score_dataframe(df)"
NEW = "GNNScorer.from_full_graph(trainer, full).score_dataframe(df)"


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root."); sys.exit(2)
    original = TARGET.read_text(encoding="utf-8")

    if NEW in original:
        print("  SKIP  ablate_gnn already uses from_full_graph. No changes.")
        return

    n = original.count(OLD)
    if n == 0:
        print("  ABORT: from_trainer probe line not found (file drifted). No change."); sys.exit(2)
    if n != 1:
        print(f"  ABORT: anchor found {n}x (expected 1). Manual review."); sys.exit(2)

    text = original.replace(OLD, NEW, 1)
    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"ABORT: patched file fails AST parse: {exc}"); sys.exit(3)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{stamp}")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print("  OK    probe scorer from_trainer -> from_full_graph (mirrors production)")
    print(f"PATCHED {TARGET}  (backup {backup.name})")
    print("AST parse: OK")


if __name__ == "__main__":
    main()
