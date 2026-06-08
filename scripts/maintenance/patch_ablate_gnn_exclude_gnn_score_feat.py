#!/usr/bin/env python3
"""
patch_ablate_gnn_exclude_gnn_score_feat.py
==========================================
Make the GNN probe's node-feature set match production.

Bug: ablate_gnn._assemble builds
    feat = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
which INCLUDES the gnn_score column. Production excludes it
(run_phase2_eval.py: node_feat_cols = [c for c in X_train.columns if c != "gnn_score"]),
because gnn_score is the GNN's own output and must not be fed back as a node
feature. With it included, the probe trains a different model than Run 15, so its
wall-clock / VRAM / AUROC are not a valid baseline for Tier-1/Tier-2 ablations.

Fix: add `and c != "gnn_score"` to the comprehension. Safe whether or not the
column is present (no-op filter if absent). Single substring substitution,
count == 1, .bak, AST-verified, idempotent, BOM-free. Run from repo root.
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("scripts/ablate_gnn.py")
OLD = 'feat = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]'
NEW = 'feat = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c != "gnn_score"]'


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root."); sys.exit(2)
    original = TARGET.read_text(encoding="utf-8")

    if 'c != "gnn_score"' in original and OLD not in original:
        print("  SKIP  feat already excludes gnn_score. No changes.")
        return

    n = original.count(OLD)
    if n == 0:
        print("  ABORT: feat comprehension not found (file drifted). No change."); sys.exit(2)
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
    print('  OK    feat now excludes gnn_score (mirrors production node_feat_cols)')
    print(f"PATCHED {TARGET}  (backup {backup.name})")
    print("AST parse: OK")


if __name__ == "__main__":
    main()
