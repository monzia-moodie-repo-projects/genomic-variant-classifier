#!/usr/bin/env python3
"""Tier-2 edge-denoise wiring for gnn.py (idempotent, count-guarded, backup-first, ast-validated).

Threads opt-in STRING edge-denoising (threshold mode) through build_pyg_dataset and
train_gnn_pipeline. Default edge_denoise="none" is behavior-identical to pre-patch.
Requires the Tier-1 patch already applied (imports bf16_autocast from gnn_optim).
"""
from __future__ import annotations
import ast, shutil, sys
from pathlib import Path

MARKER = 'edge_denoise: str = "none"'

# (old, new, expected_count) — each old must appear exactly expected_count times
EDITS = [
    # E1: extend the Tier-1 import to also bring denoise_string_edges
    ("from genomic_variant_classifier.models.gnn_optim import bf16_autocast\n",
     "from genomic_variant_classifier.models.gnn_optim import bf16_autocast, denoise_string_edges\n",
     1),
    # E2: build_pyg_dataset signature
    ('    node_feature_cols: list[str],\n'
     '    label_col: str = "acmg_label",\n'
     ') -> SharedFocalGraph:',
     '    node_feature_cols: list[str],\n'
     '    label_col: str = "acmg_label",\n'
     '    edge_denoise: str = "none",\n'
     '    edge_denoise_tau: float = 0.0,\n'
     ') -> SharedFocalGraph:',
     1),
    # E3: apply denoise after the edge-construction block, before the node tensor
    ('        edge_attr = torch.zeros((0, 3), dtype=torch.float)\n'
     '\n'
     '    x = torch.tensor(gene_features, dtype=torch.float)',
     '        edge_attr = torch.zeros((0, 3), dtype=torch.float)\n'
     '\n'
     '    edge_index, edge_attr = denoise_string_edges(\n'
     '        edge_index, edge_attr, mode=edge_denoise, tau=edge_denoise_tau\n'
     '    )\n'
     '\n'
     '    x = torch.tensor(gene_features, dtype=torch.float)',
     1),
    # E4: train_gnn_pipeline signature (after the Tier-1 precision param)
    ('    graph: Optional[nx.Graph] = None,\n'
     '    precision: str = "fp32",\n'
     ') -> tuple[VariantGAT, GNNTrainer, list[dict]]:',
     '    graph: Optional[nx.Graph] = None,\n'
     '    precision: str = "fp32",\n'
     '    edge_denoise: str = "none",\n'
     '    edge_denoise_tau: float = 0.0,\n'
     ') -> tuple[VariantGAT, GNNTrainer, list[dict]]:',
     1),
    # E5: thread the params into the build_pyg_dataset call inside the pipeline
    ('    ds = build_pyg_dataset(variant_df, graph, node_feature_cols)\n',
     '    ds = build_pyg_dataset(\n'
     '        variant_df, graph, node_feature_cols,\n'
     '        edge_denoise=edge_denoise, edge_denoise_tau=edge_denoise_tau,\n'
     '    )\n',
     1),
]

def main(path_str: str) -> int:
    path = Path(path_str)
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"SKIP: marker already present in {path} (idempotent no-op)")
        return 0
    # pre-count guard
    for old, _new, n in EDITS:
        c = src.count(old)
        if c != n:
            print(f"ABORT: expected {n} occurrence(s) of:\n---\n{old}\n---\ngot {c}. No changes written.")
            return 2
    backup = path.with_suffix(path.suffix + ".tier2.bak")
    shutil.copy2(path, backup)
    out = src
    for old, new, _n in EDITS:
        out = out.replace(old, new, 1)
    # ast-validate before writing
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched source fails to parse: {e}. No changes written (backup at {backup}).")
        return 3
    path.write_text(out, encoding="utf-8")
    print(f"patched {path}")
    print(f"backup  {backup}")
    print(f"applied {len(EDITS)} edits")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "src/genomic_variant_classifier/models/gnn.py"))
