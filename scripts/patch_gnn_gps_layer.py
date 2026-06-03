#!/usr/bin/env python3
"""Add opt-in layer_type='gps' to train_gnn_pipeline (default 'gat' = behavior-identical).
Imports VariantGATGPS and switches the model at instantiation. Idempotent, backup-first,
count-guarded, ast-validated, line-ending agnostic."""
from __future__ import annotations
import ast, shutil, sys
from pathlib import Path

MARKER = 'layer_type: str = "gat"'
EDITS = [
    ("from genomic_variant_classifier.models.gnn_optim import bf16_autocast, denoise_string_edges\n",
     "from genomic_variant_classifier.models.gnn_optim import bf16_autocast, denoise_string_edges, VariantGATGPS\n",
     1),
    ('    edge_denoise: str = "none",\n'
     "    edge_denoise_tau: float = 0.0,\n"
     ") -> tuple[VariantGAT, GNNTrainer, list[dict]]:",
     '    edge_denoise: str = "none",\n'
     "    edge_denoise_tau: float = 0.0,\n"
     '    layer_type: str = "gat",\n'
     ") -> tuple[VariantGAT, GNNTrainer, list[dict]]:",
     1),
    ("    model = VariantGAT(in_channels=in_channels, hidden_channels=128, heads=8)\n",
     '    if layer_type == "gps":\n'
     "        model = VariantGATGPS(in_channels=in_channels, hidden_channels=128, heads=4)\n"
     '    elif layer_type == "gat":\n'
     "        model = VariantGAT(in_channels=in_channels, hidden_channels=128, heads=8)\n"
     "    else:\n"
     '        raise ValueError(f"unknown layer_type: {layer_type!r} (expected \'gat\' or \'gps\')")\n',
     1),
]

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    data = raw.replace("\r\n", "\n")
    if MARKER in data:
        print(f"SKIP: {path} already has layer_type (idempotent no-op)"); return 0
    for old, _new, n in EDITS:
        c = data.count(old)
        if c != n:
            print(f"ABORT: expected {n} of an anchor, got {c}; no change. Head:\n{old[:70]!r}"); return 2
    out = data
    for old, new, _n in EDITS:
        out = out.replace(old, new, 1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched source invalid: {e}; no change"); return 3
    final = out.replace("\n", nl) if nl == "\r\n" else out
    backup = path.with_suffix(path.suffix + ".gps.bak")
    shutil.copy2(path, backup)
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"patched {path} (backup {backup}); applied {len(EDITS)} edits; endings={'CRLF' if nl==chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "src/genomic_variant_classifier/models/gnn.py"))
