#!/usr/bin/env python3
"""Append VariantGATGPS (GraphGPS hybrid) to gnn_optim.py. Idempotent, backup-first,
ast-validated, line-ending preserving."""
from __future__ import annotations
import ast, shutil, sys
from pathlib import Path

MARKER = "class VariantGATGPS"
BLOCK = '# ---------------------------------------------------------------------------\n# GraphGPS hybrid variant: local edge-aware GATConv + global Performer attention.\n# Drop-in for VariantGAT (identical forward signature) so GNNTrainer/GNNScorer\n# need no changes. Opt-in via train_gnn_pipeline(layer_type="gps").\n# ---------------------------------------------------------------------------\nfrom torch_geometric.nn import GATConv as _GATConv, GPSConv as _GPSConv\n\n\nclass VariantGATGPS(nn.Module):\n    """GraphGPS hybrid: per layer, a local edge-aware GATConv plus global Performer\n    (linear) attention. forward(x, edge_index, gene_idx, edge_attr) -> (n_focal, 2)."""\n\n    def __init__(\n        self,\n        in_channels: int,\n        hidden_channels: int = 128,\n        num_layers: int = 3,\n        heads: int = 4,\n        dropout: float = 0.3,\n        attn_type: str = "performer",\n        gate_edges: bool = False,\n    ) -> None:\n        super().__init__()\n        self.dropout = dropout\n        self.input_proj = nn.Linear(in_channels, hidden_channels)\n        self.edge_gate = EdgeGate(edge_dim=3) if gate_edges else None\n        self.convs = nn.ModuleList()\n        for _ in range(num_layers):\n            local = _GATConv(\n                hidden_channels, hidden_channels, heads=heads, concat=False,\n                dropout=dropout, edge_dim=3, add_self_loops=False,\n            )\n            self.convs.append(\n                _GPSConv(\n                    channels=hidden_channels, conv=local, heads=heads,\n                    dropout=dropout, attn_type=attn_type,\n                    attn_kwargs={"dropout": dropout},\n                )\n            )\n        self.classifier = nn.Sequential(\n            nn.Linear(hidden_channels, 32),\n            nn.ReLU(),\n            nn.Dropout(dropout),\n            nn.Linear(32, 2),\n        )\n\n    def forward(self, x, edge_index, gene_idx, edge_attr=None):\n        if self.edge_gate is not None and edge_attr is not None and edge_attr.numel() > 0:\n            edge_attr, _ = self.edge_gate(edge_attr)\n        h = self.input_proj(x)\n        for conv in self.convs:\n            h = conv(h, edge_index, edge_attr=edge_attr)\n        focal = h[gene_idx]\n        return self.classifier(focal)\n'

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    if MARKER in raw:
        print(f"SKIP: {path} already has VariantGATGPS (idempotent no-op)"); return 0
    body = raw.replace("\r\n", "\n")
    if not body.endswith("\n"):
        body += "\n"
    out = body + "\n" + BLOCK
    if not out.endswith("\n"):
        out += "\n"
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: appended source invalid: {e}; no change"); return 3
    final = out.replace("\n", nl) if nl == "\r\n" else out
    backup = path.with_suffix(path.suffix + ".gpsappend.bak")
    shutil.copy2(path, backup)
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"appended VariantGATGPS to {path} (backup {backup}); endings={'CRLF' if nl==chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "src/genomic_variant_classifier/models/gnn_optim.py"))
