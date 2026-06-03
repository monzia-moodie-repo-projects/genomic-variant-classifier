from __future__ import annotations
from contextlib import nullcontext
import torch
import torch.nn as nn

def bf16_autocast(device, enabled: bool = True):
    dev_type = getattr(device, "type", device)
    if enabled and dev_type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

def denoise_string_edges(edge_index, edge_attr, mode="none", tau=0.0, weights=(1.0,1.0,1.0)):
    if mode == "none":
        return edge_index, edge_attr
    if mode != "threshold":
        raise ValueError(f"unknown denoise mode: {mode!r}")
    if edge_attr is None or edge_attr.numel() == 0:
        return edge_index, edge_attr
    w = torch.tensor(weights, dtype=edge_attr.dtype, device=edge_attr.device)
    score = (edge_attr * w).sum(dim=1) / w.sum()
    keep = score >= tau
    return edge_index[:, keep], edge_attr[keep]

class EdgeGate(nn.Module):
    def __init__(self, edge_dim: int = 3):
        super().__init__()
        self.lin = nn.Linear(edge_dim, 1)
    def forward(self, edge_attr):
        gate = torch.sigmoid(self.lin(edge_attr))
        return edge_attr * gate, gate

# ---------------------------------------------------------------------------
# GraphGPS hybrid variant: local edge-aware GATConv + global Performer attention.
# Drop-in for VariantGAT (identical forward signature) so GNNTrainer/GNNScorer
# need no changes. Opt-in via train_gnn_pipeline(layer_type="gps").
# ---------------------------------------------------------------------------
from torch_geometric.nn import GATConv as _GATConv, GPSConv as _GPSConv


class VariantGATGPS(nn.Module):
    """GraphGPS hybrid: per layer, a local edge-aware GATConv plus global Performer
    (linear) attention. forward(x, edge_index, gene_idx, edge_attr) -> (n_focal, 2)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        attn_type: str = "performer",
        gate_edges: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_gate = EdgeGate(edge_dim=3) if gate_edges else None
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local = _GATConv(
                hidden_channels, hidden_channels, heads=heads, concat=False,
                dropout=dropout, edge_dim=3, add_self_loops=False,
            )
            self.convs.append(
                _GPSConv(
                    channels=hidden_channels, conv=local, heads=heads,
                    dropout=dropout, attn_type=attn_type,
                    attn_kwargs={"dropout": dropout},
                )
            )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x, edge_index, gene_idx, edge_attr=None):
        if self.edge_gate is not None and edge_attr is not None and edge_attr.numel() > 0:
            edge_attr, _ = self.edge_gate(edge_attr)
        h = self.input_proj(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr=edge_attr)
        focal = h[gene_idx]
        return self.classifier(focal)
