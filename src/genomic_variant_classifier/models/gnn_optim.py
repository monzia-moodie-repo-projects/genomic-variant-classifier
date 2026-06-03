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
