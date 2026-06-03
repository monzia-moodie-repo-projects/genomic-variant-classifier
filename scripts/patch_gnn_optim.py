r"""
patch_gnn_optim.py - apply Tier-1 GNN optimizations to gnn.py (bf16 autocast + device-once).
Idempotent, count-guarded (each edit must match exactly once), backup-first, ast-validated.
Usage:  python patch_gnn_optim.py [path-to-gnn.py]
"""
from __future__ import annotations
import ast
import shutil
import sys
from pathlib import Path

MARKER = 'precision: str = "fp32",'

EDITS = [
    # 1. import the helper
    ("from torch_geometric.nn import GATConv\n",
     "from torch_geometric.nn import GATConv\n\nfrom genomic_variant_classifier.models.gnn_optim import bf16_autocast\n"),
    # 2. add precision param to GNNTrainer.__init__
    ('        device: Optional[str] = None,\n        checkpoint_path: str = "models/best_gat.pt",\n    ) -> None:',
     '        device: Optional[str] = None,\n        checkpoint_path: str = "models/best_gat.pt",\n        precision: str = "fp32",\n    ) -> None:'),
    # 3. store precision
    ("        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)\n        self.history: list[dict] = []",
     "        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)\n        self.precision = precision\n        self.history: list[dict] = []"),
    # 4. device-once cache in _graph_tensors
    ("    def _graph_tensors(self, ds: SharedFocalGraph):\n        return (\n            ds.x.to(self.device),\n            ds.edge_index.to(self.device),\n            ds.edge_attr.to(self.device),\n        )",
     "    def _graph_tensors(self, ds: SharedFocalGraph):\n        cache = getattr(self, \"_gt_cache\", None)\n        if cache is None:\n            cache = {}\n            self._gt_cache = cache\n        key = id(ds)\n        if key not in cache:\n            cache[key] = (\n                ds.x.to(self.device),\n                ds.edge_index.to(self.device),\n                ds.edge_attr.to(self.device),\n            )\n        return cache[key]"),
    # 5. bf16 wrap in train_epoch
    ("        self.optimizer.zero_grad()\n        out = self.model(x, ei, focal, edge_attr=ea)   # one forward over the whole graph\n        loss = F.cross_entropy(out, y)\n        loss.backward()",
     "        self.optimizer.zero_grad()\n        with bf16_autocast(self.device, enabled=(self.precision == \"bf16\")):\n            out = self.model(x, ei, focal, edge_attr=ea)   # one forward over the whole graph\n            loss = F.cross_entropy(out, y)\n        loss.backward()"),
    # 6. bf16 wrap in evaluate (+ float before softmax)
    ("        out = self.model(x, ei, focal, edge_attr=ea)\n        proba = F.softmax(out, dim=-1)[:, 1].cpu().numpy()\n        labels = ds.y.cpu().numpy()",
     "        with bf16_autocast(self.device, enabled=(self.precision == \"bf16\")):\n            out = self.model(x, ei, focal, edge_attr=ea)\n        proba = F.softmax(out.float(), dim=-1)[:, 1].cpu().numpy()\n        labels = ds.y.cpu().numpy()"),
    # 7. bf16 wrap in predict_proba (+ float before softmax)
    ("            x, ei, ea = self._graph_tensors(ds)\n            focal = ds.focal_idx.to(self.device)\n            out = self.model(x, ei, focal, edge_attr=ea)\n            return F.softmax(out, dim=-1)[:, 1].cpu().numpy()",
     "            x, ei, ea = self._graph_tensors(ds)\n            focal = ds.focal_idx.to(self.device)\n            with bf16_autocast(self.device, enabled=(self.precision == \"bf16\")):\n                out = self.model(x, ei, focal, edge_attr=ea)\n            return F.softmax(out.float(), dim=-1)[:, 1].cpu().numpy()"),
    # 8. train_gnn_pipeline: add precision param
    ("    batch_size: int = 32,\n    graph: Optional[nx.Graph] = None,\n) -> tuple[VariantGAT, GNNTrainer, list[dict]]:",
     "    batch_size: int = 32,\n    graph: Optional[nx.Graph] = None,\n    precision: str = \"fp32\",\n) -> tuple[VariantGAT, GNNTrainer, list[dict]]:"),
    # 9. pass precision into GNNTrainer
    ("    trainer = GNNTrainer(model, epochs=epochs, batch_size=batch_size)",
     "    trainer = GNNTrainer(model, epochs=epochs, batch_size=batch_size, precision=precision)"),
]

def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src/genomic_variant_classifier/models/gnn.py")
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print("already patched (idempotent) - no change"); return 0
    for old, _ in EDITS:
        c = src.count(old)
        if c != 1:
            print(f"ABORT: expected exactly 1 match, found {c} for:\n---\n{old[:90]}\n---"); return 2
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copyfile(path, bak)
    out = src
    for old, new in EDITS:
        out = out.replace(old, new, 1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched source fails to parse ({e}); restoring backup"); shutil.copyfile(bak, path); return 3
    path.write_text(out, encoding="utf-8")
    print(f"patched {path}\nbackup  {bak}\napplied {len(EDITS)} edits"); return 0

if __name__ == "__main__":
    raise SystemExit(main())
