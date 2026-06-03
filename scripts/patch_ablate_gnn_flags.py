#!/usr/bin/env python3
"""Wire --layer-type / --edge-denoise / --edge-denoise-tau into ablate_gnn.py and
thread them into BOTH train_gnn_pipeline and the scorer's build_pyg_dataset
(scorer-consistency: the scored graph must match the trained graph). Records the
config in the JSON row. Idempotent, backup-first, ast-validated, line-ending agnostic."""
from __future__ import annotations
import ast, shutil, sys
from pathlib import Path

MARKER = '"--layer-type"'
EDITS = [
    ('    ap.add_argument("--seed", type=int, default=42)\n',
     '    ap.add_argument("--seed", type=int, default=42)\n'
     '    ap.add_argument("--layer-type", choices=["gat", "gps"], default="gat")\n'
     '    ap.add_argument("--edge-denoise", choices=["none", "threshold"], default="none")\n'
     '    ap.add_argument("--edge-denoise-tau", type=float, default=0.0)\n',
     1),
    ("    model, trainer, hist = train_gnn_pipeline(df, feat, graph=graph, epochs=a.epochs, test_split=0.2)\n",
     "    model, trainer, hist = train_gnn_pipeline(\n"
     "        df, feat, graph=graph, epochs=a.epochs, test_split=0.2,\n"
     "        layer_type=a.layer_type, edge_denoise=a.edge_denoise, edge_denoise_tau=a.edge_denoise_tau,\n"
     "    )\n",
     1),
    ("    full = build_pyg_dataset(df, graph, feat)\n",
     "    full = build_pyg_dataset(\n"
     "        df, graph, feat,\n"
     "        edge_denoise=a.edge_denoise, edge_denoise_tau=a.edge_denoise_tau,  # scorer-consistency\n"
     "    )\n",
     1),
    ("    row = summarize(a.tag, hist, float(sc.std()), peak, wall, len(df), device)\n",
     "    row = summarize(a.tag, hist, float(sc.std()), peak, wall, len(df), device)\n"
     '    row["layer_type"] = a.layer_type\n'
     '    row["edge_denoise"] = a.edge_denoise\n'
     '    row["edge_denoise_tau"] = a.edge_denoise_tau\n',
     1),
]

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    data = raw.replace("\r\n", "\n")
    if MARKER in data:
        print(f"SKIP: {path} already wired (idempotent)"); return 0
    for old, _new, n in EDITS:
        if data.count(old) != n:
            print(f"ABORT: expected {n} of an anchor, got {data.count(old)}; no change. Head:\n{old[:60]!r}"); return 2
    out = data
    for old, new, _n in EDITS:
        out = out.replace(old, new, 1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched source invalid: {e}; no change"); return 3
    final = out.replace("\n", nl) if nl == "\r\n" else out
    backup = path.with_suffix(path.suffix + ".flags.bak")
    shutil.copy2(path, backup)
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"patched {path} (backup {backup}); applied {len(EDITS)} edits; endings={'CRLF' if nl==chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "scripts/ablate_gnn.py"))
