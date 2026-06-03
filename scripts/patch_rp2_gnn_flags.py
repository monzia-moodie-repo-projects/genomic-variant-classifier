#!/usr/bin/env python3
"""Wire --layer-type / --edge-denoise / --edge-denoise-tau into run_phase2_eval.py:
add the three argparse flags after --string-db, thread them into the keyword-style
train_gnn_pipeline call, and add edge_denoise/tau to the scorer's build_pyg_dataset
(scorer-consistency; the STRING threshold is already shared via _string_kwargs).
Idempotent, backup-first, ast-validated, line-ending agnostic."""
from __future__ import annotations
import ast, shutil, sys
from pathlib import Path

MARKER = '"--layer-type"'

ARG_OLD = (
    '    p.add_argument(\n'
    '        "--string-db",\n'
    '        default=None,\n'
    '        help="Path to STRING DB file, or \'auto\' to use config default",\n'
    '    )\n'
)
ARG_NEW = ARG_OLD + (
    '    p.add_argument(\n'
    '        "--layer-type",\n'
    '        choices=["gat", "gps"],\n'
    '        default="gat",\n'
    '        help="GNN layer type: gat (default) or gps (GraphGPS hybrid).",\n'
    '    )\n'
    '    p.add_argument(\n'
    '        "--edge-denoise",\n'
    '        choices=["none", "threshold"],\n'
    '        default="none",\n'
    '        help="STRING edge denoising applied before GNN train AND score.",\n'
    '    )\n'
    '    p.add_argument(\n'
    '        "--edge-denoise-tau",\n'
    '        type=float,\n'
    '        default=0.0,\n'
    '        help="Weighted-mean STRING confidence cutoff for --edge-denoise threshold.",\n'
    '    )\n'
)

TRAIN_OLD = (
    "                gnn_model, gnn_trainer, gnn_history = train_gnn_pipeline(\n"
    "                    variant_df=gnn_df,\n"
    "                    node_feature_cols=node_feat_cols,\n"
    "                    string_threshold=string_threshold,\n"
    "                    string_kwargs=_string_kwargs,\n"
    "                    test_split=0.15,\n"
    "                    epochs=100,\n"
    "                    batch_size=32,\n"
    "                )\n"
)
TRAIN_NEW = (
    "                gnn_model, gnn_trainer, gnn_history = train_gnn_pipeline(\n"
    "                    variant_df=gnn_df,\n"
    "                    node_feature_cols=node_feat_cols,\n"
    "                    string_threshold=string_threshold,\n"
    "                    string_kwargs=_string_kwargs,\n"
    "                    test_split=0.15,\n"
    "                    epochs=100,\n"
    "                    batch_size=32,\n"
    "                    layer_type=args.layer_type,\n"
    "                    edge_denoise=args.edge_denoise,\n"
    "                    edge_denoise_tau=args.edge_denoise_tau,\n"
    "                )\n"
)

SCORE_OLD = "                full_dataset = build_pyg_dataset(gnn_df, graph, node_feat_cols)\n"
SCORE_NEW = (
    "                full_dataset = build_pyg_dataset(\n"
    "                    gnn_df, graph, node_feat_cols,\n"
    "                    edge_denoise=args.edge_denoise,\n"
    "                    edge_denoise_tau=args.edge_denoise_tau,  # scorer-consistency\n"
    "                )\n"
)

EDITS = [(ARG_OLD, ARG_NEW, 1), (TRAIN_OLD, TRAIN_NEW, 1), (SCORE_OLD, SCORE_NEW, 1)]

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    data = raw.replace("\r\n", "\n")
    if MARKER in data:
        print(f"SKIP: {path} already wired (idempotent)"); return 0
    for old, _new, n in EDITS:
        c = data.count(old)
        if c != n:
            print(f"ABORT: expected {n} of an anchor, got {c}; no change. Head:\n{old.splitlines()[0]!r}"); return 2
    out = data
    for old, new, _n in EDITS:
        out = out.replace(old, new, 1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched source invalid: {e}; no change"); return 3
    final = out.replace("\n", nl) if nl == "\r\n" else out
    backup = path.with_suffix(path.suffix + ".rp2flags.bak")
    shutil.copy2(path, backup)
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"patched {path} (backup {backup}); applied {len(EDITS)} edits; endings={'CRLF' if nl==chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "scripts/run_phase2_eval.py"))
