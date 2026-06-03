#!/usr/bin/env python3
"""Add `pytest.importorskip("torch_geometric")` to GNN test modules that import the
GNN stack without a guard (matches the repo convention used for torch/catboost/etc.).

Idempotent, backup-first, ast-validated, line-ending preserving. Run:
    python scripts/add_gnn_importorskip.py tests/unit/test_gnn_optim.py tests/unit/test_gnn_shared_graph.py
"""
from __future__ import annotations
import ast, shutil, sys
from pathlib import Path

MARKER = 'importorskip("torch_geometric")'
GNN_IMPORT = "from genomic_variant_classifier.models.gnn import"

def fix(path_str: str) -> int:
    path = Path(path_str)
    if not path.exists():
        print(f"ABORT: {path} does not exist"); return 2
    data = path.open(encoding="utf-8", newline="").read()  # newline="" preserves CRLF/LF exactly
    nl = "\r\n" if "\r\n" in data else "\n"
    if MARKER in data:
        print(f"SKIP: {path} already guarded (idempotent no-op)"); return 0
    lines = data.splitlines(keepends=True)
    idx = next((i for i, l in enumerate(lines) if l.startswith(GNN_IMPORT)), None)
    if idx is None:
        print(f"ABORT: no top-level '{GNN_IMPORT}' line found in {path}; shape unexpected, no change"); return 3
    block = ("import pytest" + nl
             + 'pytest.importorskip("torch_geometric")  # GNN tests need PyG; skip where absent (e.g. CI)' + nl)
    lines.insert(idx, block)
    out = "".join(lines)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched {path} fails to parse: {e}; no change"); return 4
    backup = path.with_suffix(path.suffix + ".ciguard.bak")
    shutil.copy2(path, backup)
    path.open("w", encoding="utf-8", newline="").write(out)
    print(f"guarded {path}  (backup {backup})")
    return 0

if __name__ == "__main__":
    targets = sys.argv[1:] or [
        "tests/unit/test_gnn_optim.py",
        "tests/unit/test_gnn_shared_graph.py",
    ]
    rc = 0
    for t in targets:
        rc = fix(t) or rc
    sys.exit(rc)
