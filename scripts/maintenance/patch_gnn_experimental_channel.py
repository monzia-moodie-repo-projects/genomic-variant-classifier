#!/usr/bin/env python3
"""Fix the STRING edge-channel column name in StringDBGraph.build().

Bug (confirmed against the downloaded protein.links.detailed.v12.0 columns):
the file's experimental-evidence channel is named "experimental", but build()
references "experiments" in both the _CHANNELS guard list and the add_edge
call. The guard therefore zero-fills a phantom "experiments" column and the
edge attr reads those zeros, silently nulling the experimental channel of the
3-channel edge weights. Real columns:
    [... 'coexpression', 'experimental', 'database', 'textmining', ...]

Two exact, unique anchors. Count-guarded (each must match exactly once),
backup-first, idempotent, AST-verified, line-ending preserving.

Usage:  python patch_gnn_experimental_channel.py [path/to/gnn.py]
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

DEFAULT = "src/genomic_variant_classifier/models/gnn.py"

REPLACEMENTS = [
    (
        '        _CHANNELS = ["experiments", "database", "coexpression"]',
        '        _CHANNELS = ["experimental", "database", "coexpression"]',
    ),
    (
        '                experimental=float(row["experiments"]) / 1000.0,',
        '                experimental=float(row["experimental"]) / 1000.0,',
    ),
]

APPLIED_MARKERS = [new for _old, new in REPLACEMENTS]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT)
    if not path.exists():
        print(f"FATAL: {path} not found (run from repo root or pass the path).")
        return 2

    raw = path.read_bytes()
    original = raw.decode("utf-8")

    if all(marker in original for marker in APPLIED_MARKERS):
        print("Already patched (all markers present); no changes made.")
        return 0

    text = original
    for old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            print(f"FATAL: anchor matched {n} times (expected exactly 1): {old!r}")
            return 3
        text = text.replace(old, new, 1)

    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"FATAL: patched source fails AST parse: {exc}")
        return 4

    backup = path.with_suffix(path.suffix + f".bak_chan_{time.strftime('%Y%m%d_%H%M%S')}")
    backup.write_bytes(raw)
    path.write_bytes(text.encode("utf-8"))
    print(f"Patched {path}")
    print(f"  backup: {backup.name}")
    print("  2 anchors replaced; AST OK; line endings preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
