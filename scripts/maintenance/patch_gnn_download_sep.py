#!/usr/bin/env python3
"""Fix StringDBGraph._download_gz to accept an explicit CSV separator and parse
the STRING protein.info (names) file as TAB-delimited on the DOWNLOAD path.

Bug: _download_gz hardcodes sep=" ". That is correct for the space-delimited
protein.links.detailed file, but wrong for protein.info (TAB-delimited, with a
free-text annotation column full of spaces), so the download path raises
pandas.errors.ParserError. The LOCAL info path already uses sep="\\t" (gnn.py
line ~112); this aligns the download path with it.

Three exact, unique anchors. Count-guarded (each must match exactly once),
backup-first (.bak_<ts>), idempotent (skips if already applied), AST-verified,
line-ending preserving (operates on bytes/decoded text without newline
translation).

Usage:  python patch_gnn_download_sep.py [path/to/gnn.py]
Default path: src/genomic_variant_classifier/models/gnn.py (run from repo root).
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

DEFAULT = "src/genomic_variant_classifier/models/gnn.py"

REPLACEMENTS = [
    (
        '    def _download_gz(self, url: str) -> pd.DataFrame:',
        '    def _download_gz(self, url: str, sep: str = " ") -> pd.DataFrame:',
    ),
    (
        '            return pd.read_csv(fh, sep=" ")',
        '            return pd.read_csv(fh, sep=sep)',
    ),
    (
        '            df = self._download_gz(STRING_NAMES_URL)',
        '            df = self._download_gz(STRING_NAMES_URL, sep="\\t")',
    ),
]

# If every "new" string is already present, the patch has been applied.
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

    backup = path.with_suffix(path.suffix + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")
    backup.write_bytes(raw)
    path.write_bytes(text.encode("utf-8"))
    print(f"Patched {path}")
    print(f"  backup: {backup.name}")
    print("  3 anchors replaced; AST OK; line endings preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
