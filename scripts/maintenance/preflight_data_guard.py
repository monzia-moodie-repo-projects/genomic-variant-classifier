#!/usr/bin/env python3
"""
scripts/maintenance/preflight_data_guard.py  --  Monzia Moodie

Fail-loud guard: verify data/ is a usable REAL local directory (or a junction
that currently resolves) with the canonical subtrees present, BEFORE any run or
test touches it. Catches the dangling-junction / shadow-file incident class
early with a clear message instead of a deep mkdir traceback.

Returns 0 if usable, 1 if the data/ path is broken. Importable as
`assert_data_usable()` to wire into run scripts / conftest.

Usage:
  python scripts/maintenance/preflight_data_guard.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REQUIRED = ["external", "raw", "processed"]


def assert_data_usable(data_dir: str | Path = "data") -> None:
    p = Path(data_dir)
    isjunction = getattr(os.path, "isjunction", lambda _p: False)
    if os.path.lexists(p) and not p.exists():
        raise SystemExit(
            f"[data-guard] '{p}' is a DANGLING junction/symlink (target gone -- e.g. Google "
            "Drive G: not mounted/synced). Mount/sync the target or re-point data/, then retry."
        )
    if p.exists() and not p.is_dir():
        raise SystemExit(f"[data-guard] '{p}' exists but is NOT a directory (a stray file shadows it). "
                         "Remove/rename it and restore data/ (git or setup_data_tree.py).")
    if not p.exists():
        raise SystemExit(f"[data-guard] '{p}' is missing. Run scripts/maintenance/setup_data_tree.py.")
    missing = [s for s in _REQUIRED if not (p / s).is_dir()]
    if missing:
        raise SystemExit(f"[data-guard] '{p}' is missing canonical subtrees {missing}. "
                         "Run scripts/maintenance/setup_data_tree.py.")
    kind = "junction(resolves)" if (isjunction(p) or os.path.islink(p)) else "real dir"
    print(f"[data-guard] OK -- '{p}' usable ({kind}); subtrees present.")


if __name__ == "__main__":
    try:
        assert_data_usable(sys.argv[1] if len(sys.argv) > 1 else "data")
    except SystemExit as e:
        print(e)
        sys.exit(1)
    sys.exit(0)
