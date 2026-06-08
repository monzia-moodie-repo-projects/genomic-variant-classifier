#!/usr/bin/env python3
"""
patch_datapreprep_mkdir_guard.py   (OPTIONAL defense-in-depth)
==============================================================
DataPrepConfig.__post_init__ has the same eager-mkdir antipattern as
FetchConfig: it calls

    self.output_dir.mkdir(parents=True, exist_ok=True)

on a CWD-relative default (data/splits). When a stray file / dangling
symlink shadows data/, this raises the cryptic WinError 183 at construction.

This patcher wraps ONLY that one line in a try/except that re-raises a clear,
actionable NotADirectoryError. It is self-contained (no new imports) and
preserves the happy-path behaviour exactly. Strictly defense-in-depth: the
guard test (tests/unit/test_data_dir_not_shadowed.py) already converts a
recurrence into a single clear failure, so this is optional.

Target: src/genomic_variant_classifier/data/real_data_prep.py
Run from the repo root. Guarded count==1 abort; .bak backup; AST verify.
Safe to re-run (no-ops if already wrapped).
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")

OLD = "        self.output_dir.mkdir(parents=True, exist_ok=True)\n"
NEW = (
    "        try:\n"
    "            self.output_dir.mkdir(parents=True, exist_ok=True)\n"
    "        except FileExistsError as _exc:  # 'data/' shadowed by a non-dir\n"
    "            raise NotADirectoryError(\n"
    "                f\"Cannot create {self.output_dir!s}: a path component \"\n"
    "                f\"exists as a non-directory (stray file or dangling \"\n"
    "                f\"symlink/junction shadowing data/). Remove or rename it \"\n"
    "                f\"and restore data/ from git, then retry.\"\n"
    "            ) from _exc\n"
)


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root.")
        sys.exit(2)

    original = TARGET.read_text(encoding="utf-8")

    if "stray file or dangling" in original:
        print("  SKIP  output_dir mkdir already guarded.")
        return

    n = original.count(OLD)
    if n == 0:
        print("  ABORT anchor not found. real_data_prep.py drifted from the "
              "expected `self.output_dir.mkdir(parents=True, exist_ok=True)` "
              "line. Inspect DataPrepConfig.__post_init__ manually.")
        sys.exit(2)
    if n > 1:
        print(f"  ABORT anchor found {n}x (expected 1). Manual review needed.")
        sys.exit(2)

    text = original.replace(OLD, NEW, 1)

    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"ABORT: patched file fails AST parse: {exc}")
        sys.exit(3)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{stamp}")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print(f"PATCHED {TARGET}  (backup {backup.name})")
    print("  OK    output_dir mkdir wrapped in clear-error guard")
    print("AST parse: OK")


if __name__ == "__main__":
    main()
