#!/usr/bin/env python3
"""
patch_protein_pipeline_mkdir_guard.py
=====================================
Close the SAME data/-shadow vector in ProteinStructurePipeline that we already
closed in FetchConfig and DataPrepConfig.

ProteinStructurePipeline.__init__ eagerly calls

    self.cache_dir.mkdir(parents=True, exist_ok=True)

on a CWD-relative default (data/raw/cache/alphafold). Step 14 of
_annotate_scores constructs this pipeline on EVERY data-prep run (even in stub
mode, since protein_cache_dir defaults to that path), so a stray file or a
dangling junction named 'data' re-detonates with the cryptic
FileExistsError [WinError 183] here — a path NOT covered by
tests/unit/test_data_dir_not_shadowed.py.

This wraps the eager mkdir in the same clear-error guard used by
DataPrepConfig, turning WinError 183 into an actionable NotADirectoryError.
Behaviour when data/ is healthy is unchanged. No new imports.

Target: src/genomic_variant_classifier/pipelines/protein_pipeline.py
Guarded count==1 abort; .bak backup; AST verify; idempotent.
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/pipelines/protein_pipeline.py")

MARKER = "except FileExistsError as _exc:  # 'data/' shadowed"

OLD = "        self.cache_dir.mkdir(parents=True, exist_ok=True)\n"
NEW = (
    "        try:\n"
    "            self.cache_dir.mkdir(parents=True, exist_ok=True)\n"
    "        except FileExistsError as _exc:  # 'data/' shadowed by a non-dir\n"
    "            raise NotADirectoryError(\n"
    "                f\"Cannot create {self.cache_dir!s}: a path component exists as a \"\n"
    "                f\"non-directory (stray file or dangling symlink/junction shadowing \"\n"
    "                f\"data/). Remove or rename it and restore data/ from git, then retry.\"\n"
    "            ) from _exc\n"
)


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root.")
        sys.exit(2)
    original = TARGET.read_text(encoding="utf-8")
    if MARKER in original:
        print("  SKIP  protein_pipeline mkdir already guarded. No changes.")
        return
    n = original.count(OLD)
    if n == 0:
        print("  ABORT anchor not found. protein_pipeline.py drifted from the "
              "expected `self.cache_dir.mkdir(parents=True, exist_ok=True)` line.")
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
    print("  OK    cache_dir mkdir wrapped in clear-error guard")
    print("AST parse: OK")


if __name__ == "__main__":
    main()
