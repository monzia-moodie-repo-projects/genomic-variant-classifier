#!/usr/bin/env python3
"""patch_variant_ensemble_ascii.py -- ASCII-clean variant_ensemble.py so log
messages can't crash a Windows cp1252 console.

The all-models smoke completed but logged a non-fatal UnicodeEncodeError at the
'OOF blend AUROC ... Delta=' line: under PowerShell 2>&1 the logging handler's
stderr stays cp1252, which cannot encode Greek Delta (U+0394). cp1252 *can*
encode the 5 em-dashes (U+2014) in this file, so they only mojibake -- but we
remove them too for cleanliness. Replaces U+0394 -> 'delta', U+2014 -> '-'.

evaluator.py's box-drawing report is intentionally left Unicode: it prints via
stdout, which train.py's _force_utf8_stdio reconfigures successfully.

Idempotent (skips if already ASCII), py_compile/ast-gated, newline-preserving,
ASCII patcher. Author: Monzia Moodie."""
from __future__ import annotations

import ast
import py_compile
import shutil
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/models/variant_ensemble.py")
REPLACEMENTS = [("\u0394", "delta"), ("\u2014", "-")]


def _read(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def _write(p, text):
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def _non_ascii_count(text):
    return sum(1 for ch in text if ord(ch) > 127)


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root).")
        return 2
    raw = _read(TARGET)
    if _non_ascii_count(raw) == 0:
        print("variant_ensemble.py: already ASCII-clean (idempotent)")
        return 0
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    before = _non_ascii_count(text)
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    after = _non_ascii_count(text)
    if after != 0:
        leftovers = sorted({hex(ord(c)) for c in text if ord(c) > 127})
        print(f"ERROR: {after} non-ASCII char(s) remain after known replacements: {leftovers}")
        return 2
    ast.parse(text)
    shutil.copy2(TARGET, TARGET.with_suffix(TARGET.suffix + ".bak"))
    _write(TARGET, text.replace("\n", nl))
    py_compile.compile(str(TARGET), doraise=True)
    print(f"variant_ensemble.py: ASCII-cleaned ({before} non-ASCII -> 0); py_compile OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
