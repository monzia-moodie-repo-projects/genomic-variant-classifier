#!/usr/bin/env python3
"""patch_train_utf8_stdio.py -- force UTF-8 stdout/stderr in scripts/train.py so
Unicode in reports/logs (evaluator's box-drawing separator U+2500, the Greek
delta in variant_ensemble's blend-AUROC log, connector '->' arrows) cannot crash
on a Windows cp1252 console (the smoke's PHASE-4 UnicodeEncodeError).

Inserts a small _force_utf8_stdio() helper + call immediately BEFORE
logging.basicConfig so the StreamHandler inherits the reconfigured stderr.
No-op on platforms/streams without .reconfigure. Idempotent, py_compile-gated,
newline-preserving, ASCII. Author: Monzia Moodie."""
from __future__ import annotations

import ast
import py_compile
import shutil
import sys
from pathlib import Path

TRAIN = Path("scripts/train.py")
MARKER = "_force_utf8_stdio"
ANCHOR = "logging.basicConfig("

BLOCK = (
    "def _force_utf8_stdio(streams=None):\n"
    '    """Make stdout/stderr UTF-8 so Unicode in reports/logs does not crash a\n'
    '    Windows cp1252 console. No-op where .reconfigure is unavailable."""\n'
    "    if streams is None:\n"
    "        streams = (sys.stdout, sys.stderr)\n"
    "    for _stream in streams:\n"
    '        _reconfigure = getattr(_stream, "reconfigure", None)\n'
    "        if _reconfigure is not None:\n"
    "            try:\n"
    '                _reconfigure(encoding="utf-8", errors="replace")\n'
    "            except (ValueError, OSError):\n"
    "                pass\n"
    "\n"
    "\n"
    "_force_utf8_stdio()\n"
    "\n"
    "\n"
)


def _read(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def _write(p, text):
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def main() -> int:
    if not TRAIN.exists():
        print("ERROR: run from repo root (scripts/train.py not found).")
        return 2
    raw = _read(TRAIN)
    if MARKER in raw:
        print("train.py: already patched (idempotent)")
        return 0
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if text.count("\n" + ANCHOR) != 1:
        print(f"ERROR: anchor '{ANCHOR}' at line start count "
              f"{text.count(chr(10) + ANCHOR)} != 1")
        return 2
    text = text.replace("\n" + ANCHOR, "\n" + BLOCK + ANCHOR, 1)
    shutil.copy2(TRAIN, TRAIN.with_suffix(TRAIN.suffix + ".bak"))
    _write(TRAIN, text.replace("\n", nl))
    py_compile.compile(str(TRAIN), doraise=True)
    ast.parse(_read(TRAIN).replace("\r\n", "\n"))
    print("train.py: patched (+_force_utf8_stdio before basicConfig); py_compile OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
