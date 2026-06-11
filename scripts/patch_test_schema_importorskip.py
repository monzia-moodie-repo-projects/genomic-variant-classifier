#!/usr/bin/env python3
"""patch_test_schema_importorskip.py -- gate test_schema_drift_monitor_agent.py on
pandera via pytest.importorskip so it SKIPS (not errors) where pandera is absent (CI).
Idempotent, backup-first, py_compile-gated, ASCII-only. Author: Monzia Moodie.
"""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

TARGET = Path("tests/unit/test_schema_drift_monitor_agent.py")
OLD = "import pandera.pandas as pa"
NEW = 'import pytest\npa = pytest.importorskip("pandera.pandas")'
MARKER = 'pytest.importorskip("pandera.pandas")'

def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found"); return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if MARKER in text:
        print("already applied (importorskip present); no change."); return 0
    if text.count(OLD) != 1:
        print(f"ABORT: expected exactly 1 `{OLD}`; found {text.count(OLD)}"); return 1
    text = text.replace(OLD, NEW, 1)
    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(text.replace("\n", nl), encoding="utf-8", newline="")
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, TARGET)
        print(f"ABORT: py_compile failed, restored backup:\n{exc}"); return 1
    print(f"OK: importorskip applied; backup at {bak}"); return 0

if __name__ == "__main__":
    sys.exit(main())
