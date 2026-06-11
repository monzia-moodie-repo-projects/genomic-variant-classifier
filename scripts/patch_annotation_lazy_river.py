#!/usr/bin/env python3
"""patch_annotation_lazy_river.py -- guard the optional `river` import in
annotation_policy_agent.py so the module/orchestrator import without river installed.
Idempotent, backup-first, py_compile-gated, ASCII-only. Author: Monzia Moodie.
"""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/agent_layer/agents/annotation_policy_agent.py")
OLD = "from river import drift as river_drift"
NEW = (
    "try:\n"
    "    from river import drift as river_drift\n"
    "except ModuleNotFoundError:  # optional dep: required only when detection runs\n"
    "    river_drift = None"
)
MARKER = "river_drift = None"

def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found"); return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if MARKER in text:
        print("already applied (river guard present); no change."); return 0
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
    print(f"OK: river guard applied; backup at {bak}"); return 0

if __name__ == "__main__":
    sys.exit(main())
