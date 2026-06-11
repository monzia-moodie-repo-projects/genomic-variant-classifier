#!/usr/bin/env python3
"""patch_schema_lazy_pandera.py -- make pandera a lazy (run-time) import in
schema_drift_agent.py so the module/orchestrator import without pandera installed.
Idempotent, backup-first, py_compile-gated, ASCII-only. Author: Monzia Moodie.
"""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/agent_layer/agents/schema_drift_agent.py")
MODULE_IMPORT = "import pandas as pd\nimport pandera.pandas as pa"
MODULE_IMPORT_NEW = "import pandas as pd"
DETECT_SIG = "    def detect(self, df: pd.DataFrame) -> SchemaDriftResult:\n"
LAZY_LINE = "        import pandera.pandas as pa  # lazy: required only when detection runs\n"
MARKER = "import pandera.pandas as pa  # lazy"

def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found"); return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if MARKER in text:
        print("already applied (lazy pandera marker present); no change."); return 0
    if text.count(MODULE_IMPORT) != 1:
        print(f"ABORT: expected exactly 1 module-level pandera import block; found {text.count(MODULE_IMPORT)}"); return 1
    if text.count(DETECT_SIG) != 1:
        print(f"ABORT: expected exactly 1 detect() signature; found {text.count(DETECT_SIG)}"); return 1
    text = text.replace(MODULE_IMPORT, MODULE_IMPORT_NEW, 1)
    text = text.replace(DETECT_SIG, DETECT_SIG + LAZY_LINE, 1)
    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(text.replace("\n", nl), encoding="utf-8", newline="")
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, TARGET)
        print(f"ABORT: py_compile failed, restored backup:\n{exc}"); return 1
    print(f"OK: lazy pandera applied; backup at {bak}"); return 0

if __name__ == "__main__":
    sys.exit(main())
