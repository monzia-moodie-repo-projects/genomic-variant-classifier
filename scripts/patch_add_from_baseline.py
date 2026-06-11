#!/usr/bin/env python3
"""patch_add_from_baseline.py -- add SchemaDriftAgent.from_baseline(), a classmethod that
reconstructs a detector from a schema-baseline JSON (scripts/build_schema_baseline.py output).
Idempotent, backup-first, py_compile-gated, ASCII-only. Author: Monzia Moodie.
"""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/agent_layer/agents/schema_drift_agent.py")
ANCHOR = "    def detect(self, df: pd.DataFrame) -> SchemaDriftResult:\n"
MARKER = "def from_baseline("
BLOCK = (
    "    @classmethod\n"
    "    def from_baseline(cls, baseline_path, output_dir):\n"
    "        \"\"\"Reconstruct a detector from a schema-baseline JSON.\n"
    "\n"
    "        The pandera schema is rebuilt from expected_dtypes with nullable columns so that\n"
    "        degenerate-but-present (all-NaN) columns do not raise false nullability violations\n"
    "        against their own baseline. pandera is imported lazily (optional dep).\n"
    "        \"\"\"\n"
    "        import pandera.pandas as pa\n"
    "        data = json.loads(Path(baseline_path).read_text(encoding=\"utf-8\"))\n"
    "        expected_dtypes = {str(k): str(v) for k, v in data[\"expected_dtypes\"].items()}\n"
    "        schema = pa.DataFrameSchema(\n"
    "            {col: pa.Column(dtype, nullable=True) for col, dtype in expected_dtypes.items()}\n"
    "        )\n"
    "        return cls(\n"
    "            schema=schema,\n"
    "            expected_dtypes=expected_dtypes,\n"
    "            expected_schema_hash=data[\"expected_schema_hash\"],\n"
    "            output_dir=Path(output_dir),\n"
    "        )\n"
    "\n"
)

def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found"); return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if MARKER in text:
        print("already applied (from_baseline present); no change."); return 0
    if text.count(ANCHOR) != 1:
        print(f"ABORT: expected exactly 1 detect() anchor; found {text.count(ANCHOR)}"); return 1
    text = text.replace(ANCHOR, BLOCK + ANCHOR, 1)
    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(text.replace("\n", nl), encoding="utf-8", newline="")
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, TARGET)
        print(f"ABORT: py_compile failed, restored backup:\n{exc}"); return 1
    print(f"OK: from_baseline added; backup at {bak}"); return 0

if __name__ == "__main__":
    sys.exit(main())
