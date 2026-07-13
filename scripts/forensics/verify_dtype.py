#!/usr/bin/env python
"""verify_dtype.py (2026-07-11) -- verify the leak-remap dtype fix in split_protocol_v2.py: the int64
promotions are present before the partition loop, the loop is otherwise unchanged, and the module
imports. Read-only + compile + import. ASCII-safe.
"""
from __future__ import annotations
import io, importlib, py_compile, sys
from pathlib import Path
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

def a(s): return s.encode("ascii","replace").decode("ascii")

def main() -> int:
    print("="*78); print("LEAK-REMAP DTYPE-FIX VERIFICATION"); print("="*78)
    sp = Path("src/genomic_variant_classifier/data/split_protocol_v2.py")
    if not sp.exists():
        print("ABORT: split_protocol_v2.py missing"); return 2
    src = sp.read_text(encoding="utf-8", errors="replace")
    checks = []
    def chk(n, c): checks.append((n,c)); print(a(f"  {'ok ' if c else 'FAIL'} {n}"))

    chk("dtype-fix marker present", "Dtype-safe promotion (2026-07-11)" in src)
    chk("count_col int64 promotion", 'out[cfg.count_col] = out[cfg.count_col].astype("int64")' in src)
    chk("derived_flag_col int64 promotion",
        'out[cfg.derived_flag_col] = out[cfg.derived_flag_col].astype("int64")' in src)
    chk("promotion guarded on column presence", "if cfg.count_col in out.columns:" in src)
    chk("partition loop still present (unchanged)", "for p in PARTITIONS:" in src
        and "out.iloc[ix, out.columns.get_loc(cfg.count_col)] = cnt" in src)
    chk("out = df.copy() intact", "out = df.copy()" in src)
    print("-"*78)
    try:
        py_compile.compile(str(sp), doraise=True); chk("split_protocol_v2 compiles", True)
    except Exception as e:
        print(a(f"  FAIL compile: {e}")); checks.append(("compile", False))
    sys.path.insert(0, "src")
    try:
        importlib.import_module("genomic_variant_classifier.data.split_protocol_v2")
        chk("split_protocol_v2 imports", True)
    except Exception as e:
        chk("split_protocol_v2 imports", False); print(a(f"    {e}"))
    npass = sum(1 for _,c in checks if c)
    print("-"*78); print(a(f"dtype-fix verification: {npass}/{len(checks)} checks pass")); print("="*78)
    return 0 if npass == len(checks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
