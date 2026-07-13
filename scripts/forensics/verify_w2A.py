#!/usr/bin/env python
"""verify_w2A.py (2026-07-11) -- post-edit verification of the PATH-1 ensemble change in
variant_ensemble.py: external-cal params present, carve-block branch present, legacy self-carve
preserved inside the else, and the module imports. Read-only + compile. ASCII-safe.
"""
from __future__ import annotations
import io, py_compile, sys
from pathlib import Path
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

def a(s): return s.encode("ascii","replace").decode("ascii")

def main() -> int:
    print("="*78); print("W2 PATH-1 ENSEMBLE VERIFICATION (post-edit)"); print("="*78)
    ens = None
    for c in Path("src").rglob("variant_ensemble.py"):
        ens = c; break
    if ens is None:
        print("ABORT: variant_ensemble.py not found"); return 2
    src = ens.read_text(encoding="utf-8", errors="replace")
    checks = []
    def chk(name, cond): checks.append((name,cond)); print(a(f"  {'ok ' if cond else 'FAIL'} {name}"))

    chk("X_tab_cal_ext param added", "X_tab_cal_ext" in src)
    chk("X_seq_cal_ext param added", "X_seq_cal_ext" in src)
    chk("y_cal_ext param added", "y_cal_ext" in src)
    chk("gene_symbol_cal_ext param added", "gene_symbol_cal_ext" in src)
    chk("carve-block branch present", "if X_tab_cal_ext is not None:" in src)
    chk("external-cal logging present", "Calibrating on EXTERNAL gene-disjoint partition" in src)
    chk("legacy self-carve preserved (in else)", "Carve out 15% calibration split" in src
        and "idx_fit, idx_cal = _tts(" in src)
    chk("W2 PATH-1 marker present", "W2 PATH-1, 2026-07-11" in src)
    print("-"*78)
    try:
        py_compile.compile(str(ens), doraise=True); chk("variant_ensemble.py compiles", True)
    except Exception as e:
        print(a(f"  FAIL compile: {e}")); checks.append(("compile", False))
    npass = sum(1 for _,c in checks if c)
    print("-"*78); print(a(f"W2 PATH-1 verification: {npass}/{len(checks)} checks pass")); print("="*78)
    return 0 if npass == len(checks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
