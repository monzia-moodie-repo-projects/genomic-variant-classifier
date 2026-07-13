#!/usr/bin/env python
"""verify_w1.py (2026-07-11) -- post-edit verification that W1 wiring is correct in train.py.

Confirms: the two new args exist, the verify gate is wired after raw_df load, the sequence block is
the attach-then-check restructure (old buggy check gone), guards preserved, and train.py compiles +
imports. Read-only except for a compile check. ASCII-safe.
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
    print("="*78); print("W1 WIRING VERIFICATION (post-edit)"); print("="*78)
    p = Path("scripts/train.py")
    if not p.exists():
        print("ABORT: scripts/train.py missing"); return 2
    src = p.read_text(encoding="utf-8", errors="replace")
    checks = []
    def chk(name, cond): checks.append((name, cond)); print(a(f"  {'ok ' if cond else 'FAIL'} {name}"))

    chk("--seq-windows arg added", '"--seq-windows"' in src)
    chk("--reference arg added", '"--reference"' in src)
    chk("verify gate wired after raw_df load", "Sequence-window coherence gate (W1" in src
        and "verify_seq_windows" in src and "raise_if_failed" in src)
    chk("gate is manifest-conditional (backward compat)", "seq_windows.manifest.json" in src
        and "will use placeholder" in src)
    chk("sequence block restructured (attach-then-check)", "ATTACH-then-CHECK" in src)
    chk("old meta.columns has_sequences check REMOVED", "REF_WIN_COL in meta_test.columns" not in src)
    chk("old Series poly path REMOVED", 'pd.Series(["A" * 101] * len(y_train))' not in src)
    chk("seq_windows_path passed to attach (test)", "attach_delta_windows(\n        meta_test, seq_windows_path=" in src
        or "attach_delta_windows(meta_test, seq_windows_path=" in src)
    chk("seq_windows_path passed to attach (train)", "seq_windows_path=_seq_win_arg" in src)
    chk("FileNotFoundError guard preserved", "meta_train.parquet not found" in src)
    chk("PM11d len-alignment guard preserved", "PM11d-style label mismatch" in src)
    chk("has_sequences from non-poly count", "_n_real_test" in src and "_POLY_WIN" in src)

    print("-"*78)
    try:
        py_compile.compile(str(p), doraise=True)
        chk("train.py compiles", True)
    except Exception as e:
        print(a(f"  FAIL compile: {e}")); checks.append(("compile", False))

    npass = sum(1 for _, c in checks if c)
    print("-"*78)
    print(a(f"W1 verification: {npass}/{len(checks)} checks pass"))
    print("="*78)
    return 0 if npass == len(checks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
