#!/usr/bin/env python
"""verify_w2b2.py (2026-07-11) -- post-edit verification of W2-B2 in scripts/train.py: the
--split-protocol arg, DataPrepConfig threading, the run()/run_v2 branch, the X_seq_tune v2 block with
PM11d guard, and the fit branch (v2 external-cal + gene_symbol; legacy preserved). Read-only +
compile. ASCII-safe.
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
    print("="*78); print("W2-B2 VERIFICATION (post-edit)"); print("="*78)
    tp = Path("scripts/train.py")
    if not tp.exists():
        print("ABORT: scripts/train.py missing"); return 2
    src = tp.read_text(encoding="utf-8", errors="replace")
    checks = []
    def chk(n, c): checks.append((n,c)); print(a(f"  {'ok ' if c else 'FAIL'} {n}"))

    chk("--split-protocol arg added", '"--split-protocol"' in src)
    chk("arg choices legacy/v2_conformal", 'choices=["legacy", "v2_conformal"]' in src)
    chk("arg default legacy", 'default="legacy"' in src)
    chk("DataPrepConfig split_protocol threaded", "split_protocol=args.split_protocol," in src)
    chk("run branch _v2 flag", '_v2 = (args.split_protocol == "v2_conformal")' in src)
    chk("v2 calls run_v2", "pipeline.run_v2(" in src)
    chk("bundle X_test -> clean holdout", "X_test, y_test = _bundle.X_test, _bundle.y_test" in src)
    chk("bundle tune extracted", "X_tune, y_tune, meta_tune = _bundle.X_tune" in src)
    chk("bundle genes extracted", "genes_train, genes_tune = _bundle.genes_train" in src)
    chk("bundle conformal reserved", "X_conformal = _bundle.X_conformal" in src)
    chk("legacy-only names bound None", "X_val = y_val = meta_val = None" in src)
    chk("legacy run() branch preserved",
        "X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test = pipeline.run(" in src)
    chk("X_seq_tune v2 block", "Tune side (W2 v2 only)" in src)
    chk("meta_tune.parquet read", 'config.output_dir / "meta_tune.parquet"' in src)
    chk("PM11d tune guard", "meta_tune rows" in src)
    chk("fit branch external-cal (X_tab_cal_ext)", "X_tab_cal_ext=X_tune," in src)
    chk("fit branch gene_symbol_cal_ext", "gene_symbol_cal_ext=genes_tune," in src)
    chk("fit branch passes gene_symbol", "gene_symbol=genes_train," in src)
    chk("legacy fit preserved", "        ensemble.fit(X_train, X_seq_train, y_train)" in src)
    print("-"*78)
    try:
        py_compile.compile(str(tp), doraise=True); chk("train.py compiles", True)
    except Exception as e:
        print(a(f"  FAIL compile: {e}")); checks.append(("compile", False))
    npass = sum(1 for _,c in checks if c)
    print("-"*78); print(a(f"W2-B2 verification: {npass}/{len(checks)} checks pass")); print("="*78)
    return 0 if npass == len(checks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
