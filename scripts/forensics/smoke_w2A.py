#!/usr/bin/env python
"""smoke_w2A.py (2026-07-11) -- prove the PATH-1 ensemble edit RUNS (not just compiles):
  1. the real variant_ensemble module imports (heavy deps included).
  2. VariantEnsemble.fit exposes the 4 external-cal params (inspect.signature).
  3. any existing ensemble test suite still passes (legacy behavior unchanged).
ASCII-safe. No full training run.
"""
from __future__ import annotations
import importlib
import inspect
import io
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass


def a(s): return s.encode("ascii", "replace").decode("ascii")


def main() -> int:
    print("=" * 78)
    print("W2 PATH-1 SMOKE TEST (imports + signature + existing tests)")
    print("=" * 78)
    results = []

    def chk(name, cond, detail=""):
        results.append(cond)
        print(a(f"  {'ok  ' if cond else 'FAIL'} {name}{('  -- ' + detail) if detail and not cond else ''}"))

    # 1. import the real module
    sys.path.insert(0, "src")
    try:
        mod = importlib.import_module("genomic_variant_classifier.models.variant_ensemble")
        chk("variant_ensemble imports (heavy deps included)", True)
    except Exception as e:
        chk("variant_ensemble imports", False, repr(e)[:200])
        print("-" * 78)
        print("  (import failed -- cannot proceed to signature check)")
        return 1

    # 2. signature exposes the 4 external-cal params
    try:
        sig = inspect.signature(mod.VariantEnsemble.fit)
        params = list(sig.parameters.keys())
        for p in ["X_tab_cal_ext", "X_seq_cal_ext", "y_cal_ext", "gene_symbol_cal_ext"]:
            chk(f"fit() exposes {p}", p in params)
        # confirm they default to None (backward-compat)
        defaults_none = all(
            sig.parameters[p].default is None
            for p in ["X_tab_cal_ext", "X_seq_cal_ext", "y_cal_ext", "gene_symbol_cal_ext"]
            if p in sig.parameters
        )
        chk("external-cal params default to None (backward-compat)", defaults_none)
    except Exception as e:
        chk("fit signature inspect", False, repr(e)[:200])

    # 3. existing ensemble tests (if any)
    test_dir = Path("tests")
    if test_dir.exists():
        # find ensemble-related tests
        ens_tests = [str(p) for p in test_dir.rglob("*ensemble*.py")]
        if ens_tests:
            print(a(f"  running ensemble tests: {ens_tests}"))
            r = subprocess.run([sys.executable, "-m", "pytest", *ens_tests, "-q"],
                               capture_output=True, text=True, timeout=600)
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            for ln in tail:
                print(a(f"    {ln}"))
            chk("existing ensemble tests pass", r.returncode == 0, f"rc={r.returncode}")
        else:
            print("  (no *ensemble* test files found -- signature + import + prior self-test cover it)")
    else:
        print("  (no tests/ dir -- signature + import cover it)")

    print("-" * 78)
    npass = sum(1 for x in results if x)
    print(a(f"W2 PATH-1 smoke: {npass}/{len(results)} checks pass"))
    print("=" * 78)
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
