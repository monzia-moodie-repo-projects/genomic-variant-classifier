#!/usr/bin/env python
"""verify_w2b1.py (2026-07-11) -- post-edit verification of W2-B1 in real_data_prep.py: the config
field, SplitBundle dataclass, _save_splits tune/conformal params+writes, and run_v2 with the clean
inline scaler (no double-_scale). Confirms legacy run() untouched. Read-only + import + compile.
ASCII-safe.
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
    print("="*78); print("W2-B1 VERIFICATION (post-edit)"); print("="*78)
    rdp = Path("src/genomic_variant_classifier/data/real_data_prep.py")
    if not rdp.exists():
        print("ABORT: real_data_prep.py missing"); return 2
    src = rdp.read_text(encoding="utf-8", errors="replace")
    checks = []
    def chk(n, c): checks.append((n,c)); print(a(f"  {'ok ' if c else 'FAIL'} {n}"))

    chk("DataPrepConfig.split_protocol field added", "split_protocol: str" in src)
    chk("SplitBundle dataclass added", "class SplitBundle:" in src)
    chk("SplitBundle role-named fields", "X_tune:" in src and "X_conformal:" in src)
    chk("_save_splits meta_tune param", "meta_tune: pd.DataFrame | None" in src)
    chk("_save_splits meta_conformal param", "meta_conformal: pd.DataFrame | None" in src)
    chk("_save_splits writes meta_tune", 'meta_tune.to_parquet(out / "meta_tune.parquet"' in src)
    chk("_save_splits writes meta_conformal", 'meta_conformal.to_parquet(out / "meta_conformal.parquet"' in src)
    chk("run_v2 method added", "def run_v2(" in src)
    chk("run_v2 imports split_protocol_v2", "from genomic_variant_classifier.data.split_protocol_v2 import" in src)
    chk("run_v2 applies leakage remap", "_leak_remap_v2(combo, result.indices, cfg2)" in src)
    chk("run_v2 clean inline scaler (fit on train)", "self.scaler.fit_transform(X_train)" in src)
    chk("NO double-_scale / dead code", "X_train = X_train  # unchanged" not in src
        and "X_train2, _Xtest2" not in src)
    chk("legacy run() intact", "return X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test" in src)
    chk("legacy _gene_aware_split intact", "def _gene_aware_split(" in src)
    print("-"*78)
    try:
        py_compile.compile(str(rdp), doraise=True); chk("real_data_prep.py compiles", True)
    except Exception as e:
        print(a(f"  FAIL compile: {e}")); checks.append(("compile", False))
    # import test
    sys.path.insert(0, "src")
    try:
        m = importlib.import_module("genomic_variant_classifier.data.real_data_prep")
        chk("real_data_prep imports", True)
        chk("SplitBundle importable", hasattr(m, "SplitBundle"))
        chk("DataPrepPipeline.run_v2 exists", hasattr(m.DataPrepPipeline, "run_v2"))
        chk("DataPrepPipeline.run (legacy) exists", hasattr(m.DataPrepPipeline, "run"))
    except Exception as e:
        chk("real_data_prep imports", False)
        print(a(f"    import error: {e}"))
    npass = sum(1 for _,c in checks if c)
    print("-"*78); print(a(f"W2-B1 verification: {npass}/{len(checks)} checks pass")); print("="*78)
    return 0 if npass == len(checks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
