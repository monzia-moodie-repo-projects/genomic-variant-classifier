#!/usr/bin/env python
"""smoke_w1.py (2026-07-11) -- end-to-end-lite proof that W1 wiring RUNS (not just compiles).

Checks, without a training run:
  1. train.py imports as a module (executes module-level code; catches import errors the compile
     check cannot).
  2. train.py's argparse exposes --seq-windows and --reference with the expected dest names.
  3. verify_seq_windows imports and is callable from the same context train.py uses it.
  4. The gate FIRES: PASS on the real cohort, RAISES on a deliberately mutated cohort.
ASCII-safe. Does not train, does not write to the real artifact.
"""
from __future__ import annotations
import importlib.util
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
    print("W1 SMOKE TEST (proves the wiring RUNS, not just compiles)")
    print("=" * 78)
    results = []

    def chk(name, cond, detail=""):
        results.append(cond)
        print(a(f"  {'ok  ' if cond else 'FAIL'} {name}{('  -- ' + detail) if detail and not cond else ''}"))

    # 1 + 2: train.py --help (full import + argparse construction + flag exposure)
    try:
        r = subprocess.run([sys.executable, "scripts/train.py", "--help"],
                           capture_output=True, text=True, timeout=120)
        help_out = r.stdout + r.stderr
        chk("train.py --help runs (full import + argparse)", r.returncode == 0,
            f"rc={r.returncode}: {help_out[-300:]}")
        chk("--seq-windows exposed in --help", "--seq-windows" in help_out)
        chk("--reference exposed in --help", "--reference" in help_out)
    except Exception as e:
        chk("train.py --help runs", False, str(e))

    # 3: verify_seq_windows importable + callable
    try:
        sys.path.insert(0, "src")
        from genomic_variant_classifier.data.seq_window_manifest import (
            verify_seq_windows, cohort_key_hash,
        )
        chk("verify_seq_windows imports", True)
    except Exception as e:
        chk("verify_seq_windows imports", False, str(e))
        verify_seq_windows = None

    # 4: gate FIRES -- PASS on real cohort, RAISES on mutated cohort
    if verify_seq_windows is not None:
        import pandas as pd
        cohort_path = Path("data/processed/clinvar_grch38_pathfix.parquet")
        wdir = Path("data/processed/seq_windows")
        ref = "data/external/grch38/GRCh38.fa"
        if cohort_path.exists() and (wdir / "seq_windows.manifest.json").exists():
            cohort = pd.read_parquet(cohort_path, columns=["chrom", "pos", "ref", "alt"])
            # PASS on real cohort
            res = verify_seq_windows(cohort, wdir, ref)
            chk("gate PASSES on the real pathfix cohort", res.ok,
                f"reasons={res.reasons}")
            # RAISES on a mutated cohort (flip one alt -> cohort hash changes)
            mutated = cohort.copy()
            mutated.loc[mutated.index[0], "alt"] = (
                "T" if str(mutated.loc[mutated.index[0], "alt"]) != "T" else "G"
            )
            res2 = verify_seq_windows(mutated, wdir, ref)
            chk("gate REJECTS a mutated (stale) cohort", (not res2.ok)
                and any("cohort" in x for x in res2.reasons),
                f"unexpectedly ok={res2.ok}")
        else:
            chk("real cohort + manifest available for gate test",
                False, "cohort or manifest missing")

    print("-" * 78)
    npass = sum(1 for x in results if x)
    print(a(f"W1 smoke: {npass}/{len(results)} checks pass"))
    print("=" * 78)
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
