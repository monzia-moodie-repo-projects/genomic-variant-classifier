#!/usr/bin/env python3
"""verify_schema_seal_inputs.py -- read-only pre-seal check before re-sealing the
schema baseline 78->81. Confirms the three smoke splits are mutually schema-identical
(names + dtypes + order), reports exactly which columns are new vs the current sealed
baseline, and flags any dtype inconsistency that would make the seal unstable.

Why: build_schema_baseline.py seals from X_train only. If X_train/X_val/X_test disagree
on a column's dtype, the seal would encode X_train's dtype and the full-regen drift
check could later go red on a split that differs. Catch that now, not on Vast.ai.

STRICTLY READ-ONLY (parquet footer schemas + baseline JSON only).

Usage:  python scripts/verify_schema_seal_inputs.py
        [--splits models/smoke_run16b/splits] [--baseline data/reference/schema/schema_baseline.json]
Author: Monzia Moodie."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SPLITS = ["X_train.parquet", "X_val.parquet", "X_test.parquet"]


def _schema_map(p: Path) -> "dict[str,str]":
    # Read the way build_schema_baseline.py + the drift check do: first batch -> pandas,
    # so dtype STRINGS match the baseline ("float64", not pyarrow's "double"). Bounded memory.
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(p)
    try:
        df = next(pf.iter_batches(batch_size=4096)).to_pandas()
    except StopIteration:
        df = pf.read().to_pandas()
    return {c: str(df[c].dtype) for c in df.columns}  # preserves order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=Path("models/smoke_run16b/splits"))
    ap.add_argument("--baseline", type=Path, default=Path("data/reference/schema/schema_baseline.json"))
    args = ap.parse_args()

    print("=" * 76)
    print(" Schema seal pre-check (read-only)")
    print("=" * 76)

    # 1) load the three smoke splits' schemas
    maps = {}
    for fn in SPLITS:
        fp = args.splits / fn
        if not fp.exists():
            print(f" FAIL: missing split {fp}")
            return 2
        maps[fn] = _schema_map(fp)
    train = maps["X_train.parquet"]
    print(f"\n[1] smoke splits under {args.splits}")
    for fn in SPLITS:
        print(f"    {fn:<16} {len(maps[fn])} cols")

    # 2) mutual consistency: identical column set, order, and dtypes
    ok = True
    base_cols = list(train.keys())
    for fn in SPLITS[1:]:
        m = maps[fn]
        if list(m.keys()) != base_cols:
            only_train = [c for c in base_cols if c not in m]
            only_other = [c for c in m if c not in train]
            print(f"    MISMATCH names/order X_train vs {fn}: "
                  f"train-only={only_train[:5]} {fn}-only={only_other[:5]}")
            ok = False
        else:
            dt_diff = [(c, train[c], m[c]) for c in base_cols if train[c] != m[c]]
            if dt_diff:
                print(f"    DTYPE MISMATCH X_train vs {fn}: {dt_diff[:5]}")
                ok = False
    if ok:
        print(f"    PASS: all 3 splits share an identical {len(base_cols)}-column schema (names+order+dtype)")

    # 3) dtype histogram
    from collections import Counter
    hist = Counter(train.values())
    print(f"\n[2] X_train dtype histogram: {dict(hist)}")

    # 4) diff vs current sealed baseline
    print(f"\n[3] vs current baseline {args.baseline}")
    if not args.baseline.exists():
        print("    (no current baseline -- this will be the first seal)")
    else:
        b = json.loads(args.baseline.read_text(encoding="utf-8"))
        b_dtypes = b.get("expected_dtypes", {})
        print(f"    current: n_columns={b.get('n_columns')}  run_label={b.get('run_label')}")
        added = [c for c in base_cols if c not in b_dtypes]
        removed = [c for c in b_dtypes if c not in train]
        retyped = [(c, b_dtypes[c], train[c]) for c in base_cols
                   if c in b_dtypes and b_dtypes[c] != train[c]]
        print(f"    new columns ({len(added)}): {added}")
        if removed:
            print(f"    REMOVED columns ({len(removed)}): {removed}  <-- investigate before sealing")
        if retyped:
            print(f"    RETYPED columns ({len(retyped)}): {retyped[:8]}  <-- dtype changed vs sealed")

    print("\n" + "=" * 76)
    if ok:
        print(f" READY TO SEAL at {len(base_cols)} columns from {args.splits / 'X_train.parquet'}")
        return 0
    print(" NOT READY -- resolve the split inconsistency above before sealing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
