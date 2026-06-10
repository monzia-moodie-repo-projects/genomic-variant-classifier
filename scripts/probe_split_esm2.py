#!/usr/bin/env python3
"""probe_split_esm2.py -- READ-ONLY check of esm2_delta_norm in the on-disk splits.

Author: Monzia Moodie

Confirms (or breaks) the stale-splits hypothesis for the Run 15 ESM-2 = 3,451 result:
if Run 15 trained on pre-coords splits, the *current* data/splits/X_*.parquet should
show esm2_delta_norm as effectively dead (one dominant value after StandardScaler;
~3,451 rows differ), and the split files should pre-date the 2.41M coord cache.

X_*.parquet feature columns are StandardScaler-transformed, so a near-all-zero raw
feature collapses to a single modal z-value; "rows != mode" ~= the active count.
meta_*.parquet carries the RAW annotated rows, so esm2_delta_norm there is the raw
delta and protein_pos (if present) is the coord-coverage ground truth.

Strictly read-only. No writes, no deletes.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _mtime(p: str) -> str:
    try:
        import datetime as dt
        return dt.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")
    except Exception:
        return "?"


def report_x(path: str) -> None:
    import pandas as pd
    if not os.path.isfile(path):
        print(f"  {os.path.basename(path)}: MISSING")
        return
    try:
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(path).schema.names
    except Exception:
        cols = None
    if cols is not None and "esm2_delta_norm" not in cols:
        print(f"  {os.path.basename(path)}: NO esm2_delta_norm column"
              f"  (mtime {_mtime(path)})  <- predates ESM-2 wiring entirely")
        return
    s = pd.read_parquet(path, columns=["esm2_delta_norm"])["esm2_delta_norm"]
    n = len(s)
    nuniq = int(s.nunique(dropna=True))
    mode = s.mode(dropna=True)
    n_off_mode = int((s != mode.iloc[0]).sum()) if len(mode) else n
    print(f"  {os.path.basename(path)}: rows={n:,}  distinct(esm2)={nuniq:,}  "
          f"rows!=mode={n_off_mode:,}  (mtime {_mtime(path)})")
    if nuniq <= 1:
        print("      -> DEAD feature (single value): esm2_delta_norm carried NO signal "
              "in these splits.")
    elif n_off_mode < 50_000:
        print(f"      -> NEAR-DEAD: only ~{n_off_mode:,} rows differ from the modal "
              "value -- consistent with the frozen 3,451 from pre-coords splits.")


def report_meta(path: str) -> None:
    import pandas as pd
    if not os.path.isfile(path):
        print(f"  {os.path.basename(path)}: MISSING")
        return
    try:
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(path).schema.names
    except Exception:
        cols = []
    want = [c for c in ("esm2_delta_norm", "protein_pos", "wt_aa", "is_missense") if c in cols]
    if not want:
        print(f"  {os.path.basename(path)}: none of esm2_delta_norm/protein_pos present"
              f"  (mtime {_mtime(path)})")
        return
    df = pd.read_parquet(path, columns=want)
    bits = [f"rows={len(df):,}"]
    if "esm2_delta_norm" in df:
        bits.append(f"esm2>0={int((df['esm2_delta_norm'] > 0).sum()):,}")
    if "protein_pos" in df:
        bits.append(f"protein_pos notna={int(df['protein_pos'].notna().sum()):,}")
    if "is_missense" in df:
        bits.append(f"is_missense={int(df['is_missense'].fillna(0).astype(int).sum()):,}")
    print(f"  {os.path.basename(path)}: " + "  ".join(bits) + f"  (mtime {_mtime(path)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--coord-cache",
                    default="data/external/alphamissense/alphamissense_protein_index.parquet")
    args = ap.parse_args()
    sd = Path(args.splits_dir)
    print("READ-ONLY. No files written or deleted.\n")
    print(f"coord cache: {args.coord_cache}  (mtime {_mtime(args.coord_cache)})")
    print("=" * 64)
    print("X splits (StandardScaler-transformed; mode-collapse reveals dead feature):")
    for name in ("X_train.parquet", "X_val.parquet", "X_test.parquet"):
        report_x(str(sd / name))
    print("\nmeta splits (RAW annotated rows; protein_pos = coord coverage ground truth):")
    for name in ("meta_train.parquet", "meta_val.parquet", "meta_test.parquet"):
        report_meta(str(sd / name))
    print("\nInterpretation:")
    print("  If X splits show esm2 dead/near-dead AND splits pre-date the coord cache,")
    print("  the Run 15 3,451 is a STALE-SPLITS artifact -> fix = regen splits (no code")
    print("  change to protein_coords/esm2), then verify coverage climbs, then Run 16.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
