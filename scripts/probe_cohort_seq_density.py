#!/usr/bin/env python3
"""probe_cohort_seq_density.py PARQUET [PARQUET ...]

READ-ONLY. Reports, for each parquet: which of fasta_seq / fasta_seq_ref /
fasta_seq_alt exist, and how densely each is populated (notna, nonempty, dummy
'A'*101, real). Tells us whether train.py's CNN raise can fire and which column
form the train-side fix must consume. Does not modify anything.
Author: Monzia Moodie
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SEQ_COLS = ["fasta_seq", "fasta_seq_ref", "fasta_seq_alt"]
DUMMY = "A" * 101


def report(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"\n[MISSING] {path}")
        return
    schema_cols = list(pq.read_schema(p).names)
    nrows = pq.ParquetFile(p).metadata.num_rows
    present = [c for c in SEQ_COLS if c in schema_cols]
    print(f"\n=== {path} ===")
    print(f"  rows={nrows}  total_cols={len(schema_cols)}  seq_cols_present={present or 'NONE'}")
    if not present:
        print("  -> no fasta_seq* columns present: CNN drops to placeholders "
              "(no raise; no real-sequence training). NOT a Run 16 blocker.")
        return
    df = pd.read_parquet(p, columns=present)
    for c in present:
        s = df[c].astype("string")
        notna = int(s.notna().sum())
        nonempty = int((s.fillna("").str.len() > 0).sum())
        ndummy = int((s == DUMMY).sum())
        real = int(((s.notna()) & (s != DUMMY) & (s.fillna("").str.len() > 0)).sum())
        print(f"  {c:16s} notna={notna}/{nrows}  nonempty={nonempty}  "
              f"dummy={ndummy}  real={real}")
    if "fasta_seq" in present:
        s = df["fasta_seq"].astype("string")
        print(f"  NOTE: train.py raises when TEST-split single 'fasta_seq' notna > 100. "
              f"Whole-cohort 'fasta_seq' notna={int(s.notna().sum())} "
              f"(test split ~= test_fraction of this).")
    else:
        print("  NOTE: single 'fasta_seq' absent -> train.py has_sequences=False -> "
              "CNN drops to placeholders (no raise). 2-col delta path needs wiring separately.")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python probe_cohort_seq_density.py PARQUET [PARQUET ...]")
        return 2
    for a in sys.argv[1:]:
        report(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
