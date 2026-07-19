#!/usr/bin/env python3
"""probe_cohort_seq_density.py PARQUET [PARQUET ...]

READ-ONLY. Reports, for each parquet: which of fasta_seq / fasta_seq_ref /
fasta_seq_alt exist, how densely each is populated (notna, nonempty, length
range), and -- from the builder's own `ok` column -- how many rows carry a
placeholder window. Reports placeholder counts from PROVENANCE, never from
content: a window of one repeated base may be genuine biology, so content cannot
distinguish it from a window the builder failed to build. Where `ok` is absent the
probe says so explicitly rather than reporting zero. Does not modify anything.
Author: Monzia Moodie
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SEQ_COLS = ["fasta_seq", "fasta_seq_ref", "fasta_seq_alt"]
PROV_COLS = ["ok", "reason"]


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
    has_prov = "ok" in schema_cols
    df = pd.read_parquet(p, columns=present + ([c for c in PROV_COLS if c in schema_cols]
                                               if has_prov else []))
    for c in present:
        s = df[c].astype("string")
        notna = int(s.notna().sum())
        nonempty = int((s.fillna("").str.len() > 0).sum())
        lens = s.fillna("").str.len()
        print(f"  {c:16s} notna={notna}/{nrows}  nonempty={nonempty}  "
              f"len_min={int(lens.min()) if len(lens) else 0} "
              f"len_max={int(lens.max()) if len(lens) else 0}")

    # PROVENANCE, not content. A window whose bases are one repeated letter may be real
    # biology; only the builder knows whether it gave up. Before 2026-07-18 this block
    # compared against "A" * 101 and reported dummy=0 once the placeholder base became
    # "N" -- which reads as "clean" when the truth is "cannot tell".
    if has_prov:
        okc = df["ok"].fillna(False).astype(bool)
        n_bad = int((~okc).sum())
        print(f"  provenance       ok column PRESENT -> "
              f"usable={int(okc.sum())}/{nrows}  placeholder={n_bad}")
        if n_bad and "reason" in df.columns:
            counts = (df.loc[~okc, "reason"].astype(str)
                      .str.split("(").str[0].value_counts())
            for reason, k in counts.items():
                print(f"                     {reason:<24} {int(k):>8}")
    else:
        print("  provenance       ok column ABSENT -- placeholder rows CANNOT be "
              "identified.")
        print("                   This is NOT the same as zero placeholders. Rebuild "
              "with")
        print("                   scripts/build_seq_windows.py, then join with")
        print("                   scripts/build_clean_seq_from_windows.py, to restore it.")
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
