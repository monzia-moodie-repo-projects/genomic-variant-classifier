#!/usr/bin/env python3
"""
scripts/clinvar_name_probe.py
=============================
Confirms the HGVSp is recoverable from ClinVar's RAW `Name` field (where the
cleaned parquet dropped it). Reports, for the raw variant_summary:
  - rows total / with a non-empty Name
  - rows whose Name carries a missense protein change  p.(Xxx###Yyy)  [3-letter]
  - rows with any p. consequence (incl. Ter/fs/del/dup/=)
These are the variants whose protein_pos/wt_aa/mut_aa a parser could populate.

Accepts the raw .txt.gz, a plain .txt, or a cached .parquet that still has `Name`.

    python scripts/clinvar_name_probe.py --raw data/raw/variant_summary.txt.gz
    python scripts/clinvar_name_probe.py --raw data/raw/cache/clinvar_summary_GRCh38.parquet
"""
from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import pandas as pd

# 3-letter missense: p.Lys177Glu  (both sides AA triplets, not Ter/fs/del/=)
MISSENSE_P = re.compile(r"p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}\b")
# any protein consequence: includes Ter, fs, del, dup, =, and 1-letter forms
ANY_P = re.compile(r"p\.\(?\s*([A-Z][a-z]{2}|[A-Z])\d+")


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    # tab-delimited; pull only the columns we need if they exist
    open_fn = gzip.open if path.suffix == ".gz" else open
    # peek header
    with open_fn(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    want = [c for c in ("Name", "Type", "Assembly") if c in header]
    if "Name" not in want:
        raise SystemExit(f"No 'Name' column in {path}. Header was: {header[:8]} ...")
    return pd.read_csv(path, sep="\t", usecols=want, low_memory=False,
                       compression="gzip" if path.suffix == ".gz" else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()
    path = Path(args.raw)
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    df = _read(path)
    if "Assembly" in df.columns:
        before = len(df)
        df = df[df["Assembly"].astype(str) == "GRCh38"].copy()
        print(f"Assembly filter GRCh38: {len(df):,} / {before:,}")

    name_col = "Name" if "Name" in df.columns else df.columns[0]
    name = df[name_col].fillna("").astype(str)
    n = len(df)

    def pct(x):
        return f"{(100.0*x/n):.2f}%" if n else "n/a"

    nonempty = int(name.str.strip().ne("").sum())
    miss = int(name.str.contains(MISSENSE_P).sum())
    anyp = int(name.str.contains(ANY_P).sum())

    print(f"\nrows total            : {n:,}")
    print(f"Name non-empty        : {nonempty:,} ({pct(nonempty)})")
    print(f"missense p.(3-letter) : {miss:,} ({pct(miss)})   <- HGVSp recoverable by a parser")
    print(f"any p. consequence    : {anyp:,} ({pct(anyp)})")
    ex = name[name.str.contains(MISSENSE_P)]
    if len(ex):
        print(f"example Name          : {ex.iloc[0][:90]!r}")
    print("\nUse 'missense p.(3-letter)' as the upper bound the parser can populate.\n")


if __name__ == "__main__":
    main()
