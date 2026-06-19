#!/usr/bin/env python3
"""
merge_1kg_parquets.py  --  Monzia Moodie

Concatenate 1000G AF shards (e.g. the chr1-22 autosome parquet + a chrX shard) into one
kg parquet, deduplicating on variant_id and re-validating that every per-superpopulation AF
column survives with non-zero coverage. Atomic write (tmp + os.replace) so it is safe even when
--out equals one of the inputs. Fails LOUD on: missing superpop columns, an all-zero superpop
after merge, schema mismatch across shards, or a row-count regression.

Usage:
  python scripts/merge_1kg_parquets.py \
      --inputs data/external/1kgp/kg_grch38_af.parquet data/external/1kgp/kg_grch38_af_chrX.parquet \
      --out    data/external/1kgp/kg_grch38_af.parquet
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import pandas as pd

SUPERPOP = ["AFR_AF", "EUR_AF", "EAS_AF", "SAS_AF", "AMR_AF"]


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr); raise SystemExit(2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    frames, schemas = [], []
    for src in args.inputs:
        p = Path(src)
        if not p.exists():
            fail(f"input not found: {p}")
        df = pd.read_parquet(p)
        if "variant_id" not in df.columns:
            fail(f"{p} lacks variant_id")
        miss = [c for c in SUPERPOP if c not in df.columns]
        if miss:
            fail(f"{p} missing super-pop columns {miss}")
        print(f"[in] {p.name}: {len(df):,} rows, cols={list(df.columns)}")
        frames.append(df)
        schemas.append(tuple(df.columns))

    if len(set(schemas)) != 1:
        fail(f"shard column schemas differ: {set(schemas)}")

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset="variant_id", keep="first").reset_index(drop=True)
    dropped = before - len(merged)
    print(f"[merge] {before:,} concatenated -> {len(merged):,} unique variant_id ({dropped:,} dup rows dropped)")

    if len(merged) < max(len(f) for f in frames):
        fail("merged row count < largest input; concat/dedup regression")

    nz = {c: int((merged[c].fillna(0) != 0).sum()) for c in SUPERPOP}
    print(f"[validate] non-zero super-pop AF counts: {nz}")
    zero = [c for c, n in nz.items() if n == 0]
    if zero:
        fail(f"super-pop column(s) all-zero after merge: {zero} (silent-zero -> abort)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, out)  # atomic; safe even if out was an input
    print(f"[ok] wrote {out} ({len(merged):,} variants, {len(merged.columns)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
