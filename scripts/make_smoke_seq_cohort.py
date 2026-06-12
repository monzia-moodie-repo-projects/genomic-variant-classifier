#!/usr/bin/env python3
"""make_smoke_seq_cohort.py -- carve a tiny ref/alt cohort from the full clean_seq
cohort for the Run-16 all-models smoke.

train.py has no --max-train (it reads the whole --clinvar parquet), and the existing
clinvar_smoke.parquet lacks ref/alt (CNN + maxentscan_delta would be inert). This
samples a small WITH-ref/alt subset that preserves the input schema, so the smoke
exercises every feature the real regen will. Author: Monzia Moodie."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# columns that may encode the pathogenic/benign signal in a raw clinvar parquet
_LABELISH = ["label", "clinical_significance", "clnsig", "clnsig_simple", "significance", "y"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Carve a tiny ref/alt smoke cohort.")
    ap.add_argument("--src", default="data/processed/clinvar_grch38_clean_seq.parquet")
    ap.add_argument("--out", default="data/processed/clinvar_smoke_seq.parquet")
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args(argv)

    try:
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        print(f"ENV: pandas import failed ({e})")
        return 3
    src = Path(args.src)
    if not src.exists():
        print(f"ENV: src not found: {src}")
        return 3

    df = pd.read_parquet(src)
    n0 = len(df)
    for c in ("ref", "alt"):
        if c in df.columns:
            df = df[df[c].notna() & (df[c].astype(str).str.len() > 0)]
    for c in ("fasta_seq_ref", "fasta_seq_alt"):
        if c in df.columns:
            df = df[df[c].notna()]
    if len(df) == 0:
        print("FAIL: no rows with populated ref/alt in src.")
        return 2

    lab = next((c for c in _LABELISH if c in df.columns), None)
    if lab is not None and df[lab].nunique() > 1:
        per = max(args.n // int(df[lab].nunique()), 1)
        parts = [g.sample(min(len(g), per), random_state=0) for _, g in df.groupby(lab)]
        out = pd.concat(parts).sample(frac=1, random_state=0).reset_index(drop=True)
    else:
        out = df.sample(min(len(df), args.n), random_state=0).reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    ng = int(out["gene_symbol"].nunique()) if "gene_symbol" in out.columns else "n/a"
    print(f"src rows with ref/alt: {len(df):,} of {n0:,}")
    print(f"wrote {args.out}: {len(out):,} rows, genes={ng}")
    if lab:
        print(f"label-ish '{lab}' distribution: {out[lab].value_counts().head(6).to_dict()}")
    else:
        print("note: no label column in the raw cohort (pipeline derives it downstream);")
        print("      confirm both classes appear in the smoke's Train/Test log lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
