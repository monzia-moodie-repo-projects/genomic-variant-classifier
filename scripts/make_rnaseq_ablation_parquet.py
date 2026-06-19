#!/usr/bin/env python3
"""
make_rnaseq_ablation_parquet.py  --  Monzia Moodie

Parameterized, seed-aware builder for ONE RNA-seq ablation parquet (vs the hardcoded-path,
single-seed make_rnaseq_ablation_parquets.py). Modes:
  full          copy of --src (rnaseq features intact)
  drop_de       zero rnaseq_log2fc + rnaseq_de_neglog10p (DE block off)
  drop_all      zero all five rnaseq_* features (rnaseq fully off == no_rnaseq floor)
  gene_shuffle  permute the five rnaseq_* columns together by --seed (breaks gene<->expression linkage)

Validates required columns + gene_symbol uniqueness before writing. Run anywhere (paths are CLI args).
  python scripts/make_rnaseq_ablation_parquet.py --src <full.parquet> --out <abl.parquet> --mode gene_shuffle --seed 11
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNASEQ_COLS = ["rnaseq_mean_log_tpm", "rnaseq_detection_rate", "rnaseq_log2_cv",
               "rnaseq_log2fc", "rnaseq_de_neglog10p"]
DE_COLS = ["rnaseq_log2fc", "rnaseq_de_neglog10p"]
REQUIRED = {"gene_symbol", *RNASEQ_COLS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=["full", "drop_de", "drop_all", "gene_shuffle"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.src)
    miss = REQUIRED - set(df.columns)
    if miss:
        print(f"ERROR: --src missing columns {sorted(miss)}; available={sorted(df.columns)}", file=sys.stderr)
        return 2
    if not df["gene_symbol"].is_unique:
        print("ERROR: gene_symbol is not unique in --src", file=sys.stderr); return 3

    out = df.copy()
    if args.mode == "drop_de":
        out[DE_COLS] = 0.0
    elif args.mode == "drop_all":
        out[RNASEQ_COLS] = 0.0
    elif args.mode == "gene_shuffle":
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(out))
        out[RNASEQ_COLS] = out.loc[out.index[perm], RNASEQ_COLS].to_numpy()
    # mode == full -> unchanged copy

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    nz = int((out[RNASEQ_COLS].abs().to_numpy() > 0).any(axis=1).sum())
    print(f"[ok] mode={args.mode} seed={args.seed} -> {args.out} "
          f"({len(out)} genes, {nz} with any non-zero rnaseq feature)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
