#!/usr/bin/env python3
"""
build_rnaseq_canonical_real.py  --  Monzia Moodie

Replaces build_rnaseq_canonical_from_gtex_de.py (which approximated mean_log_tpm from the
two tissue means and HARDCODED rnaseq_log2_cv = 0.0). This builds the canonical
rnaseq_gene_expression.parquet so EVERY feature is real per-sample data:

  * rnaseq_mean_log_tpm, rnaseq_detection_rate, rnaseq_log2_cv
        -> computed from the raw gene x sample TPM matrix via the validated
           build_rnaseq_parquet.summarise_matrix (real CV = std/mean over samples).
  * rnaseq_log2fc, rnaseq_de_neglog10p
        -> taken UNCHANGED from the validated DE artifact (de_features.tsv), deduped to
           one row per symbol by strongest DE evidence (same order as the prior builder).

Ensembl(Name) -> symbol(Description) collapse is tximport-style SUM. Fails LOUD on any
NaN, any all-zero/constant feature, duplicate symbols, residual ENSG ids, or missing
TP53/HBB sentinels.

Usage:
  python scripts/build_rnaseq_canonical_real.py \
      --matrix  "<...>.gene_tpm.tsv"  --de "<...>.de_features.tsv" \
      --out     data/external/rnaseq_gene_expression.parquet
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling builder
from build_rnaseq_parquet import summarise_matrix  # validated per-sample summariser

FEATURES = ["rnaseq_mean_log_tpm", "rnaseq_detection_rate", "rnaseq_log2_cv",
            "rnaseq_log2fc", "rnaseq_de_neglog10p"]
NAME_COL, SYM_COL = "Name", "Description"


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr); raise SystemExit(2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", required=True, type=Path, help="gene_tpm.tsv (Name, Description, samples...)")
    ap.add_argument("--de", required=True, type=Path, help="de_features.tsv (validated DE)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-detect", type=float, default=1.0)
    args = ap.parse_args(argv)

    if not args.matrix.exists(): fail(f"missing matrix: {args.matrix}")
    if not args.de.exists():     fail(f"missing DE table: {args.de}")

    # --- raw matrix -> symbol-collapsed gene x sample (SUM) ---
    mat = pd.read_csv(args.matrix, sep="\t")
    for c in (NAME_COL, SYM_COL):
        if c not in mat.columns: fail(f"matrix missing '{c}' column; found {list(mat.columns)[:5]}")
    sym = mat[SYM_COL].astype(str).str.strip()
    sample_cols = [c for c in mat.columns if c not in (NAME_COL, SYM_COL)]
    samples = mat[sample_cols].apply(pd.to_numeric, errors="coerce")
    nonnum = samples.columns[samples.isna().all()].tolist()
    if nonnum:
        print(f"[note] dropping non-numeric sample col(s): {nonnum}")
        samples = samples.drop(columns=nonnum)
    if samples.shape[1] == 0: fail("no numeric sample columns in matrix")
    print(f"[matrix] {len(mat)} rows x {samples.shape[1]} samples")

    keep = (sym.str.len() > 0) & (~sym.str.startswith("ENSG"))
    gene_mat = samples[keep].copy()
    gene_mat.insert(0, "gene_symbol", sym[keep].values)
    gene_mat = gene_mat.groupby("gene_symbol").sum()  # tximport-style SUM per symbol
    print(f"[collapse] {int(keep.sum())} symbol rows -> {gene_mat.shape[0]} unique symbols (summed)")

    expr = summarise_matrix(gene_mat, case_cols=None, control_cols=None, min_detect=args.min_detect)
    expr = expr[["gene_symbol", "rnaseq_mean_log_tpm", "rnaseq_detection_rate", "rnaseq_log2_cv"]]

    # --- validated DE from de_features, deduped to one row per symbol (strongest evidence) ---
    de = pd.read_csv(args.de, sep="\t", dtype={"gene_id": str, "gene_symbol": str}).fillna("")
    need = {"gene_id", "gene_symbol", "rnaseq_log2fc", "rnaseq_de_neglog10p"}
    miss = need - set(de.columns)
    if miss: fail(f"de_features missing columns: {sorted(miss)}")
    de = de[(de["gene_symbol"].str.len() > 0) & (~de["gene_symbol"].str.startswith("ENSG"))].copy()
    for c in ("rnaseq_log2fc", "rnaseq_de_neglog10p"):
        de[c] = pd.to_numeric(de[c], errors="coerce")
    de["_abs"] = de["rnaseq_log2fc"].abs()
    de = (de.sort_values(["gene_symbol", "rnaseq_de_neglog10p", "_abs", "gene_id"],
                         ascending=[True, False, False, True])
            .drop_duplicates("gene_symbol", keep="first")
            [["gene_symbol", "rnaseq_log2fc", "rnaseq_de_neglog10p"]])

    out = expr.merge(de, on="gene_symbol", how="inner")  # symbols present in BOTH

    # --- LOUD validation: nothing zero, nothing degenerate, nothing NaN ---
    for c in FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out[FEATURES].isna().any().any(): fail("NaN in canonical features")
    for c in FEATURES:
        s = out[c]
        if s.nunique(dropna=True) <= 1:
            fail(f"feature '{c}' is constant/degenerate (nunique={s.nunique()}, all-zero={bool((s==0).all())}) "
                 f"-- refusing to write a zero-firing feature")
    if out["gene_symbol"].duplicated().any(): fail("duplicate gene_symbol")
    if out["gene_symbol"].str.startswith("ENSG").any(): fail("residual ENSG symbols")
    for g in ("TP53", "HBB"):
        if not (out["gene_symbol"] == g).any(): fail(f"{g} missing from canonical parquet")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = out[["gene_symbol"] + FEATURES].sort_values("gene_symbol").reset_index(drop=True)
    out.to_parquet(args.out, index=False)
    print(f"OK wrote {args.out}  ({len(out)} symbols)")
    for c in FEATURES:
        print(f"  {c}: nunique={out[c].nunique()} min={out[c].min():.4g} max={out[c].max():.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
