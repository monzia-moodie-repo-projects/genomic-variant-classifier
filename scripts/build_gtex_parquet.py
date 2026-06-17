#!/usr/bin/env python3
"""
scripts/build_gtex_parquet.py
=============================
Aggregate the GTEx bulk *median gene-level TPM* GCT into the per-gene parquet
that the data pipeline joins for RNA-expression features:

    gene_symbol               str    HGNC gene symbol (GCT "Description" column)
    gtex_max_tpm              float  max median TPM across tissues
    gtex_n_tissues_expressed  int    tissues with median TPM >= 1.0
    gtex_tissue_specificity   float  1 - mean_tpm / max_tpm   (0 when max==0)

These three are gene-level and match GTExConnector._summarise_expression exactly
(same GTEX_EXPR_MIN_TPM threshold and specificity formula), so the offline bulk
path and the live-API path produce identical features. The variant-level eQTL
trio (gtex_is_eqtl / gtex_min_eqtl_pval / gtex_max_abs_effect) is NOT produced
here -- it needs the (much larger) significant-eQTL bulk files and stays at its
0 defaults.

Input
-----
GTEx bulk median-TPM GCT (https://gtexportal.org/home/downloads/adult-gtex):
    GTEx_Analysis_*_gene_median_tpm.gct.gz
GCT layout: line 1 = "#1.2", line 2 = "<nrows>\t<ncols>", line 3 = header
(Name, Description, <tissue 1>, <tissue 2>, ...), then one row per gene.

This script does NOT download anything; download the GCT yourself and point
--gct at it. .gz is read transparently.

Usage
-----
    python scripts/build_gtex_parquet.py \
        --gct data/external/gtex/GTEx_Analysis_v10_gene_median_tpm.gct.gz \
        --out data/external/gtex_gene_expression.parquet
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd

from genomic_variant_classifier.data.gtex import GTEX_EXPR_MIN_TPM


def _read_gct(gct_path: Path) -> pd.DataFrame:
    """Read a GCT (gzip-aware), returning the gene x tissue TPM table.

    Fails LOUDLY if the file is not a GCT (first line must be '#1.2').
    """
    opener = gzip.open if gct_path.suffix == ".gz" else open
    with opener(gct_path, "rt", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
        if not first.startswith("#1.2"):
            raise SystemExit(
                f"{gct_path.name}: not a GCT -- first line is {first!r}, expected '#1.2'. "
                "Point --gct at GTEx_Analysis_*_gene_median_tpm.gct(.gz)."
            )
        fh.readline()  # dims line "<nrows>\t<ncols>"; not relied upon
        df = pd.read_csv(fh, sep="\t")
    if "Name" not in df.columns or "Description" not in df.columns:
        raise SystemExit(
            f"{gct_path.name}: GCT header missing Name/Description "
            f"(found {list(df.columns)[:4]}...)."
        )
    return df


def summarise_gct(gct_path: Path) -> pd.DataFrame:
    """GCT -> per-gene (gtex_max_tpm, gtex_n_tissues_expressed, gtex_tissue_specificity).

    Duplicate gene symbols (multiple Ensembl ids -> one HGNC symbol) are collapsed
    by taking the MAX median TPM per tissue across the duplicate rows (an explicit,
    non-silent choice: a gene counts as expressed in a tissue if any of its mapped
    ids is). Symbols that are empty/NA are dropped.
    """
    df = _read_gct(gct_path)
    tissue_cols = [c for c in df.columns if c not in ("Name", "Description")]
    if not tissue_cols:
        raise SystemExit(f"{gct_path.name}: no tissue columns after Name/Description.")

    mat = df[tissue_cols].apply(pd.to_numeric, errors="coerce")
    sym = df["Description"].astype(str).str.strip()
    keep = sym.str.len() > 0
    mat = mat[keep]
    sym = sym[keep]

    grouped = mat.groupby(sym).max()  # max-per-tissue across duplicate symbols
    max_tpm = grouped.max(axis=1)
    mean_tpm = grouped.mean(axis=1)
    n_expr = (grouped >= GTEX_EXPR_MIN_TPM).sum(axis=1).astype(int)
    spec = (1.0 - mean_tpm / max_tpm)
    spec = spec.where(max_tpm > 0, 0.0).round(4)

    out = pd.DataFrame(
        {
            "gene_symbol": grouped.index.astype(str),
            "gtex_max_tpm": max_tpm.round(4).values,
            "gtex_n_tissues_expressed": n_expr.values,
            "gtex_tissue_specificity": spec.values,
        }
    ).reset_index(drop=True)
    out = out.sort_values("gene_symbol").reset_index(drop=True)
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gct", required=True, type=Path,
                    help="GTEx_Analysis_*_gene_median_tpm.gct(.gz)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    if not args.gct.exists():
        raise SystemExit(f"--gct not found: {args.gct}")

    agg = summarise_gct(args.gct)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(args.out, index=False)
    n_expr_genes = int((agg["gtex_max_tpm"] > 0).sum())
    print(
        f"Wrote {args.out}  ({len(agg)} genes; {n_expr_genes} with max_tpm>0; "
        f"max gtex_max_tpm={float(agg['gtex_max_tpm'].max()) if len(agg) else 0:.1f})."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
