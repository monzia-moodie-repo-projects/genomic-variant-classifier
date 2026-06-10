#!/usr/bin/env python3
"""probe_uniprot_index.py -- READ-ONLY sizing of the ESM-2 UniProt gene-symbol gap.

Author: Monzia Moodie

Run 15 logged: "ESM-2: gene(s) absent from the UniProt index ... (first missing:
MYH11;NDE1)". Both are standard reviewed-human genes, so their absence is a clue:
either they are genuinely not in data/external/uniprot/uniprot_human_reviewed.parquet
(builder/rebuild problem) or they are present under a different key (alias/case fix).

This probe answers that and -- crucially -- sizes the gap by the number of MISSENSE
VARIANTS affected (not just distinct genes), because that is the true silent-zero
exposure once coord coverage is fixed and ESM-2 runs on ~2.4M missense.

Strictly read-only. No writes, no deletes.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

INDEX_DEFAULT = "data/external/uniprot/uniprot_human_reviewed.parquet"
COHORT_DEFAULT = "data/processed/clinvar_grch38_clean.parquet"
PROBE_GENES = ["MYH11", "NDE1", "BRCA1", "TP53", "MYH7", "CFTR", "TTN", "SCN5A"]


def _load_index_genes(path: str) -> tuple[set, int, int, int, list]:
    """Return (upper_gene_set, n_rows, n_null_seq, n_dup_genes, schema)."""
    import pyarrow.parquet as pq
    schema = pq.ParquetFile(path).schema.names
    cols = [c for c in ("gene_symbol", "sequence") if c in schema] or None
    df = pd.read_parquet(path, columns=cols)
    g = df["gene_symbol"].astype(str).str.strip().str.upper()
    null_seq = 0
    if "sequence" in df.columns:
        seq = df["sequence"].astype(str)
        null_seq = int((seq.str.len() == 0).sum() + seq.str.lower().isin(["nan", "none"]).sum())
    n_dup = int(g.duplicated().sum())
    return set(g[g != ""].tolist()), len(df), null_seq, n_dup, schema


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=INDEX_DEFAULT)
    ap.add_argument("--clinvar", default=COHORT_DEFAULT)
    ap.add_argument("--probe-genes", default=",".join(PROBE_GENES))
    ap.add_argument("--sample-missing", type=int, default=25,
                    help="how many top-by-variant-count missing genes to print")
    args = ap.parse_args()

    if not os.path.isfile(args.index):
        print(f"index not found: {args.index}")
        return 2
    print("READ-ONLY. No writes/deletes.\n")

    genes, n_rows, n_null_seq, n_dup, schema = _load_index_genes(args.index)
    print("=" * 68)
    print(f"[1] UniProt index: {args.index}")
    print(f"    rows={n_rows:,}  unique upper gene_symbol={len(genes):,}  schema={schema}")
    print(f"    null/empty sequence rows={n_null_seq:,}  duplicate gene_symbol rows={n_dup:,}")

    print("\n[2] probe genes present in index?")
    for gene in [x.strip().upper() for x in args.probe_genes.split(",") if x.strip()]:
        print(f"    {gene:<10}: {'PRESENT' if gene in genes else 'ABSENT'}")

    if not os.path.isfile(args.clinvar):
        print(f"\n(cohort not found: {args.clinvar}; skipping variant-weighted gap)")
        return 0

    print("\n[3] cohort cross-check (missense variants weighted)")
    coh = pd.read_parquet(args.clinvar, columns=["gene_symbol", "consequence"])
    mm = coh[coh["consequence"].fillna("").str.contains("missense", case=False)].copy()
    n_mm = len(mm)
    gsym = mm["gene_symbol"].astype(str).str.strip()
    n_null_gene = int((gsym == "").sum() + gsym.str.lower().isin(["nan", "none"]).sum())
    gsym_u = gsym.str.upper()
    in_index = gsym_u.isin(genes)
    n_in = int(in_index.sum())
    n_out = n_mm - n_in
    print(f"    missense variants                 : {n_mm:,}")
    print(f"    ... in a gene present in index    : {n_in:,} ({100*n_in/max(n_mm,1):.2f}%)")
    print(f"    ... in a gene ABSENT from index   : {n_out:,} ({100*n_out/max(n_mm,1):.2f}%)  <- silent ESM-2 zeros")
    print(f"    ... with null/empty gene_symbol   : {n_null_gene:,}")

    missing = (
        mm.loc[~in_index]
        .assign(_g=gsym_u[~in_index].values)
    )
    missing = missing[missing["_g"] != ""]
    top = missing.groupby("_g").size().sort_values(ascending=False)
    n_missing_genes = int(top.shape[0])
    print(f"\n    distinct genes absent from index  : {n_missing_genes:,}")
    print(f"    top {min(args.sample_missing, n_missing_genes)} missing genes by missense-variant count:")
    for gene, cnt in top.head(args.sample_missing).items():
        print(f"      {gene:<14} {cnt:,}")

    print("\nINTERPRETATION:")
    print("  - If MYH11/NDE1 show ABSENT in [2] and appear in [3]'s top list, the index")
    print("    simply lacks them -> rebuild/expand the index (or add an HGNC alias map).")
    print("  - If the missing genes look like old/withdrawn symbols or aliases, an")
    print("    alias-resolution map is the fix; if they are current primary symbols, the")
    print("    index build is incomplete and should be regenerated.")
    print("  - The [3] 'ABSENT from index' percent is the variant-weighted silent-zero")
    print("    exposure to size against effort.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
