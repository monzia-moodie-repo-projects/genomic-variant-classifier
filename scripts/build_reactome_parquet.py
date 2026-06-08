#!/usr/bin/env python3
"""
scripts/build_reactome_parquet.py
=================================
Aggregate a Reactome bulk mapping file into the per-gene parquet that
ReactomeConnector consumes:

    gene_symbol             str   HGNC gene symbol
    reactome_pathway_count  int   number of DISTINCT Reactome pathways for the gene

Input
-----
A Reactome "All_Levels" mapping TSV downloaded from
https://reactome.org/download-data , one of:
    Ensembl2Reactome_All_Levels.txt   (source id = Ensembl gene/transcript/protein)
    UniProt2Reactome_All_Levels.txt   (source id = UniProt accession)
    NCBI2Reactome_All_Levels.txt      (source id = NCBI Gene id)

These files are tab-separated with NO header and (standard layout) columns:
    0 source_id   1 pathway_stable_id   2 url   3 pathway_name
    4 evidence_code   5 species

Reactome does NOT key these files by HGNC symbol, so you must say how the
source id maps to a gene symbol:

  --symbol-map PATH      a TSV or parquet with columns (source_id, gene_symbol)
                         to translate the source id space into HGNC symbols.
  --source-is-symbol     treat column 0 as already being the gene symbol
                         (use only for a custom symbol-keyed export).

Exactly one of those is required, so the output key space is never guessed.

This script does NOT download anything (the egress is intentionally narrow);
download the file yourself, then point --input at it.

Usage
-----
    python scripts/build_reactome_parquet.py \
        --input data/external/reactome/Ensembl2Reactome_All_Levels.txt \
        --symbol-map data/external/reactome/ensembl_to_symbol.tsv \
        --out data/external/reactome_gene_pathways.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_DEFAULT_COLS = ["source_id", "pathway_stable_id", "url", "pathway_name",
                 "evidence_code", "species"]


def _load_symbol_map(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        m = pd.read_parquet(path)
    else:
        m = pd.read_csv(path, sep="\t", dtype=str)
    cols = {c.lower(): c for c in m.columns}
    if "source_id" not in cols or "gene_symbol" not in cols:
        raise SystemExit(
            f"--symbol-map {path} must have columns (source_id, gene_symbol); "
            f"found {list(m.columns)}"
        )
    m = m.rename(columns={cols["source_id"]: "source_id",
                          cols["gene_symbol"]: "gene_symbol"})
    return m[["source_id", "gene_symbol"]].dropna().drop_duplicates()


def build(
    input_path: Path,
    out_path: Path,
    symbol_map: Path | None,
    source_is_symbol: bool,
    species: str,
) -> pd.DataFrame:
    raw = pd.read_csv(
        input_path, sep="\t", header=None, names=_DEFAULT_COLS,
        dtype=str, usecols=range(len(_DEFAULT_COLS)),
    )
    raw = raw[raw["species"].str.strip().str.lower() == species.strip().lower()]
    raw = raw.dropna(subset=["source_id", "pathway_stable_id"])

    if source_is_symbol:
        raw = raw.rename(columns={"source_id": "gene_symbol"})
    else:
        if symbol_map is None:
            raise SystemExit(
                "Pass --symbol-map (source_id,gene_symbol) or --source-is-symbol; "
                "the gene-symbol key space must not be guessed."
            )
        m = _load_symbol_map(symbol_map)
        raw = raw.merge(m, on="source_id", how="inner")  # drop unmapped ids

    agg = (
        raw.groupby("gene_symbol")["pathway_stable_id"]
        .nunique()
        .rename("reactome_pathway_count")
        .reset_index()
    )
    agg["reactome_pathway_count"] = agg["reactome_pathway_count"].astype(int)
    agg = agg[agg["gene_symbol"].astype(str).str.len() > 0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out_path, index=False)
    return agg


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--symbol-map", type=Path, default=None)
    ap.add_argument("--source-is-symbol", action="store_true")
    ap.add_argument("--species", default="Homo sapiens")
    args = ap.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"--input not found: {args.input}")

    agg = build(args.input, args.out, args.symbol_map, args.source_is_symbol, args.species)
    print(f"Wrote {args.out}  ({len(agg)} genes; "
          f"max pathways/gene={int(agg['reactome_pathway_count'].max()) if len(agg) else 0}).")


if __name__ == "__main__":
    main(sys.argv[1:])
