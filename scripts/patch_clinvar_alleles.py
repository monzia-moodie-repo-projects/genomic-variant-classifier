"""
scripts/patch_clinvar_alleles.py
=================================
The ClinVar variant_summary.txt has ref="na"/alt="na" for 99.99% of rows.
This script reads the ClinVar VCF (which has proper REF/ALT), extracts
(VariationID, chrom, pos, ref, alt, consequence), and left-joins it onto
the existing clinvar_grch38.parquet to populate those columns.

Replaces: data/processed/clinvar_grch38.parquet  (in-place)

Usage:
  .venv\Scripts\python scripts\patch_clinvar_alleles.py \
      --vcf   data/raw/clinvar/clinvar_GRCh38.vcf.gz \
      --parquet data/processed/clinvar_grch38.parquet
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("patch_clinvar_alleles")

# VCF INFO field parsers
_CLNSIG_RE   = re.compile(r"CLNSIG=([^;]+)")
_CLNVC_RE    = re.compile(r"CLNVC=([^;]+)")   # variant type
_MC_RE       = re.compile(r"MC=([^;]+)")       # molecular consequence
_GENEINFO_RE = re.compile(r"GENEINFO=([^:;]+)") # gene symbol


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vcf",     required=True)
    p.add_argument("--parquet", required=True)
    return p.parse_args()


def parse_consequence(info: str) -> str | None:
    """Extract VEP-style consequence from MC= field."""
    m = _MC_RE.search(info)
    if not m:
        return None
    # MC field: SO:0001583|missense_variant,SO:0001627|intron_variant
    # Take the first consequence term
    first = m.group(1).split(",")[0]
    parts = first.split("|")
    return parts[1] if len(parts) > 1 else None


def extract_vcf_alleles(vcf_path: str) -> pd.DataFrame:
    """
    Parse ClinVar VCF and return DataFrame with columns:
      source_id (VariationID as str), chrom, pos, ref, alt, consequence
    """
    rows: list[dict] = []
    n = 0

    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            n += 1
            if n % 500_000 == 0:
                logger.info("  Parsed %dK VCF lines, %d records kept",
                            n // 1000, len(rows))

            parts = line.split("\t", 8)
            if len(parts) < 8:
                continue

            chrom   = parts[0].lstrip("chr")
            pos_str = parts[1]
            vid     = parts[2]   # VariationID
            ref     = parts[3]
            alt_raw = parts[4]
            info    = parts[7]

            # Skip multi-allelic (rare in ClinVar VCF, but possible)
            if "," in alt_raw:
                alt_raw = alt_raw.split(",")[0]

            try:
                pos = int(pos_str)
            except ValueError:
                continue

            consequence = parse_consequence(info)

            rows.append({
                "source_id":   vid,
                "chrom":       chrom,
                "pos":         pos,
                "ref":         ref,
                "alt":         alt_raw,
                "consequence": consequence,
            })

    logger.info("VCF parse complete: %d variants with allele info", len(rows))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    logger.info("Loading existing ClinVar parquet...")
    df = pd.read_parquet(args.parquet)
    logger.info("  %d rows loaded", len(df))

    # Backup
    backup = args.parquet.replace(".parquet", "_noalleles.parquet")
    df.to_parquet(backup, index=False)
    logger.info("Backup saved to %s", backup)

    logger.info("Parsing ClinVar VCF for allele info...")
    vcf_df = extract_vcf_alleles(args.vcf)

    # Ensure source_id types match for join
    df["source_id"]    = df["source_id"].astype(str)
    vcf_df["source_id"] = vcf_df["source_id"].astype(str)

    # Drop existing ref/alt/consequence (all "na") before merge
    drop_cols = [c for c in ["ref", "alt", "consequence"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Left join on source_id (VariationID)
    df = df.merge(
        vcf_df[["source_id", "ref", "alt", "consequence"]],
        on="source_id",
        how="left",
    )

    # Rebuild variant_id with real alleles where available
    has_alleles = df["ref"].notna() & (df["ref"] != "na") & \
                  df["alt"].notna() & (df["alt"] != "na")

    new_id = (
        "clinvar:" +
        df["chrom"].astype(str) + ":" +
        df["pos"].astype(str)   + ":" +
        df["ref"].astype(str)   + ":" +
        df["alt"].astype(str)
    )
    df["variant_id"] = np.where(has_alleles, new_id, df["variant_id"].astype(str))

    n_with = has_alleles.sum()
    logger.info("Variants with real REF/ALT: %d / %d (%.1f%%)",
                n_with, len(df), n_with / len(df) * 100)
    logger.info("Consequence coverage: %d / %d (%.1f%%)",
                df["consequence"].notna().sum(), len(df),
                df["consequence"].notna().mean() * 100)

    df.to_parquet(args.parquet, index=False)
    logger.info("Updated parquet saved to %s", args.parquet)

    # Quick check
    logger.info("Sample variant_ids after patch:")
    sample = df[has_alleles]["variant_id"].head(5).tolist()
    for v in sample:
        logger.info("  %s", v)


if __name__ == "__main__":
    main()
