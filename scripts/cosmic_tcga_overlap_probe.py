#!/usr/bin/env python
"""cosmic_tcga_overlap_probe.py -- VERIFY (not assume) how much TCGA somatic overlaps COSMIC.

Tests the claim "a TCGA somatic-recurrence feature would largely duplicate COSMIC" by
intersecting a TCGA open-access Masked Somatic Mutation MAF against the COSMIC CMC index
we already built (data/external/cosmic/cosmic_cmc_grch38_index.parquet), on the same
GRCh38 chrom:pos:ref:alt substitution key. Reports overlap fractions. READ-ONLY.

Get one open-access MAF first (no dbGaP needed for MASKED somatic MAFs), e.g. via the GDC
API (see the printed instructions), then:
    python scripts/cosmic_tcga_overlap_probe.py --maf <TCGA...masked.maf.gz>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_INDEX = Path("data/external/cosmic/cosmic_cmc_grch38_index.parquet")


def _norm_chrom(c: str) -> str:
    c = str(c).strip()
    if c[:3].lower() == "chr":
        c = c[3:]
    return "MT" if c in ("M", "MT") else c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maf", required=True, type=Path, help="TCGA masked somatic MAF (.maf/.maf.gz)")
    ap.add_argument("--cosmic-index", type=Path, default=DEFAULT_INDEX)
    args = ap.parse_args()

    if not args.maf.exists():
        print(f"FAIL: MAF not found: {args.maf}"); return 2
    if not args.cosmic_index.exists():
        print(f"FAIL: COSMIC index not found: {args.cosmic_index} (run the activation probe first)."); return 2

    comp = "gzip" if str(args.maf).endswith(".gz") else None
    maf = pd.read_csv(args.maf, sep="\t", comment="#", dtype=str, compression=comp, low_memory=False)
    need = ["Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"]
    miss = [c for c in need if c not in maf.columns]
    if miss:
        print(f"FAIL: MAF missing columns {miss}; found first 12: {list(maf.columns)[:12]}"); return 2
    print(f"MAF rows: {len(maf)}")
    if "Variant_Type" in maf.columns:
        print("Variant_Type counts:", maf["Variant_Type"].value_counts().to_dict())

    ref = maf["Reference_Allele"].astype(str).str.upper()
    alt = maf["Tumor_Seq_Allele2"].astype(str).str.upper()
    snv = ref.str.fullmatch(r"[ACGT]") & alt.str.fullmatch(r"[ACGT]")
    m = maf[snv].copy()
    print(f"SNV substitutions in MAF: {len(m)} ({100*len(m)/max(len(maf),1):.1f}%)")

    keys = (m["Chromosome"].map(_norm_chrom).astype(str) + ":"
            + m["Start_Position"].astype(str) + ":"
            + ref[snv] + ":" + alt[snv])
    maf_keys = set(keys)

    idx = pd.read_parquet(args.cosmic_index, columns=["_key"])
    cosmic_keys = set(idx["_key"].astype(str))
    print(f"COSMIC index substitutions: {len(cosmic_keys)}")

    inter = maf_keys & cosmic_keys
    print("\n-- OVERLAP --")
    print(f"unique MAF SNV keys        : {len(maf_keys)}")
    print(f"MAF (intersect) COSMIC               : {len(inter)}")
    if maf_keys:
        print(f"fraction of TCGA-MAF in COSMIC : {100*len(inter)/len(maf_keys):.1f}%")
    if cosmic_keys:
        print(f"fraction of COSMIC hit by MAF  : {100*len(inter)/len(cosmic_keys):.3f}%")
    print("\nINTERPRETATION: high 'fraction of TCGA-MAF in COSMIC' -> a TCGA somatic-recurrence")
    print("feature would largely duplicate COSMIC (COSMIC ingests TCGA). Low -> it would add signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
