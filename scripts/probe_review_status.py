#!/usr/bin/env python3
"""DIAGNOSTIC (read-only, no VM): quantify the impact of the missing ReviewStatus
column on the --min-review-tier filter.

Joins CLNREVSTAT from the raw ClinVar VCF onto the processed cohort by
chrom:pos:ref:alt, maps it to the pipeline's tier scale, and reports:
  - labeled cohort size (clinical_sig in PATHOGENIC/BENIGN)
  - VCF<->cohort join rate (are coordinates compatible? -> Path A feasibility)
  - tier distribution among labeled variants
  - tier<=3 count == the would-be tier>=3 training size (budget input)

Writes nothing. Pure measurement to inform Path A vs Path B.
Usage: python probe_review_status.py [cohort.parquet] [clinvar.vcf.gz]
"""
from __future__ import annotations
import gzip
import sys
from pathlib import Path
import pandas as pd

REVIEW_STATUS_TIER = {
    "practice guideline": 1,
    "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "no assertion criteria provided": 4,
    "no classification provided": 5,
    "no classification for the individual variant": 5,
}
PATHOGENIC_TERMS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN_TERMS = {"Benign", "Likely benign", "Benign/Likely benign"}


def _norm_chrom(c: str) -> str:
    c = str(c)
    return c[3:] if c.lower().startswith("chr") else c


def _tier_of(revstat_raw: str) -> int:
    # ClinVar VCF encodes CLNREVSTAT with underscores for spaces
    s = revstat_raw.replace("_", " ").lower()
    return next((v for k, v in REVIEW_STATUS_TIER.items() if k in s), 5)


def _parse_info(info: str) -> dict[str, str]:
    out = {}
    for kv in info.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def build_vcf_tier_map(vcf_path: Path) -> dict[str, int]:
    m: dict[str, int] = {}
    n = 0
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, _id, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            info = _parse_info(parts[7])
            rev = info.get("CLNREVSTAT")
            if not rev:
                continue
            # ALT may be comma-separated; index each allele
            for a in alt.split(","):
                key = f"{_norm_chrom(chrom)}:{pos}:{ref}:{a}"
                m[key] = _tier_of(rev)
            n += 1
            if n % 500000 == 0:
                print(f"  ... parsed {n:,} CLNREVSTAT records", flush=True)
    return m


def main(cohort_path: str, vcf_path: str) -> int:
    coh = pd.read_parquet(cohort_path, columns=["chrom", "pos", "ref", "alt", "clinical_sig"])
    coh["clinical_sig"] = coh["clinical_sig"].fillna("").str.strip()
    labeled = coh[coh["clinical_sig"].isin(PATHOGENIC_TERMS | BENIGN_TERMS)].copy()
    print(f"cohort rows={len(coh):,} | labeled (P/LP/B/LB)={len(labeled):,}")

    coh_keys = (labeled["chrom"].map(_norm_chrom) + ":" + labeled["pos"].astype("int64").astype(str)
                + ":" + labeled["ref"].astype(str) + ":" + labeled["alt"].astype(str))
    print("sample cohort keys:", list(coh_keys.head(3)))

    print(f"parsing VCF {vcf_path} ...")
    tier_map = build_vcf_tier_map(Path(vcf_path))
    print(f"VCF CLNREVSTAT keys={len(tier_map):,}")
    print("sample VCF keys:", list(list(tier_map.keys())[:3]))

    tiers = coh_keys.map(tier_map)
    matched = int(tiers.notna().sum())
    print(f"\njoin: matched {matched:,}/{len(labeled):,} labeled ({100*matched/max(len(labeled),1):.1f}%)")
    if matched:
        dist = tiers.dropna().astype(int).value_counts().sort_index().to_dict()
        print("tier distribution (labeled+matched):", dist)
        keep = int((tiers.dropna().astype(int) <= 3).sum())
        print(f"tier<=3 (== --min-review-tier 3 KEEP) among matched: {keep:,}")
        print(f"would-be reduction: labeled {len(labeled):,} -> tier<=3 ~{keep:,} "
              f"({100*keep/max(len(labeled),1):.1f}% kept)")
    print("\nINTERPRETATION: high join% + meaningful tier<=3 reduction => Path A feasible; "
          "tier<=3 count is the real tier>=3 training size for budget/VRAM.")
    return 0


if __name__ == "__main__":
    cohort = sys.argv[1] if len(sys.argv) > 1 else "data/processed/clinvar_grch38_clean.parquet"
    vcf = sys.argv[2] if len(sys.argv) > 2 else "data/raw/clinvar/clinvar_GRCh38.vcf.gz"
    sys.exit(main(cohort, vcf))
