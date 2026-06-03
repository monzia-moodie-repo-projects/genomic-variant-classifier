#!/usr/bin/env python3
"""Part 1 of Path A: attach a ReviewStatus column to the clean cohort from the
ClinVar VCF's CLNREVSTAT, so DataPrepPipeline's --min-review-tier filter (which
reads df["ReviewStatus"]) finally fires.

CRITICAL: ClinVar VCF encodes CLNREVSTAT with underscores; the pipeline substring-
matches SPACE-form REVIEW_STATUS_TIER keys after .lower(). So we store the
DECODED (underscore->space) status. Unmatched variants get "" -> pipeline maps to
tier 5 -> excluded by <=3 (conservative: only confirmed tier>=3 are kept).

Safety: backup-first, idempotent (skips if ReviewStatus already present), atomic
write (tmp + os.replace), and verifies rows/null/dup unchanged + tier<=3 count.
Read VCF + write augmented cohort in place. Usage:
  python augment_reviewstatus.py [cohort.parquet] [clinvar.vcf.gz]
"""
from __future__ import annotations
import gzip, os, shutil, sys
from pathlib import Path
import pandas as pd

REVIEW_STATUS_TIER = {
    "practice guideline": 1, "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "no assertion criteria provided": 4, "no classification provided": 5,
    "no classification for the individual variant": 5,
}
PATHOGENIC_TERMS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN_TERMS = {"Benign", "Likely benign", "Benign/Likely benign"}

def _norm_chrom(c): c = str(c); return c[3:] if c.lower().startswith("chr") else c
def _tier_of(s): s = str(s).lower(); return next((v for k, v in REVIEW_STATUS_TIER.items() if k in s), 5)

def build_vcf_map(vcf_path: Path) -> dict[str, str]:
    m = {}
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"): continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 8: continue
            info = dict(kv.split("=", 1) for kv in p[7].split(";") if "=" in kv)
            rev = info.get("CLNREVSTAT")
            if not rev: continue
            decoded = rev.replace("_", " ")  # store SPACE-form for the pipeline's matcher
            for a in p[4].split(","):
                m[f"{_norm_chrom(p[0])}:{p[1]}:{p[3]}:{a}"] = decoded
    return m

def main(cohort_path: str, vcf_path: str) -> int:
    cp = Path(cohort_path)
    df = pd.read_parquet(cp)
    if "ReviewStatus" in df.columns:
        print("SKIP: ReviewStatus already present (idempotent)"); return 0
    n0 = len(df)
    null0 = int(df["ref"].isna().sum() + df["alt"].isna().sum())
    dup0 = int(df["variant_id"].duplicated().sum())

    print(f"parsing VCF {vcf_path} ...")
    vmap = build_vcf_map(Path(vcf_path))
    print(f"VCF CLNREVSTAT keys={len(vmap):,}")

    key = (df["chrom"].map(_norm_chrom) + ":" + df["pos"].astype("int64").astype(str)
           + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str))
    df["ReviewStatus"] = key.map(vmap).fillna("")  # unmatched -> "" -> tier 5
    n_matched = int((df["ReviewStatus"] != "").sum())

    # verify rows/null/dup unchanged
    assert len(df) == n0, "row count changed!"
    assert int(df["ref"].isna().sum() + df["alt"].isna().sum()) == null0, "null changed!"
    assert int(df["variant_id"].duplicated().sum()) == dup0, "dup changed!"

    # simulate the pipeline filter to confirm it WILL fire
    sig = df["clinical_sig"].fillna("").str.strip()
    labeled = df[sig.isin(PATHOGENIC_TERMS | BENIGN_TERMS)]
    tier = labeled["ReviewStatus"].map(_tier_of)
    keep3 = int((tier <= 3).sum())
    print(f"rows={n0:,} (unchanged) | null={null0} dup={dup0} (unchanged)")
    print(f"ReviewStatus non-empty (matched)={n_matched:,}")
    print(f"labeled={len(labeled):,} | tier<=3 (pipeline KEEP @ min_review_tier 3)={keep3:,}")
    print(f"tier dist (labeled): {tier.value_counts().sort_index().to_dict()}")

    # backup + atomic write
    bak = cp.with_suffix(cp.suffix + ".pre_reviewstatus.bak")
    if not bak.exists():
        shutil.copy2(cp, bak); print(f"backup -> {bak}")
    else:
        print(f"backup already exists -> {bak} (left as-is)")
    tmp = cp.with_suffix(cp.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, cp)
    print(f"WROTE augmented cohort -> {cp} (+ReviewStatus, {len(df.columns)} cols)")
    return 0

if __name__ == "__main__":
    coh = sys.argv[1] if len(sys.argv) > 1 else "data/processed/clinvar_grch38_clean.parquet"
    vcf = sys.argv[2] if len(sys.argv) > 2 else "data/raw/clinvar/clinvar_GRCh38.vcf.gz"
    sys.exit(main(coh, vcf))
