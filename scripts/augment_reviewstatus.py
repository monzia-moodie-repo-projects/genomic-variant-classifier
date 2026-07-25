#!/usr/bin/env python3
"""Part 1 of Path A: attach a ReviewStatus column to the clean cohort from the
ClinVar VCF's CLNREVSTAT, so DataPrepPipeline's --min-review-tier filter (which
reads df["ReviewStatus"]) finally fires.

CRITICAL: ClinVar VCF encodes CLNREVSTAT with underscores; the shared resolver
normalises underscores to spaces and lowercases before an EXACT lookup. So we
store the DECODED (underscore->space) status. A row whose status is a recognised
missing token ("" here) resolves to TIER_MISSING; a row whose status is real but
unknown to the map raises -- construction fails closed rather than silently
demoting it. See src/genomic_variant_classifier/data/review_status.py.

Step 1b (2026-07-24): the local REVIEW_STATUS_TIER map and substring _tier_of were
removed. This script now consumes the single canonical resolver. Unknown VCF
vocabulary is validated in ONE aggregate preflight (Option C): every distinct raw
CLNREVSTAT value is resolved once; any that the map does not recognise are
collected and reported together in a single raise, so the fix is one map edit, not
a fix-rerun-fix loop. No permissive resolver path is used anywhere, and no
fallback tier is fabricated.

Safety: backup-first, idempotent (skips if ReviewStatus already present), atomic
write (tmp + os.replace), and verifies rows/null/dup unchanged + tier<=3 count.
Read VCF + write augmented cohort in place. Usage:
  python augment_reviewstatus.py [cohort.parquet] [clinvar.vcf.gz]
"""
from __future__ import annotations
import gzip, os, shutil, sys
from collections import Counter
from pathlib import Path
import pandas as pd

from genomic_variant_classifier.data.review_status import (
    UnmatchedReviewStatusError,
    normalise,
    resolve,
    tier_of,
)

PATHOGENIC_TERMS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN_TERMS = {"Benign", "Likely benign", "Benign/Likely benign"}


def _norm_chrom(c): c = str(c); return c[3:] if c.lower().startswith("chr") else c


def _build_strict_tier_lookup(series: pd.Series) -> dict[object, int]:
    """Resolve every DISTINCT review status once and return a raw-value -> tier map.

    Option C aggregate preflight: strictly resolve each distinct value; collect any
    unmatched ones (keyed by their canonical normalised form, with raw spellings and
    counts retained); and if any exist, raise ONCE with the complete inventory rather
    than aborting on the first. Missing tokens resolve normally (they are recognised),
    so a recognised absence never lands in the unmatched inventory. No fallback tier
    is ever assigned.
    """
    lookup: dict[object, int] = {}
    # normalised term -> {raw spelling -> row count}
    unmatched: dict[str, dict[str, int]] = {}

    counts = series.value_counts(dropna=False)
    for raw_value, count in counts.items():
        try:
            lookup[raw_value] = tier_of(raw_value)
        except UnmatchedReviewStatusError:
            key = normalise(raw_value)
            raw_repr = "<NA>" if pd.isna(raw_value) else str(raw_value)
            unmatched.setdefault(key, {})
            unmatched[key][raw_repr] = unmatched[key].get(raw_repr, 0) + int(count)

    if unmatched:
        blocks = []
        for norm_key in sorted(unmatched):
            total = sum(unmatched[norm_key].values())
            blocks.append(f"  {norm_key!r}: {total:,} row(s)")
            for raw_form, n in sorted(unmatched[norm_key].items(), key=lambda kv: -kv[1]):
                blocks.append(f"      raw {raw_form!r}: {n:,}")
        raise UnmatchedReviewStatusError(
            "Unmatched ClinVar CLNREVSTAT vocabulary prevents cohort augmentation:\n"
            + "\n".join(blocks)
            + "\nAdd every normalised value to REVIEW_STATUS_TIER and "
            "REVIEW_STATUS_SEMANTICS in\n"
            "src/genomic_variant_classifier/data/review_status.py before rerunning."
        )
    return lookup


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
            decoded = rev.replace("_", " ")  # store SPACE-form for the resolver's matcher
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
    df["ReviewStatus"] = key.map(vmap).fillna("")  # unmatched-in-VCF -> "" -> TIER_MISSING
    n_matched = int((df["ReviewStatus"] != "").sum())

    # verify rows/null/dup unchanged
    assert len(df) == n0, "row count changed!"
    assert int(df["ref"].isna().sum() + df["alt"].isna().sum()) == null0, "null changed!"
    assert int(df["variant_id"].duplicated().sum()) == dup0, "dup changed!"

    # simulate the pipeline filter to confirm it WILL fire, using the strict resolver.
    # Aggregate preflight first: every distinct labeled ReviewStatus must resolve.
    sig = df["clinical_sig"].fillna("").str.strip()
    labeled = df[sig.isin(PATHOGENIC_TERMS | BENIGN_TERMS)]
    tier_lookup = _build_strict_tier_lookup(labeled["ReviewStatus"])
    tier = labeled["ReviewStatus"].map(tier_lookup)
    keep3 = int((tier <= 3).sum())
    print(f"rows={n0:,} (unchanged) | null={null0} dup={dup0} (unchanged)")
    print(f"ReviewStatus non-empty (matched)={n_matched:,}")
    print(f"distinct labeled statuses resolved (unmatched=0): {len(tier_lookup):,}")
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
