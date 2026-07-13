#!/usr/bin/env python
"""probe_vcf_columns.py (2026-07-09)
Confirm the root cause of the corrupt fresh parquet: is the non-VCF ReferenceAllele column
deprecated (mostly 'na') in the 2026-07 variant_summary, with the REAL alleles now in
ReferenceAlleleVCF/AlternateAlleleVCF? And how does PositionVCF relate to Start? Read-only,
samples a modest number of GRCh38 rows. Reports, among GRCh38 rows in the sample:
  * ReferenceAllele: %% 'na' vs populated
  * ReferenceAlleleVCF: %% 'na' vs populated
  * AlternateAllele vs AlternateAlleleVCF likewise
  * For rows where BOTH the non-VCF and VCF alleles are populated, do they AGREE? (they encode
    the same variant differently: non-VCF uses '-' for indels/no pad; VCF is padded.)
  * Start vs PositionVCF: distribution of (Start - PositionVCF) to see the offset convention.
  * A few concrete example rows showing all six columns side by side.
This tells us definitively which columns carry the alleles in 2026-07 and how coordinates map,
so we can design a fresh ingestion that is COORDINATE-COMPATIBLE with the stale parquet.
"""
import sys, gzip, argparse
from collections import Counter
from pathlib import Path
print("=== probe_vcf_columns START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def is_na_token(v):
    if v is None: return True
    try:
        if pd.isna(v): return True
    except (TypeError, ValueError): pass
    return str(v).strip().lower() in {"", "na", "nan", "none", ".", "-", "null"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", default="data/external/clinvar/variant_summary.txt.gz")
    ap.add_argument("--nrows", type=int, default=500000)
    a = ap.parse_args()

    op = gzip.open if str(a.fresh).endswith(".gz") else open
    print(f"reading first {a.nrows:,} rows of {a.fresh} (all columns as str) ...", flush=True)
    with op(a.fresh, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, sep="\t", low_memory=False, nrows=a.nrows, dtype=str)
    g = df[df["Assembly"] == "GRCh38"].copy()
    print(f"GRCh38 rows in sample: {len(g):,}", flush=True)
    print()

    cols = ["ReferenceAllele","AlternateAllele","ReferenceAlleleVCF","AlternateAlleleVCF",
            "Start","PositionVCF"]
    have = [c for c in cols if c in g.columns]
    print("columns present:", have, flush=True)
    print()

    for c in ("ReferenceAllele","ReferenceAlleleVCF","AlternateAllele","AlternateAlleleVCF"):
        if c in g.columns:
            na = g[c].map(is_na_token).sum()
            pop = len(g) - na
            print(f"  {c:22s}: na/empty {na:,} ({100*na/len(g):.2f}%)  populated {pop:,} ({100*pop/len(g):.2f}%)", flush=True)
    print()

    # Agreement where both populated
    if all(c in g.columns for c in ("ReferenceAllele","ReferenceAlleleVCF")):
        both = g[(~g["ReferenceAllele"].map(is_na_token)) & (~g["ReferenceAlleleVCF"].map(is_na_token))]
        print(f"  rows with BOTH ReferenceAllele and ReferenceAlleleVCF populated: {len(both):,}", flush=True)
        if len(both):
            agree = (both["ReferenceAllele"].str.upper() == both["ReferenceAlleleVCF"].str.upper()).sum()
            print(f"    exact match: {agree:,}/{len(both):,}", flush=True)

    # Start vs PositionVCF offset
    if all(c in g.columns for c in ("Start","PositionVCF")):
        s = pd.to_numeric(g["Start"], errors="coerce")
        pv = pd.to_numeric(g["PositionVCF"], errors="coerce")
        both_pos = (~s.isna()) & (~pv.isna())
        diff = (s[both_pos] - pv[both_pos])
        print(f"\n  Start - PositionVCF distribution (rows with both numeric: {int(both_pos.sum()):,}):", flush=True)
        for k, v in Counter(diff.astype(int)).most_common(8):
            print(f"      diff={k:+d}: {v:,}", flush=True)

    # concrete examples: rows where VCF alleles are populated
    print("\n  --- example rows (VCF alleles populated) ---", flush=True)
    ex = g[~g["ReferenceAlleleVCF"].map(is_na_token)].head(6) if "ReferenceAlleleVCF" in g.columns else g.head(0)
    for _, r in ex.iterrows():
        parts = []
        for c in have:
            parts.append(f"{c}={r[c]}")
        print("    " + " | ".join(parts), flush=True)

    print("\n--- VERDICT ---", flush=True)
    if "ReferenceAlleleVCF" in g.columns and "ReferenceAllele" in g.columns:
        ra_na = g["ReferenceAllele"].map(is_na_token).mean()
        rav_na = g["ReferenceAlleleVCF"].map(is_na_token).mean()
        print(f"  ReferenceAllele empty rate {100*ra_na:.1f}% ; ReferenceAlleleVCF empty rate {100*rav_na:.1f}%", flush=True)
        if ra_na > 0.9 and rav_na < 0.5:
            print("  CONFIRMED: ReferenceAllele is deprecated/mostly-empty in 2026-07; the REAL", flush=True)
            print("  alleles are in ReferenceAlleleVCF. The connector reads the wrong column for", flush=True)
            print("  this snapshot. Fresh ingestion must use the *VCF columns + PositionVCF.", flush=True)
        else:
            print("  Pattern not as hypothesized -- inspect the numbers above before deciding.", flush=True)
    print("=== probe_vcf_columns DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
