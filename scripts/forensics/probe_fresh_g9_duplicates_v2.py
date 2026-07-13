#!/usr/bin/env python
"""probe_fresh_g9_duplicates_v2.py (2026-07-10)
Re-characterize the fresh duplicate-variant_id groups AFTER the Conflicting->uncertain label
fix, to establish the CURRENT genuine-conflict set that the dedup collapse must handle. Runs on
the label-corrected, finalized fresh parquet. For every fully-alleled duplicate variant_id group
(the ones that survive quarantine and reach the builder's G9 check), prints:
  * the variant_id, chrom:pos:ref:alt
  * every participating row's gene_symbol, source_id (VariationID), clinical_sig, pathogenicity
  * whether the group is an IDENTICAL twin (all pathogenicity equal) or a CONFLICT (differ)
Prediction to TEST (not assume): the TTN group, formerly pathogenic-vs-uncertain, is now an
IDENTICAL twin (both uncertain) because its 'Conflicting' row became uncertain; SEC23B and GLA
(both likely_pathogenic-vs-pathogenic) remain genuine conflicts. Read-only.
Acronyms: ClinVar Variation Identifier (VariationID), Variant Call Format (VCF).
"""
import sys
from pathlib import Path
print("=== probe_fresh_g9_duplicates_v2 START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

FRESH = Path("data/processed/clinvar_grch38_fresh.parquet")
if not FRESH.exists():
    print(f"FATAL: {FRESH} missing", flush=True); sys.exit(2)

df = pd.read_parquet(FRESH)

def is_empty(x):
    if x is None: return True
    s = str(x).strip().lower()
    return s in ("", "none", "nan", ".", "-", "na")

# fully-alleled rows only (these are what reach the clean cohort / G9)
alleleful = df[~(df["ref"].map(is_empty) | df["alt"].map(is_empty))].copy()
dmask = alleleful["variant_id"].duplicated(keep=False)
dups = alleleful[dmask].sort_values("variant_id")
groups = dups.groupby("variant_id")
print(f"  fully-alleled duplicate variant_id groups: {dups['variant_id'].nunique()} "
      f"({len(dups)} rows)", flush=True)

identical, conflict = [], []
for vid, g in groups:
    paths = list(g["pathogenicity"].astype("string").fillna("<NA>"))
    is_conflict = len(set(paths)) > 1
    (conflict if is_conflict else identical).append(vid)
    kind = "CONFLICT" if is_conflict else "identical-twin"
    row0 = g.iloc[0]
    print(f"\n  [{kind}] {vid}", flush=True)
    print(f"    chrom:pos:ref:alt = {row0['chrom']}:{row0['pos']}:{row0['ref']}:{row0['alt']}", flush=True)
    for _, r in g.iterrows():
        print(f"      gene={str(r.get('gene_symbol'))[:16]:16s} "
              f"VariationID(source_id)={str(r.get('source_id'))[:12]:12s} "
              f"pathogenicity={str(r.get('pathogenicity')):18s} "
              f"clinical_sig={str(r.get('clinical_sig'))[:48]}", flush=True)

print(f"\n  SUMMARY: {len(identical)} identical-twin groups, {len(conflict)} genuine-conflict groups", flush=True)
print(f"    identical-twin ids: {identical}", flush=True)
print(f"    genuine-conflict ids: {conflict}", flush=True)
print("\n  This is the authoritative CURRENT conflict set the dedup collapse must resolve", flush=True)
print("  (post label-fix). The collapse keeps ALL VariationIDs + original calls in metadata;", flush=True)
print("  for conflicts it selects the most-severe survivor and flags classification_conflict.", flush=True)
print("=== probe_fresh_g9_duplicates_v2 DONE ===", flush=True)
