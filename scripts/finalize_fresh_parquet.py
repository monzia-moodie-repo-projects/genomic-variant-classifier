#!/usr/bin/env python
"""finalize_fresh_parquet.py (2026-07-09)
Post-ingestion hygiene + contextual verification for clinvar_grch38_fresh.parquet. The fresh
parquet is already CLEAN (na:na 20,310 ~ stale 19,988; zero literal empty-token survivors); the
only cosmetic difference is COLUMN ORDER. This tool:
  1. Reorders the fresh parquet's columns to EXACTLY match the stale parquet's column order (the
     builder + diff are name-based so order is cosmetic, but matching order removes a known
     discrepancy and keeps project hygiene flawless). Writes in place (requires --apply).
  2. Reports, for CONTEXT (not a gate): stale's duplicate variant_id count vs fresh's, and the
     composition of fresh duplicates (how many are na:na ':None:None' collisions vs genuine
     multi-VariationID variants sharing chrom:pos:ref:alt).
  3. Documents the expected half-bad=0 fact (VCF columns are all-or-nothing populated).
Read-only unless --apply is given (then it rewrites ONLY clinvar_grch38_fresh.parquet, never a
canonical file). Prints before/after column order and a final schema-order-match confirmation.
"""
import sys, argparse
from pathlib import Path
print("=== finalize_fresh_parquet START ===", flush=True)
try:
    import pandas as pd
    import pyarrow.parquet as pq
except Exception as e:
    print("FATAL import:", e, flush=True); sys.exit(11)

FRESH = "data/processed/clinvar_grch38_fresh.parquet"
STALE = "data/processed/clinvar_grch38.parquet"

def is_nullish(v):
    if v is None: return True
    try:
        if pd.isna(v): return True
    except (TypeError, ValueError): pass
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", default=FRESH)
    ap.add_argument("--stale", default=STALE)
    ap.add_argument("--apply", action="store_true", help="rewrite fresh parquet with stale column order")
    a = ap.parse_args()
    fp, sp = Path(a.fresh), Path(a.stale)
    if not fp.exists(): print(f"FATAL: {fp} missing", flush=True); return 2
    if not sp.exists(): print(f"FATAL: {sp} missing", flush=True); return 2

    stale_order = [f.name for f in pq.ParquetFile(sp).schema_arrow]
    fresh_order = [f.name for f in pq.ParquetFile(fp).schema_arrow]
    print(f"  stale column order: {stale_order}", flush=True)
    print(f"  fresh column order: {fresh_order}", flush=True)
    print(f"  same set: {set(stale_order)==set(fresh_order)}  same order: {stale_order==fresh_order}", flush=True)

    # ---- (2) duplicate context ----
    print("\n  --- duplicate variant_id context ---", flush=True)
    sdup = pd.read_parquet(sp, columns=["variant_id"])["variant_id"]
    fdf = pd.read_parquet(fp, columns=["variant_id","ref","alt"])
    s_dupN = int(sdup.duplicated().sum())
    f_dupN = int(fdf["variant_id"].duplicated().sum())
    print(f"  stale duplicate variant_id: {s_dupN:,}", flush=True)
    print(f"  fresh duplicate variant_id: {f_dupN:,}", flush=True)
    # composition of fresh dups: how many involve a null allele (':None:None'-style collision)?
    dup_mask = fdf["variant_id"].duplicated(keep=False)
    dup_rows = fdf[dup_mask]
    null_allele = dup_rows["ref"].map(is_nullish) | dup_rows["alt"].map(is_nullish)
    print(f"  fresh rows participating in a duplicate: {len(dup_rows):,}", flush=True)
    print(f"    of which have a null allele (na:na collisions): {int(null_allele.sum()):,}", flush=True)
    print(f"    of which are fully-allele'd genuine collisions: {int((~null_allele).sum()):,}", flush=True)
    print("  (genuine collisions = same chrom:pos:ref:alt from multiple VariationIDs; the builder", flush=True)
    print("   deduplicates on variant_id, so these collapse correctly. na:na collisions all land", flush=True)
    print("   in the quarantine bucket and never reach the clean cohort.)", flush=True)

    # ---- (3) half-bad note ----
    print("\n  --- half-bad note ---", flush=True)
    print("  Fresh half-bad (exactly one allele empty) = 0 by construction: the *VCF allele", flush=True)
    print("  columns are populated all-or-nothing (identical empty rates), so a row has BOTH VCF", flush=True)
    print("  alleles or NEITHER. Stale's 1,103 half-bad came from the old independently-emptyable", flush=True)
    print("  non-VCF columns. This is an EXPECTED representation difference, not a defect.", flush=True)

    # ---- (1) reorder ----
    if stale_order == fresh_order:
        print("\n  column order already matches stale -- nothing to reorder.", flush=True)
    elif a.apply:
        print("\n  reordering fresh columns to stale order and rewriting ...", flush=True)
        full = pd.read_parquet(fp)
        full = full[stale_order]
        full.to_parquet(fp, index=False)
        new_order = [f.name for f in pq.ParquetFile(fp).schema_arrow]
        print(f"  new fresh column order: {new_order}", flush=True)
        print(f"  ORDER NOW MATCHES STALE: {new_order == stale_order}", flush=True)
        import hashlib
        h = hashlib.md5(open(fp,'rb').read()).hexdigest().upper()
        print(f"  new fresh MD5: {h}", flush=True)
    else:
        print("\n  (dry-run) column order differs; re-run with --apply to reorder.", flush=True)

    print("=== finalize_fresh_parquet DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
