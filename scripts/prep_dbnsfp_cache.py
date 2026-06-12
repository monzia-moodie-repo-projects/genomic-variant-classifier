#!/usr/bin/env python3
"""prep_dbnsfp_cache.py -- verify the small ClinVar dbNSFP index is a safe drop-in
for the DbNSFPConnector cache, so Run 16 gets real SIFT/PolyPhen/REVEL/CADD scores
WITHOUT the 85.3M-row full-index OOM, and WITHOUT a silent all-default deadzone.

Why this exists: DbNSFPConnector hardcodes its cache filename to
``dbnsfp_full_index.parquet`` and (for a genome-wide cohort) loads it whole via
pd.read_parquet -> OOM on the 895 MB / 85.3M-row file. The 34.57 MB ClinVar index
is the same 10-col schema; promoting it to the cache name is memory-safe -- IF its
schema matches AND its chrom-value format matches the cohort's normalised chroms
(the connector filters the cache with [("chrom","in", cohort_chroms)]; a format
mismatch silently yields zero hits = every variant gets default scores).

STRICTLY READ-ONLY. Footer schema + one row-group sample only; never full-loads.
Prints a SAFE / UNSAFE verdict and the exact (manual) promote commands.

Usage:  python scripts/prep_dbnsfp_cache.py
Author: Monzia Moodie."""
from __future__ import annotations

import sys
from pathlib import Path

DBNSFP_DIR = Path("data/external/dbnsfp")
CLINVAR_IDX = DBNSFP_DIR / "dbnsfp_clinvar_index.parquet"
FULL_IDX = DBNSFP_DIR / "dbnsfp_full_index.parquet"
COHORT = Path("data/processed/clinvar_grch38_clean_seq.parquet")

# Columns the connector's merge requires (chrom/pos/ref/alt + the 6 score outputs).
REQUIRED = {"chrom", "pos", "ref", "alt",
            "sift_score", "polyphen2_score", "revel_score",
            "cadd_phred", "phylop_score", "gerp_score"}


def _norm(c: str) -> str:
    """Mirror the connector's chrom normalisation closely enough to compare formats:
    strip a leading 'chr', upper-case. (The connector strips 'chr' and keeps 1..22/X/Y/M.)"""
    c = str(c).strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    return c.upper()


def _schema_cols(p: Path):
    import pyarrow.parquet as pq
    return list(pq.read_schema(p).names)


def _sample_chroms(p: Path, limit_rows: int = 200_000):
    """Distinct chrom values from the first row group(s) only -- bounded read."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(p)
    seen, taken = set(), 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["chrom"])
        for v in tbl.column(0).to_pylist():
            if v is not None:
                seen.add(str(v))
        taken += tbl.num_rows
        if taken >= limit_rows:
            break
    return seen


def main() -> int:
    ok = True
    print("=" * 74)
    print(" dbNSFP cache prep / verify (read-only)")
    print("=" * 74)

    for p, label in [(CLINVAR_IDX, "ClinVar index"), (FULL_IDX, "FULL index"), (COHORT, "cohort")]:
        print(f"  {label:<14}: {'FOUND ' + str(p) if p.exists() else 'MISSING ' + str(p)}"
              + (f"  ({round(p.stat().st_size/1048576,2)} MB)" if p.exists() else ""))
    if not CLINVAR_IDX.exists():
        print("\nFAIL: ClinVar index not found -- cannot promote. Stop.")
        return 2
    if not COHORT.exists():
        print("\nWARN: cohort not found at expected path; chrom-format check skipped.")

    # 1) schema completeness + (if present) match against the full index
    cv_cols = _schema_cols(CLINVAR_IDX)
    print(f"\n[1] ClinVar index schema: {len(cv_cols)} cols")
    missing = REQUIRED - set(cv_cols)
    if missing:
        print(f"    FAIL: missing connector-required columns: {sorted(missing)}")
        ok = False
    else:
        print(f"    PASS: all {len(REQUIRED)} connector-required columns present")
    if FULL_IDX.exists():
        full_cols = _schema_cols(FULL_IDX)
        if set(cv_cols) == set(full_cols):
            print(f"    PASS: ClinVar index schema == full index schema (drop-in compatible)")
        else:
            print(f"    WARN: schema differs from full index "
                  f"(clinvar-only: {sorted(set(cv_cols)-set(full_cols))}, "
                  f"full-only: {sorted(set(full_cols)-set(cv_cols))})")

    # 2) chrom-format match (the silent-deadzone guard)
    if COHORT.exists():
        idx_raw = _sample_chroms(CLINVAR_IDX)          # literal values stored in the index
        coh_raw = _sample_chroms(COHORT)               # literal values in the cohort
        coh_norm = {_norm(c) for c in coh_raw}         # what the connector passes to the filter
        # The connector applies [("chrom","in", coh_norm)] to the index's RAW chrom column,
        # so the rows that survive are exactly idx_raw & coh_norm. Empty => deadzone.
        effective = idx_raw & coh_norm
        raw_overlap = idx_raw & coh_raw
        print(f"\n[2] chrom-format check (sampled; mirrors the pushdown filter)")
        print(f"    index raw chroms:        {sorted(idx_raw)[:8]}")
        print(f"    cohort raw chroms:       {sorted(coh_raw)[:8]}")
        print(f"    cohort normalised:       {sorted(coh_norm)[:8]}  <- filter values")
        print(f"    survivors (idx_raw & cohort_norm): {len(effective)} chroms")
        if not effective and not raw_overlap:
            print("    FAIL: index raw chroms match NEITHER the cohort's normalised NOR raw "
                  "chroms -> pushdown returns 0 rows = every variant gets DEFAULT scores "
                  "(silent deadzone). Re-key the index chrom column before promoting.")
            ok = False
        elif not effective and raw_overlap:
            print("    WARN: index matches the cohort's RAW chroms but not its NORMALISED "
                  "values; whether rows survive depends on the connector's exact "
                  "_normalise_chrom. Treat as unverified -> confirm dbNSFP n_hit>0 in the "
                  "re-smoke before trusting the scores.")
        else:
            print(f"    PASS: {len(effective)} cohort chroms survive the filter against the index.")

    print("\n" + "=" * 74)
    if ok:
        print(" VERDICT: SAFE TO PROMOTE. Run these THREE commands manually:")
        print("   # 1. quarantine the 85.3M-row full index (OOM landmine)")
        print(r"   Move-Item data\external\dbnsfp\dbnsfp_full_index.parquet data\external\dbnsfp\dbnsfp_full_index.parquet.OOMbak -Force")
        print("   # 2. promote the ClinVar index to the connector's hardcoded cache name")
        print(r"   Copy-Item data\external\dbnsfp\dbnsfp_clinvar_index.parquet data\external\dbnsfp\dbnsfp_full_index.parquet -Force")
        print("   # 3. then pass:  --dbnsfp-path data\\external\\dbnsfp\\dbnsfp_full_index.parquet")
        print(" Confirm in the re-smoke: dbNSFP n_hit > 0 and sift_score not all == 0.5.")
        return 0
    print(" VERDICT: UNSAFE -- resolve the FAIL line(s) above before promoting.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
