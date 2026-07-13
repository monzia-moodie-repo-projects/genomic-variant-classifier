#!/usr/bin/env python
"""probe_fresh_parquet_verify.py (2026-07-09)
Verify the freshly-ingested clinvar_grch38_fresh.parquet is a CLEAN, schema-identical input
before any cohort is built from it. Read-only. Checks:
  1. SCHEMA: exact column names + order vs the stale parquet (must be identical 16 cols).
  2. NORMALIZATION: ZERO literal empty-tokens ('na','NA','nan','none','.','-','') survive in
     ref/alt -- every empty must be null (None/NaN). Any survivor => normalization failed at
     scale. Reports survivor counts per token, per column.
  3. ALLELE-LESS COUNT: real na:na count (both ref+alt null) and half-bad count (exactly one
     null), so we know the fresh quarantine size empirically. Compares to stale's 19,988 + 1,103.
  4. MANIFEST echo: prints the sidecar manifest's normalization + row-count fields.
  5. DUPES/SANITY: duplicate variant_id count; a few sample allele-less variant_ids.
"""
import sys, json
from pathlib import Path
from collections import Counter
print("=== probe_fresh_parquet_verify START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

FRESH = "data/processed/clinvar_grch38_fresh.parquet"
STALE = "data/processed/clinvar_grch38.parquet"
EMPTY_TOKENS = {"", "na", "nan", "none", ".", "-", "null", "n/a"}

def is_null(v):
    if v is None: return True
    try:
        if pd.isna(v): return True
    except (TypeError, ValueError): pass
    return False

def literal_empty_survivors(series):
    """Count values that are NON-null but whose stripped-lower form is an empty token."""
    c = Counter()
    for v in series:
        if is_null(v): continue
        s = str(v).strip().lower()
        if s in EMPTY_TOKENS:
            c[repr(str(v))] += 1
    return c

def main():
    fp, sp = Path(FRESH), Path(STALE)
    if not fp.exists():
        print(f"FATAL: {FRESH} not found. Run the ingestion first.", flush=True); return 2

    print("\n--- (1) SCHEMA IDENTITY ---", flush=True)
    import pyarrow.parquet as pq
    fcols = [f.name for f in pq.ParquetFile(fp).schema_arrow]
    scols = [f.name for f in pq.ParquetFile(sp).schema_arrow] if sp.exists() else None
    print(f"  fresh cols ({len(fcols)}): {fcols}", flush=True)
    if scols is not None:
        print(f"  stale cols ({len(scols)}): {scols}", flush=True)
        print(f"  IDENTICAL order+names: {fcols == scols}", flush=True)
        print(f"  same SET: {set(fcols) == set(scols)}", flush=True)
    else:
        print("  (stale parquet not found for comparison)", flush=True)

    print("\n--- (2) NORMALIZATION: literal empty-token survivors in ref/alt (must be 0) ---", flush=True)
    df = pd.read_parquet(fp, columns=["ref", "alt", "variant_id"])
    print(f"  fresh rows: {len(df):,}", flush=True)
    for col in ("ref", "alt"):
        surv = literal_empty_survivors(df[col])
        total = sum(surv.values())
        status = "CLEAN (0 survivors)" if total == 0 else f"*** {total} SURVIVORS -- normalization FAILED"
        print(f"  {col}: {status}", flush=True)
        for tok, n in surv.most_common(10):
            print(f"      {tok}: {n:,}", flush=True)

    print("\n--- (3) ALLELE-LESS COUNTS (empirical) ---", flush=True)
    ref_null = df["ref"].map(is_null)
    alt_null = df["alt"].map(is_null)
    nana = int((ref_null & alt_null).sum())
    half = int((ref_null ^ alt_null).sum())
    print(f"  na:na (both null)      : {nana:,}   (stale had 19,988)", flush=True)
    print(f"  half-bad (exactly one) : {half:,}   (stale had 1,103)", flush=True)
    print(f"  total allele-less      : {nana + half:,}   (stale had 21,091)", flush=True)
    print(f"  clean (both present)   : {len(df) - nana - half:,}", flush=True)
    print("  sample na:na variant_ids (should end ':None:None' or ':nan:nan'):", flush=True)
    sample = df[ref_null & alt_null].head(3)
    for _, r in sample.iterrows():
        print(f"      {r['variant_id']}", flush=True)

    print("\n--- (4) DUPLICATE variant_id ---", flush=True)
    dup = int(df["variant_id"].duplicated().sum())
    print(f"  duplicate variant_id: {dup:,}", flush=True)

    print("\n--- (5) MANIFEST echo ---", flush=True)
    mpath = Path(str(fp) + ".manifest.json")
    if mpath.exists():
        man = json.loads(mpath.read_text())
        for k in ("input", "input_md5", "rows_raw", "rows_after_assembly_filter", "rows_out",
                  "ref_empty_normalized", "alt_empty_normalized", "protein_change_all_null",
                  "missing_rename_sources", "output_md5"):
            print(f"  {k}: {man.get(k)}", flush=True)
    else:
        print(f"  (no manifest at {mpath})", flush=True)

    print("\n--- VERDICT ---", flush=True)
    ref_surv = sum(literal_empty_survivors(df['ref']).values())
    alt_surv = sum(literal_empty_survivors(df['alt']).values())
    schema_ok = (scols is not None and fcols == scols)
    if ref_surv == 0 and alt_surv == 0 and schema_ok:
        print("  PASS: schema identical to stale, zero literal empty-token survivors.", flush=True)
        print("  Fresh parquet is a CLEAN input -> ready for builder --audit --genome.", flush=True)
    else:
        print("  ATTENTION: one or more checks need review before building a cohort (see above).", flush=True)
    print("=== probe_fresh_parquet_verify DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
