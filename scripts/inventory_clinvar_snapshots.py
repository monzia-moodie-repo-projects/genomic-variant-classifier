#!/usr/bin/env python
"""inventory_clinvar_snapshots.py (2026-07-09)
Read-only disk inventory to determine, WITHOUT assumption, whether a FRESH processed
ClinVar (Clinical Variant database) parquet exists in the canonical cohort schema -- the
prerequisite for the build-both-and-diff plan. Lists:
  1. every *.parquet under data/processed and data/external whose name mentions clinvar,
     with row count, column count, column names, and pos convention hint (min/max pos).
  2. the raw fresh ClinVar files at data/external/clinvar (VCF + variant_summary) with size
     and modified date, to confirm the fresh snapshot is raw (VCF/TSV) not processed parquet.
  3. an explicit VERDICT: case A (fresh processed parquet present -> build both now) or case
     B (only raw fresh present -> must ingest to parquet first).
Writes nothing. Pure inventory.
"""
import sys, glob, os
from pathlib import Path
from datetime import datetime
print("=== inventory_clinvar_snapshots START ===", flush=True)
try:
    import pandas as pd
    import pyarrow.parquet as pq
except Exception as e:
    print("FATAL import:", e, flush=True); sys.exit(11)

def human(n):
    for u in ("B","KB","MB","GB"):
        if n < 1024: return f"{n:.1f}{u}"
        n/=1024
    return f"{n:.1f}TB"

def main():
    roots = ["data/processed", "data/external", "data/raw"]
    print("\n--- (1) ClinVar-related PARQUET files (schema + rowcount) ---", flush=True)
    found_parquets = []
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)):
            base = os.path.basename(p).lower()
            if "clinvar" not in base and "cohort" not in base:
                continue
            try:
                pf = pq.ParquetFile(p)
                nrows = pf.metadata.num_rows
                cols = [f.name for f in pf.schema_arrow]
                # pos convention hint: read just pos min/max cheaply
                pos_hint = ""
                if "pos" in cols:
                    dfp = pd.read_parquet(p, columns=["pos"])
                    pos_hint = f" pos[min={int(dfp['pos'].min())},max={int(dfp['pos'].max())}]"
                mtime = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d")
                sz = human(os.path.getsize(p))
                print(f"  {p}", flush=True)
                print(f"      rows={nrows:,} cols={len(cols)} size={sz} modified={mtime}{pos_hint}", flush=True)
                print(f"      columns: {cols}", flush=True)
                found_parquets.append((p, nrows, tuple(cols), mtime))
            except Exception as e:
                print(f"  {p}  <error reading: {e}>", flush=True)

    print("\n--- (2) RAW fresh ClinVar source files ---", flush=True)
    raw_globs = ["data/external/clinvar/*", "data/raw/clinvar/*"]
    raw_found = []
    for g in raw_globs:
        for p in sorted(glob.glob(g)):
            if os.path.isdir(p): continue
            mtime = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d")
            sz = human(os.path.getsize(p))
            print(f"  {p}  size={sz} modified={mtime}", flush=True)
            raw_found.append((p, mtime))

    print("\n--- (3) VERDICT ---", flush=True)
    # canonical stale parquet
    stale = "data/processed/clinvar_grch38.parquet"
    stale_cols = None
    for p, nrows, cols, mt in found_parquets:
        if p.replace("\\","/").endswith("data/processed/clinvar_grch38.parquet"):
            stale_cols = cols
            print(f"  canonical STALE input: {p} ({nrows:,} rows, modified {mt})", flush=True)
    # any OTHER processed parquet in the SAME schema that is NOT the stale one and is a full cohort
    fresh_candidates = []
    for p, nrows, cols, mt in found_parquets:
        pp = p.replace("\\","/")
        if pp.endswith("data/processed/clinvar_grch38.parquet"): continue
        if "structural" in pp or "clean" in pp or "seq" in pp or "conflict" in pp: continue
        # a fresh full snapshot would have ~same schema + millions of rows
        if stale_cols and cols == stale_cols and nrows > 1_000_000:
            fresh_candidates.append((p, nrows, mt))
    if fresh_candidates:
        print("  CASE A -- a fresh processed parquet in the canonical schema EXISTS:", flush=True)
        for p, nrows, mt in fresh_candidates:
            print(f"      {p} ({nrows:,} rows, modified {mt}) -> can build both + diff NOW.", flush=True)
    else:
        print("  CASE B -- NO fresh processed parquet in the canonical schema found.", flush=True)
        print("      The fresh snapshot exists only as raw VCF/variant_summary (see section 2).", flush=True)
        print("      To build-both-and-diff, the fresh raw must FIRST be ingested into a parquet", flush=True)
        print("      in the IDENTICAL schema as clinvar_grch38.parquet (same ingestion step that", flush=True)
        print("      produced the stale parquet). That ingestion is a separate, verified stage.", flush=True)
    print("=== inventory_clinvar_snapshots DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
