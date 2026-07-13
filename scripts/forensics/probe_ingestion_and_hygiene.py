#!/usr/bin/env python
"""probe_ingestion_and_hygiene.py (2026-07-09)
Three read-only diagnostics ahead of the fresh-snapshot ingestion:
  (1) INGESTION LOCATOR: scan scripts/ + src/ for the code that WRITES clinvar_grch38.parquet
      (the raw VCF + variant_summary -> processed parquet stage). Reports every file that
      references that exact output path or parses variant_summary, with line numbers, so we
      can read the true ingestion and re-run it on the fresh raw in the SAME schema.
  (2) CONFLICTS FILE DIAGNOSIS: clinvar_grch38_conflicts.parquet fails to read with
      'cannot convert float NaN to integer'. Read it defensively (no dtype coercion) and
      report rowcount, schema, and any NaN columns -- so we know if it is empty-with-bad-dtype
      or has real NaN-pos rows (a corruption to fix).
  (3) V3 DUPLICATE CHECK: compare data/processed/clinvar_grch38_clean_v3_verified.parquet
      (live) against data/processed/_invalidated_2026-07-09/clinvar_grch38_clean_v3_verified
      .parquet (invalidated) by MD5 -- identical? different? -> know which v3 is canonical.
Writes nothing. Pure evidence.
"""
import sys, glob, os, hashlib
from pathlib import Path
print("=== probe_ingestion_and_hygiene START ===", flush=True)
try:
    import pandas as pd
    import pyarrow.parquet as pq
except Exception as e:
    print("FATAL import:", e, flush=True); sys.exit(11)

def md5(p):
    h=hashlib.md5()
    with open(p,"rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest().upper()

def sec1_ingestion_locator():
    print("\n--- (1) INGESTION LOCATOR: who writes clinvar_grch38.parquet? ---", flush=True)
    pats = ["clinvar_grch38.parquet", "variant_summary", "ReferenceAlleleVCF",
            "AlternateAlleleVCF", "PositionVCF", "to_parquet"]
    hits = {}
    for root in ("scripts", "src"):
        for f in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
            try:
                txt = open(f, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            # a true ingestion writes clinvar_grch38.parquet (not _clean, not _v2)
            for i, line in enumerate(txt.splitlines(), 1):
                if "clinvar_grch38.parquet" in line and "to_parquet" in line:
                    hits.setdefault(f, []).append((i, "WRITES clinvar_grch38.parquet", line.strip()[:100]))
                elif "clinvar_grch38.parquet" in line and (".parquet'" in line or '.parquet"' in line) and ("out" in line.lower() or "write" in line.lower() or "=" in line):
                    hits.setdefault(f, []).append((i, "refs output path", line.strip()[:100]))
                elif "variant_summary" in line and ("read" in line.lower() or "open" in line.lower() or "gzip" in line.lower() or "parse" in line.lower()):
                    hits.setdefault(f, []).append((i, "reads variant_summary", line.strip()[:100]))
    if not hits:
        print("  NO python file both parses variant_summary AND writes clinvar_grch38.parquet.", flush=True)
        print("  The ingestion may be a notebook, a one-off, or an external script. Widen search:", flush=True)
        # fallback: any file mentioning the output name at all
        for root in ("scripts", "src"):
            for f in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
                try: txt=open(f,encoding="utf-8",errors="replace").read()
                except Exception: continue
                if "clinvar_grch38.parquet" in txt:
                    ln=[i for i,l in enumerate(txt.splitlines(),1) if "clinvar_grch38.parquet" in l]
                    print(f"    {f}: mentions output at lines {ln[:8]}", flush=True)
    else:
        for f, rows in sorted(hits.items()):
            print(f"  {f}", flush=True)
            for ln, kind, txt in rows[:12]:
                print(f"      L{ln} [{kind}] {txt}", flush=True)

def sec2_conflicts():
    print("\n--- (2) CONFLICTS FILE DIAGNOSIS ---", flush=True)
    p = "data/processed/clinvar_grch38_conflicts.parquet"
    if not os.path.exists(p):
        print(f"  {p} does not exist.", flush=True); return
    try:
        meta = pq.ParquetFile(p).metadata
        print(f"  metadata: {meta.num_rows} rows, {meta.num_columns} cols", flush=True)
    except Exception as e:
        print(f"  metadata read error: {e}", flush=True)
    try:
        df = pd.read_parquet(p)
        print(f"  full read OK: {len(df)} rows, cols={list(df.columns)}", flush=True)
        if len(df):
            for c in df.columns:
                n_na = int(df[c].isna().sum())
                if n_na: print(f"    column '{c}': {n_na} NaN", flush=True)
    except Exception as e:
        print(f"  full read FAILED: {e}", flush=True)
        # try column-by-column with pyarrow to find the offender
        try:
            import pyarrow.parquet as pq2
            t = pq2.read_table(p)
            print(f"  pyarrow read OK: {t.num_rows} rows. pandas coercion is the issue, not the file.", flush=True)
            print(f"    schema: {[f.name+':'+str(f.type) for f in t.schema]}", flush=True)
        except Exception as e2:
            print(f"  pyarrow read ALSO failed: {e2} -- file may be genuinely corrupt.", flush=True)

def sec3_v3dup():
    print("\n--- (3) V3 DUPLICATE CHECK ---", flush=True)
    live = "data/processed/clinvar_grch38_clean_v3_verified.parquet"
    inv  = "data/processed/_invalidated_2026-07-09/clinvar_grch38_clean_v3_verified.parquet"
    for label, p in (("live", live), ("invalidated", inv)):
        if os.path.exists(p):
            print(f"  {label}: {p}  MD5={md5(p)}  rows={pq.ParquetFile(p).metadata.num_rows:,}", flush=True)
        else:
            print(f"  {label}: {p} NOT FOUND", flush=True)
    if os.path.exists(live) and os.path.exists(inv):
        same = md5(live)==md5(inv)
        print(f"  -> identical: {same}  ({'same bytes, safe' if same else 'DIFFERENT -- must determine which is canonical 5871AE9C'})", flush=True)

def main():
    sec1_ingestion_locator()
    sec2_conflicts()
    sec3_v3dup()
    print("\n=== probe_ingestion_and_hygiene DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
