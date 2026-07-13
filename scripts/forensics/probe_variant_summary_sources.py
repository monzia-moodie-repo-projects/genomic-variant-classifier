#!/usr/bin/env python
"""probe_variant_summary_sources.py (2026-07-09)
Read-only preflight before building the fresh ingestion. Answers three questions with zero
assumptions:
  (1) WHICH variant_summary.txt.gz files exist on disk (stale? fresh?) -- so we know whether an
      exact stale-reproduction gate is even possible.
  (2) For each, read ONLY the header line (first row) and report the exact column names, so we
      can confirm the columns ClinVarConnector.fetch() relies on are present and detect any NCBI
      schema drift between snapshots:
        required by connector: Assembly, Chromosome, Start, ReferenceAllele, AlternateAllele,
                               GeneSymbol, ClinicalSignificance, ProteinChange, VariationID,
                               'RS# (dbSNP)', ReviewStatus
  (3) Peek the first few DATA rows of the fresh file to sanity-check Assembly values + that Start
      is an integer-like position (not PositionVCF). Also report file sizes/mtimes.
Reads only the header + a tiny sample (nrows=5) -- does NOT load the 400MB file.
"""
import sys, glob, os, gzip
from pathlib import Path
from datetime import datetime
print("=== probe_variant_summary_sources START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

REQUIRED = ["Assembly","Chromosome","Start","ReferenceAllele","AlternateAllele",
            "GeneSymbol","ClinicalSignificance","ProteinChange","VariationID",
            "RS# (dbSNP)","ReviewStatus"]

def human(n):
    for u in ("B","KB","MB","GB"):
        if n<1024: return f"{n:.1f}{u}"
        n/=1024
    return f"{n:.1f}TB"

def find_files():
    pats = ["data/**/variant_summary*.txt.gz", "data/**/variant_summary*.txt",
            "data/**/*variant_summary*"]
    seen=set(); out=[]
    for p in pats:
        for f in glob.glob(p, recursive=True):
            if os.path.isfile(f) and f not in seen:
                seen.add(f); out.append(f)
    return sorted(out)

def header_of(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        first = f.readline().rstrip("\n")
    return [c.lstrip("#") for c in first.split("\t")]

def main():
    print("\n--- (1) variant_summary files on disk ---", flush=True)
    files = find_files()
    if not files:
        print("  NONE FOUND. (No variant_summary anywhere under data/.)", flush=True)
    for f in files:
        mt = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d")
        print(f"  {f}  size={human(os.path.getsize(f))}  modified={mt}", flush=True)

    print("\n--- (2) header column check per file ---", flush=True)
    for f in files:
        try:
            cols = header_of(f)
            missing = [c for c in REQUIRED if c not in cols]
            print(f"  {f}", flush=True)
            print(f"      {len(cols)} columns", flush=True)
            if missing:
                print(f"      *** MISSING required columns: {missing}", flush=True)
            else:
                print(f"      all {len(REQUIRED)} connector-required columns present.", flush=True)
            # show whether PositionVCF also exists (drift indicator)
            extra = [c for c in ("PositionVCF","ReferenceAlleleVCF","AlternateAlleleVCF") if c in cols]
            if extra:
                print(f"      (also has VCF-style cols: {extra} -- connector ignores these, uses Start/ReferenceAllele)", flush=True)
        except Exception as e:
            print(f"  {f}  <header read error: {e}>", flush=True)

    print("\n--- (3) fresh-file data sanity (first 5 GRCh38 rows) ---", flush=True)
    fresh = [f for f in files if "external" in f.replace("\\","/")]
    target = fresh[0] if fresh else (files[0] if files else None)
    if target:
        print(f"  sampling: {target}", flush=True)
        try:
            op = gzip.open if str(target).endswith(".gz") else open
            with op(target, "rt", encoding="utf-8", errors="replace") as f:
                df = pd.read_csv(f, sep="\t", low_memory=False, nrows=2000)
            g = df[df["Assembly"]=="GRCh38"] if "Assembly" in df.columns else df
            print(f"      read 2000 rows; Assembly=='GRCh38' in first 2000: {len(g)}", flush=True)
            if "Assembly" in df.columns:
                print(f"      Assembly value counts (first 2000): {dict(df['Assembly'].value_counts().head(6))}", flush=True)
            cols_show = [c for c in ("Chromosome","Start","ReferenceAllele","AlternateAllele","VariationID") if c in df.columns]
            print(f"      sample rows [{cols_show}]:", flush=True)
            for _, r in g.head(3).iterrows():
                print("        " + " | ".join(f"{c}={r[c]}" for c in cols_show), flush=True)
        except Exception as e:
            print(f"      <sample read error: {e}>", flush=True)
    else:
        print("  no file to sample.", flush=True)
    print("\n=== probe_variant_summary_sources DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
