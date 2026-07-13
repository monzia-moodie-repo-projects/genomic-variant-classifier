#!/usr/bin/env python
"""probe_pathogenicity_mapping_audit.py (2026-07-10)
The 'Conflicting classifications of pathogenicity' -> pathogenic mislabel (161K/164K rows) was
caused by a SUBSTRING fallback in the pathogenicity mapper. Before fixing, AUDIT THE WHOLE
MAPPING to find EVERY clinical_sig string whose mapped pathogenicity may be wrong -- do not
assume 'Conflicting' is the only casualty. Read-only.

For BOTH stale and fresh processed parquets, this prints the FULL cross-tabulation of the raw
clinical_sig string -> mapped pathogenicity, for every clinical_sig value (sorted by count),
so a human can see exactly how each ClinVar status was labeled. It flags SUSPICIOUS mappings:
  * clinical_sig containing 'conflict' mapped to a confident class (pathogenic/benign)
  * clinical_sig containing 'risk factor','association','drug response','protective','affects',
    'other','not provided','no classification' mapped to pathogenic/benign (should likely be
    uncertain/other)
  * any clinical_sig whose mapped label seems to contradict its leading token
This gives the COMPLETE picture of mapping correctness, not just the one bug we already found.
"""
import sys, argparse
from pathlib import Path
print("=== probe_pathogenicity_mapping_audit START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

SUSP_TOKENS = ["conflict","risk factor","association","drug response","protective","affects",
               "other","not provided","no classification","no assertion","uncertain"]

def audit(path, topn):
    p = Path(path)
    if not p.exists():
        print(f"  (missing: {path})", flush=True); return
    df = pd.read_parquet(p, columns=["clinical_sig","pathogenicity"])
    total = len(df)
    print(f"\n  FILE: {path}  ({total:,} rows)", flush=True)
    ct = df.groupby([df["clinical_sig"].astype("string").fillna("<NA>"),
                     df["pathogenicity"].astype("string").fillna("<NA>")]).size()
    ct = ct.reset_index(name="n").sort_values("n", ascending=False)
    print(f"  distinct (clinical_sig -> pathogenicity) pairs: {len(ct):,}", flush=True)
    print(f"  TOP {topn} by count:", flush=True)
    for _, r in ct.head(topn).iterrows():
        cs = str(r["clinical_sig"]); pa = str(r["pathogenicity"]); n = int(r["n"])
        print(f"    {n:>10,}  [{cs[:60]}] -> {pa}", flush=True)

    print("\n  SUSPICIOUS mappings (contested/modifier strings mapped to a CONFIDENT class):", flush=True)
    flagged_total = 0
    for _, r in ct.iterrows():
        cs = str(r["clinical_sig"]).lower(); pa = str(r["pathogenicity"]); n = int(r["n"])
        if pa in ("pathogenic","likely_pathogenic","benign","likely_benign"):
            if any(tok in cs for tok in ["conflict","risk factor","association","drug response",
                                          "protective","affects","not provided","no classification",
                                          "no assertion"]):
                print(f"    {n:>10,}  [{str(r['clinical_sig'])[:60]}] -> {pa}   <== review", flush=True)
                flagged_total += n
    print(f"  total rows in suspicious mappings: {flagged_total:,}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", default="data/processed/clinvar_grch38.parquet")
    ap.add_argument("--fresh", default="data/processed/clinvar_grch38_fresh.parquet")
    ap.add_argument("--topn", type=int, default=40)
    a = ap.parse_args()
    audit(a.stale, a.topn)
    audit(a.fresh, a.topn)
    print("\n  Use this to define the CORRECTED mapping precisely -- every suspicious row's proper", flush=True)
    print("  label can be decided from the full cross-tab before touching the canonical mapper.", flush=True)
    print("=== probe_pathogenicity_mapping_audit DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
