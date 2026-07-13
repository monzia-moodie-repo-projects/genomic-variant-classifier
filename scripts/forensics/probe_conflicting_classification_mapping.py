#!/usr/bin/env python
"""probe_conflicting_classification_mapping.py (2026-07-10)
The duplicate-collapse design surfaced a possible UPSTREAM labeling issue: ClinVar's aggregate
status "Conflicting classifications of pathogenicity" may be mapped to pathogenicity='pathogenic'
by the ingestion's prefix-mapper (because the string contains 'pathogenic'). If so, that is a
latent label bug affecting potentially MANY rows project-wide, not just the TTN duplicate.

This probe QUANTIFIES it on BOTH the stale and fresh processed parquets (read-only). For each,
it reports:
  * how many rows have clinical_sig containing 'Conflicting classifications' (case-insensitive)
  * the pathogenicity these rows were mapped to (value_counts) -- exposing whether 'Conflicting'
    became 'pathogenic'
  * a few example rows
  * the same for 'Conflicting classifications of pathogenicity and ...' variants and any other
    clinical_sig strings that contain 'pathogenic' but are NOT a clean pathogenic call (e.g.
    'Uncertain significance', 'no classification', 'not provided' won't match; but 'Likely
    pathogenic, low penetrance' etc. will be surfaced).
This tells us the SCOPE of any mis-mapping before we decide whether/how to fix it -- a decision
that is SEPARATE from the dedup and higher-value.
"""
import sys, argparse
from pathlib import Path
from collections import Counter
print("=== probe_conflicting_classification_mapping START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def analyze(path):
    p = Path(path)
    if not p.exists():
        print(f"  (missing: {path})", flush=True); return
    df = pd.read_parquet(p, columns=["clinical_sig","pathogenicity"])
    cs = df["clinical_sig"].astype("string").fillna("")
    total = len(df)
    print(f"\n  FILE: {path}  ({total:,} rows)", flush=True)

    # rows whose clinical_sig mentions 'conflicting'
    conf = cs.str.contains("conflicting", case=False, na=False)
    print(f"  clinical_sig contains 'conflicting': {int(conf.sum()):,} rows", flush=True)
    if int(conf.sum()):
        vc = df[conf]["pathogenicity"].astype("string").fillna("<NA>").value_counts()
        print("    -> mapped pathogenicity distribution:", flush=True)
        for k,v in vc.items():
            print(f"         {k:20s} {v:,}", flush=True)
        # show the distinct clinical_sig strings that mention conflicting
        distinct = df[conf]["clinical_sig"].astype("string").value_counts().head(8)
        print("    -> distinct 'conflicting' clinical_sig strings (top 8):", flush=True)
        for k,v in distinct.items():
            print(f"         [{k}]  x{v:,}", flush=True)

    # broader: any clinical_sig containing 'pathogenic' that mapped to pathogenicity='pathogenic'
    # but is NOT simply 'Pathogenic'/'Likely pathogenic' -- i.e. suspicious mappings
    has_patho_word = cs.str.contains("pathogenic", case=False, na=False)
    mapped_patho = df["pathogenicity"].astype("string") == "pathogenic"
    suspicious = has_patho_word & mapped_patho & conf  # conflicting + mapped pathogenic
    print(f"  'conflicting' AND mapped to 'pathogenic': {int(suspicious.sum()):,} rows "
          f"<== these are the potential mis-maps", flush=True)

    # For calibration: how many total mapped to each pathogenicity
    print("  overall pathogenicity distribution:", flush=True)
    for k,v in df["pathogenicity"].astype("string").fillna("<NA>").value_counts().items():
        print(f"       {k:20s} {v:,} ({100*v/total:.2f}%)", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", default="data/processed/clinvar_grch38.parquet")
    ap.add_argument("--fresh", default="data/processed/clinvar_grch38_fresh.parquet")
    a = ap.parse_args()
    analyze(a.stale)
    analyze(a.fresh)
    print("\n  INTERPRETATION:", flush=True)
    print("  If a large count of 'Conflicting classifications of pathogenicity' rows are mapped to", flush=True)
    print("  pathogenicity='pathogenic', the prefix-mapper is over-calling them. 'Conflicting' is", flush=True)
    print("  ClinVar's way of saying submitters DISAGREE -- semantically closer to 'uncertain'.", flush=True)
    print("  This is a SEPARATE fix from the dedup (it changes labels for ALL such rows in BOTH", flush=True)
    print("  snapshots) and should be decided explicitly. The dedup merely surfaced it via TTN.", flush=True)
    print("=== probe_conflicting_classification_mapping DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
