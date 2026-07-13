#!/usr/bin/env python
"""probe_nana_provenance.py (2026-07-09)
Decide WHERE the 19,988 na:na rows entered. clean_cohort.py provably routes na:na rows to
the structural table and fail-loud-asserts none remain in its clean output. This probe
checks the ACTUAL artifacts to confirm that and locate the real source:
  * clinvar_grch38_clean.parquet         (clean_cohort output)   -> expect 0 na:na
  * clinvar_grch38_structural.parquet    (clean_cohort output)   -> expect the na:na here
  * clinvar_grch38_clean_v2_verified.parquet (canonical/v2)      -> has 19,988 na:na
  * clinvar_grch38.parquet / _source, if present                 -> upstream source
For each, report row count, na:na count, and whether variant_id/source_id columns exist.
If clean.parquet has 0 na:na but v2 has 19,988, the v2 builder (NOT clean_cohort) is the
source. Pure evidence; writes nothing.
"""
import sys, os, argparse
print("=== probe_nana_provenance START ===", flush=True)
try:
    import pandas as pd
    print("python:", sys.version.split()[0], "pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

NULL = {"", "na", "nan", "none", ".", "null", "-", "<na>"}
def is_na(x):
    if x is None: return True
    if isinstance(x, float) and pd.isna(x): return True
    return str(x).strip().lower() in NULL

def nana_count(df):
    if "ref" not in df.columns or "alt" not in df.columns:
        return None
    return int((df["ref"].map(is_na) & df["alt"].map(is_na)).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    a = ap.parse_args()
    d = a.processed_dir
    targets = [
        "clinvar_grch38_clean.parquet",
        "clinvar_grch38_structural.parquet",
        "clinvar_grch38_conflicts.parquet",
        "clinvar_grch38_clean_v2_verified.parquet",
        "clinvar_grch38_clean_v3_verified.parquet",
        "clinvar_grch38.parquet",
    ]
    print(f"\nscanning {d}", flush=True)
    for name in targets:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            print(f"  {name:44s}: (absent)", flush=True); continue
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"  {name:44s}: READ ERROR {e}", flush=True); continue
        nn = nana_count(df)
        cols = []
        for c in ("variant_id","source_id","ref","alt"):
            if c in df.columns: cols.append(c)
        print(f"  {name:44s}: rows={len(df):>10,}  na:na={nn}  has={cols}", flush=True)

    print("\nINTERPRETATION:", flush=True)
    print("  clean.parquet na:na == 0 AND structural.parquet na:na > 0", flush=True)
    print("    -> clean_cohort.py routing is CORRECT; na:na are quarantined there.", flush=True)
    print("  clean_v2_verified na:na == 19,988 while clean.parquet na:na == 0", flush=True)
    print("    -> the v2 builder (NOT clean_cohort.py) introduced them; fix belongs there.", flush=True)
    print("  If clean.parquet ALSO has 19,988 na:na -> clean_cohort DID emit them, contradicting", flush=True)
    print("    its post-condition -> would mean the file predates the guard; investigate.", flush=True)
    print("=== probe_nana_provenance DONE ===", flush=True)

if __name__ == "__main__":
    main()
