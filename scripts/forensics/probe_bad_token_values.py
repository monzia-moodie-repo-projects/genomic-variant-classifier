#!/usr/bin/env python
"""probe_bad_token_values.py (2026-07-09)
Verify the EXACT values of bad-allele tokens in (1) the real STALE processed parquet and
(2) the FRESH raw variant_summary, so the fresh ingestion reproduces token handling exactly
instead of relying on an assumption. A prior turn assumed pandas default-NaN turns ClinVar
'na' -> None; that assumption FAILED a unit test (lowercase 'na' is NOT a pandas default NA
token). This probe settles what the tokens really are.

Reports:
 (A) STALE parquet (data/processed/clinvar_grch38.parquet): among rows where ref or alt is
     'bad', the distinct repr() of ref and alt values + counts. Specifically distinguishes
     Python None vs the literal strings 'na','nan','.','','-'. Uses .isna() AND explicit
     string comparison so we see EXACTLY what is stored.
 (B) FRESH raw variant_summary: read a modest sample (nrows) of GRCh38 rows the SAME way the
     connector does (pd.read_csv default NaN, no dtype=str), then report the distinct repr()
     of ReferenceAllele/AlternateAllele values that are empty-ish, so we can see whether the
     connector-style read yields None or literal 'na' on the FRESH data.
This tells us if stale and fresh encode empty alleles identically. Read-only.
"""
import sys, gzip, argparse
from pathlib import Path
from collections import Counter
print("=== probe_bad_token_values START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def classify(v):
    if v is None: return "PY_None"
    try:
        if pd.isna(v): return f"NaN({type(v).__name__})"
    except Exception: pass
    s = str(v)
    if s in ("na","NA","nan","NaN","N/A",".","","-","null","NULL"): return f"str:{s!r}"
    if len(s) <= 3 and not s.isalpha(): return f"str:{s!r}"
    return "VALID_ALLELE"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", default="data/processed/clinvar_grch38.parquet")
    ap.add_argument("--fresh", default="data/external/clinvar/variant_summary.txt.gz")
    ap.add_argument("--nrows", type=int, default=300000)
    a = ap.parse_args()

    print("\n--- (A) STALE parquet bad-token values ---", flush=True)
    df = pd.read_parquet(a.stale, columns=["ref","alt"])
    print(f"  total rows: {len(df):,}", flush=True)
    # a row is 'bad' if ref or alt is null OR a known junk token
    def is_bad(v):
        if v is None: return True
        try:
            if pd.isna(v): return True
        except Exception: pass
        return str(v) in ("na","NA","nan","NaN","N/A",".","","-","null","NULL")
    ref_bad = df["ref"].map(is_bad); alt_bad = df["alt"].map(is_bad)
    bad = df[ref_bad | alt_bad]
    print(f"  rows with bad ref or alt: {len(bad):,}", flush=True)
    print(f"  distinct ref classifications among bad rows:", flush=True)
    for k,v in Counter(bad["ref"].map(classify)).most_common():
        print(f"      {k:20s} {v:,}", flush=True)
    print(f"  distinct alt classifications among bad rows:", flush=True)
    for k,v in Counter(bad["alt"].map(classify)).most_common():
        print(f"      {k:20s} {v:,}", flush=True)
    # show a few raw reprs
    print("  sample raw (repr) of first 5 bad rows:", flush=True)
    for _, r in bad.head(5).iterrows():
        print(f"      ref={r['ref']!r}  alt={r['alt']!r}", flush=True)

    print("\n--- (B) FRESH raw, connector-style read (default NaN, no dtype) ---", flush=True)
    op = gzip.open if str(a.fresh).endswith(".gz") else open
    with op(a.fresh, "rt", encoding="utf-8", errors="replace") as f:
        fresh = pd.read_csv(f, sep="\t", low_memory=False, nrows=a.nrows)
    fresh = fresh[fresh["Assembly"]=="GRCh38"]
    print(f"  GRCh38 rows in first {a.nrows:,}: {len(fresh):,}", flush=True)
    rb = fresh["ReferenceAllele"].map(is_bad); ab = fresh["AlternateAllele"].map(is_bad)
    fbad = fresh[rb | ab]
    print(f"  bad-allele rows: {len(fbad):,}", flush=True)
    print(f"  distinct ReferenceAllele classifications among bad:", flush=True)
    for k,v in Counter(fbad["ReferenceAllele"].map(classify)).most_common():
        print(f"      {k:20s} {v:,}", flush=True)
    print(f"  distinct AlternateAllele classifications among bad:", flush=True)
    for k,v in Counter(fbad["AlternateAllele"].map(classify)).most_common():
        print(f"      {k:20s} {v:,}", flush=True)
    print("  sample raw (repr) of first 5 fresh bad rows:", flush=True)
    for _, r in fbad.head(5).iterrows():
        print(f"      ref={r['ReferenceAllele']!r}  alt={r['AlternateAllele']!r}", flush=True)

    print("\n--- RECONCILE ---", flush=True)
    print("  Compare (A) vs (B): do STALE and FRESH encode empty alleles the SAME way?", flush=True)
    print("  If STALE shows PY_None but FRESH (connector-style) shows str:'na', then the stale", flush=True)
    print("  parquet's None did NOT come from the connector's read alone -> there was another", flush=True)
    print("  step, OR the 2026-03 ClinVar used a different token. Either way the fresh ingestion", flush=True)
    print("  must be made to reproduce the stale encoding EXACTLY (map 'na'->None if needed).", flush=True)
    print("=== probe_bad_token_values DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
