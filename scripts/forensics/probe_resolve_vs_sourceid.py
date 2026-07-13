#!/usr/bin/env python
"""probe_resolve_vs_sourceid.py (2026-07-09)
Decisive test: does the recovery's resolved cohort_varid EQUAL the cohort row's own
source_id? Focus on the 301 singleton rows (exactly one source_id at their variant_id):
if source_id == cohort_varid there, the recovery IS keyed on the true id and its allele is
trustworthy; if not, resolve_varid returned a DIFFERENT variant and the allele may be wrong
even for 'clean' rows. Also verifies the audit's comparison isn't a dtype artifact by
showing raw repr() of both keys. Pure evidence.
"""
import sys, os, argparse
print("=== probe_resolve_vs_sourceid START ===", flush=True)
try:
    import pandas as pd
    print("python:", sys.version.split()[0], "pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def bad(x):
    if x is None: return True
    if isinstance(x, float) and pd.isna(x): return True
    return str(x).strip().lower() in {"","na","nan","none","-",".","<na>"}
def clean(s):
    s = str(s).strip(); return s[:-2] if s.endswith(".0") else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--recovered-by-id", default="outputs/alleleless_recovered_by_id.tsv")
    a = ap.parse_args()
    coh = pd.read_parquet(a.cohort)
    al = coh[coh["ref"].map(bad) & coh["alt"].map(bad)].copy()
    al["source_id"] = al["source_id"].astype(str).map(clean)
    al["variant_id"] = al["variant_id"].astype(str)
    sids_by_vid = al.groupby("variant_id")["source_id"].apply(lambda s: sorted(set(s)))

    rec = pd.read_csv(a.recovered_by_id, sep="\t", dtype=str)
    rec["variant_id"] = rec["variant_id"].astype(str)
    rec["cohort_varid"] = rec["cohort_varid"].map(clean)

    # DTYPE SANITY: show raw values on a few rows
    print("\n--- dtype sanity: repr of keys on first 5 recovered rows ---", flush=True)
    for _, r in rec.head(5).iterrows():
        vid = r["variant_id"]; cv = r["cohort_varid"]; sset = sids_by_vid.get(vid, [])
        print(f"  cohort_varid={cv!r}  sids_at_vid={[s for s in sset[:5]]!r}  cv_in={'Y' if cv in sset else 'N'}", flush=True)

    # SINGLETON test: rows whose variant_id has exactly one source_id
    rec["sids_at_vid"] = rec["variant_id"].map(lambda v: sids_by_vid.get(v, []))
    rec["n_sids"] = rec["sids_at_vid"].map(len)
    singles = rec[rec["n_sids"] == 1].copy()
    singles["the_sid"] = singles["sids_at_vid"].map(lambda l: l[0] if l else None)
    singles["cv_eq_sid"] = singles["cohort_varid"] == singles["the_sid"]
    n_single = len(singles); n_eq = int(singles["cv_eq_sid"].sum())
    print(f"\nSINGLETON recovered rows (n_sids==1): {n_single}", flush=True)
    print(f"  of those, cohort_varid == the single source_id: {n_eq}", flush=True)
    print(f"  of those, cohort_varid != source_id           : {n_single - n_eq}", flush=True)

    if n_single - n_eq > 0:
        print("\n  --- singleton rows where resolved varid != source_id (up to 15) ---", flush=True)
        bad_s = singles[~singles["cv_eq_sid"]].head(15)
        print(bad_s[["variant_id","cohort_varid","the_sid","rec_ref","rec_alt","verdict"]].to_string(index=False, max_colwidth=30), flush=True)
    else:
        print("\n  ALL singleton rows have cohort_varid == source_id.", flush=True)
        print("  => resolve_varid IS returning the row's own VariationID for clean rows;", flush=True)
        print("     the earlier '544 not-a-source_id' was because sids_at_vid only holds the", flush=True)
        print("     ALLELE-LESS source_ids, but the resolved varid may be a NON-alleleless", flush=True)
        print("     variant at the same locus (the real SNV). Alleles are then trustworthy for", flush=True)
        print("     singletons and the re-key is a clean per-row source_id merge.", flush=True)

    # For a few MULTI rows, show whether cohort_varid is among sids or is an outside (SNV) id
    multi = rec[rec["n_sids"] > 1].head(10)
    print("\n--- multi rows: is resolved varid one of the alleleless sids, or an outside id? ---", flush=True)
    for _, r in multi.iterrows():
        cv = r["cohort_varid"]; sset = r["sids_at_vid"]
        print(f"  vid={r['variant_id']} cv={cv} in_alleleless_sids={'Y' if cv in sset else 'N'} "
              f"n_sids={r['n_sids']} allele={r['rec_ref']}>{r['rec_alt']}", flush=True)

    print("=== probe_resolve_vs_sourceid DONE ===", flush=True)

if __name__ == "__main__":
    main()
