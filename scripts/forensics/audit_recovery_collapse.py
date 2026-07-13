#!/usr/bin/env python
"""audit_recovery_collapse.py (2026-07-09) -- measure identity-ambiguity of recovered rows."""
import sys, os, argparse
print("=== audit_recovery_collapse START ===", flush=True)
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
    s = str(s); return s[:-2] if s.endswith(".0") else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--recovered-by-id", default="outputs/alleleless_recovered_by_id.tsv")
    ap.add_argument("--out", default="outputs/recovery_collapse_audit.tsv")
    a = ap.parse_args()
    for p in (a.cohort, a.recovered_by_id):
        if not os.path.exists(p): print("FATAL missing:", p, flush=True); sys.exit(12)

    coh = pd.read_parquet(a.cohort)
    al = coh[coh["ref"].map(bad) & coh["alt"].map(bad)].copy()
    al["source_id"] = al["source_id"].astype(str).map(clean)
    # distinct source_ids per variant_id (the collision group membership)
    sids_by_vid = al.groupby("variant_id")["source_id"].apply(lambda s: sorted(set(s)))
    npath_by_vid = al.groupby("variant_id")["pathogenicity"].nunique()

    rec = pd.read_csv(a.recovered_by_id, sep="\t", dtype=str)
    rec["cohort_varid"] = rec["cohort_varid"].map(clean)
    print(f"\nrecovered-by-id rows              : {len(rec):,}", flush=True)
    print(f"  unique variant_id               : {rec['variant_id'].nunique():,}", flush=True)
    print(f"  unique cohort_varid (resolved)  : {rec['cohort_varid'].nunique():,}", flush=True)
    print(f"  duplicate variant_id rows        : {int(rec['variant_id'].duplicated().sum()):,}", flush=True)

    # For each recovered row: how many distinct source_ids live at its variant_id?
    rec["n_sids_at_vid"] = rec["variant_id"].map(lambda v: len(sids_by_vid.get(v, [])))
    rec["sids_at_vid"] = rec["variant_id"].map(lambda v: ";".join(sids_by_vid.get(v, [])[:6]))
    rec["npath_at_vid"] = rec["variant_id"].map(lambda v: int(npath_by_vid.get(v, 0)))
    # is the resolved cohort_varid actually one of the source_ids at that vid?
    rec["resolved_is_real_sid"] = rec.apply(
        lambda r: r["cohort_varid"] in sids_by_vid.get(r["variant_id"], []), axis=1)
    # AMBIGUOUS = the recovered allele cannot be uniquely attributed to one source_id
    rec["ambiguous_identity"] = rec["n_sids_at_vid"] > 1

    n_amb = int(rec["ambiguous_identity"].sum())
    n_badresolve = int((~rec["resolved_is_real_sid"]).sum())
    print(f"\nrecovered rows on a MULTI-source_id variant_id (identity ambiguous): {n_amb:,}", flush=True)
    print(f"recovered rows whose resolved cohort_varid is NOT a source_id at that vid: {n_badresolve:,}", flush=True)
    print(f"recovered rows with clean 1:1 identity (n_sids==1)                 : {int((rec['n_sids_at_vid']==1).sum()):,}", flush=True)

    print("\ndistribution: distinct source_ids at each recovered row's variant_id:", flush=True)
    print(rec["n_sids_at_vid"].value_counts().sort_index().to_string(), flush=True)

    print("\n--- recovered rows with AMBIGUOUS identity (multi-source_id vid), up to 25 ---", flush=True)
    sub = rec[rec["ambiguous_identity"]].head(25)
    if len(sub):
        print(sub[["variant_id","cohort_varid","n_sids_at_vid","npath_at_vid","sids_at_vid",
                   "resolved_is_real_sid","rec_ref","rec_alt"]].to_string(index=False, max_colwidth=34), flush=True)
    else:
        print("  (none -- every recovered row maps to a unique source_id)", flush=True)

    rec.to_csv(a.out, sep="\t", index=False)
    print(f"\naudit -> {a.out}", flush=True)
    print("\nINTERPRETATION:", flush=True)
    print("  n_sids==1 rows: recovery is unambiguous -- the allele belongs to the sole source_id.", flush=True)
    print("  n_sids>1 rows : the recovered allele was fetched for the locus-resolved varid, but", flush=True)
    print("    the physical row could be ANY of the co-located source_ids; on merge-by-variant_id", flush=True)
    print("    the allele would splatter across all of them. These MUST be re-keyed per source_id.", flush=True)
    print("=== audit_recovery_collapse DONE ===", flush=True)

if __name__ == "__main__":
    main()
