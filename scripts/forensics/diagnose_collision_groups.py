#!/usr/bin/env python
"""diagnose_collision_groups.py (2026-07-09) -- stratified view of variant_id collision groups."""
import sys, os, argparse
print("=== diagnose_collision_groups START ===", flush=True)
try:
    import pandas as pd
    print("python:", sys.version.split()[0], "pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def bad(x):
    if x is None: return True
    if isinstance(x, float) and pd.isna(x): return True
    return str(x).strip().lower() in {"","na","nan","none","-",".","<na>"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--recovered-by-id", default="outputs/alleleless_recovered_by_id.tsv")
    ap.add_argument("--ncbi-resolved", default="outputs/alleleless_ncbi_resolved.tsv")
    args = ap.parse_args()
    if not os.path.exists(args.cohort):
        print("FATAL cohort not found:", args.cohort, flush=True); sys.exit(12)
    coh = pd.read_parquet(args.cohort)
    al = coh[coh["ref"].map(bad) & coh["alt"].map(bad)].copy()
    al["source_id"] = al["source_id"].astype(str)
    print(f"allele-less rows: {len(al):,}  unique variant_id: {al['variant_id'].nunique():,}  unique source_id: {al['source_id'].nunique():,}", flush=True)

    vc = al["variant_id"].value_counts(); multi = vc[vc > 1]

    # Which source_ids were previously flagged RECOVERABLE (raw + ncbi)? If any live inside
    # a collision group, the re-key MATTERS for an actual allele (not just CNV re-confirm).
    rec_ids = set()
    for p, col in [(args.recovered_by_id, "cohort_varid"), (args.ncbi_resolved, "cohort_varid")]:
        if os.path.exists(p):
            try:
                d = pd.read_csv(p, sep="\t", dtype=str)
                for c in ("cohort_varid","source_id","variation_id"):
                    if c in d.columns:
                        rec_ids |= set(d[c].dropna().astype(str).str.replace(r"\.0$","",regex=True))
            except Exception as e:
                print("warn read", p, e, flush=True)
    print(f"recovered source_ids loaded for cross-check: {len(rec_ids):,}", flush=True)

    # how many collision groups contain >=1 recovered source_id?
    al["_rec"] = al["source_id"].isin(rec_ids)
    grp_has_rec = al[al["variant_id"].isin(multi.index)].groupby("variant_id")["_rec"].sum()
    n_groups_with_rec = int((grp_has_rec > 0).sum())
    n_rows_rec_in_multi = int(al[al["variant_id"].isin(multi.index)]["_rec"].sum())
    print(f"\ncollision groups containing >=1 recovered source_id: {n_groups_with_rec:,}", flush=True)
    print(f"recovered rows that sit inside a collision group      : {n_rows_rec_in_multi:,}", flush=True)
    print("  (these are the rows where keying by variant_id vs source_id changes the merge)", flush=True)

    cols = [c for c in ["variant_id","source_id","chrom","pos","gene_symbol","pathogenicity"] if c in al.columns]

    # sample SMALL groups (size 2 and 3) -- the dominant, less-examined case
    for size in (2, 3, 4):
        ids = multi[multi == size].index.tolist()
        print(f"\n===== sample of variant_id groups of size {size} (there are {len(ids):,}) =====", flush=True)
        for vid in ids[:4]:
            g = al[al["variant_id"] == vid][cols]
            print(f"\n  {vid}  ({len(g)} rows, {g['source_id'].nunique()} distinct source_id, {g['pathogenicity'].nunique()} paths):", flush=True)
            print(g.to_string(index=False, max_colwidth=42), flush=True)

    # if any recovered row is inside a collision group, SHOW those groups (highest priority)
    if n_groups_with_rec:
        print("\n===== collision groups that CONTAIN a recovered source_id (re-key matters here) =====", flush=True)
        shown = 0
        for vid in grp_has_rec[grp_has_rec > 0].index:
            g = al[al["variant_id"] == vid][cols + ["_rec"]]
            print(f"\n  {vid}  ({len(g)} rows; recovered rows marked _rec=True):", flush=True)
            print(g.to_string(index=False, max_colwidth=42), flush=True)
            shown += 1
            if shown >= 15: 
                print(f"  ... ({n_groups_with_rec - 15} more groups)", flush=True)
                break
    else:
        print("\nNO recovered source_id sits inside any collision group.", flush=True)
        print("=> re-keying will NOT change any recovered allele; collisions are ALL in the", flush=True)
        print("   CONFIRMED_ALLELELESS (CNV/structural) set. Re-key still needed for correct", flush=True)
        print("   per-row EXCLUSION accounting, but no recovered allele was ever mis-merged.", flush=True)
    print("=== diagnose_collision_groups DONE ===", flush=True)

if __name__ == "__main__":
    main()
