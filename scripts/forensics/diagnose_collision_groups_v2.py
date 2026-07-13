#!/usr/bin/env python
"""diagnose_collision_groups_v2.py (2026-07-09) -- CORRECT recovered-set scoping by verdict."""
import sys, os, argparse
print("=== diagnose_collision_groups_v2 START ===", flush=True)
try:
    import pandas as pd
    print("python:", sys.version.split()[0], "pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def bad(x):
    if x is None: return True
    if isinstance(x, float) and pd.isna(x): return True
    return str(x).strip().lower() in {"","na","nan","none","-",".","<na>"}

REC_VERDICTS = {"RECOVER_BY_ID_RAW","RECOVER_BY_ID_FRESH","REPEAT_RECOVER_BY_ID"}

def _clean(s):
    s = str(s)
    return s[:-2] if s.endswith(".0") else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--recovered-by-id", default="outputs/alleleless_recovered_by_id.tsv")
    ap.add_argument("--disposition", default="outputs/alleleless_identity_recovery_full.tsv")
    ap.add_argument("--ncbi-resolved", default="outputs/alleleless_ncbi_resolved.tsv")
    args = ap.parse_args()
    if not os.path.exists(args.cohort):
        print("FATAL cohort not found:", args.cohort, flush=True); sys.exit(12)
    coh = pd.read_parquet(args.cohort)
    al = coh[coh["ref"].map(bad) & coh["alt"].map(bad)].copy()
    al["source_id"] = al["source_id"].astype(str)

    # --- build the TRUE recovered source_id set, filtered by verdict ---
    rec_ids = set()
    # from recovered-by-id (already only recovered rows), take source_id if present else cohort_varid
    if os.path.exists(args.recovered_by_id):
        d = pd.read_csv(args.recovered_by_id, sep="\t", dtype=str)
        # this file is ALREADY only RECOVER rows; use cohort_varid (== source_id/VariationID)
        key = "cohort_varid" if "cohort_varid" in d.columns else ("source_id" if "source_id" in d.columns else None)
        if key:
            rec_ids |= set(d[key].dropna().map(_clean))
        print(f"recovered-by-id rows: {len(d):,}  ids from '{key}': {len(rec_ids):,}", flush=True)
    # cross-check against disposition verdicts (authoritative), if available
    if os.path.exists(args.disposition):
        disp = pd.read_csv(args.disposition, sep="\t", dtype=str)
        if "verdict" in disp.columns:
            recv = disp[disp["verdict"].isin(REC_VERDICTS)]
            col = "cohort_varid" if "cohort_varid" in recv.columns else None
            disp_ids = set(recv[col].dropna().map(_clean)) if col else set()
            print(f"disposition RECOVER-verdict rows: {len(recv):,}  ids: {len(disp_ids):,}", flush=True)
            # prefer the intersection/union as appropriate; report both
            print(f"  ids in recovered-by-id but NOT disp-verdict: {len(rec_ids - disp_ids):,}", flush=True)
            print(f"  ids in disp-verdict but NOT recovered-by-id: {len(disp_ids - rec_ids):,}", flush=True)
            rec_ids |= disp_ids
    # ncbi resolved (only RESOLVED_HAS_ALLELE)
    if os.path.exists(args.ncbi_resolved):
        nc = pd.read_csv(args.ncbi_resolved, sep="\t", dtype=str)
        vcol = "ncbi_verdict" if "ncbi_verdict" in nc.columns else None
        if vcol:
            nrec = nc[nc[vcol] == "RESOLVED_HAS_ALLELE"]
            for c in ("source_id","cohort_varid","variation_id"):
                if c in nrec.columns:
                    rec_ids |= set(nrec[c].dropna().map(_clean)); break
            print(f"ncbi RESOLVED_HAS_ALLELE rows: {len(nrec):,}", flush=True)

    print(f"\nTRUE recovered source_id set size: {len(rec_ids):,}  (expected ~547)", flush=True)

    vc = al["variant_id"].value_counts(); multi = vc[vc > 1]
    al["_rec"] = al["source_id"].isin(rec_ids)
    in_multi = al[al["variant_id"].isin(multi.index)]
    grp_has_rec = in_multi.groupby("variant_id")["_rec"].sum()
    n_groups = int((grp_has_rec > 0).sum())
    n_rows = int(in_multi["_rec"].sum())
    print(f"\nrecovered source_ids that sit in a collision group: {n_rows:,}", flush=True)
    print(f"collision groups containing a recovered source_id : {n_groups:,}", flush=True)
    print("  (THIS is the true blast radius of the old merge-by-variant_id bug)", flush=True)

    total_rec_in_al = int(al["_rec"].sum())
    print(f"\nrecovered source_ids present in allele-less set at all: {total_rec_in_al:,}", flush=True)
    print(f"  -> of those, {n_rows:,} are in a collision group, "
          f"{total_rec_in_al - n_rows:,} are in singleton variant_ids (safe either key)", flush=True)

    cols = [c for c in ["variant_id","source_id","chrom","pos","gene_symbol","pathogenicity","clinical_sig"] if c in al.columns]
    if n_groups:
        print("\n--- collision groups containing a TRUE recovered variant (up to 25) ---", flush=True)
        shown = 0
        for vid in grp_has_rec[grp_has_rec > 0].index:
            g = al[al["variant_id"] == vid][cols + ["_rec"]]
            print(f"\n  {vid} ({len(g)} rows, {int(g['_rec'].sum())} recovered):", flush=True)
            print(g.to_string(index=False, max_colwidth=32), flush=True)
            shown += 1
            if shown >= 25:
                print(f"  ... ({n_groups - 25} more)", flush=True); break
    else:
        print("\nZERO truly-recovered variants sit in a collision group.", flush=True)
        print("=> the old merge-by-variant_id never mis-assigned a RECOVERED allele;", flush=True)
        print("   collisions are entirely within the excluded CONFIRMED_ALLELELESS set.", flush=True)
    print("=== diagnose_collision_groups_v2 DONE ===", flush=True)

if __name__ == "__main__":
    main()
