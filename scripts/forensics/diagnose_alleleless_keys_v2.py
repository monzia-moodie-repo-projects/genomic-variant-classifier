#!/usr/bin/env python
"""diagnose_alleleless_keys_v2.py (2026-07-09) -- paste-proof, self-announcing."""
import sys, argparse
print("=== diagnose_alleleless_keys_v2 START ===", flush=True)
print("python:", sys.version.split()[0], flush=True)
try:
    import pandas as pd
    print("pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL: pandas import failed:", e, flush=True); sys.exit(11)

def norm_chrom(c): return str(c).strip().lstrip("chr")

NULL = {"", "na", "nan", "none", "-", ".", "<na>", "None"}
def is_alleleless_series(ref, alt):
    def bad(x):
        return x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip().lower() in {"","na","nan","none","-",".","<na>"}
    return ref.map(bad) & alt.map(bad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--variant-summary", default=None)
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--out", default="outputs/alleleless_key_diagnosis.tsv")
    a = ap.parse_args()
    import os
    if not os.path.exists(a.cohort):
        print("FATAL: cohort not found:", a.cohort, flush=True); sys.exit(12)
    print("reading cohort ...", flush=True)
    coh = pd.read_parquet(a.cohort)
    print("cohort rows:", f"{len(coh):,}", "cols:", list(coh.columns), flush=True)
    al = coh[is_alleleless_series(coh["ref"], coh["alt"])].copy()
    n_rows = len(al); n_ids = al["variant_id"].nunique()
    print("", flush=True)
    print(f"allele-less rows           : {n_rows:,}", flush=True)
    print(f"unique variant_ids         : {n_ids:,}", flush=True)
    print(f"duplicate-keyed rows (gap) : {n_rows - n_ids:,}", flush=True)

    vc = al["variant_id"].value_counts(); multi = vc[vc > 1]
    print(f"\nvariant_ids with 2+ rows   : {len(multi):,}", flush=True)
    print(f"  rows in those groups     : {int(multi.sum()):,}", flush=True)
    print("  group-size distribution  :", flush=True)
    print(multi.value_counts().sort_index().to_string(), flush=True)

    print("\nper-allele-less-row uniqueness of each column:", flush=True)
    for col in al.columns:
        try:
            u = al[col].nunique(dropna=False)
            flag = "  <-- UNIQUE per row" if u == n_rows else ""
            print(f"  {col:26s}: {u:,} unique{flag}", flush=True)
        except Exception as e:
            print(f"  {col:26s}: (err {e})", flush=True)

    # exact-duplicate rows (all columns identical) vs same-id-different-content
    dup_all = int(al.duplicated(keep=False).sum())
    print(f"\nrows that are EXACT duplicates (all cols identical): {dup_all:,}", flush=True)

    if len(multi):
        cols = [c for c in ["variant_id","chrom","pos","ref","alt","gene_symbol","pathogenicity"] if c in al.columns]
        print("\n--- up to 8 colliding groups (are they distinct variants?) ---", flush=True)
        for vid in list(multi.index[:8]):
            grp = al[al["variant_id"] == vid][cols]
            npath = grp["pathogenicity"].nunique() if "pathogenicity" in grp else "?"
            print(f"\n  {vid}  ({len(grp)} rows, {npath} distinct pathogenicity):", flush=True)
            print(grp.to_string(index=False, max_colwidth=36), flush=True)

    if a.variant_summary and __import__("os").path.exists(a.variant_summary) and len(multi):
        print("\nreading variant_summary for VariationID-at-locus check ...", flush=True)
        vs = pd.read_csv(a.variant_summary, sep="\t", dtype=str, compression="gzip",
                         usecols=lambda c: c in {"VariationID","Assembly","Chromosome","Start"})
        if "Assembly" in vs.columns: vs = vs[vs["Assembly"].isin([a.assembly,"na"])]
        at = {}
        for c,s,vid in zip(vs["Chromosome"].map(norm_chrom), vs["Start"].astype(str), vs["VariationID"]):
            at.setdefault((c,s), set()).add(vid)
        print("--- distinct VariationIDs at colliding loci ---", flush=True)
        for vid in list(multi.index[:5]):
            r0 = al[al["variant_id"] == vid].iloc[0]
            ids_here = at.get((norm_chrom(r0["chrom"]), str(int(r0["pos"]))), set())
            print(f"  {vid}: {len(al[al['variant_id']==vid])} cohort rows; {len(ids_here)} distinct VariationIDs at locus", flush=True)

    try:
        __import__("os").makedirs(__import__("os").path.dirname(a.out) or ".", exist_ok=True)
        al.to_csv(a.out, sep="\t", index=False)
        print(f"\nfull allele-less dump -> {a.out}", flush=True)
    except Exception as e:
        print("WARN: could not write dump:", e, flush=True)
    print("=== diagnose_alleleless_keys_v2 DONE ===", flush=True)

if __name__ == "__main__":
    main()
