#!/usr/bin/env python
"""diagnose_alleleless_keys_v3.py (2026-07-09) -- paste-proof, self-announcing, key-focused."""
import sys, os, argparse
print("=== diagnose_alleleless_keys_v3 START ===", flush=True)
print("python:", sys.version.split()[0], flush=True)
try:
    import pandas as pd
    print("pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL: pandas import failed:", e, flush=True); sys.exit(11)

def norm_chrom(c): return str(c).strip().lstrip("chr")
def bad(x):
    if x is None: return True
    if isinstance(x, float) and pd.isna(x): return True
    return str(x).strip().lower() in {"","na","nan","none","-",".","<na>"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--variant-summary", default=None)
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--out", default="outputs/alleleless_key_diagnosis.tsv")
    a = ap.parse_args()
    if not os.path.exists(a.cohort):
        print("FATAL: cohort not found:", a.cohort, flush=True); sys.exit(12)
    print("reading cohort ...", flush=True)
    coh = pd.read_parquet(a.cohort)
    al = coh[coh["ref"].map(bad) & coh["alt"].map(bad)].copy()
    n_rows = len(al)
    print(f"\nallele-less rows           : {n_rows:,}", flush=True)
    print(f"unique variant_id          : {al['variant_id'].nunique():,}", flush=True)
    print(f"unique source_id           : {al['source_id'].nunique():,}", flush=True)

    # --- source_id characterization ---
    print("\n--- source_id semantics ---", flush=True)
    print("dtype:", al["source_id"].dtype, flush=True)
    print("sample values:", al["source_id"].head(8).tolist(), flush=True)
    sid_dup = al["source_id"].value_counts()
    sid_multi = sid_dup[sid_dup > 1]
    print(f"source_id values with 2+ rows: {len(sid_multi)} "
          f"(covering {int(sid_multi.sum())} rows)", flush=True)
    if len(sid_multi):
        cols = [c for c in ["variant_id","source_id","chrom","pos","ref","alt","gene_symbol","pathogenicity","clinical_sig"] if c in al.columns]
        print("the non-unique source_id rows (examine for true dup vs distinct):", flush=True)
        for sid in list(sid_multi.index[:15]):
            g = al[al["source_id"] == sid][cols]
            print(f"\n  source_id={sid} ({len(g)} rows):", flush=True)
            print(g.to_string(index=False, max_colwidth=30), flush=True)

    # --- exact-duplicate rows (hashable cols only) ---
    hashable = [c for c in al.columns if c not in ("metadata",)]
    try:
        dup_all = int(al[hashable].duplicated(keep=False).sum())
        print(f"\nEXACT-duplicate rows (hashable cols): {dup_all:,}", flush=True)
    except Exception as e:
        print("dup check skipped:", e, flush=True)

    # --- colliding variant_id groups: are they distinct by source_id / pathogenicity? ---
    vc = al["variant_id"].value_counts(); multi = vc[vc > 1]
    print(f"\nvariant_id collision groups: {len(multi):,} covering {int(multi.sum()):,} rows", flush=True)
    cols = [c for c in ["variant_id","source_id","chrom","pos","gene_symbol","pathogenicity","clinical_sig"] if c in al.columns]
    print("--- up to 6 sample groups (distinct source_id within same variant_id?) ---", flush=True)
    for vid in list(multi.index[:6]):
        g = al[al["variant_id"] == vid][cols]
        print(f"\n  {vid}  ({len(g)} rows, {g['source_id'].nunique()} distinct source_id, "
              f"{g['pathogenicity'].nunique() if 'pathogenicity' in g else '?'} distinct pathogenicity):", flush=True)
        print(g.to_string(index=False, max_colwidth=28), flush=True)

    # --- optional variant_summary cross-check ---
    if a.variant_summary and os.path.exists(a.variant_summary) and len(multi):
        print("\nreading variant_summary ...", flush=True)
        vs = pd.read_csv(a.variant_summary, sep="\t", dtype=str, compression="gzip",
                         usecols=lambda c: c in {"VariationID","Assembly","Chromosome","Start"})
        if "Assembly" in vs.columns: vs = vs[vs["Assembly"].isin([a.assembly,"na"])]
        at = {}
        for c,s,vid in zip(vs["Chromosome"].map(norm_chrom), vs["Start"].astype(str), vs["VariationID"]):
            at.setdefault((c,s), set()).add(vid)
        # does source_id match a VariationID at that locus?
        sample = al.head(10)
        print("--- does source_id equal a VariationID at the row's locus? (first 10) ---", flush=True)
        for _, r in sample.iterrows():
            ids_here = at.get((norm_chrom(r["chrom"]), str(int(r["pos"]))), set())
            hit = str(r["source_id"]) in ids_here
            print(f"  source_id={r['source_id']} at {r['chrom']}:{r['pos']} -> in variant_summary VariationIDs at locus: {hit} (locus has {len(ids_here)} ids)", flush=True)

    try:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        al.drop(columns=["metadata"], errors="ignore").to_csv(a.out, sep="\t", index=False)
        print(f"\ndump -> {a.out}", flush=True)
    except Exception as e:
        print("WARN dump:", e, flush=True)
    print("=== diagnose_alleleless_keys_v3 DONE ===", flush=True)

if __name__ == "__main__":
    main()
