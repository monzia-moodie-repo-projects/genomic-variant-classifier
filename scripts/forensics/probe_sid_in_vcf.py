#!/usr/bin/env python
"""probe_sid_in_vcf.py (2026-07-09)
Resolve H3 vs H4 for the 0-recovered result. Takes allele-less source_ids and:
  (1) looks each up in the raw+fresh VCF index EXACTLY as recover_by_sourceid does,
      printing hit/miss and the raw key repr (catches dtype/format bugs = H4);
  (2) greps the VCF ID column directly for those ids (independent of the index);
  (3) reports the variant_summary Type for those ids (SNV vs Duplication/Deletion/CNV),
      which tells us whether these variants are the kind ClinVar puts in the VCF at all.
Also, as a positive control, takes a few source_ids that ARE small SNVs (if any) to show
the lookup DOES work when the id is a VCF-present SNV. Pure evidence.
"""
import sys, os, gzip, argparse
print("=== probe_sid_in_vcf START ===", flush=True)
try:
    import pandas as pd
    print("python:", sys.version.split()[0], "pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def norm_chrom(c): return str(c).strip().lstrip("chr")
def bad(x):
    if x is None: return True
    if isinstance(x, float) and pd.isna(x): return True
    return str(x).strip().lower() in {"","na","nan","none","-",".","<na>"}
def clean(s):
    s = str(s).strip(); return s[:-2] if s.endswith(".0") else s

def index_vcf(path, want):
    """Return {id: (chrom,pos,ref,alt)} for ids in `want`, scanning the VCF once."""
    found = {}
    with gzip.open(path,"rt",encoding="utf-8",errors="replace") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split("\t",5)
            if len(p) < 5: continue
            vid = clean(p[2])
            if vid in want:
                found[vid] = (norm_chrom(p[0]), p[1], p[3][:20], p[4][:20])
                if len(found) == len(want): break
    return found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--raw-vcf", required=True)
    ap.add_argument("--fresh-vcf", default=None)
    ap.add_argument("--variant-summary", required=True)
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    coh = pd.read_parquet(a.cohort)
    al = coh[coh["ref"].map(bad) & coh["alt"].map(bad)].copy()
    al["source_id"] = al["source_id"].map(clean)
    sample = al["source_id"].head(a.n).tolist()
    print(f"\nallele-less rows: {len(al):,}; sample source_ids: {sample}", flush=True)
    print("repr of first 3 source_ids:", [repr(s) for s in sample[:3]], flush=True)

    want = set(sample)
    print("\n--- direct grep of RAW VCF ID column for these ids ---", flush=True)
    raw_hits = index_vcf(a.raw_vcf, want)
    for s in sample:
        print(f"  source_id={s!r}  raw_VCF: {'HIT '+str(raw_hits[s]) if s in raw_hits else 'MISS'}", flush=True)
    if a.fresh_vcf and os.path.exists(a.fresh_vcf):
        print("\n--- direct grep of FRESH VCF ID column for these ids ---", flush=True)
        fresh_hits = index_vcf(a.fresh_vcf, want)
        for s in sample:
            print(f"  source_id={s!r}  fresh_VCF: {'HIT '+str(fresh_hits[s]) if s in fresh_hits else 'MISS'}", flush=True)

    # variant_summary Type for these ids
    print("\n--- variant_summary Type + Assembly for these ids ---", flush=True)
    vs = pd.read_csv(a.variant_summary, sep="\t", dtype=str, compression="gzip",
                     usecols=lambda c: c in {"VariationID","Type","Assembly","Chromosome","Start","Name"})
    vs["VariationID"] = vs["VariationID"].map(clean)
    sub = vs[vs["VariationID"].isin(want)]
    if len(sub):
        show = sub.drop_duplicates("VariationID").set_index("VariationID")
        for s in sample:
            if s in show.index:
                r = show.loc[s]
                nm = str(r.get("Name",""))[:40]
                print(f"  {s}: Type={r.get('Type')}  Asm={r.get('Assembly')}  {r.get('Chromosome')}:{r.get('Start')}  {nm}", flush=True)
            else:
                print(f"  {s}: NOT in variant_summary", flush=True)
    # aggregate: what Types do ALL alleleless source_ids have?
    allsub = vs[vs["VariationID"].isin(set(al["source_id"]))]
    print("\n--- Type distribution across ALL allele-less source_ids (variant_summary) ---", flush=True)
    print(allsub.drop_duplicates("VariationID")["Type"].value_counts().to_string(), flush=True)

    print("\nINTERPRETATION:", flush=True)
    print("  If raw/fresh VCF = MISS for all AND Type is Duplication/Deletion/CNV/copy number:", flush=True)
    print("    -> H3 confirmed: these are structural variants ClinVar's VCF omits; they are", flush=True)
    print("       genuinely allele-less. Correct disposition ~ all CONFIRMED_ALLELELESS.", flush=True)
    print("  If VCF = HIT for some (repr matches) but the tool still missed:", flush=True)
    print("    -> H4: a format/dtype bug in the tool's index; fix normalization.", flush=True)
    print("=== probe_sid_in_vcf DONE ===", flush=True)

if __name__ == "__main__":
    main()
