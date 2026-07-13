#!/usr/bin/env python
"""characterize_flagged_rows.py (2026-07-09)
Full disposition of EVERY genome-inconsistent row (ref_genome_consistent == False) and every
unchecked row (== <NA>) found by build_cohort_from_source.py's per-row flag. The audit
reported 53 False + 2 <NA> on the stale snapshot; the indel-only G8 guard only saw 13 of
them, so 40 are non-indel (SNV/MNV/delins) disagreements plus the 2 unchecked. This reader
re-derives the flag on the CLEAN cohort rows (reproducing the builder's exact per-row genome
check) so every flagged row is enumerated with:
  variant_id, chrom, pos, ref, alt, variant_class, pathogenicity, genome_at_pos, reason
Reasons:
  INDEL_MISMATCH       ref len != alt len and ref != genome@pos (the 13 already studied)
  SUBSTITUTION_MISMATCH ref len == alt len and ref != genome@pos (SNV/MNV disagreement)
  UNCHECKED_CONTIG     contig absent from the FASTA (the <NA> rows) -- a coverage gap
Special attention: lists ALL pathogenic/likely_pathogenic flagged rows individually.
Writes outputs/flagged_rows_disposition.tsv. Read-only on the cohort.

NOTE: runs on the CLEAN cohort produced by an --apply build. If you have not yet --apply'd,
this script rebuilds the clean frame in-memory from the raw source using the SAME builder,
so it needs the raw parquet + genome. Pass --cohort to read an already-written cohort instead.
"""
import sys, argparse
from pathlib import Path
print("=== characterize_flagged_rows START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

def _norm_chrom(c):
    s=str(c); return s[3:] if s.lower().startswith("chr") else s
def _load_genome(path):
    try:
        import pysam
        fa=pysam.FastaFile(str(path)); return (lambda c,s,e: fa.fetch(c,s,e)), set(fa.references)
    except ImportError:
        import pyfaidx
        fa=pyfaidx.Fasta(str(path)); return (lambda c,s,e: str(fa[c][s:e])), set(fa.keys())

def vclass(ref, alt):
    r=str(ref); a=str(alt)
    if len(r)==1 and len(a)==1: return "SNV"
    if len(r)==len(a): return "MNV"
    if len(r)>len(a): return "deletion"
    return "insertion"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None,
                    help="an already-written cohort parquet (with ref_genome_consistent). If omitted, rebuild from --raw.")
    ap.add_argument("--raw", default="data/processed/clinvar_grch38.parquet")
    ap.add_argument("--genome", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--out", default="outputs/flagged_rows_disposition.tsv")
    a=ap.parse_args()

    fetch, contigs = _load_genome(Path(a.genome))
    def contig_of(c):
        c=_norm_chrom(c)
        for cand in (c,f"chr{c}"):
            if cand in contigs: return cand
        return None

    if a.cohort and Path(a.cohort).exists():
        df = pd.read_parquet(a.cohort)
        print(f"read cohort {a.cohort}: {len(df):,} rows", flush=True)
        if "ref_genome_consistent" in df.columns:
            flagged = df[df["ref_genome_consistent"] == False]     # noqa: E712
            unchecked = df[df["ref_genome_consistent"].isna()]
        else:
            print("cohort lacks ref_genome_consistent; recomputing per-row.", flush=True)
            flagged = unchecked = None
    else:
        # rebuild clean frame from raw using the real builder
        import build_cohort_from_source as B
        raw = pd.read_parquet(a.raw)
        recon = B.BuildReconciliation()
        df, _ = B.build(raw, recon)
        print(f"rebuilt clean cohort from raw: {len(df):,} rows", flush=True)
        flagged = unchecked = None

    # (Re)compute the flag if we don't already have the split
    if flagged is None:
        rows_false=[]; rows_na=[]
        for _, r in df.iterrows():
            cc=contig_of(str(r["chrom"]))
            ref=str(r["ref"])
            if cc is None:
                rows_na.append(r); continue
            got=fetch(cc, int(r["pos"])-1, int(r["pos"])-1+len(ref)).upper()
            if got != ref.upper():
                rr=r.copy(); rr["_genome_at_pos"]=got; rows_false.append(rr)
        flagged=pd.DataFrame(rows_false); unchecked=pd.DataFrame(rows_na)

    print(f"\nflagged (False): {len(flagged):,}    unchecked (<NA>): {len(unchecked):,}", flush=True)

    def enrich(fr, reason_default):
        out=[]
        for _, r in fr.iterrows():
            cc=contig_of(str(r["chrom"]))
            ref=str(r["ref"]); alt=str(r["alt"])
            vc=vclass(ref, alt)
            if cc is None:
                gat="<contig absent>"; reason="UNCHECKED_CONTIG"
            else:
                gat=r.get("_genome_at_pos") or fetch(cc,int(r["pos"])-1,int(r["pos"])-1+len(ref)).upper()
                reason = "INDEL_MISMATCH" if len(ref)!=len(alt) else "SUBSTITUTION_MISMATCH"
            out.append(dict(variant_id=r["variant_id"], chrom=r["chrom"], pos=int(r["pos"]),
                            ref=ref[:30], alt=alt[:30], variant_class=vc,
                            pathogenicity=r.get("pathogenicity","?"),
                            genome_at_pos=str(gat)[:30], reason=reason))
        return pd.DataFrame(out)

    fdf=enrich(flagged, "MISMATCH")
    udf=enrich(unchecked, "UNCHECKED_CONTIG") if len(unchecked) else pd.DataFrame()
    allrows=pd.concat([fdf,udf], ignore_index=True) if len(udf) else fdf

    print("\n--- reason summary ---", flush=True)
    for k,v in allrows["reason"].value_counts().items():
        print(f"  {k:24s} {v}", flush=True)
    print("\n--- variant_class summary ---", flush=True)
    for k,v in allrows["variant_class"].value_counts().items():
        print(f"  {k:12s} {v}", flush=True)
    print("\n--- pathogenicity summary ---", flush=True)
    for k,v in allrows["pathogenicity"].astype(str).value_counts().items():
        print(f"  {k:16s} {v}", flush=True)

    patho = allrows[allrows["pathogenicity"].astype(str).str.lower().isin(["pathogenic","likely_pathogenic"])]
    print(f"\n--- ALL {len(patho)} PATHOGENIC/LIKELY_PATHOGENIC flagged rows (each documented) ---", flush=True)
    for _, r in patho.iterrows():
        print(f"  {r['variant_id'][:60]:60s} {r['variant_class']:9s} ref={r['ref'][:12]:12s} "
              f"genome={r['genome_at_pos'][:12]:12s} {r['reason']}", flush=True)

    if len(udf):
        print(f"\n--- {len(udf)} UNCHECKED (<NA>) rows -- contig absent from FASTA ---", flush=True)
        for _, r in udf.iterrows():
            print(f"  {r['variant_id'][:60]:60s} chrom={r['chrom']} -- contig not in genome", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    allrows.to_csv(a.out, sep="\t", index=False)
    print(f"\nwrote {a.out} ({len(allrows)} rows)", flush=True)
    print("=== characterize_flagged_rows DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
