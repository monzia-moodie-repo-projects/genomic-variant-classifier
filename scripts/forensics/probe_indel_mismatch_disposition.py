#!/usr/bin/env python
"""probe_indel_mismatch_disposition.py (2026-07-09)
Classify EACH indel genome-mismatch from build_cohort_from_source.py into a definite
disposition, so the 13 (esp. the 2 pathogenic) are resolved on evidence, not tolerated.

For every mismatch variant (clinvar:chrom:pos:ref:alt) this checks the genome at a window
of offsets around pos and reports the FIRST offset (if any) at which the reference allele
matches GRCh38. Interpretation:
  offset 0    : ref matches at pos (would mean the mismatch was spurious -- not expected here)
  offset -1   : true coordinate is pos-1 -> SOURCE OFF-BY-ONE (recoverable by a shift). For an
                insertion this means the anchor base sits at pos-1; for a deletion, the ref
                string starts at pos-1.
  offset +1   : true coordinate is pos+1 (rare; right-shift)
  none in +-5 : GENUINE ClinVar-vs-GRCh38 disagreement (variant differs from primary assembly,
                alt-locus/patch/left-alignment). Cannot be recovered by a shift; must be FLAGGED.
Also reverse-complements the ref to detect strand issues (report-only).

Pure evidence. Writes an annotated TSV to outputs/. Requires the mismatch TSV, the raw
parquet (for pathogenicity/type), and the GRCh38 FASTA.
"""
import sys, argparse
from pathlib import Path
print("=== probe_indel_mismatch_disposition START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def _norm_chrom(c):
    s = str(c); return s[3:] if s.lower().startswith("chr") else s

def _load_genome(path):
    try:
        import pysam
        fa = pysam.FastaFile(str(path))
        return (lambda c, s0, e0: fa.fetch(c, s0, e0)), set(fa.references)
    except ImportError:
        import pyfaidx
        fa = pyfaidx.Fasta(str(path))
        return (lambda c, s0, e0: str(fa[c][s0:e0])), set(fa.keys())

_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
def revcomp(s): return s.translate(_COMP)[::-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mismatches", default="outputs/clinvar_grch38_cohort_v4_indel_mismatches.tsv")
    ap.add_argument("--raw", default="data/processed/clinvar_grch38.parquet")
    ap.add_argument("--genome", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--out", default="outputs/indel_mismatch_disposition.tsv")
    a = ap.parse_args()

    mm_path = Path(a.mismatches)
    if not mm_path.exists():
        print(f"no mismatch file at {mm_path}; nothing to classify.", flush=True)
        print("=== probe_indel_mismatch_disposition DONE ===", flush=True); return 0
    mm = pd.read_csv(mm_path, sep="\t")
    print(f"mismatch rows: {len(mm):,}", flush=True)

    raw = pd.read_parquet(a.raw)
    extra = [c for c in ("pathogenicity","clinical_sig","gene_symbol") if c in raw.columns]
    ann = mm.merge(raw[["variant_id"]+extra].drop_duplicates("variant_id"),
                   on="variant_id", how="left")

    fetch, contigs = _load_genome(Path(a.genome))
    def contig_of(c):
        c=_norm_chrom(c)
        for cand in (c,f"chr{c}"):
            if cand in contigs: return cand
        return None

    def parse_vid(vid):
        # clinvar:chrom:pos:ref:alt  (ref/alt may themselves contain ':' ? no -- alleles are ACGT)
        p = str(vid).split(":")
        # p[0]=clinvar, p[1]=chrom, p[2]=pos, p[3]=ref, p[4]=alt
        return p[1], int(p[2]), p[3], p[4]

    W=a.window
    rows=[]
    print("\n--- per-variant disposition ---", flush=True)
    for _, r in ann.iterrows():
        vid=r["variant_id"]
        chrom,pos,ref,alt = parse_vid(vid)
        cc=contig_of(chrom)
        is_ins = len(alt)>len(ref)
        vtype = "insertion" if is_ins else ("deletion" if len(ref)>len(alt) else "other")
        match_off=None; rc_match_off=None
        if cc is not None:
            for off in range(-W, W+1):
                s0=pos-1+off
                if s0<0: continue
                got=fetch(cc, s0, s0+len(ref)).upper()
                if got==ref.upper():
                    match_off=off; break
            if match_off is None:
                for off in range(-W, W+1):
                    s0=pos-1+off
                    if s0<0: continue
                    got=fetch(cc, s0, s0+len(ref)).upper()
                    if got==revcomp(ref).upper():
                        rc_match_off=off; break
        if match_off==0: disp="MATCHES_AT_POS(spurious?)"
        elif match_off==-1: disp="SOURCE_OFF_BY_ONE(pos-1)"
        elif match_off==1: disp="SOURCE_OFF_BY_ONE(pos+1)"
        elif match_off is not None: disp=f"MATCH_AT_OFFSET({match_off:+d})"
        elif rc_match_off is not None: disp=f"REVCOMP_MATCH({rc_match_off:+d})"
        else: disp="GENUINE_DISAGREEMENT(no match +-%d)"%W
        patho=r.get("pathogenicity","?")
        rows.append(dict(variant_id=vid, chrom=chrom, pos=pos, type=vtype,
                         ref_head=ref[:12], alt_head=alt[:12], pathogenicity=patho,
                         match_offset=match_off, revcomp_offset=rc_match_off, disposition=disp))
        flag = "  <<< PATHOGENIC" if str(patho).lower() in ("pathogenic","likely_pathogenic") else ""
        print(f"  {chrom}:{pos} {vtype:9s} ref={ref[:10]:10s} patho={str(patho):11s} -> {disp}{flag}", flush=True)

    out=pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, sep="\t", index=False)

    print("\n--- disposition summary ---", flush=True)
    for k,v in out["disposition"].value_counts().items():
        print(f"  {k:32s} {v}", flush=True)
    npath_bad = int(((out["pathogenicity"].astype(str).str.lower().isin(["pathogenic","likely_pathogenic"]))
                     & (out["disposition"].str.startswith("GENUINE"))).sum())
    npath_shift = int(((out["pathogenicity"].astype(str).str.lower().isin(["pathogenic","likely_pathogenic"]))
                     & (out["disposition"].str.startswith("SOURCE_OFF_BY_ONE"))).sum())
    print(f"\nPATHOGENIC that are GENUINE disagreements (flag, cannot shift): {npath_bad}", flush=True)
    print(f"PATHOGENIC that are SOURCE off-by-one (recoverable by shift)   : {npath_shift}", flush=True)
    print(f"wrote {a.out}", flush=True)
    print("=== probe_indel_mismatch_disposition DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
