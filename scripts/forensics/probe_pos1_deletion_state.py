#!/usr/bin/env python
"""probe_pos1_deletion_state.py (2026-07-09)
Three indel mismatches classified SOURCE_OFF_BY_ONE(pos-1) are DELETIONS with long refs
(2:32063719, 1:231351570, 10:68199589). At least one (1:231351570 AAGGT>A) satisfies
ref.startswith(alt), so is_padded_deletion SHOULD have flagged it and the builder SHOULD
have shifted pos-=1. Yet its ref still mismatches the genome at the CORRECTED pos, matching
instead at pos-1 (i.e. pos-2 from the original Start). This probe determines, per row, the
EXACT state so we know whether the padded-deletion mask is missing a legitimate population
(a real builder finding) or these are genuine repeat/left-alignment artefacts.

For each of the 3 rows it reports, reading DIRECTLY from the raw source parquet (uncorrected
Start) and re-deriving:
  * raw pos (Start), ref, alt
  * is_padded_deletion(ref, alt) per the canonical predicate
  * genome[pos-1 : pos-1+len(ref)]         (does ref sit at the raw Start?)
  * genome[pos-2 : pos-2+len(ref)]         (does ref sit at Start-1, i.e. the corrected pos?)
  * genome[pos-3 : pos-3+len(ref)]         (Start-2)
so we can see EXACTLY which offset holds the ref and whether one uniform rule explains all 3.
Pure evidence; writes nothing but a tiny TSV to outputs/.
"""
import sys, argparse
from pathlib import Path
print("=== probe_pos1_deletion_state START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def _norm_chrom(c):
    s=str(c); return s[3:] if s.lower().startswith("chr") else s
def _load_genome(path):
    try:
        import pysam
        fa=pysam.FastaFile(str(path)); return (lambda c,s,e: fa.fetch(c,s,e)), set(fa.references)
    except ImportError:
        import pyfaidx
        fa=pyfaidx.Fasta(str(path)); return (lambda c,s,e: str(fa[c][s:e])), set(fa.keys())

# canonical padded-deletion predicate (with non-empty guard)
def is_padded_deletion(ref, alt):
    r=str(ref) if ref is not None else ""; a=str(alt) if alt is not None else ""
    if len(r)<1 or len(a)<1: return False
    return len(a)<len(r) and r.startswith(a)

TARGETS = [("2",32063719),("1",231351570),("10",68199589)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/processed/clinvar_grch38.parquet")
    ap.add_argument("--genome", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--out", default="outputs/pos1_deletion_state.tsv")
    a=ap.parse_args()

    raw=pd.read_parquet(a.raw, columns=["variant_id","chrom","pos","ref","alt"])
    raw["chrom"]=raw["chrom"].map(_norm_chrom)
    fetch,contigs=_load_genome(Path(a.genome))
    def contig_of(c):
        c=_norm_chrom(c)
        for cand in (c,f"chr{c}"):
            if cand in contigs: return cand
        return None

    rows=[]
    print("\nNOTE: raw parquet pos is the UNCORRECTED Start (before the builder's pos-=1).", flush=True)
    print("      If a row is a padded deletion, the builder's clean cohort stores pos = Start-1.\n", flush=True)
    for chrom,startpos in TARGETS:
        # the raw row may be stored at Start (uncorrected). Find rows at this chrom whose pos
        # is near the mismatch's reported (corrected) pos. The mismatch pos was the CORRECTED
        # one, so raw Start = corrected+1 for a padded deletion. Search a small window.
        cand=raw[(raw["chrom"]==chrom) & (raw["pos"].between(startpos-1, startpos+2))]
        cc=contig_of(chrom)
        for _,r in cand.iterrows():
            ref=r["ref"]; alt=r["alt"]
            if ref is None: continue
            reflen=len(str(ref))
            pad=is_padded_deletion(ref,alt)
            reads={}
            for off in (-1,-2,-3,0,1):   # genome index = pos-1+off  (off=-? relative to pos-1)
                s0=int(r["pos"])-1+off
                if s0<0: reads[off]="<neg>"; continue
                reads[off]=fetch(cc, s0, s0+reflen).upper()
            match_off=None
            for off in (-1,-2,-3,0,1):
                if reads[off]==str(ref).upper(): match_off=off; break
            print(f"chrom {chrom} rawpos {int(r['pos'])} ref={str(ref)[:14]}({reflen}bp) alt={str(alt)[:8]} "
                  f"is_padded_del={pad}", flush=True)
            print(f"    ref matches genome at pos{('%+d'%match_off) if match_off is not None else ' NOWHERE in -3..+1'}", flush=True)
            print(f"    (pos-1 slice = genome at Start; a padded-del corrected row stores Start-1)", flush=True)
            rows.append(dict(chrom=chrom, rawpos=int(r["pos"]), reflen=reflen,
                             is_padded_deletion=pad, match_offset=match_off,
                             ref_head=str(ref)[:20], alt_head=str(alt)[:12]))
    out=pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, sep="\t", index=False)
    print(f"\nINTERPRETATION KEY:", flush=True)
    print("  * is_padded_del=True AND ref matches at pos-1 (Start-1): the CANONICAL builder", flush=True)
    print("    already shifts to Start-1 and it's CORRECT -> should NOT be a mismatch. If it", flush=True)
    print("    still shows as a mismatch, the builder's genome check ran at a different pos", flush=True)
    print("    than where it stored the row -> a REAL builder inconsistency to fix.", flush=True)
    print("  * is_padded_del=True AND ref matches at pos-2 (Start-2): the row needs a DOUBLE", flush=True)
    print("    shift -> a repeat/left-alignment case the single pos-=1 rule cannot fix; FLAG it.", flush=True)
    print("  * is_padded_del=False: not a padded deletion at all (delins etc.) -> correctly not", flush=True)
    print("    shifted; the genome mismatch is genuine -> FLAG it.", flush=True)
    print(f"wrote {a.out}", flush=True)
    print("=== probe_pos1_deletion_state DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
