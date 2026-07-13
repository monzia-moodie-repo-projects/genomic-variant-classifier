#!/usr/bin/env python
"""probe_snv_alleleless.py (2026-07-09)
Isolate the allele-less rows whose ClinVar Type is 'single nucleotide variant' (the only
rows that SHOULD have a simple allele) and fully characterise each: cohort locus, the
variant_summary record (Type, Assembly, Name, Start), whether the source_id is in the raw
or fresh VCF by ID, and if so the VCF allele + genome check. This decides, per row, whether
it is genuinely recoverable or must stay CONFIRMED_ALLELELESS. Also lists the small
non-CNV types (Insertion/Indel/Microsatellite) counts so we know the full recoverable-
candidate envelope. Pure evidence.
"""
import sys, os, gzip, argparse
print("=== probe_snv_alleleless START ===", flush=True)
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

def find_in_vcf(path, want):
    found = {}
    if not path or not os.path.exists(path): return found
    with gzip.open(path,"rt",encoding="utf-8",errors="replace") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split("\t",5)
            if len(p) < 5: continue
            vid = clean(p[2])
            if vid in want:
                found[vid] = (norm_chrom(p[0]), p[1], p[3], p[4])
                if len(found) == len(want): break
    return found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--raw-vcf", required=True)
    ap.add_argument("--fresh-vcf", default=None)
    ap.add_argument("--variant-summary", required=True)
    ap.add_argument("--fasta", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--assembly", default="GRCh38")
    a = ap.parse_args()

    coh = pd.read_parquet(a.cohort)
    al = coh[coh["ref"].map(bad) & coh["alt"].map(bad)].copy()
    al["source_id"] = al["source_id"].map(clean)
    al["chrom"] = al["chrom"].astype(str); al["pos"] = al["pos"].astype(int)

    # variant_summary Type per source_id (prefer the requested assembly row)
    vs = pd.read_csv(a.variant_summary, sep="\t", dtype=str, compression="gzip",
                     usecols=lambda c: c in {"VariationID","Type","Assembly","Chromosome","Start","Name"})
    vs["VariationID"] = vs["VariationID"].map(clean)
    vs_all = vs[vs["VariationID"].isin(set(al["source_id"]))].copy()
    # a source_id may have GRCh37 and GRCh38 rows; keep a Type per id (types agree across asm)
    type_by_id = {}
    rec_by_id = {}
    for _, r in vs_all.iterrows():
        vid = r["VariationID"]
        type_by_id.setdefault(vid, r["Type"])
        # prefer requested-assembly record for display
        if vid not in rec_by_id or r.get("Assembly") == a.assembly:
            rec_by_id[vid] = r

    al["vs_type"] = al["source_id"].map(type_by_id).fillna("(not in variant_summary)")

    print("\n--- allele-less Type distribution (per row) ---", flush=True)
    print(al["vs_type"].value_counts().to_string(), flush=True)

    # THE SNVs
    snv = al[al["vs_type"] == "single nucleotide variant"].copy()
    print(f"\n=== single-nucleotide-variant allele-less rows: {len(snv)} ===", flush=True)

    want = set(snv["source_id"])
    raw_hit = find_in_vcf(a.raw_vcf, want)
    fresh_hit = find_in_vcf(a.fresh_vcf, want) if a.fresh_vcf else {}

    ref_genome = None
    if os.path.exists(a.fasta):
        from pyfaidx import Fasta
        ref_genome = Fasta(a.fasta, rebuild=False)
    contigs = set(ref_genome.keys()) if ref_genome else set()
    def gcheck(chrom, pos, ref):
        if ref_genome is None or pos is None: return None
        c = norm_chrom(chrom)
        if c not in contigs: return None
        try:
            return str(ref_genome[c][int(pos)-1:int(pos)-1+len(ref)]).upper() == str(ref).upper()
        except Exception:
            return None

    print("\nper-SNV detail:", flush=True)
    for _, r in snv.iterrows():
        sid = r["source_id"]; hit = raw_hit.get(sid) or fresh_hit.get(sid)
        src = "raw" if sid in raw_hit else ("fresh" if sid in fresh_hit else "MISS")
        vsr = rec_by_id.get(sid, {})
        line = (f"  sid={sid} cohort={r['chrom']}:{r['pos']} "
                f"vs=({vsr.get('Assembly','?')} {vsr.get('Chromosome','?')}:{vsr.get('Start','?')}) "
                f"Name={str(vsr.get('Name',''))[:36]}")
        if hit:
            vc, vp, vref, valt = hit
            g = gcheck(vc, vp, vref)
            line += f"  VCF[{src}]={vc}:{vp} {vref}>{valt} genome_ok={g}"
        else:
            line += f"  VCF=MISS(both)"
        print(line, flush=True)

    print("\n--- also: counts of other small (non-CNV) types among allele-less ---", flush=True)
    for t in ["Insertion","Indel","Microsatellite","Complex","Variation","Inversion","Translocation"]:
        print(f"  {t:16s}: {int((al['vs_type']==t).sum())}", flush=True)
    print("\nINTERPRETATION: SNV rows with a VCF hit + genome_ok=True are genuinely recoverable", flush=True)
    print("by their OWN source_id. SNV rows with VCF=MISS are withdrawn/updated ids -> stay", flush=True)
    print("CONFIRMED_ALLELELESS. All CNV/SV/repeat types are CONFIRMED_ALLELELESS_SV by nature.", flush=True)
    print("=== probe_snv_alleleless DONE ===", flush=True)

if __name__ == "__main__":
    main()
