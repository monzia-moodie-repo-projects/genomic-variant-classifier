#!/usr/bin/env python
"""probe_stale_allele_representation.py (2026-07-09)
Before switching the fresh ingestion to the *VCF allele columns (keeping pos=Start), CONFIRM
that the STALE parquet's alleles are already in the padded VCF-style representation -- i.e.
that stale ReferenceAllele (2026-03) == fresh ReferenceAlleleVCF (2026-07) in REPRESENTATION.
If so, Design 1 (VCF alleles + pos=Start) yields matching allele strings across snapshots and
the diff is valid. If stale used an unpadded '-' representation, every indel would differ
spuriously and we'd need a normalization layer.

Checks on the STALE parquet (data/processed/clinvar_grch38.parquet):
  1. Among clean (both alleles present) rows, count how many deletions look PADDED
     (len(ref)>len(alt) AND ref.startswith(alt)) vs UNPADDED ('-' present, or alt empty for a
     deletion). Padded-style => VCF representation.
  2. Report the fraction of indels using '-' anywhere (the non-VCF hallmark). Should be ~0 if
     stale is VCF-style.
  3. Cross-snapshot spot check by source_id (VariationID): pick a handful of stale rows with
     multi-base indels, and if the fresh raw is provided, look up the SAME VariationID in the
     fresh *VCF columns and compare ref/alt strings directly. This is the decisive per-variant
     test that stale ref/alt == fresh *VCF ref/alt.
Read-only.
"""
import sys, gzip, argparse
from pathlib import Path
print("=== probe_stale_allele_representation START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def is_nullish(v):
    if v is None: return True
    try:
        if pd.isna(v): return True
    except (TypeError, ValueError): pass
    return str(v).strip().lower() in {"", "na", "nan", "none", ".", "-", "null"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", default="data/processed/clinvar_grch38.parquet")
    ap.add_argument("--fresh", default="data/external/clinvar/variant_summary.txt.gz")
    ap.add_argument("--fresh-nrows", type=int, default=2000000,
                    help="rows of fresh raw to scan for the VariationID cross-check")
    a = ap.parse_args()

    st = pd.read_parquet(a.stale, columns=["ref","alt","source_id","chrom","pos"])
    print(f"stale rows: {len(st):,}", flush=True)
    clean = st[(~st["ref"].map(is_nullish)) & (~st["alt"].map(is_nullish))].copy()
    print(f"stale clean (both alleles present): {len(clean):,}", flush=True)

    rl = clean["ref"].astype(str).str.len()
    al = clean["alt"].astype(str).str.len()
    dele = clean[rl > al]
    ins  = clean[al > rl]
    snv  = clean[(rl == 1) & (al == 1)]
    print(f"  SNV (1->1): {len(snv):,}", flush=True)
    print(f"  deletions (ref longer): {len(dele):,}", flush=True)
    print(f"  insertions (alt longer): {len(ins):,}", flush=True)

    # padded-style deletion: ref.startswith(alt) and multi-base ref
    d_ref = dele["ref"].astype(str); d_alt = dele["alt"].astype(str)
    padded = [r.startswith(al_) for r, al_ in zip(d_ref, d_alt)]
    import numpy as np
    padded = np.array(padded)
    print(f"  of deletions, padded-style (ref.startswith(alt)): {int(padded.sum()):,} "
          f"({100*padded.mean():.1f}%)", flush=True)
    # any '-' in clean alleles?
    dash = ((clean['ref'].astype(str)=='-') | (clean['alt'].astype(str)=='-')).sum()
    print(f"  clean rows using '-' as an allele: {int(dash):,} (VCF-style should be ~0)", flush=True)
    print("  sample padded deletions (stale):", flush=True)
    for _, r in dele.head(4).iterrows():
        print(f"      ref={str(r['ref'])[:20]} alt={str(r['alt'])[:8]} source_id={r['source_id']} pos={r['pos']}", flush=True)

    # Cross-snapshot spot check by VariationID
    print("\n  --- cross-snapshot spot check by VariationID (source_id) ---", flush=True)
    targets = dele.head(8)[["source_id","ref","alt","chrom","pos"]].copy()
    tset = set(str(x) for x in targets["source_id"].tolist())
    if Path(a.fresh).exists() and tset:
        op = gzip.open if str(a.fresh).endswith(".gz") else open
        found = {}
        with op(a.fresh, "rt", encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\n").split("\t")
            hidx = {c.lstrip("#"): i for i, c in enumerate(header)}
            vi = hidx.get("VariationID"); asmi = hidx.get("Assembly")
            rvi = hidx.get("ReferenceAlleleVCF"); avi = hidx.get("AlternateAlleleVCF")
            si = hidx.get("Start"); pvi = hidx.get("PositionVCF")
            n = 0
            for line in f:
                n += 1
                if n > a.fresh_nrows: break
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(vi, asmi, rvi, avi): continue
                if parts[asmi] != "GRCh38": continue
                if parts[vi] in tset and parts[vi] not in found:
                    found[parts[vi]] = (parts[rvi], parts[avi], parts[si], parts[pvi])
                    if len(found) == len(tset): break
        print(f"  matched {len(found)}/{len(tset)} VariationIDs in fresh (first {a.fresh_nrows:,} rows):", flush=True)
        for _, t in targets.iterrows():
            sid = str(t["source_id"])
            if sid in found:
                fref, falt, fstart, fpos = found[sid]
                match = (str(t["ref"]).upper()==str(fref).upper() and str(t["alt"]).upper()==str(falt).upper())
                print(f"    VID {sid}: STALE ref={str(t['ref'])[:16]} alt={str(t['alt'])[:8]} | "
                      f"FRESH *VCF ref={str(fref)[:16]} alt={str(falt)[:8]} | MATCH={match}", flush=True)
            else:
                print(f"    VID {sid}: not found in fresh sample", flush=True)
    else:
        print("  (fresh raw not available or no targets)", flush=True)

    print("\n--- VERDICT ---", flush=True)
    print("  If padded-style % is high, '-' count ~0, and the VariationID spot checks MATCH,", flush=True)
    print("  then stale ref/alt == fresh *VCF ref/alt in representation -> Design 1 is sound:", flush=True)
    print("  fresh ingestion uses *VCF alleles + pos=Start, and the diff is a true control.", flush=True)
    print("=== probe_stale_allele_representation DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
