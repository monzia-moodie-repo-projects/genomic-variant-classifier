#!/usr/bin/env python3
"""Audit persisted splits for leakage and structural contamination (read-only)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--splits-dir",default="outputs/run15_baseline/full/splits")
    ap.add_argument("--structural",default="data/processed/clinvar_grch38_structural.parquet")
    a=ap.parse_args(); sd=Path(a.splits_dir)
    paths={n:sd/f"meta_{n}.parquet" for n in ("train","val","test")}
    for n,p in paths.items():
        if not p.exists(): print(f"ABORT: missing {p}"); return 1
    m={n:pd.read_parquet(p) for n,p in paths.items()}
    c=list(m["train"].columns); print("meta cols:",c)
    v="variant_id" if "variant_id" in c else None; g="gene_symbol" if "gene_symbol" in c else None
    res={}
    if v:
        for n,d in m.items():
            dup=int(d[v].duplicated().sum()); print(f"  {n}: rows={len(d):,} dup_vid={dup}"); res[f"no_dup_{n}"]=dup==0
        s={n:set(d[v]) for n,d in m.items()}
        tt=len(s['train']&s['test']);tv=len(s['train']&s['val']);vt=len(s['val']&s['test'])
        print(f"  vid overlap train&test={tt} train&val={tv} val&test={vt}"); res["no_cross_split"]= (tt+tv+vt)==0
    if g:
        gtt=len(set(m['train'][g].dropna())&set(m['test'][g].dropna())); print(f"  gene overlap train&test={gtt}"); res["gene_disjoint"]=gtt==0
    if v and Path(a.structural).exists():
        st=set(pd.read_parquet(a.structural,columns=["variant_id"])["variant_id"])
        allv=set().union(*[set(d[v]) for d in m.values()]); con=len(st&allv)
        print(f"  structural_in_splits={con}"); res["no_structural"]=con==0
    print("\n=== RESULT ===")
    for k,ok in res.items(): print(f"  {'PASS' if ok else 'FAIL'} {k}")
    allp=all(res.values()) if res else False
    print("\nVERDICT:", "ALL PASS -- splits leak-free" if allp else "FAIL present -- splits NOT clean")
    return 0 if allp else 1
if __name__=="__main__": sys.exit(main())
