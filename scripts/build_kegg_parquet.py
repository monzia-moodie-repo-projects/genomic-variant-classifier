#!/usr/bin/env python3
"""
scripts/build_kegg_parquet.py  --  build the KEGG gene->pathway mapping (2026-07-06)

Pulls the human (hsa) pathway maps from the KEGG REST API and writes
data/external/kegg_gene_pathways.parquet with columns:
    gene_symbol               str    HGNC symbol (from KEGG's own hsa gene list)
    kegg_pathway_count        int    distinct KEGG pathways the gene is in
    kegg_disease_pathway_flag int    1 if in >=1 "Human Diseases" pathway (hsa05xxx)

Entrez->symbol mapping comes from KEGG's OWN gene list (`/list/hsa`) -- no external
dependency (per the chosen design). NETWORK-AT-BUILD-TIME: this must run where
rest.kegg.jp is reachable (NOT the build sandbox). KEGG REST is free for academic use;
be polite (the endpoints below are 3 bulk calls, not per-gene).

REST endpoints used (documented, stable):
    GET https://rest.kegg.jp/list/pathway/hsa   -> "hsa00010\tGlycolysis ..."
    GET https://rest.kegg.jp/link/hsa/pathway   -> "path:hsa00010\thsa:10327"   (pathway<->gene)
    GET https://rest.kegg.jp/list/hsa           -> "hsa:10327\t... ; SYMBOL, ...; description"

Disease pathways = KEGG "Human Diseases" category, map id range hsa05xxx (documented,
overridable via --disease-prefix-range). Fails LOUD if the pull is empty, if sentinel
cancer genes (TP53, BRCA1) are missing / not flagged, or if a feature is all-zero.

Usage:
    python scripts/build_kegg_parquet.py --out data/external/kegg_gene_pathways.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    print("ERROR: requests not installed (pip install requests).", file=sys.stderr)
    raise SystemExit(2)

KEGG_BASE = "https://rest.kegg.jp"
DISEASE_LO, DISEASE_HI = 5000, 5999  # hsa05xxx = "Human Diseases" maps


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _get(path: str, timeout: int = 60) -> str:
    url = f"{KEGG_BASE}{path}"
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200 or not r.text.strip():
        fail(f"KEGG REST {url} -> HTTP {r.status_code}, {len(r.text)} bytes")
    return r.text


def _pathway_num(pathway_id: str) -> int | None:
    """'path:hsa05200' or 'hsa05200' -> 5200 ; None if not hsa#####."""
    pid = pathway_id.split(":")[-1]
    if not pid.startswith("hsa"):
        return None
    digits = pid[3:]
    return int(digits) if digits.isdigit() else None


def _symbol_from_list_hsa(desc_field: str) -> str | None:
    """
    KEGG /list/hsa row (after the 'hsa:####\\t'): e.g.
      'ALDOA, ALDA, ...; fructose-bisphosphate aldolase A'  (older)
      'ALDOA; ...'                                          (newer)
    Symbol = first comma/semicolon-delimited token, uppercased & stripped.
    """
    first = desc_field.split(";")[0]
    sym = first.split(",")[0].strip()
    return sym or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/external/kegg_gene_pathways.parquet"))
    ap.add_argument("--disease-prefix-range", default=f"{DISEASE_LO}-{DISEASE_HI}",
                    help="hsa map-number range treated as 'Human Diseases' (default 5000-5999)")
    args = ap.parse_args(argv)
    lo, hi = (int(x) for x in args.disease_prefix_range.split("-"))

    # 1. hsa gene id -> symbol (KEGG's own list)
    id2sym: dict[str, str] = {}
    for line in _get("/list/hsa").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        gid = parts[0].strip()                       # 'hsa:10327'
        sym = _symbol_from_list_hsa(parts[-1])
        if gid and sym:
            id2sym[gid] = sym
    if len(id2sym) < 10000:
        fail(f"/list/hsa yielded only {len(id2sym)} gene->symbol rows (expected ~20k); format drift?")

    # 2. pathway <-> gene links -> per-gene distinct pathways + disease membership
    from collections import defaultdict
    gene_paths: dict[str, set[str]] = defaultdict(set)
    gene_disease: dict[str, bool] = defaultdict(bool)
    n_links = 0
    for line in _get("/link/hsa/pathway").splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        pth, gid = parts[0].strip(), parts[1].strip()
        num = _pathway_num(pth)
        if num is None or gid not in id2sym:
            continue
        sym = id2sym[gid]
        gene_paths[sym].add(pth)
        if lo <= num <= hi:
            gene_disease[sym] = True
        n_links += 1
    if n_links < 10000:
        fail(f"/link/hsa/pathway yielded only {n_links} usable links (expected >30k); format drift?")

    rows = [
        {"gene_symbol": sym,
         "kegg_pathway_count": len(paths),
         "kegg_disease_pathway_flag": int(gene_disease.get(sym, False))}
        for sym, paths in gene_paths.items()
    ]
    df = pd.DataFrame(rows).sort_values("gene_symbol").reset_index(drop=True)

    # 3. LOUD self-check -- refuse to write garbage
    if df.empty:
        fail("no gene rows assembled")
    if df["kegg_pathway_count"].max() <= 0:
        fail("kegg_pathway_count all-zero -- pull/parse failed")
    for g in ("TP53", "BRCA1"):
        row = df[df["gene_symbol"] == g]
        if row.empty:
            fail(f"sentinel gene {g} missing -- symbol mapping likely drifted")
        if int(row["kegg_pathway_count"].iloc[0]) <= 0:
            fail(f"sentinel gene {g} has 0 pathways -- link parse failed")
        if int(row["kegg_disease_pathway_flag"].iloc[0]) != 1:
            fail(f"sentinel cancer gene {g} not flagged as disease-pathway -- hsa05xxx range wrong?")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    n_dis = int((df["kegg_disease_pathway_flag"] == 1).sum())
    print(f"OK wrote {args.out}  ({len(df)} genes; {n_dis} in a KEGG disease pathway)")
    print(f"  kegg_pathway_count: min={df['kegg_pathway_count'].min()} "
          f"median={int(df['kegg_pathway_count'].median())} max={df['kegg_pathway_count'].max()}")
    print(f"  sentinels: TP53={int(df[df.gene_symbol=='TP53'].kegg_pathway_count.iloc[0])} paths, "
          f"BRCA1={int(df[df.gene_symbol=='BRCA1'].kegg_pathway_count.iloc[0])} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
