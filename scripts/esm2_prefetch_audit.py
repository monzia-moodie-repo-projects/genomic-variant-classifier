"""
scripts/esm2_prefetch_audit.py
==============================
Read-only triage of the genes still missing from the ESM-2 sequence cache after
pre-fetch, measured in VARIANTS (the real signal-loss unit), not just genes.
No network, no writes.

Each distinct cohort missense gene string is classified:
  resolved   sequence present in the cache (annotate will score it)
  compound   contains ';' (ClinVar multi-gene annotation) -> candidate for
             split-first resolution; we also test whether the FIRST split part
             is itself already cached (= recoverable with no new fetch)
  noncoding  matches antisense / non-coding name patterns (heuristic, for triage
             only): -AS<n>, -DT, LINC*, LOC<n>, MIR*, SNOR*, -IT<n>, -OT<n>
  other      single clean symbol still unresolved (transient, or no reviewed
             human UniProt entry)

USAGE (repo root, .venv312 active)
----------------------------------
  python scripts/esm2_prefetch_audit.py --clinvar data/processed/clinvar_grch38.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Heuristic non-coding name patterns -- used only to split the unresolved bucket
# for human triage; it does not drive any fix.
_NONCODING = re.compile(r"(-AS\d+$|-DT$|^LINC|^LOC\d+|^MIR|^SNOR|-IT\d+$|-OT\d+$)")


def classify(gene: str, cached: set[str]) -> str:
    if gene in cached:
        return "resolved"
    if ";" in gene:
        return "compound"
    if _NONCODING.search(gene):
        return "noncoding"
    return "other"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit unresolved ESM-2 cache genes (read-only)")
    ap.add_argument("--clinvar", required=True)
    ap.add_argument("--cache-path", default=None)
    args = ap.parse_args(argv)

    from genomic_variant_classifier.data import esm2 as esm2_mod

    cache_path = Path(args.cache_path) if args.cache_path else esm2_mod._DEFAULT_CACHE
    conn = esm2_mod._open_cache(cache_path)
    cached = {r[0] for r in conn.execute("SELECT gene FROM sequences").fetchall()}
    print(f"cache    : {cache_path}  ({len(cached):,} sequences)")

    df = pd.read_parquet(args.clinvar, columns=["gene_symbol", "consequence"])
    miss = df[df["consequence"].fillna("").str.contains("missense", case=False)].copy()
    miss["gene_symbol"] = miss["gene_symbol"].astype("string")
    miss = miss[miss["gene_symbol"].notna() & (miss["gene_symbol"] != "")]
    n_var = int(len(miss))
    var_by_gene = miss.groupby("gene_symbol").size()
    genes = list(var_by_gene.index)

    cls = {g: classify(g, cached) for g in genes}
    gene_ct: dict[str, int] = {}
    var_ct: dict[str, int] = {}
    for g in genes:
        c = cls[g]
        gene_ct[c] = gene_ct.get(c, 0) + 1
        var_ct[c] = var_ct.get(c, 0) + int(var_by_gene[g])

    print(f"\ncohort missense: {n_var:,} variants across {len(genes):,} distinct gene strings\n")
    print(f"{'class':<11}{'genes':>9}{'variants':>13}{'% variants':>12}")
    for c in ("resolved", "compound", "noncoding", "other"):
        gc = gene_ct.get(c, 0)
        vc = var_ct.get(c, 0)
        print(f"{c:<11}{gc:>9,}{vc:>13,}{100 * vc / max(n_var, 1):>11.2f}%")

    # Split-first recoverability for compound genes.
    comp = [g for g in genes if cls[g] == "compound"]
    rec_g = rec_v = 0
    for g in comp:
        if g.split(";")[0] in cached:
            rec_g += 1
            rec_v += int(var_by_gene[g])
    comp_v = var_ct.get("compound", 0)
    print(f"\ncompound genes whose FIRST split-part is already cached: "
          f"{rec_g:,}/{len(comp):,} genes, {rec_v:,}/{comp_v:,} variants")
    print("  => split-first resolution recovers those with NO new fetch.")
    print("\nThe 'noncoding' bucket is correctly 0.0 (no protein); 'other' is the "
          "residual to retry or accept. 'compound' is the recoverable silent-zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
