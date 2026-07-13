#!/usr/bin/env python
"""read_indel_mismatches.py (2026-07-09)
Read the indel mismatch TSV produced by build_cohort_from_source.py's genome-consistency
check and characterize EACH disagreement, so the 13 (or however many) ClinVar-vs-GRCh38
mismatches are documented individually, not merely tolerated as "under threshold".

For each mismatch variant_id (clinvar:chrom:pos:ref:alt) it joins back to the cohort's
raw parquet to recover pathogenicity/type, and prints ref-vs-genome side by side. Flags any
mismatch whose pathogenicity is 'pathogenic' or 'likely_pathogenic' (a lost pathogenic label
would matter). Pure evidence; writes a small annotated TSV to outputs/. 
"""
import sys, argparse
from pathlib import Path
print("=== read_indel_mismatches START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mismatches", default="outputs/clinvar_grch38_cohort_v4_indel_mismatches.tsv",
                    help="the *_indel_mismatches.tsv from the builder")
    ap.add_argument("--raw", default="data/processed/clinvar_grch38.parquet")
    ap.add_argument("--out", default="outputs/indel_mismatches_annotated.tsv")
    a = ap.parse_args()

    mm_path = Path(a.mismatches)
    if not mm_path.exists():
        print(f"no mismatch file at {mm_path} -- if the build reported 0 mismatches this is expected.", flush=True)
        print("=== read_indel_mismatches DONE (nothing to read) ===", flush=True)
        return 0
    mm = pd.read_csv(mm_path, sep="\t")
    print(f"mismatch rows: {len(mm):,}", flush=True)
    print(f"columns: {list(mm.columns)}", flush=True)

    # variant_id -> parse chrom:pos to rejoin to raw for pathogenicity/type
    raw_cols = ["variant_id", "chrom", "pos", "ref", "alt"]
    raw = pd.read_parquet(a.raw)
    extra = [c for c in ("pathogenicity", "clinical_sig", "gene_symbol", "metadata") if c in raw.columns]
    # match on the mismatch variant_id string directly if present, else on chrom:pos:ref:alt
    key = "variant_id"
    ann = mm.merge(raw[raw_cols + extra].drop_duplicates("variant_id"),
                   on="variant_id", how="left", suffixes=("", "_raw"))
    print("\n--- per-mismatch detail ---", flush=True)
    for _, r in ann.iterrows():
        patho = r.get("pathogenicity", "?")
        line = (f"  {r['variant_id']}  expected_ref={r.get('ref', r.get('expected_label','?'))}  "
                f"genome={r.get('genome_seq', r.get('genome_label','?'))}  patho={patho}")
        print(line, flush=True)

    if "pathogenicity" in ann.columns:
        p = ann["pathogenicity"].astype("string").str.lower()
        n_patho = int(p.isin(["pathogenic", "likely_pathogenic"]).sum())
        print(f"\nmismatches labeled pathogenic/likely_pathogenic: {n_patho}", flush=True)
        if n_patho:
            print("  ^ these lost a real label to a ClinVar-vs-genome disagreement; document individually.", flush=True)
        else:
            print("  none are pathogenic/likely_pathogenic -- no pathogenic label lost to a mismatch.", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ann.to_csv(a.out, sep="\t", index=False)
    print(f"\nwrote {a.out}", flush=True)
    print("=== read_indel_mismatches DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
