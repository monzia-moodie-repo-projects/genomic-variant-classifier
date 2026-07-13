#!/usr/bin/env python
"""probe_fresh_g9_duplicates.py (2026-07-10)
The fresh cohort build failed G9 (post-condition): 8 duplicate variant_id in the clean set.
This probe extracts and fully characterizes those duplicates so we can design a PRINCIPLED
resolution (collapse-if-identical vs reconcile-conflicts) from evidence -- never a blind drop.

It reproduces the builder's clean-set derivation faithfully using the CANONICAL allele_classify
module: (1) quarantine allele-less rows (is_empty_allele on ref OR alt), (2) coordinate-correct
padded deletions (pos -= 1) and rebuild variant_id, then (3) find variant_ids that remain
duplicated in the clean set. For each duplicated variant_id it prints EVERY row's source_id
(VariationID), chrom, pos, ref, alt, pathogenicity, clinical_sig, and gene_symbol side by side,
and flags whether the duplicate rows are IDENTICAL on the decision-relevant fields
(pathogenicity, clinical_sig) or CONFLICTING. It also cross-checks each variant_id against the
STALE parquet (was it present? unique?). Read-only.
"""
import sys, argparse
from pathlib import Path
print("=== probe_fresh_g9_duplicates START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

# import the canonical classifier the builder uses
sys.path.insert(0, "src")
try:
    from genomic_variant_classifier.data import allele_classify as AC
except Exception as e:
    print("FATAL: cannot import allele_classify:", e, flush=True); sys.exit(12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", default="data/processed/clinvar_grch38_fresh.parquet")
    ap.add_argument("--stale", default="data/processed/clinvar_grch38.parquet")
    a = ap.parse_args()

    df = pd.read_parquet(a.fresh)
    print(f"fresh rows: {len(df):,}", flush=True)

    # (1) quarantine allele-less rows
    empty = AC.is_empty_allele(df["ref"]) | AC.is_empty_allele(df["alt"])
    clean = df[~empty].copy()
    print(f"after quarantine (allele-less removed): {len(clean):,}", flush=True)

    # (2) coordinate-correct padded deletions: pos -= 1, rebuild variant_id
    padded = AC.is_padded_deletion(clean["ref"], clean["alt"])
    clean.loc[padded, "pos"] = clean.loc[padded, "pos"].astype("int64") - 1
    clean["variant_id"] = ("clinvar:" + clean["chrom"].astype(str) + ":" +
                           clean["pos"].astype(str) + ":" +
                           clean["ref"].astype(str) + ":" + clean["alt"].astype(str))
    print(f"padded deletions corrected: {int(padded.sum()):,}", flush=True)

    # (3) find remaining duplicate variant_ids
    dupmask = clean["variant_id"].duplicated(keep=False)
    dups = clean[dupmask].sort_values("variant_id")
    dup_ids = sorted(dups["variant_id"].unique())
    print(f"\nDUPLICATE variant_id in clean set: {len(dup_ids)} distinct ({len(dups)} rows)", flush=True)
    print("="*70, flush=True)

    # stale lookup
    st = pd.read_parquet(a.stale, columns=["variant_id","source_id","pathogenicity","clinical_sig"])
    st_counts = st["variant_id"].value_counts()

    fields = ["source_id","chrom","pos","ref","alt","pathogenicity","clinical_sig","gene_symbol"]
    fields = [f for f in fields if f in dups.columns]
    n_identical = 0; n_conflict = 0
    for vid in dup_ids:
        rows = dups[dups["variant_id"] == vid]
        print(f"\nvariant_id: {vid}   ({len(rows)} rows)", flush=True)
        for _, r in rows.iterrows():
            vals = " | ".join(f"{f}={r[f]}" for f in fields)
            print(f"    {vals}", flush=True)
        # identical vs conflicting on decision-relevant fields
        pset = set(rows["pathogenicity"].astype(str))
        cset = set(rows["clinical_sig"].astype(str))
        aset = set(rows["ref"].astype(str) + ">" + rows["alt"].astype(str))
        verdict = "IDENTICAL (path+sig+allele)" if (len(pset)==1 and len(cset)==1 and len(aset)==1) else \
                  f"CONFLICT (pathogenicity={pset if len(pset)>1 else '-'}, clinical_sig={'differs' if len(cset)>1 else 'same'})"
        if verdict.startswith("IDENTICAL"): n_identical += 1
        else: n_conflict += 1
        print(f"    -> {verdict}", flush=True)
        # stale cross-check
        in_stale = int(st_counts.get(vid, 0))
        print(f"    -> in STALE: {in_stale} row(s) {'(unique)' if in_stale==1 else '(absent)' if in_stale==0 else '(ALSO DUP in stale!)'}", flush=True)

    print("\n" + "="*70, flush=True)
    print(f"SUMMARY: {n_identical} identical-twin variant_ids, {n_conflict} conflicting.", flush=True)
    print("  * identical twins -> safe to collapse to one row (same path/sig/allele; only source_id differs)", flush=True)
    print("  * conflicting twins -> MUST reconcile (e.g. keep most-pathogenic, or merge metadata); NEVER blind-drop", flush=True)
    print("=== probe_fresh_g9_duplicates DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
