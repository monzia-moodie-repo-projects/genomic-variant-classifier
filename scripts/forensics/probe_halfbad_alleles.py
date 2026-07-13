#!/usr/bin/env python
"""probe_halfbad_alleles.py (2026-07-09)
The provenance probe showed the canonical v2 (and thus v3) was built from the RAW source,
bypassing clean_cohort.py's structural split. v3 removed the 19,988 both-absent na:na rows
but clean_cohort quarantines 21,091 (= 19,988 na:na + 1,103 other bad-allele). This probe
isolates and characterises those ~1,103 rows STILL PRESENT in v3 that clean_cohort would
reject: rows where exactly ONE of ref/alt is bad (half-missing), or where a token like
'null'/'-' appears. Reports how many, their ref/alt patterns, ClinVar Type (via source_id),
and pathogenicity mix, so we can decide whether v3 should exclude them too. Pure evidence.
"""
import sys, os, argparse
print("=== probe_halfbad_alleles START ===", flush=True)
try:
    import pandas as pd
    print("python:", sys.version.split()[0], "pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

# clean_cohort.py's exact bad-allele token set (the authoritative upstream predicate)
BAD = {"", "nan", "none", "na", ".", "null", "-"}
def is_bad(x):
    if x is None: return True
    if isinstance(x, float) and pd.isna(x): return True
    return str(x).strip().lower() in BAD

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", required=True)
    ap.add_argument("--variant-summary", default=None)
    ap.add_argument("--assembly", default="GRCh38")
    a = ap.parse_args()
    v3 = pd.read_parquet(a.v3)
    print(f"v3 rows: {len(v3):,}", flush=True)

    bad_ref = v3["ref"].map(is_bad)
    bad_alt = v3["alt"].map(is_bad)
    both = bad_ref & bad_alt            # na:na (should be 0 in v3)
    either = bad_ref | bad_alt
    half = either & ~both              # exactly one side bad
    print(f"rows with BOTH ref&alt bad (na:na) : {int(both.sum()):,}  (expect 0 in v3)", flush=True)
    print(f"rows with EITHER bad               : {int(either.sum()):,}", flush=True)
    print(f"rows with EXACTLY ONE bad (half)   : {int(half.sum()):,}", flush=True)

    hb = v3[half].copy()
    if len(hb):
        hb["which_bad"] = ["ref" if is_bad(r) else "alt" for r, _ in zip(hb["ref"], hb["alt"])]
        # actually compute per row correctly
        hb["which_bad"] = [("ref" if is_bad(r) else "alt") for r, aa in zip(hb["ref"], hb["alt"])]
        print("\n--- which side is bad ---", flush=True)
        print(hb["which_bad"].value_counts().to_string(), flush=True)

        print("\n--- sample ref/alt values (up to 20) ---", flush=True)
        cols = [c for c in ["variant_id","source_id","chrom","pos","ref","alt","pathogenicity"] if c in hb.columns]
        print(hb[cols].head(20).to_string(index=False, max_colwidth=28), flush=True)

        # distinct bad tokens actually present
        badtokens = set()
        for r, al in zip(hb["ref"], hb["alt"]):
            if is_bad(r): badtokens.add(repr(r))
            if is_bad(al): badtokens.add(repr(al))
        print("\ndistinct bad-token values seen:", sorted(badtokens)[:20], flush=True)

        if "pathogenicity" in hb.columns:
            print("\n--- pathogenicity mix of half-bad rows ---", flush=True)
            print(hb["pathogenicity"].value_counts(dropna=False).to_string(), flush=True)

        # ClinVar Type via source_id
        if a.variant_summary and os.path.exists(a.variant_summary):
            vs = pd.read_csv(a.variant_summary, sep="\t", dtype=str, compression="gzip",
                             usecols=lambda c: c in {"VariationID","Type"})
            def clean(s):
                s=str(s).strip(); return s[:-2] if s.endswith(".0") else s
            tmap = {}
            for vid,t in zip(vs["VariationID"].map(clean), vs["Type"]): tmap.setdefault(vid,t)
            hb["_sid"] = hb["source_id"].map(clean) if "source_id" in hb.columns else None
            hb["vs_type"] = hb["_sid"].map(tmap)
            print("\n--- ClinVar Type of half-bad rows ---", flush=True)
            print(hb["vs_type"].value_counts(dropna=False).to_string(), flush=True)
    else:
        print("\nNo half-bad rows in v3.", flush=True)

    print("\nINTERPRETATION: half-bad-allele rows (one of ref/alt absent) are unusable for a", flush=True)
    print("variant classifier the same way na:na are -- clean_cohort quarantines them. If v3", flush=True)
    print("still carries them, the canonical clean cohort (clinvar_grch38_clean.parquet,", flush=True)
    print("4,399,089) is the correct training base, and v3 should also exclude these.", flush=True)
    print("=== probe_halfbad_alleles DONE ===", flush=True)

if __name__ == "__main__":
    main()
