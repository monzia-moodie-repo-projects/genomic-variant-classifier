#!/usr/bin/env python
"""probe_raw_bad_tokens.py (2026-07-09)
Before building the unified cohort builder, VERIFY (not assume) the exact bad-allele token
set present in the raw source clinvar_grch38.parquet, and confirm whether any literal
empty-string alt with a longer ref exists -- the one case where build_cohort_v2.py's inline
is_padded_deletion (no non-empty guard) would diverge from the canonical allele_classify
version and silently mis-shift a position. Reports:
  1. distinct ref tokens and alt tokens among bad-allele rows (repr, so '' vs '.' vs 'na'
     are distinguishable).
  2. count of rows where alt is a bad token AND len(ref) > len(alt-as-written) -- i.e. rows
     the inline mask could shift but the canonical mask would not.
  3. the na:na count and half-bad count, to cross-check against prior probes.
Pure evidence; writes nothing. Requires only the raw parquet.
"""
import sys, argparse
print("=== probe_raw_bad_tokens START ===", flush=True)
try:
    import pandas as pd
    print("python:", sys.version.split()[0], "pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

# allele_classify canonical null tokens (single source of truth)
NULL = {"", "na", "nan", "none", "."}
def is_empty(x):
    if x is None: return True
    if isinstance(x, float) and pd.isna(x): return True
    return str(x).strip().lower() in NULL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/processed/clinvar_grch38.parquet")
    a = ap.parse_args()
    df = pd.read_parquet(a.raw, columns=["chrom","pos","ref","alt"])
    print(f"raw rows: {len(df):,}", flush=True)

    ref_bad = df["ref"].map(is_empty)
    alt_bad = df["alt"].map(is_empty)
    both = ref_bad & alt_bad
    either = ref_bad | alt_bad
    half = either & ~both
    print(f"na:na (both empty)     : {int(both.sum()):,}", flush=True)
    print(f"half-bad (exactly one) : {int(half.sum()):,}", flush=True)
    print(f"any bad (either)       : {int(either.sum()):,}", flush=True)

    bad_rows = df[either]
    ref_tokens = sorted({repr(x) for x in bad_rows["ref"] if is_empty(x)})
    alt_tokens = sorted({repr(x) for x in bad_rows["alt"] if is_empty(x)})
    print(f"\ndistinct BAD ref tokens: {ref_tokens[:20]}", flush=True)
    print(f"distinct BAD alt tokens: {alt_tokens[:20]}", flush=True)

    # THE divergence case: alt is literally empty-string '' (len 0) with a longer, real ref.
    def is_literal_empty(x):
        return x is not None and not (isinstance(x,float) and pd.isna(x)) and str(x) == ""
    alt_empty_str = df["alt"].map(is_literal_empty)
    ref_len = df["ref"].astype("string").fillna("").str.len()
    alt_len = df["alt"].astype("string").fillna("").str.len()
    divergent = alt_empty_str & (ref_len > alt_len)
    print(f"\nDIVERGENCE-RISK rows (alt is literal '' AND len(ref)>len(alt)): {int(divergent.sum()):,}", flush=True)
    print("  (these are the ONLY rows where build_cohort_v2 inline mask would shift but the", flush=True)
    print("   canonical allele_classify mask would not. Expect 0 if tokens are only 'na'/'.'.)", flush=True)
    if int(divergent.sum()) > 0:
        print(df[divergent].head(10).to_string(index=False), flush=True)

    # also: could ANY bad-allele row be caught by the v2 inline padded-del mask?
    # inline: len(alt)<len(ref) AND ref.startswith(alt). Test on the bad rows.
    r = bad_rows["ref"].astype("string").fillna("")
    al = bad_rows["alt"].astype("string").fillna("")
    starts = [rr.startswith(aa) for rr, aa in zip(r, al)]
    inline_mask = (al.str.len() < r.str.len()).to_numpy() & pd.Series(starts, index=r.index).to_numpy()
    print(f"\nbad-allele rows the v2 INLINE padded-del mask would catch (and shift): {int(inline_mask.sum()):,}", flush=True)
    print("  (canonical mask catches 0 of these due to the non-empty guard; any >0 here is a", flush=True)
    print("   row that quarantine-FIRST ordering must remove before coordinate correction.)", flush=True)
    print("=== probe_raw_bad_tokens DONE ===", flush=True)

if __name__ == "__main__":
    main()
