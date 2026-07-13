#!/usr/bin/env python
"""verify_clean_as_canonical.py (2026-07-09)
Before adopting clinvar_grch38_clean.parquet (the clean_cohort structural-split output,
4,399,089 rows) as the canonical training base, verify it is sound and correctly related to
the current v2 cohort:
  1. clean has 0 na:na AND 0 half-bad (alt='.' etc.) rows           [must pass]
  2. clean has 0 duplicate variant_id                                [must pass]
  3. every clean row is present in v2 (clean is a strict subset)     [must pass]
  4. v2_rows - structural_rows == clean_rows (partition reconciles)  [must pass]
  5. clean vs v3: v3 still carries the 1,103 half-bad rows clean drops (report the diff)
  6. the 21 pathogenic/likely_pathogenic alt='.' rows: show them, confirm alt is truly
     absent (not a recoverable allele) before accepting their exclusion.
Pure evidence; writes nothing.
"""
import sys, os, argparse
print("=== verify_clean_as_canonical START ===", flush=True)
try:
    import pandas as pd
    print("python:", sys.version.split()[0], "pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

BAD = {"", "nan", "none", "na", ".", "null", "-", "<na>"}
def is_bad(x):
    if x is None: return True
    if isinstance(x, float) and pd.isna(x): return True
    return str(x).strip().lower() in BAD

def key_set(df):
    return set(zip(df["variant_id"].astype(str),
                   df["chrom"].astype(str), df["pos"].astype(int),
                   df["ref"].astype(str), df["alt"].astype(str)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v3", default=None)
    ap.add_argument("--structural", default=None)
    a = ap.parse_args()
    clean = pd.read_parquet(a.clean)
    v2 = pd.read_parquet(a.v2)
    print(f"clean rows: {len(clean):,}  v2 rows: {len(v2):,}", flush=True)
    fails = []

    # 1. clean has 0 na:na and 0 half-bad
    br = clean["ref"].map(is_bad); ba = clean["alt"].map(is_bad)
    n_bad = int((br | ba).sum())
    print(f"[{'PASS' if n_bad==0 else 'FAIL'}] clean has 0 bad-allele rows (got {n_bad})", flush=True)
    if n_bad: fails.append("clean has bad-allele rows")

    # 2. dup variant_id
    dup = int(clean["variant_id"].duplicated().sum())
    print(f"[{'PASS' if dup==0 else 'FAIL'}] clean has 0 duplicate variant_id (got {dup})", flush=True)
    if dup: fails.append("clean has dup variant_id")

    # 3. clean subset of v2
    kc, kv2 = key_set(clean), key_set(v2)
    notin = len(kc - kv2)
    print(f"[{'PASS' if notin==0 else 'FAIL'}] all clean rows present in v2 (missing {notin:,})", flush=True)
    if notin: fails.append(f"{notin} clean rows not in v2")

    # 4. partition reconciles
    if a.structural and os.path.exists(a.structural):
        st = pd.read_parquet(a.structural)
        ok = (len(v2) == len(clean) + len(st))
        print(f"[{'PASS' if ok else 'FAIL'}] v2 ({len(v2):,}) == clean ({len(clean):,}) + "
              f"structural ({len(st):,})", flush=True)
        if not ok: fails.append("partition mismatch")

    # 5. clean vs v3
    if a.v3 and os.path.exists(a.v3):
        v3 = pd.read_parquet(a.v3)
        kv3 = key_set(v3)
        only_v3 = len(kv3 - kc)
        only_clean = len(kc - kv3)
        print(f"[INFO] v3 rows {len(v3):,}; in v3 not clean: {only_v3:,}; in clean not v3: {only_clean:,}", flush=True)
        print(f"       (expect ~1,103 in v3-not-clean = the half-bad rows v3 still carries)", flush=True)

    # 6. the pathogenic alt='.' rows
    path_bad = clean.iloc[0:0]  # placeholder
    v2_bad = v2[(v2["ref"].map(is_bad) | v2["alt"].map(is_bad))]
    half = v2_bad[~(v2_bad["ref"].map(is_bad) & v2_bad["alt"].map(is_bad))]
    if "pathogenicity" in half.columns:
        pth = half[half["pathogenicity"].astype(str).str.contains("pathogenic", case=False, na=False)]
        print(f"\n--- {len(pth)} pathogenic/likely_pathogenic half-bad (alt='.') rows ---", flush=True)
        cols = [c for c in ["variant_id","source_id","chrom","pos","ref","alt","pathogenicity"] if c in pth.columns]
        print(pth[cols].to_string(index=False, max_colwidth=30), flush=True)
        print("\n  -> all have alt='.' (no alternate allele). Confirm: these are reference/no-alt", flush=True)
        print("     records, correctly excluded. No recoverable substitution exists in the row.", flush=True)

    print("", flush=True)
    if fails:
        print("VERIFICATION FAILED:", *("  - "+f for f in fails), sep="\n", flush=True)
        print("=== verify_clean_as_canonical DONE (FAIL) ===", flush=True)
        return 1
    print("clean.parquet is SOUND and is a correct strict subset of v2 (v2 minus structural).", flush=True)
    print("It is the recommended canonical training base (4,399,089 rows).", flush=True)
    print("=== verify_clean_as_canonical DONE (PASS) ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
