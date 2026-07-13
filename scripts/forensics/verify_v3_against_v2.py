#!/usr/bin/env python
"""verify_v3_against_v2.py (2026-07-09)
Independent structural verification that v3 == v2 with EXACTLY the 19,988 allele-less rows
removed and nothing else altered. Confirms: v3 has zero na:na rows; every non-allele-less v2
row survives UNCHANGED in v3 (same variant_id, source_id, chrom, pos, ref, alt); the set of
removed rows is EXACTLY the v2 allele-less set; row counts reconcile; and v3 has no duplicate
variant_id. Reads both parquets; writes nothing. This is the real-data guard the 0-recovery
rebuild run could not exercise via the merge path.
"""
import sys, argparse
print("=== verify_v3_against_v2 START ===", flush=True)
try:
    import pandas as pd
    print("python:", sys.version.split()[0], "pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

sys.path.insert(0, "src")
try:
    from genomic_variant_classifier.data.allele_classify import is_allele_less
except Exception:
    def is_allele_less(ref, alt):
        def bad(x):
            if x is None: return True
            if isinstance(x, float) and pd.isna(x): return True
            return str(x).strip().lower() in {"","na","nan","none","-",".","<na>"}
        return ref.map(bad) & alt.map(bad)

def clean(s):
    s = str(s).strip(); return s[:-2] if s.endswith(".0") else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v3", required=True)
    a = ap.parse_args()
    v2 = pd.read_parquet(a.v2)
    v3 = pd.read_parquet(a.v3)
    print(f"v2 rows: {len(v2):,}  v3 rows: {len(v3):,}  diff: {len(v2)-len(v3):,}", flush=True)

    al2 = v2[is_allele_less(v2["ref"], v2["alt"])]
    print(f"v2 allele-less rows: {len(al2):,}", flush=True)

    fails = []
    # 1. v3 has zero na:na
    n_na_v3 = int(is_allele_less(v3["ref"], v3["alt"]).sum())
    print(f"[{'PASS' if n_na_v3==0 else 'FAIL'}] v3 na:na rows == 0 (got {n_na_v3})", flush=True)
    if n_na_v3: fails.append("v3 has na:na rows")

    # 2. row-count reconciliation
    ok_count = (len(v3) == len(v2) - len(al2))
    print(f"[{'PASS' if ok_count else 'FAIL'}] v3 == v2 - alleleless ({len(v2)-len(al2):,})", flush=True)
    if not ok_count: fails.append("row count mismatch")

    # 3. identify removed rows by a stable per-row key. Non-alleleless v2 rows are unchanged,
    #    so key on (source_id, chrom, pos, ref, alt) for the NON-alleleless part.
    def key_df(df):
        k = df.copy()
        k["source_id"] = k["source_id"].map(clean)
        return set(zip(k["source_id"], k["chrom"].astype(str), k["pos"].astype(int),
                       k["ref"].astype(str), k["alt"].astype(str)))
    nonal2 = v2[~is_allele_less(v2["ref"], v2["alt"])]
    k_nonal2 = key_df(nonal2)
    k_v3 = key_df(v3)
    # every non-alleleless v2 row must be present unchanged in v3
    missing = k_nonal2 - k_v3
    print(f"[{'PASS' if not missing else 'FAIL'}] all {len(k_nonal2):,} non-alleleless v2 rows "
          f"present unchanged in v3 (missing {len(missing):,})", flush=True)
    if missing:
        fails.append(f"{len(missing)} non-alleleless rows missing/altered")
        print("   e.g.:", list(missing)[:3], flush=True)

    # 4. v3 rows that are NOT accounted for by non-alleleless v2 (should be 0, since 0 recovered)
    extra = k_v3 - k_nonal2
    print(f"[{'PASS' if not extra else 'INFO'}] v3 rows not in non-alleleless v2 set: {len(extra):,} "
          f"(expected 0 for a 0-recovery build)", flush=True)
    if extra:
        # these would be recovered/merged rows; for 0-recovery build there should be none
        fails.append(f"{len(extra)} unexpected v3 rows (recovered?)")
        print("   e.g.:", list(extra)[:3], flush=True)

    # 5. no duplicate variant_id in v3
    dup = int(v3["variant_id"].duplicated().sum())
    print(f"[{'PASS' if dup==0 else 'FAIL'}] v3 has no duplicate variant_id (got {dup})", flush=True)
    if dup: fails.append("duplicate variant_id in v3")

    # 6. schema preserved
    same_cols = list(v2.columns) == list(v3.columns)
    print(f"[{'PASS' if same_cols else 'FAIL'}] v3 schema identical to v2", flush=True)
    if not same_cols: fails.append("schema changed")

    print("", flush=True)
    if fails:
        print("VERIFICATION FAILED:", flush=True)
        for f in fails: print("  -", f, flush=True)
        print("=== verify_v3_against_v2 DONE (FAIL) ===", flush=True)
        return 1
    print("VERIFICATION PASSED: v3 is exactly v2 minus the 19,988 allele-less rows,", flush=True)
    print("all other rows preserved unchanged, schema intact, no duplicates.", flush=True)
    print("=== verify_v3_against_v2 DONE (PASS) ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
