#!/usr/bin/env python
"""investigate_source_duplicates.py (2026-07-10)

Investigate the 4,203 duplicate variant_ids discovered in the processed source parquets
(clinvar_grch38.parquet and clinvar_grch38_pathfix.parquet) by the labelfix diff. Answers:

  Q1 multiplicity of duplicated variant_ids (how many appear 2x, 3x, ...)
  Q2 are duplicate rows byte-identical, or do they differ in other columns?
  Q3 which columns vary within duplicate groups (composite-key candidates)
  Q4 do duplicates cluster by chrom / gene / source_db / consequence?
  Q5 how many duplicate-involved rows carry pathogenic->uncertain transitions (the 17-row gap)?
  Q6 does pathfix preserve the same duplicate structure as raw?
  Q7 are the duplicates bad-allele / na:na rows the builder quarantines? (likely explanation
     for why the built cohort has 0 duplicates)

Read-only. Dumps a full report to stdout (redirect to a .txt). No fixes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/processed/clinvar_grch38.parquet")
FIX = Path("data/processed/clinvar_grch38_pathfix.parquet")
KEY = "variant_id"
_EMPTY = {"", "none", "nan", ".", "-", "na", "null"}


def is_empty_allele(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    return str(v).strip().lower() in _EMPTY


def line(c="-", n=78):
    print(c * n)


def dup_ids(df):
    vc = df[KEY].value_counts()
    return vc[vc > 1]


def main():
    print("=" * 78)
    print("SOURCE DUPLICATE INVESTIGATION")
    print("=" * 78)
    for p in (RAW, FIX):
        if not p.exists():
            print(f"FATAL: {p} not found")
            return 2
    raw = pd.read_parquet(RAW)
    fix = pd.read_parquet(FIX)
    print(f"raw  {RAW.name}: {len(raw):,} rows, {len(raw.columns)} cols")
    print(f"fix  {FIX.name}: {len(fix):,} rows, {len(fix.columns)} cols")
    print()

    # ---- Q1 multiplicity ----
    line("=")
    print("Q1. DUPLICATE variant_id MULTIPLICITY (raw)")
    line("=")
    d = dup_ids(raw)
    n_dup_ids = len(d)
    n_extra = int((d - 1).sum())
    print(f"  distinct variant_ids that are duplicated: {n_dup_ids:,}")
    print(f"  total rows involved in duplicates: {int(d.sum()):,}")
    print(f"  'extra' rows (sum(count-1)): {n_extra:,}   <- this is the diff's dup count")
    print(f"  multiplicity histogram:")
    for mult, cnt in d.value_counts().sort_index().items():
        print(f"    {mult}x : {cnt:,} variant_ids")
    print()

    dup_id_set = set(d.index)
    raw_dup = raw[raw[KEY].isin(dup_id_set)].copy()

    # ---- Q2 identical vs differing rows ----
    line("=")
    print("Q2. Are duplicate rows byte-identical, or do they differ?")
    line("=")
    compare_cols = [c for c in raw.columns if c not in ("metadata",)]  # dict col handled apart
    def row_sig(df):
        # null-safe: fillna then cast to str so mixed None/float/str columns join cleanly
        return df[compare_cols].astype(object).where(df[compare_cols].notna(), "<NA>").astype(str).agg("|".join, axis=1)
    raw_dup_sig = raw_dup.assign(_sig=row_sig(raw_dup))
    identical_ids, differing_ids = [], []
    for vid, grp in raw_dup_sig.groupby(KEY):
        if grp["_sig"].nunique() == 1:
            identical_ids.append(vid)
        else:
            differing_ids.append(vid)
    print(f"  duplicate variant_ids with IDENTICAL rows (excl. metadata): {len(identical_ids):,}")
    print(f"  duplicate variant_ids with DIFFERING rows: {len(differing_ids):,}")
    print("  -> identical => safe redundant dups; differing => distinct records (need composite key)")
    print()

    # ---- Q3 which columns vary ----
    line("=")
    print("Q3. Which columns VARY within duplicate groups? (composite-key candidates)")
    line("=")
    varying = {}
    for c in raw.columns:
        if c == "metadata":
            # dict col: compare via str
            nun = raw_dup.groupby(KEY)[c].apply(lambda s: s.astype(str).nunique())
        else:
            nun = raw_dup.groupby(KEY)[c].apply(lambda s: s.astype(str).nunique())
        n_groups_varying = int((nun > 1).sum())
        varying[c] = n_groups_varying
    for c, n in sorted(varying.items(), key=lambda kv: -kv[1]):
        flag = "  <- VARIES" if n > 0 else ""
        print(f"    {c:16s}: {n:,} groups vary{flag}")
    print()

    # ---- Q4 clustering ----
    line("=")
    print("Q4. Do duplicates CLUSTER? (top values among duplicate-involved rows)")
    line("=")
    for c in ("chrom", "gene_symbol", "source_db", "consequence"):
        if c in raw_dup.columns:
            print(f"  top {c}:")
            for val, cnt in raw_dup[c].astype(str).value_counts().head(8).items():
                print(f"    {val:30s} {cnt:,}")
            print()

    # ---- Q7 bad-allele / na:na cross-tab (the likely explanation) ----
    line("=")
    print("Q7. Are duplicate-involved rows BAD-ALLELE / na:na (quarantined by the builder)?")
    line("=")
    if "ref" in raw_dup.columns and "alt" in raw_dup.columns:
        ref_empty = raw_dup["ref"].apply(is_empty_allele)
        alt_empty = raw_dup["alt"].apply(is_empty_allele)
        nana = ref_empty & alt_empty
        halfbad = (ref_empty ^ alt_empty)
        clean = ~(ref_empty | alt_empty)
        print(f"  duplicate-involved rows total: {len(raw_dup):,}")
        print(f"    na:na (both alleles empty):   {int(nana.sum()):,}")
        print(f"    half-bad (one allele empty):  {int(halfbad.sum()):,}")
        print(f"    clean (both alleles present): {int(clean.sum()):,}")
        print()
        # how many DUP IDS are entirely quarantinable?
        raw_dup2 = raw_dup.assign(_bad=(ref_empty | alt_empty))
        allbad = raw_dup2.groupby(KEY)["_bad"].all()
        print(f"  duplicate variant_ids where ALL rows are bad-allele: {int(allbad.sum()):,}")
        print(f"  duplicate variant_ids where SOME rows are clean:     {int((~allbad).sum()):,}")
        print("  -> if most dups are na:na/bad-allele, the builder QUARANTINES them, explaining")
        print("     why the built cohort has 0 duplicates while the source has 4,203.")
    print()

    # ---- Q5 the 17-row gap ----
    line("=")
    print("Q5. Do duplicate rows carry the pathogenic->uncertain transitions (the 17-row gap)?")
    line("=")
    # align raw vs fix on index (same row order assumed; verify lengths match)
    if len(raw) == len(fix):
        # compare pathogenicity per row positionally
        changed = (raw["pathogenicity"].astype(str).values !=
                   fix["pathogenicity"].astype(str).values)
        p2u = ((raw["pathogenicity"].astype(str).values == "pathogenic") &
               (fix["pathogenicity"].astype(str).values == "uncertain"))
        print(f"  total pathogenicity changes (positional, full frame): {int(changed.sum()):,}")
        print(f"  pathogenic->uncertain (positional, full frame): {int(p2u.sum()):,}")
        # of those, how many are on duplicate variant_ids?
        on_dup = raw[KEY].isin(dup_id_set).values
        print(f"  ...of which on DUPLICATE variant_ids: {int((p2u & on_dup).sum()):,}")
        print(f"  ...on UNIQUE variant_ids: {int((p2u & ~on_dup).sum()):,}")
        print("  -> the dup-borne count should explain the 161,423 vs 161,406 gap (17).")
    else:
        print(f"  raw/fix length mismatch ({len(raw)} vs {len(fix)}) -- positional compare unsafe.")
    print()

    # ---- Q6 pathfix preserves dup structure ----
    line("=")
    print("Q6. Does pathfix preserve the SAME duplicate structure as raw?")
    line("=")
    d_fix = dup_ids(fix)
    print(f"  raw duplicated variant_ids: {len(d):,}")
    print(f"  fix duplicated variant_ids: {len(d_fix):,}")
    print(f"  identical dup id sets: {set(d.index) == set(d_fix.index)}")
    print()

    print("=" * 78)
    print("INVESTIGATION COMPLETE -- review above before designing the composite-key fix.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
