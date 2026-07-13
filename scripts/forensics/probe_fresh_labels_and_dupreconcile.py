#!/usr/bin/env python
"""probe_fresh_labels_and_dupreconcile.py (2026-07-10)
Two rigor checks before building cohorts, so nothing is assumed:
  (A) VERIFY the fresh re-ingest actually applied the Conflicting->uncertain label fix. Prints
      the fresh 5-class pathogenicity distribution and, for every clinical_sig starting with
      'conflicting', shows what pathogenicity it now maps to (must be uncertain for ALL). Also
      contrasts with the CORRECTED stale (pathfix) distribution for a side-by-side sanity check.
  (B) RECONCILE the duplicate-variant_id counts: the builder's G9 post-condition found 0 dups in
      the CLEAN (post-quarantine) stale cohort, yet finalize reports 4,203 dups in the RAW stale
      parquet. Prove these are consistent: show that raw-stale duplicate variant_ids are
      dominated by allele-less (na:na / :None:None) collisions that the builder QUARANTINES, so
      the clean cohort legitimately has 0. Breaks the 4,203 into (na:na collisions) vs (genuine
      fully-alleled collisions), same decomposition as fresh.
Read-only. Every acronym expanded on first use: Message-Digest-5 (MD5), ClinVar Variation
Identifier (VariationID), Variant of Uncertain Significance (VUS), Variant Call Format (VCF).
"""
import sys
from pathlib import Path
print("=== probe_fresh_labels_and_dupreconcile START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

FRESH = Path("data/processed/clinvar_grch38_fresh.parquet")
STALE_FIX = Path("data/processed/clinvar_grch38_pathfix.parquet")
STALE_RAW = Path("data/processed/clinvar_grch38.parquet")

def dist(p, label):
    if not p.exists():
        print(f"  (missing {p})", flush=True); return None
    df = pd.read_parquet(p, columns=["clinical_sig","pathogenicity"])
    print(f"\n  {label}: {len(df):,} rows", flush=True)
    vc = df["pathogenicity"].astype("string").fillna("<NA>").value_counts()
    for k,v in vc.items():
        print(f"      {k:20s} {v:,}", flush=True)
    # conflicting rows -> what label now?
    confl = df[df["clinical_sig"].astype("string").str.lower().str.startswith("conflicting", na=False)]
    print(f"    clinical_sig startswith 'conflicting': {len(confl):,}", flush=True)
    cvc = confl["pathogenicity"].astype("string").fillna("<NA>").value_counts()
    for k,v in cvc.items():
        flag = "" if k=="uncertain" else "   <== NOT uncertain! FIX DID NOT APPLY"
        print(f"        -> {k:18s} {v:,}{flag}", flush=True)
    return df

print("\n########## (A) LABEL-FIX VERIFICATION ##########", flush=True)
f = dist(FRESH, "FRESH (re-ingested with corrected replica v4)")
s = dist(STALE_FIX, "STALE-PATHFIX (re-derived)")
if f is not None:
    confl = f[f["clinical_sig"].astype("string").str.lower().str.startswith("conflicting", na=False)]
    bad = confl[confl["pathogenicity"] != "uncertain"]
    print(f"\n  FRESH conflicting rows NOT mapped to uncertain: {len(bad):,}  "
          f"({'PASS' if len(bad)==0 else 'FAIL'})", flush=True)

print("\n########## (B) DUPLICATE RECONCILIATION (raw-stale 4,203 vs clean 0) ##########", flush=True)
if STALE_RAW.exists():
    df = pd.read_parquet(STALE_RAW, columns=["variant_id","ref","alt"])
    dup_mask = df["variant_id"].duplicated(keep=False)
    dups = df[dup_mask]
    n_dup_ids = df["variant_id"][dup_mask].nunique()
    print(f"  raw-stale rows in a duplicate variant_id group: {len(dups):,} "
          f"({n_dup_ids:,} distinct ids)", flush=True)
    # decompose: how many of those duplicate ROWS are allele-less?
    def is_empty(x):
        if x is None: return True
        s = str(x).strip().lower()
        return s in ("", "none", "nan", ".", "-", "na")
    empty_ref = dups["ref"].map(is_empty)
    empty_alt = dups["alt"].map(is_empty)
    alleleless = (empty_ref | empty_alt)
    print(f"    of which allele-less (>=1 empty allele -> QUARANTINED by builder): {int(alleleless.sum()):,}", flush=True)
    print(f"    of which fully-alleled (genuine collisions, survive to clean): {int((~alleleless).sum()):,}", flush=True)
    # the genuine ones: do they still collide AFTER quarantine? group by id among fully-alleled
    genuine = dups[~alleleless]
    still_dup = genuine["variant_id"].duplicated(keep=False)
    print(f"    fully-alleled rows still sharing an id after dropping allele-less: {int(still_dup.sum()):,}", flush=True)
    print(f"  INTERPRETATION: the builder quarantines allele-less rows BEFORE the G9 dup check, so", flush=True)
    print(f"  those {int(alleleless.sum()):,} rows never reach clean. If the remaining fully-alleled", flush=True)
    print(f"  collisions are also handled (coordinate-corrected variant_id rebuild can separate", flush=True)
    print(f"  them, or they were <the 0 the audit saw>), clean-stale dup=0 is consistent.", flush=True)
    print(f"  NOTE: builder rebuilds variant_id AFTER pos correction, which can resolve apparent", flush=True)
    print(f"  raw collisions. The authoritative number is the builder's post-quarantine G9=0.", flush=True)
print("=== probe_fresh_labels_and_dupreconcile DONE ===", flush=True)
