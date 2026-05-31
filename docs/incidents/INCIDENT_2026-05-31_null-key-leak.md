# INCIDENT 2026-05-31 -- Null-key cohort leak (B1)

**Status:** RESOLVED (data layer); one hardening follow-on open
**Severity:** HIGH -- cross-split leakage + label-key collisions in every run through Run 14
**HEAD at resolution:** 18bbba1
**Resolved by:** `scripts/clean_cohort.py` (Phase 0)

---

## Summary

The source cohort `data/processed/clinvar_grch38.parquet` (4,420,180 rows) contained
**19,988 records with null `ref` and null `alt`** plus ~1,103 records whose `ref`/`alt`
were non-allele tokens (`.`, `-`, empty, etc.) -- 21,091 "structural"/allele-less rows
in total -- and **4,203 duplicate `variant_id`s**. Because `variant_id` is built as
`source:chrom:pos:ref:alt`, null/absent alleles produce degenerate keys, and the
duplicate `variant_id`s were entirely concentrated in these allele-less rows.

## Detection

- 2026-05-29 cohort audit: `null_ref=19988`, `null_alt=19988`.
- 2026-05-31 discovery: `dup_vid=4203` measured on the full cohort.
- 2026-05-31 Phase-0 audit reconciliation confirmed the 4,203 duplicates were all in
  the allele-less bucket (0 duplicate `variant_id` remained among the 4,399,089
  non-structural rows).

## Root cause

1. **Ingestion gap:** the database connectors emitted ClinVar records with null/absent
   `ref`/`alt` for ~21k entries (mostly region/structural records; a tail of ~48
   coding/splice variants whose SNV alleles were not captured).
2. **Key collapse on join:** the gnomAD allele-frequency merge in `real_data_prep.py`
   constructs its join key by stringifying alleles (`astype(str)`), which maps a null
   allele to the literal string `"None"`/`"nan"`. Distinct region records then collapse
   onto a shared key, producing cross-record contamination and label-key conflicts, and
   allowing colliding keys to fall on both sides of the train/test split.

## Impact

Runs through Run 14 trained on a cohort in which 19,988 null-key rows and 4,203
duplicate `variant_id`s could place the same or colliding keys across the train/test
boundary. This inflates apparent performance and is one contributor to the
leakage concern around the ~0.9974 test AUROC (alongside gene-prevalence memorization).

## Resolution

`scripts/clean_cohort.py` (introspective, fail-loud, dry-run/apply) operates at the
cohort source:
- Quarantines null/bad-allele rows → `clinvar_grch38_structural.parquet` (21,091).
- De-duplicates `variant_id`, resolving label conflicts by ClinVar review tier and
  quarantining irreducible conflicts → `clinvar_grch38_conflicts.parquet` (0 needed).
- Emits `clinvar_grch38_clean.parquet` (4,399,089 rows).
- Enforces a reconciliation identity (every source row accounted for) and post-conditions
  (0 null/bad key, 0 duplicate `variant_id`), raising on any violation.

**Verification (2026-05-31):**
```
clean 4399089 null 0 dup 0
reconciliation identity holds : True
```

## Residual / follow-on (OPEN)

- **~48 coding/splice variants lost** (24 synonymous, 14 missense, 8 splice-acceptor,
  2 nonsense) due to upstream null alleles -- unscoreable as-is; recovery candidate during
  the ClinVar re-pull (`time_disjoint` task).
- **Harden the gnomAD-join key** in `real_data_prep.py` to be null-safe (or assert no null
  alleles) so the `astype(str)` collapse cannot recur defensively. Source-cleaning removes
  the trigger for this cohort, but the latent code path should still be fixed. *(Exact line
  to confirm against HEAD 18bbba1 before patching.)*
- **Regenerate splits** from `clinvar_grch38_clean.parquet` and repoint the pipeline; the
  current splits derive the ~1.7M labeled subset from the pre-clean cohort.

## Learned

- Never `astype(str)` nullable allele columns when constructing join/identity keys.
- Build `variant_id` only from validated, non-null alleles; assert at construction.
- Quarantine-don't-drop with an explicit reconciliation identity makes data loss auditable
  rather than silent.
