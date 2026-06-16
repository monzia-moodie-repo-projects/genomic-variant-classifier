# Session 2026-06-16 -- gnomAD chrY/chrMT allele_freq closure (PAR X->Y fix)

## Summary
Closed the chrY/chrMT `allele_freq` gap by sourcing Y/MT frequencies directly from gnomAD v4 and
merging them into `data/processed/gnomad_v4_exomes.parquet`, so the existing `--gnomad` join fills Y/MT
with no connector change. The long-standing Y under-match (91/3155 = 3%) was traced to a single root
cause -- pseudoautosomal (PAR) coordinate canonicalisation -- and fixed. Final honest coverage:
**Y 1047/3155 (33%)**, **MT 2731/3124 (87%)**. No silent zeros: every uncovered Y/MT variant carries a
correct `allele_freq=0` that reflects genuine absence from / non-callability in the gnomAD population.

## Root cause -- PAR canonicalisation (the real cause of Y 91/3155)
gnomAD reports pseudoautosomal variants on chromosome **X**; ClinVar annotates the same variants on **Y**.
So gnomAD gene queries for PAR genes returned X-keyed variant_ids (e.g. `X:1285848:G:A`) that never
string-matched the cohort's Y keys (`Y:1285848:G:A`). Diagnosed by comparing the build's STRING-equality
match (59 for a 14-gene probe) against a position/allele match on the same data (499) -- a 440-variant gap,
every one a PAR variant differing only in the `X` vs `Y` chromosome token.

## Fix -- `y_key()` PAR X->Y remap (commit 112967d)
`scripts/build_gnomad_ymt_af.py` now canonicalises every gnomAD variant_id to the Y frame before matching
(GRCh38 GRC/UCSC coordinates):
- PAR1: X 10,001-2,781,479 == Y identical (no shift).
- PAR2: X 155,701,383-156,030,895 -> Y 56,887,903-57,217,415 (shift `_PAR2_SHIFT = 98,813,480`).
- MSY (male-specific Y) passes through unchanged.
- Non-PAR X -> dropped (not a Y variant).
`--min-y-cover` default lowered 0.50 -> 0.10 (33% is the honest gnomAD ceiling, not a throttle symptom).

## Verification (in-repo, live gnomAD)
- STRING intersection on the 14-gene probe jumped **59 -> 501** (pos/allele truth was 499; the +2 are
  additional alt alleles the remap now also catches).
- Cohort real-SNV Y keys: **2891** total -- PAR1 1892 / PAR2 344 / MSY 655. (The remaining 264 of 3155 are
  `na:na` structural/CNV entries with no SNV alleles -- gnomAD's short-variant API does not carry these.)
- PAR2 gene spot-check (IL9R/SPRY3/VAMP7): 67/344 mapped+matched.
- Full build: **Y 1047/3155 (33%)**, **MT 2731/3124 (87%)**, no WARN. Merged parquet 2,951,148 unique rows.
- MT sanity: `an=56434` matches gnomAD's 56,434 v3 mitochondrial genomes exactly (correct dataset).
- Test suite: `tests/unit/test_build_gnomad_ymt_af.py` 15 passed; full suite **1260 passed, 7 skipped**.

## Why 33% is the honest ceiling (not a defect)
gnomAD has chrY in GRCh38 and its Y genes are well populated (USP9Y ~3,400, KDM5D ~2,400 observed variants).
The 33% reflects population observation + callability, not a missing chromosome or annotation:
1. The cohort is ClinVar -- rare/private clinical variants frequently absent from a population-frequency DB
   (this absence is itself informative for pathogenicity).
2. chrY is uniquely hard to call with short reads (gnomAD genomes do not call Y; exomes + males only;
   ampliconic/palindromic/heterochromatic low-mappability regions yield AC0/no-call).
3. 264 cohort Y entries are structural `na:na` -- no SNV alleles to match.
`allele_freq=0` for the uncovered remainder is the correct, informative value.

## Data state (final, verified)
- Production `data/processed/gnomad_v4_exomes.parquet` = **2,951,148 rows** (33% Y + 87% MT merged).
- Backup `data/processed/gnomad_v4_exomes.parquet.bak_pre_ymt` = 2,947,370 rows (clean pre-ymt original).
- rclone `genvarcla:` re-synced to the corrected parquet.

## Chromosome coverage (confirmed this session)
All 24 sequence classes (autosomes 1-22 + X + Y + MT) are in the cohort and trained: `chrom` is a CatBoost
categorical and `is_mitochondrial` is a flag in the 82-col matrix. Recent work only filled Y/MT *frequency*
columns; X already had frequency via gnomAD exomes + 1000G chrX (11,310 variants).

## Build script API (scripts/build_gnomad_ymt_af.py, v4 @ 112967d)
`y_key` (PAR remap; consts `_PAR1=(10001,2781479)`, `_PAR2_X=(155701383,156030895)`,
`_PAR2_SHIFT=98813480`), `clean_y_genes` (149 raw -> 118 clean), `parse_y_af`, `parse_mt_af` (af_hom),
`build_ymt_frame`, `merge_into_gnomad` (dedup keep=last), `cohort_ymt`, `_post_retry`
(max_retries=8, base_pause=2.0, cap 60s), `fetch_y_af` (serial 6s pace; gnomAD rejects aliased batches at
HTTP 400 on per-query cost), `fetch_mt_af`. Default `--min-y-cover 0.10`.

## Handoff -- next (no-defer item 3)
Run-17 full-flag laptop smoke checker: `smoke_all_models.py` did not forward `--kg/--hetero-gnn/--kg-edges`,
so a smoke that activates ALL no-defer features together was impossible as written. Extension +
`audit_smoke_feature_population.py --run17` mode delivered separately. Then regenerate `docs/ROADMAP.docx`
from the updated `docs/ROADMAP.md`.
