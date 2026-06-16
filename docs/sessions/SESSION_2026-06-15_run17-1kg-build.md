# Session 2026-06-15 -- Run 17 prep: 1000G af_1kg build, Reactome fix, --gnn-epochs cap

## Summary
Activated `af_1kg_*` by building the 1000G per-super-population AF parquet end to end, hardened the
Reactome GMT ingest, and added a `--gnn-epochs` cap so the upcoming full-flag laptop smoke runs in minutes.
chrY/MT established as a structural 1000G data-availability limit (not a pipeline gap).

## 1000G af_1kg parquet -- BUILT + ACTIVE
- Output: `data/external/1kgp/kg_grch38_af.parquet` -- 23 shards merged, **437,668 unique variants**, 6.7 MB.
- Source: 1000G high-coverage 30x phased panel `20220422_3202_phased_SNV_INDEL_SV`.
- Method: `build_1kg_parquet.py` per-chromosome, streamed (urllib, timeout-bounded) + cohort-filtered to
  the 4,415,963 unique cohort keys (`data/processed/clinvar_grch38.parquet`), chunked ParquetWriter, then
  `merge_1kg_shards.py` (variant_id dedup + all-zero coverage gate).
- Autosomes chr1-22: 426,358 variants. chrX: 11,310 (the `.v2` panel; `AF_<POP>` floats + `AC_Hemi_*`
  hemizygous counts -> male ploidy handled correctly).
- Coverage ~9.9% of the 4.42M cohort. The ~90% absent are rare/private to 1000G; `af_1kg=0` for them is
  honest absence, NOT a dead feature. All 5 super-pops non-zero: AFR 291432 / EUR 205292 / EAS 154084 /
  SAS 188461 / AMR 251739.
- `ThousandGenomesConnector.fill_population_af` join verified in-sandbox on a parquet shaped like the build
  (bare `chrom:pos:ref:alt` key both sides, `^chr` stripped; absent variants stay 0.0).

## chrY / chrM -- structural 1000G limit (verified, not a gap to "fix")
- The 1000G high-coverage phased panel covers autosomes + chromosome X ONLY (Byrska-Bishop et al. 2022,
  Cell: 73,452,337 SNV/INDEL across autosomes and X). The chrY URL returns **HTTP 404** -- no Y file exists
  in this release. No MT either.
- Cohort has **3,191 chrY + 3,124 chrMT** variants. `af_1kg_*` is structurally 0 for them -- 1000G has no
  Y/MT data to ingest. These variants ARE in the run (full feature vector minus af_1kg; `chrom` is a
  CatBoost categorical so the model distinguishes them).
- gnomAD coverage of Y/MT is UNDER AUDIT: the project gnomAD source is `gnomad_v4_exomes.parquet`
  (EXOMES), which excludes MT entirely and covers Y exonic regions only. The gnomAD connector emits only
  global `allele_freq` (no per-ancestry AF). Decision on adding a Y/MT frequency source pending the audit.

## --gnn-epochs cap (ab81e3a)
- `run_phase2_eval` gains `--gnn-epochs` (default 100 == real-run value -> launch byte-identical), threaded
  through the GNN log line + main GNN + hetero-GNN. `smoke_all_models` forwards it only when set via the
  extracted pure `_build_eval_cmd` helper. Full-flag laptop smoke can run ~10 epochs vs 100. +5 tests.

## parse_gmt hardening + Reactome (ad18419)
- `parse_gmt` now rejects ZIP/NUL-binary inputs (the .gmt URL serves the raw zip) and transparently
  gunzips .gz; the prior errors='replace' had turned the zip into 233 junk pathways + 322 garbage edges
  that printed OK. Reactome GMT re-downloaded from the .zip + Expand-Archive: 2,855 pathways, 1,221,796
  co-membership edges, real human symbols. `--kg-edges reactome:...` now activatable.

## Durability
- `rclone genvarcla:` (parquet + per-chr shards). Parquet also committed (26342e9, force-add past the
  `data/` gitignore) for reproducibility.

## Suite
- 1239 passed / 7 skipped (full local run, 202s).

## Open / next
- AUDIT gnomAD parquet for Y/MT coverage + per-ancestry AF columns (decides whether to add a Y/MT
  frequency source or document the limit).
- No-defer item 2: extend `preflight_run17.py` emit/--check to include `--hetero-gnn` + `--kg-edges`.
- No-defer item 3: full-flag laptop smoke (`--kg` + `--hetero-gnn` + `--kg-edges reactome:...` +
  `--gnn-epochs 10` + dbNSFP/gnomAD/LOVD) -> confirm af_1kg_*, hetero_gnn_score, reactome edges all
  populate non-zero together. Then GPU full-cohort smoke -> Run 17 launch.
