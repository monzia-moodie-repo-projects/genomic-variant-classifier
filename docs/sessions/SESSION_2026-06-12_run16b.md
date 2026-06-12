# SESSION 2026-06-12 -- Run-16b smoke gate, schema re-seal, source finalization

HEAD entering: 4da4219 (schema baseline re-seal 78->81).

## Arc 1 -- Run-16b all-models smoke CLEARED
Added gnomAD-AF, dbNSFP, LOVD to the proven flag set; full --fast smoke
(models/smoke_run16b, 962s, no OOM, no crash). 13 base models, 81 features,
ENSEMBLE_STACKER test AUROC 0.9994 (up from 0.9934 without the new sources). Feature
matrix population verified in splits/X_*.parquet: af_log10, cadd_phred, sift_score,
revel_score, n_tools_pathogenic POPULATED; LOVD all-default (smoke-size; verify >0 at
full scale).

## Arc 2 -- Connector audits corrected three wrong source picks
- --lovd-path -> data/external/lovd/lovd_all_variants.parquet (train.py-canonical), NOT
  the greedy-glob am_lovd_genes.parquet (an AlphaMissense artifact).
- --uniprot OMITTED: _join_uniprot reads source_id + pathogenicity; the only on-disk
  uniprot parquet has gene_symbol/uniprot_id/sequence -> KeyError / silent-dead.
- dbNSFP: connector hard-codes its cache name to dbnsfp_clinvar_index.parquet (2.69M);
  the 895 MB dbnsfp_full_index.parquet is never read. Docstring drift (said full_index)
  fixed by patch_dbnsfp_docstring.py. OOM avoided by using the ClinVar index directly.

## Arc 3 -- Schema baseline re-sealed 78 -> 81 (run16b-smoke)
Pre-seal probe confirmed all 3 splits share an identical 81-col float64 schema; new cols
esm2_llr (live), maxentscan_delta + reactome_pathway_count (sealed dormant). Sealed from
the smoke X_train; green vs all 3 splits. Authoritative gate remains the full-regen
schema drift-check on Vast.ai.

## Arc 4 -- Feature-population gate (built + hardened)
audit_smoke_feature_population.py: v1 mis-targeted the pre-scoring checkpoint
(clinvar_enriched.parquet, 1931 rows) and checked raw connector names -> false FAIL.
Corrected to read splits/X_*.parquet + engineered names. Noted: the per-source
default-check is unsound on standard-scaled splits; the all-constant scan is the reliable
detector (36/81 columns constant: known stubs + gnn_score + af_1kg_* + uniprot + lovd).

## Arc 5 -- 1KGP + GNN investigated -> committed Run-17 scope (NOT deferred)
- No --kg-path flag in train.py; ThousandGenomesConnector fills only combined allele_freq
  (af_1kg_* never activate via it); no kg parquet staged; build_1kg_parquet.py absent.
- No --string-db flag; GNN (gnn.py: StringDBGraph / VariantGAT / GNNTrainer / GNNScorer)
  is complete but unwired; gnn_score is a df.get placeholder. Live gnn_score requires
  gene-disjoint cross-fitting to avoid label leakage.
- Both committed to docs/roadmap/RUN17_SCOPE.md with hard acceptance criteria.

## Arc 6 -- Launch contract v2 + tree hygiene
docs/launch/LAUNCH_CONTRACT_run16.md v2 (validated flag set, ship/do-not-ship manifest,
on-box blocking gates, dormant-by-design watch-items). Quarantined stale
clinvar_grch38_clean_seq (1).parquet (18-col, no ReviewStatus). dbNSFP docstring fixed;
redundant promoted dbnsfp_full_index.parquet removed.

## Tools delivered this session
audit_run16_data_sources.py, prep_dbnsfp_cache.py, audit_smoke_feature_population.py,
verify_schema_seal_inputs.py, locate_1kg.py, patch_dbnsfp_docstring.py.

## Watch-items carried to Run 16 (full scale)
- cnn_1d test 0.4782 -- at/below random across 2 smokes; full scale discriminates
  architecture-defect vs data-starvation.
- kan Brier 0.2223 (poor calibration; ranks fine for the stacker).
- CIRCULARITY: cadd_phred is #1 importance; CADD/REVEL/SIFT/AlphaMissense are
  ClinVar-trained. Run a no_meta_predictors ablation; document in the metrics glossary.
- LOVD: expect lovd_variant_class > 0 at full scale (else a join-key bug, not coverage).
- real_data_prep.py:501 FutureWarning (gnomAD fillna downcast) -- tech debt.
- Dormant-by-design (NOT bugs): gnn_score, af_1kg_*, uniprot features. Activate in Run 17.
