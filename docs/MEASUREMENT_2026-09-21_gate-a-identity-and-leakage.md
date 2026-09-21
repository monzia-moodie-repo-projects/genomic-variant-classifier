# MEASUREMENT 2026-09-21: Gate A identity, authentication and leakage

Generated from artifacts by `generate_gate_a_report.py`; no figure or verdict below was typed by hand.
Generated 2026-09-21T14:21:51.376293+00:00.

## Sources

| Artifact | Path | Bytes | SHA-256 |
|---|---|---:|---|
| mc_census | `C:\Projects\genomic-variant-classifier\outputs\mc_census_001.json` | 2,992 | `b49268bcadfc6c45b854c14dddad34e557a75f40d49e10699f9b7f7d60cfe06a` |
| geneinfo_census | `C:\Projects\genomic-variant-classifier\outputs\geneinfo_census_002.json` | 8,687 | `0234740dc202678cd6010054cb890cc265e84f01351410bfa5221b5542f2ce52` |
| provenance | `C:\Projects\genomic-variant-classifier\outputs\release_provenance_001.json` | 3,499 | `6a65e2b0324bbb285f13ae4b6304b0de298172fc25fb700249dffb7d6c650a7e` |
| authentication | `C:\Projects\genomic-variant-classifier\outputs\same_release_authentication_001.json` | 5,740 | `7f8205090ca121038483002c166cf07f00af2567911ce658f490290ab5de1748` |
| residuals | `C:\Projects\genomic-variant-classifier\outputs\residuals_001.json` | 1,385 | `5a86df0d7dce52a038f0013497cd89a1c5e93ab8d0b618cf27f9180aa86e5dd9` |
| resolution | `C:\Projects\genomic-variant-classifier\outputs\gene_id_resolution_002\resolution_summary.json` | 11,224 | `c1b730de6cc1bc802f4416ae7ccc784fea77cc3be97e289d33f2dc1ed6b04b30` |
| components | `C:\Projects\genomic-variant-classifier\outputs\gene_components_001\components_summary.json` | 9,495 | `3c678933f1001f767ce67a96aae7e20ea831999f8dfc6724638a83c365d377d6` |
| bundle | `C:\Projects\genomic-variant-classifier\data\external\identity\2026-09-21T074906Z\BUNDLE_PROVENANCE.json` | 2,050 | `cf9cb20df05b29e2165281540cfd5b722236be020fc20d939be60844cb9f3023` |
| strata_constraint | `C:\Projects\genomic-variant-classifier\outputs\leakage_strata_constraint_002.json` | 29,423 | `9cafb21d82ea69f3ad0204abd3002288b7e45ce094c77773374892161c4215dc` |
| strata_representation | `C:\Projects\genomic-variant-classifier\outputs\leakage_strata_representation_002.json` | 42,458 | `8e350f1f1a94475633d3eba3546f15faed9b3648d0000e3629cbb02d49173e32` |

## 1. Source recovery from the pinned ClinVar VCF

Molecular consequence: 4,397,869 records, 4,378,168 with an MC field, 0 malformed. 906,304 records (20.70% of annotated) list more than one consequence; the former first-term parser discarded 1,074,137 consequence entries. Labels mapping to more than one accession: 0. Unknown accessions: 0; obsolete: 0.

- label differs from SO name: `SO:0001587|nonsense|so_name=stop_gained`
- label differs from SO name: `SO:0001619|non-coding_transcript_variant|so_name=non_coding_transcript_variant`

Gene associations (GENEINFO): 4,396,982 records with gene information, 0 malformed, 475,876 (10.82%) associated with more than one gene; 4,937,323 variant-gene relation rows; 30,593 distinct source GeneIDs. Records at the maximum gene count (10): 3,321 - whether this is a cap is NOT established.

- symbol containing ':' : `HHC2:066588` GeneID 111258505 (8 entries)

## 2. Release provenance

VCF header: ##fileDate=2026-03-15, ##source=ClinVar, ##reference=GRCh38. variant_summary maximum LastEvaluated: 2026-07-03 00:00:00 (a lower bound on its release date). Cohort VariationIDs present in the VCF: 4,396,768 of 4,396,768; absent from variant_summary: 466.

## 3. Same-release authentication, keyed by VariationID

- review_status: agree 4,153,941, vcf_has_no_clnrevstat 245,148 (sum 4,399,089)
- clinical_sig: agree 4,153,554, disagree 387, vcf_has_no_clnsig 245,148 (sum 4,399,089)
- clinical_sig disagreement kinds over all rows: delimiter_only 387
- shared-VariationID groups: 2,321; NOT_xy_pair_with_same_alleles 2, binary_label_eligible_pairs 484, par1_offset_consistent 1,973, par2_offset_consistent 346, same_gene 2,319

## 4. Identity bundle (retrospective harmonisation snapshot)

| Resource | Bytes | SHA-256 | Last-Modified |
|---|---:|---|---|
| Homo_sapiens.gene_info.gz | 5,189,188 | `260f7eb7d3d91250` | Sun, 20 Sep 2026 08:07:11 GMT |
| gene_history.gz | 162,272,692 | `2de3604f6743a11e` | Sun, 20 Sep 2026 07:59:07 GMT |
| hgnc_complete_set.txt | 16,940,274 | `69bb5722d5a42bb3` | Fri, 18 Sep 2026 13:36:42 GMT |

## 5. GeneID resolution

- state_by_distinct_gene_id: current 30,559, migrated 25, discontinued_no_replacement 9
- state_by_relation_rows: current 4,936,828, migrated 259, discontinued_no_replacement 236
- symbol_state_by_relation_rows: symbol_matches_current 4,924,193, symbol_not_current_no_conflict 12,894
- hgnc_crossref_two_way: agree 44,393, gene_info_hgnc_but_hgnc_lacks_entrez 689, disagree 4, hgnc_entrez_absent_from_gene_info 1

## 6. Gene components and the old registry's leakage

Relation rows by NCBI type_of_gene: protein-coding 4,571,184, ncRNA 199,705, biological-region 156,012, other 7,507, pseudo 952, tRNA 764, snRNA 529, UNRESOLVED 236, snoRNA 229, rRNA 106, unknown 99

| Relation | Components | Evaluated validation rows sharing a component with training |
|---|---:|---:|
| all resolved records | 16,914 | 82,253 / 189,729 = 43.35% |
| excluding biological-region | 16,952 | 80,438 / 189,729 = 42.40% |

## 7. Constraint features, split by leakage

Component column: `component_genes_only`; model selection: explicit.

| Stratum | Rows | Components | Prevalence |
|---|---:|---:|---:|
| unseen | 109,291 | 1,405 | 0.1556 |
| leaked | 80,438 | 356 | 0.2545 |
| all | 189,729 | 1,761 | 0.1975 |

**lightgbm__core_plus_constraint - lightgbm__core** (Brier difference; negative favours the first)

| Stratum | Delta | 95% component interval | Verdict | Top-3 share of signed delta | Without top 3 (point) |
|---|---:|---|---|---:|---:|
| unseen | +0.001888 | [-0.000471, +0.004288] | not detectable | omitted: delta not detectable | +0.001594 |
| leaked | +0.011962 | [+0.000688, +0.019521] | detectable | 93.3% | +0.001261 |
| all | +0.006159 | [+0.001112, +0.010619] | detectable | 76.8% | +0.001688 |

Largest leaked-stratum contributors, each component labelled by the registry `gene_symbol` strings of its rows (not resolved genes): TTN (22,537 rows, +0.006450); PKD1;PKD1-AS1/PKD1;TSC2/TSC2 (5,635 rows, +0.002653); SON (905 rows, +0.002054); ALMS1 (3,557 rows, -0.000743); BRCA1 (7,432 rows, -0.000516)

**logistic_regression__core_plus_constraint - logistic_regression__core** (Brier difference; negative favours the first)

| Stratum | Delta | 95% component interval | Verdict | Top-3 share of signed delta | Without top 3 (point) |
|---|---:|---|---|---:|---:|
| unseen | -0.000074 | [-0.000360, +0.000230] | not detectable | omitted: delta not detectable | -0.000246 |
| leaked | -0.000060 | [-0.000939, +0.000589] | not detectable | omitted: delta not detectable | +0.000076 |
| all | -0.000068 | [-0.000440, +0.000235] | not detectable | omitted: delta not detectable | -0.000011 |

Largest leaked-stratum contributors, each component labelled by the registry `gene_symbol` strings of its rows (not resolved genes): GBA1/THBS3 (191 rows, +0.000192); GLA (1,391 rows, -0.000178); FLG (772 rows, -0.000148); TTN (22,537 rows, +0.000098); NSD1 (1,450 rows, -0.000049)

## 8. Representation arms, split by leakage

Component column: `component_genes_only`; model selection: explicit.

| Stratum | Rows | Components | Prevalence |
|---|---:|---:|---:|
| unseen | 109,291 | 1,405 | 0.1556 |
| leaked | 80,438 | 356 | 0.2545 |
| all | 189,729 | 1,761 | 0.1975 |

**lr_representation - lightgbm** (Brier difference; negative favours the first)

| Stratum | Delta | 95% component interval | Verdict | Top-3 share of signed delta | Without top 3 (point) |
|---|---:|---|---|---:|---:|
| unseen | +0.000328 | [+0.000127, +0.000546] | detectable | 12.6% | +0.000293 |
| leaked | +0.000579 | [+0.000364, +0.000883] | detectable | 46.1% | +0.000560 |
| all | +0.000434 | [+0.000280, +0.000594] | detectable | 27.5% | +0.000375 |

Largest leaked-stratum contributors, each component labelled by the registry `gene_symbol` strings of its rows (not resolved genes): TTN (22,537 rows, +0.000099); BRCA1 (7,432 rows, +0.000098); PKD1;PKD1-AS1/PKD1;TSC2/TSC2 (5,635 rows, +0.000069); SON (905 rows, +0.000065); EYS (3,012 rows, +0.000030)

**lr_splines - lightgbm** (Brier difference; negative favours the first)

| Stratum | Delta | 95% component interval | Verdict | Top-3 share of signed delta | Without top 3 (point) |
|---|---:|---|---|---:|---:|
| unseen | +0.000436 | [+0.000217, +0.000674] | detectable | 13.0% | +0.000388 |
| leaked | +0.000824 | [+0.000526, +0.001224] | detectable | 49.9% | +0.000666 |
| all | +0.000601 | [+0.000412, +0.000791] | detectable | 29.1% | +0.000508 |

Largest leaked-stratum contributors, each component labelled by the registry `gene_symbol` strings of its rows (not resolved genes): TTN (22,537 rows, +0.000151); BRCA1 (7,432 rows, +0.000144); FOXG1 (608 rows, +0.000117); ALMS1 (3,557 rows, +0.000077); EYS (3,012 rows, +0.000041)

**lr_representation - lr_current** (Brier difference; negative favours the first)

| Stratum | Delta | 95% component interval | Verdict | Top-3 share of signed delta | Without top 3 (point) |
|---|---:|---|---|---:|---:|
| unseen | -0.021316 | [-0.024645, -0.017577] | detectable | 9.7% | -0.020350 |
| leaked | -0.038263 | [-0.046561, -0.023749] | detectable | 60.4% | -0.025959 |
| all | -0.028501 | [-0.034773, -0.020959] | detectable | 34.4% | -0.022710 |

Largest leaked-stratum contributors, each component labelled by the registry `gene_symbol` strings of its rows (not resolved genes): TTN (22,537 rows, -0.014073); BRCA1 (7,432 rows, -0.006135); ALMS1 (3,557 rows, -0.002915); EYS (3,012 rows, -0.002634); PLEKHH1/RDH11/VTI1B (2,088 rows, -0.001273)

Top-3 shares are shown only where the delta is detectable: a share of a signed delta is unbounded near zero. Leave-out values are point estimates: the components removed were chosen after seeing the data, so no interval is attached.

## Cross-output consistency

Generation refuses unless every check passes. Passed:

- authentication disagreements == residual classifications
- census relation rows == resolution relation rows
- stratified rows == component build's evaluated and leaked validation rows
- constraint and representation runs share component column and stratum sizes

## Appendix: provenance of every figure

Each row is a key path read from an artifact; values nested inside that object are covered by its row.

| Artifact | Key path | Artifact SHA-256 |
|---|---|---|
| authentication | `clinical_sig` | `7f8205090ca12103` |
| authentication | `cohort_rows` | `7f8205090ca12103` |
| authentication | `review_status` | `7f8205090ca12103` |
| bundle | `resources` | `cf9cb20df05b29e2` |
| components | `all_resolved/components` | `3c678933f1001f76` |
| components | `all_resolved/leakage_restricted` | `3c678933f1001f76` |
| components | `exclude_types` | `3c678933f1001f76` |
| components | `genes_only/components` | `3c678933f1001f76` |
| components | `genes_only/leakage_restricted` | `3c678933f1001f76` |
| components | `relation_rows_by_type_of_gene` | `3c678933f1001f76` |
| geneinfo_census | `distinct_gene_ids` | `0234740dc202678c` |
| geneinfo_census | `genes_per_record` | `0234740dc202678c` |
| geneinfo_census | `provenance/relation_rows` | `0234740dc202678c` |
| geneinfo_census | `records_with_geneinfo` | `0234740dc202678c` |
| geneinfo_census | `records_with_malformed_geneinfo` | `0234740dc202678c` |
| geneinfo_census | `symbols_containing_colon` | `0234740dc202678c` |
| mc_census | `labels_with_more_than_one_accession` | `b49268bcadfc6c45` |
| mc_census | `mc_entries_per_record` | `b49268bcadfc6c45` |
| mc_census | `records` | `b49268bcadfc6c45` |
| mc_census | `records_with_malformed_mc` | `b49268bcadfc6c45` |
| mc_census | `records_with_mc` | `b49268bcadfc6c45` |
| mc_census | `so_validation/label_differs_from_so_name` | `b49268bcadfc6c45` |
| mc_census | `so_validation/obsolete_accessions` | `b49268bcadfc6c45` |
| mc_census | `so_validation/unknown_accessions` | `b49268bcadfc6c45` |
| provenance | `cohort_in_vcf` | `6a65e2b0324bbb28` |
| provenance | `cohort_in_vcf_not_vs` | `6a65e2b0324bbb28` |
| provenance | `cohort_source_ids` | `6a65e2b0324bbb28` |
| provenance | `vcf_meta` | `6a65e2b0324bbb28` |
| provenance | `vs_last_evaluated_max` | `6a65e2b0324bbb28` |
| residuals | `clinical_sig_disagreement_kinds` | `5a86df0d7dce52a0` |
| residuals | `shared_id_checks` | `5a86df0d7dce52a0` |
| residuals | `shared_id_ids` | `5a86df0d7dce52a0` |
| resolution | `hgnc_crossref_two_way` | `c1b730de6cc1bc80` |
| resolution | `state_by_distinct_gene_id` | `c1b730de6cc1bc80` |
| resolution | `state_by_relation_rows` | `c1b730de6cc1bc80` |
| resolution | `symbol_state_by_relation_rows` | `c1b730de6cc1bc80` |
| strata_constraint | `component_column` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/all` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/all/contrasts` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/all/contrasts/lightgbm__core_plus_constraint - lightgbm__core` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/all/contrasts/logistic_regression__core_plus_constraint - logistic_regression__core` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/all/rows` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/leaked` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/leaked/contrasts/lightgbm__core_plus_constraint - lightgbm__core` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/leaked/contrasts/logistic_regression__core_plus_constraint - logistic_regression__core` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/leaked/rows` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/unseen` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/unseen/contrasts/lightgbm__core_plus_constraint - lightgbm__core` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/unseen/contrasts/logistic_regression__core_plus_constraint - logistic_regression__core` | `9cafb21d82ea69f3` |
| strata_constraint | `strata/unseen/rows` | `9cafb21d82ea69f3` |
| strata_representation | `component_column` | `8e350f1f1a944756` |
| strata_representation | `strata/all` | `8e350f1f1a944756` |
| strata_representation | `strata/all/contrasts` | `8e350f1f1a944756` |
| strata_representation | `strata/all/contrasts/lr_representation - lightgbm` | `8e350f1f1a944756` |
| strata_representation | `strata/all/contrasts/lr_representation - lr_current` | `8e350f1f1a944756` |
| strata_representation | `strata/all/contrasts/lr_splines - lightgbm` | `8e350f1f1a944756` |
| strata_representation | `strata/all/rows` | `8e350f1f1a944756` |
| strata_representation | `strata/leaked` | `8e350f1f1a944756` |
| strata_representation | `strata/leaked/contrasts/lr_representation - lightgbm` | `8e350f1f1a944756` |
| strata_representation | `strata/leaked/contrasts/lr_representation - lr_current` | `8e350f1f1a944756` |
| strata_representation | `strata/leaked/contrasts/lr_splines - lightgbm` | `8e350f1f1a944756` |
| strata_representation | `strata/leaked/rows` | `8e350f1f1a944756` |
| strata_representation | `strata/unseen` | `8e350f1f1a944756` |
| strata_representation | `strata/unseen/contrasts/lr_representation - lightgbm` | `8e350f1f1a944756` |
| strata_representation | `strata/unseen/contrasts/lr_representation - lr_current` | `8e350f1f1a944756` |
| strata_representation | `strata/unseen/contrasts/lr_splines - lightgbm` | `8e350f1f1a944756` |
| strata_representation | `strata/unseen/rows` | `8e350f1f1a944756` |
