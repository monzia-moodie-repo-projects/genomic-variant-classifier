# Run 14 Observability Report
_Generated 2026-05-26T14:25:41.146790+00:00 on 3392d83e3f9c_

**Instance:** 37897784  
**Commit:** `80ac62ca7e83d35638274a01170d4c8f4f62c418
main
80ac62c docs(run14): plan, results template, session log, reference card; observability + preflight/postflight scripts`  
**Master log:** `/workspace/run11_master.log`  
**Outputs dir:** `/workspace/outputs/run11/full`

---
## Wall-clock + cost
- First timestamp: `2026-05-26 10:38:56`
- Last timestamp:  `2026-05-26 13:53:31`
- Elapsed: **3.24 h** (11675 s)
- Estimated cost: **$2.17** at $0.669/hr

## Per-model metrics

| Model | OOF AUROC | Test AUROC | AUPRC | F1 | MCC | Brier | Train time |
|---|---|---|---|---|---|---|---|
| random_forest | — | — | — | — | — | — | — |
| xgboost | — | — | — | — | — | — | — |
| lightgbm | — | — | — | — | — | — | — |
| gradient_boosting | — | — | — | — | — | — | — |
| logistic_regression | — | — | — | — | — | — | — |
| catboost | — | — | — | — | — | — | — |
| tabular_nn | — | — | — | — | — | — | — |
| mc_dropout | — | — | — | — | — | — | — |
| deep_ensemble | — | — | — | — | — | — | — |
| kan | — | — | — | — | — | — | — |
| cnn_1d | — | — | — | — | — | — | — |

## KAN status
- Package patch applied: **True** (fixed 3 bare-name refs)
- Backend used: **None**
- Fit succeeded: **None**

## LightGBM status
- Device: **None**
- CUDA attempted: False
- Fit succeeded: **None**

## CNN_1D status
- Skipped: **True** (no fasta_seq data)

## Feature non-zero rate (signal coverage)
- X_train shape: 1,197,216 x 78
- Dead features (non-zero rate < 0.001): **34**
  - `phylop_score`, `eve_score`, `gene_constraint_oe`, `gene_is_constrained`, `has_uniprot_annotation`, `n_known_pathogenic_protein_variants`, `gtex_is_eqtl`, `gtex_min_eqtl_pval`, `gtex_max_abs_effect`, `codon_position`, `dbsnp_af`, `omim_n_diseases`, `omim_is_autosomal_dominant`, `clingen_validity_score`, `hgmd_is_disease_mutation`, `hgmd_n_reports`, `gnn_score`, `maxentscan_score`, `dist_to_splice_site`, `exon_number`, `is_canonical_splice`, `alphafold_plddt`, `solvent_accessibility`, `secondary_structure_context`, `dist_to_active_site`, `af_1kg_afr`, `af_1kg_eur`, `af_1kg_eas`, `af_1kg_sas`, `af_1kg_amr`, `finngen_af_fin`, `finngen_af_nfsee`, `finngen_enrichment`, `esm2_delta_norm`
- Top 10 populated features:
  - `af_raw`: 1.000
  - `af_log10`: 1.000
  - `af_is_absent`: 1.000
  - `af_is_ultra_rare`: 1.000
  - `af_is_rare`: 1.000
  - `af_is_common`: 1.000
  - `ref_len`: 1.000
  - `alt_len`: 1.000
  - `len_diff`: 1.000
  - `is_snv`: 1.000

## Blend weights
- Not available

## Recent errors / warnings
- No errors detected.

## Artifact inventory
- Total size: **1837.8 MB** across 73 files
- Largest 10:
  - `models/ensemble_models/random_forest.joblib` (1105.996 MB)
  - `models/random_forest.joblib` (289.676 MB)
  - `models/ensemble.joblib` (85.405 MB)
  - `oof_predictions.parquet` (64.072 MB)
  - `splits/meta_train.parquet` (58.506 MB)
  - `splits/X_train.parquet` (22.209 MB)
  - `meta_test.parquet` (16.686 MB)
  - `splits/meta_test.parquet` (16.686 MB)
  - `models/random_forest_oof.npy` (7.764 MB)
  - `models/random_forest_oof_indices.npy` (7.764 MB)

## Host + environment
- **hostname**: `3392d83e3f9c`
- **uname**: `Linux 3392d83e3f9c 5.15.0-131-generic #141-Ubuntu SMP Fri Jan 10 21:18:28 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux`
- **python_version**: `3.11.10`
- **cuda_devices**: `NVIDIA GeForce RTX 4090, 24564 MiB, 570.86.16`
- **disk_workspace**: `overlay         100G  6.9G   94G   7% /`
- **pip_versions**: `catboost                   1.2.10
imodelsx                   1.0.13
lightgbm                   4.6.0
networkx                   3.4.2
numpy                      2.4.4
pandas                     2.3.3
pykan                      0.2.8
scikit-learn               1.8.0
torch                      2.5.1+cu124
xgboost                    3.2.0`
