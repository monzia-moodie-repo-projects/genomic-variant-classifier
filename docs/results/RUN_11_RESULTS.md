# Run 11 Results Summary
**Date:** 2026-05-25 | **Commit:** 7d91386 (training), 4e819e7 (post-run fixes)
**Instance:** Vast.ai 37687380 (RTX 4090, Norway, $0.70/hr, DLP 16.0)
**Cost:** ~$5.60 (7.9h) | **Elapsed:** 28,531s

## Test Metrics (locked holdout, n=349,067)
| Metric | Value |
|--------|-------|
| AUROC  | 0.9974 |
| AUPRC  | 0.9911 |
| F1     | 0.9776 |
| MCC    | 0.9553 |
| Brier  | 0.0134 |

## Per-Model Test AUROC
| Model | AUROC | Notes |
|-------|-------|-------|
| catboost | 0.9975 | GPU (CUDA) |
| ENSEMBLE_STACKER | 0.9974 | LR meta-learner |
| xgboost | 0.9974 | GPU (CUDA) |
| deep_ensemble | 0.9973 | |
| tabular_nn | 0.9971 | |
| mc_dropout | 0.9971 | |
| gradient_boosting | 0.9968 | CPU (sklearn) |
| kan | 0.9968 | MLP fallback (SR31 violation) |
| random_forest | 0.9964 | |
| logistic_regression | 0.9942 | |
| cnn_1d | 0.5000 | BROKEN - random chance |
| lightgbm | -- | SKIPPED (OpenCL not found) |

## Key Findings
1. **First locked test AUROC produced** (Run 9 PicklingError prevented this)
2. **gnomAD constraint recovered:** 1,600,252/1,700,687 variants (94.1%) matched
3. **Feature importance rebalanced:** n_pathogenic_in_gene dropped from #1 (Run 9) to #3; loeuf entered at #10
4. **OOF index sidecars confirmed working** (all 11 models have *_oof_indices.npy)

## Issues for Run 12
1. **CNN_1D AUROC 0.5000** - produces constant predictions, needs root cause
2. **LightGBM skipped** - device_type "gpu" needs OpenCL; fixed to "cuda" in 4e819e7
3. **FastKAN not active** - requirements.txt omission; fixed in 4e819e7 (Standing Rule 31)
4. **Instance selection** - DLP 16.0 / PCIE 0.7 caused slow RF (3.5h); add dlp>=80 pcie_bw>=12 filter

## Annotation Signal Summary (17/17)
| Source | Variants with signal |
|--------|---------------------|
| gnomAD constraint (NEW) | 1,600,252 (94.1%) |
| AlphaMissense | 206,131 |
| DbNSFP (SIFT) | 204,384 |
| SpliceAI | 148,378 |
| LOVD | 369 |
| 12 other sources | 0 (stubs/no data) |
