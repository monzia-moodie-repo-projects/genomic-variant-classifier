# RUN 16 — Results & Model Comparison

**Date:** 2026-06-13 | **Instance:** Vast.ai RTX 4090 (40728494) | **Runtime:** 49,893.9 s (13.9 h) | **Cost:** ~\$6.9
**Entrypoint:** `scripts/train.py` (run16 contract) | **HEAD at launch:** b42dfdb | **Best model:** ENSEMBLE_STACKER, test AUROC 0.99835

First full-cohort run with **real ESM-2 650M** (delta + LLR) — the capability Run 14 lacked — plus the complete 13-model ensemble and stacking meta-learner.

---

## Cohort & splits

- **1,490,014 variants** after label + review(<=3) filtering: 210,549 pathogenic / 1,279,465 benign (14.1% prevalence).
- ~201k are missense (where sequence predictors are real); the remaining ~1.29M rely on gene/frequency features.
- **81-column feature matrix; 46 live (varying), 35 constant** (see `PHASE_2_FEATURES.md`).
- **Gene-disjoint split confirmed:** 12,385 train genes, 3,539 test genes, **0 overlap (0.00%)**. The test set is genes the model never saw.

## Ensemble result (held-out gene-disjoint test)

| metric | value |
|---|---|
| AUROC | **0.99835**  [95% CI 0.99816–0.99852] |
| AUPRC | 0.99358 |
| ECE (mean calibration error) | 0.00536 |
| MCE (max calibration error) | 0.11818 |

Calibration is excellent on average (ECE 0.005) but has one weak bin (MCE 0.118) — likely a sparse high-confidence decile; flagged for a per-bin look.

## Per-model comparison (test set)

| model | auroc | auprc | f1_macro | f1_weighted | mcc | brier | OOF auroc | blend wt |
|---|---|---|---|---|---|---|---|---|
| ENSEMBLE_STACKER | 0.9983 | 0.9936 | 0.9821 | 0.9910 | 0.9641 | 0.0071 | 0.9990 (blend) | — |
| catboost | 0.9983 | 0.9935 | 0.9783 | 0.9890 | 0.9569 | 0.0090 | 0.9989 | **0.4127** |
| xgboost | 0.9982 | 0.9923 | 0.9819 | 0.9909 | 0.9638 | 0.0071 | 0.9990 | 0.1018 |
| lightgbm | 0.9982 | 0.9925 | 0.9821 | 0.9910 | 0.9641 | 0.0071 | 0.9990 | 0.1097 |
| deep_ensemble | 0.9981 | 0.9924 | 0.9812 | 0.9906 | 0.9625 | 0.0074 | 0.9986 | 0.0496 |
| tabular_nn | 0.9980 | 0.9921 | 0.9801 | 0.9901 | 0.9603 | 0.0077 | 0.9985 | 0.0790 |
| mc_dropout | 0.9980 | 0.9921 | 0.9801 | 0.9900 | 0.9602 | 0.0077 | 0.9985 | 0.0000 |
| gradient_boosting | 0.9978 | 0.9924 | 0.9805 | 0.9902 | 0.9609 | 0.0076 | 0.9983 | 0.1032 |
| random_forest | 0.9976 | 0.9913 | 0.9813 | 0.9906 | 0.9626 | 0.0074 | 0.9984 | 0.1243 |
| svm | 0.9972 | 0.9878 | 0.9786 | 0.9893 | 0.9572 | 0.0087 | 0.9979 | 0.0000 |
| svm_bagged_rbf | 0.9966 | 0.9855 | 0.9667 | 0.9832 | 0.9336 | 0.0113 | 0.9973 | 0.0014 |
| logistic_regression | 0.9962 | 0.9837 | 0.9690 | 0.9842 | 0.9388 | 0.0137 | 0.9969 | 0.0003 |
| kan | 0.9962 | 0.9864 | 0.9714 | 0.9858 | 0.9430 | 0.0111 | 0.9969 | 0.0130 |
| cnn_1d | **0.8219** | **0.5670** | 0.7161 | 0.8705 | 0.4616 | 0.0909 | 0.8536 | 0.0050 |

### Comparison findings

- **The ensemble barely beats its best member.** Stacker test AUROC 0.9983 == catboost 0.9983; AUPRC 0.9936 vs 0.9935. The stack edges ahead only on thresholded metrics (MCC 0.9641 vs 0.9569; f1_macro 0.9821 vs 0.9783). On this data a single tuned CatBoost essentially matches the 13-model stack on discrimination; stacking buys a little calibration/F1. The OOF blend (0.9990) also barely beat its LR stacker (0.9986, delta 0.0004).
- **Blend leans on CatBoost (0.41)** plus the other tree models; the optimizer zeroed out `svm` and `mc_dropout` and near-zeroed `logistic_regression`, correctly down-weighting redundant/weak members.
- **cnn_1d is the clear failure:** test AUROC 0.8219, AUPRC **0.5670**, MCC 0.4616, Brier 0.0909 — an order of magnitude worse than every other model on AUPRC/calibration. The 1D-CNN architecture is a poor fit for this unordered tabular feature vector. It stays in the ensemble (weight 0.005) per the "all models stay" policy; needs an input-representation review (PHASE_2).
- **KAN** trained successfully on CUDA (subsampling 706k->100k/fold) at 0.9962 — comparable to logistic regression; the newer architecture is measured and competitive but not leading here.

## Feature importance (PHASE 5, tree models, n_models=2 — noisy)

| rank | feature | importance | note |
|---|---|---|---|
| 1 | n_tools_pathogenic | 12.55 | dbNSFP predictor consensus (variant-level) |
| 2 | cadd_phred | 6.57 | variant-level |
| 3 | consequence_severity | 6.55 | engineered, variant-level |
| 4 | is_snv | 4.69 | variant-level |
| 5 | n_pathogenic_in_gene | 3.65 | **gene-level** |
| 6 | splice_ai_score | 2.01 | variant-level |
| 7 | is_synonymous | 1.81 | variant-level |
| 8 | gene_has_known_disease | 1.71 | **gene-level** |
| 9 | af_log10 | 1.26 | gnomAD frequency |
| 10 | is_loss_of_function | 1.19 | variant-level |

Importance std exceeds the mean (only 2 models expose importances), so treat the magnitudes as indicative, not precise.

## Leakage / validity assessment

1. **Gene memorization: ruled out.** Train/test gene sets are disjoint (0 overlap), so high AUROC is not the model re-seeing trained genes.
2. **Gene-prevalence dominance: not supported.** Gene-level features (`n_pathogenic_in_gene` #5, `gene_has_known_disease` #8) are secondary to per-variant predictor consensus in importance; the C3 permutation (prior run: observed 0.9666 vs permuted-p95 0.8016) showed `n_pathogenic_in_gene` carries genuine variant-level signal.
3. **Residual caveat — predictor/label circularity (NOT eliminated):** `n_tools_pathogenic`, CADD, REVEL, and AlphaMissense are themselves partly trained on ClinVar, so AUROC against ClinVar labels is inflated relative to an independent label source. State this as a limitation; an independent-label holdout (e.g., a curated non-ClinVar test set) is the proper future check.

## Gaps & known issues (this run)

- **`consequence_breakdown` and `gene_errors` are EMPTY in `eval_report.json`** — `train.py` did not pass `meta=meta_test` to `ClinicalEvaluator.evaluate()`, so the missense-vs-other and gene-error analyses never computed. **Fix next run** (one-line: pass `meta_test`). Missense stratification recovered post-hoc from saved OOF arrays.
- **35/81 features constant** — known data-acquisition/deferred backlog (`PHASE_2_FEATURES.md`); harmless but should be dropped from the matrix in future runs.
- **PHASE 4 eval ran ~2.5 h** — the neural models appear to be predicted on the test set twice (comparison table, then ensemble report); cache test predictions to roughly halve it. `n_bootstrap=1000` across 14 models is the other cost driver (compare() uses 500).
- **ESM-2:** 75 gene symbols absent from the UniProt index (248 delta + 2,520 LLR variants scored 0.0) — minor; widen the UniProt index next run.
- **cnn_1d** weak learner (see above).
- Cosmetic: Reactome warning string uses an em-dash that mojibakes (`ΓÇö`) in PowerShell — switch to ASCII `--` in `reactome.py`.

## Metrics glossary

- **AUROC** — area under the ROC curve; P(random pathogenic ranked above random benign). Range 0.5 (chance) to 1.0. Reported with 95% bootstrap CI (n=1000 resamples of the test set).
- **AUPRC** — area under the precision-recall curve; more informative than AUROC under class imbalance (here 14% prevalence; chance ~0.14).
- **F1 (macro/weighted)** — harmonic mean of precision and recall at the 0.5 threshold; macro averages classes equally, weighted by support.
- **MCC** — Matthews correlation coefficient; balanced even under imbalance, range -1 to 1.
- **Brier** — mean squared error of predicted probabilities; lower is better (0 = perfect).
- **ECE / MCE** — expected / maximum calibration error across 10 quantile bins; mean vs worst-bin gap between predicted and observed frequency.
- **OOF AUROC** — out-of-fold AUROC during cross-validated training (generalization estimate before the held-out test).
- **Blend weight** — Nelder-Mead-optimized weight of each base model in the convex blend; 0 = excluded.

## Artifacts (`outputs/run16/`)

`metrics_v1.json`, `METRICS.md`, `eval_report.json` (curves; breakdowns empty — see gap), `feature_importance.csv`, `ensemble_v1.joblib` (94 MB), `ensemble_v1_models/` (per-model), `catboost_model.cbm`, per-model `*.joblib` + `*_oof.npy` + `*_oof_indices.npy`, `splits/`, `val.parquet`. Master log: `logs/training/run16_master.log`.
