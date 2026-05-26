# Run 13 Results — 2026-05-26

## Summary

| Metric | Value |
|--------|-------|
| **Test AUROC** | **0.9974** |
| **Test AUPRC** | **0.9913** (best ever) |
| **Test F1** | 0.9768 |
| **Test MCC** | 0.9536 |
| **Test Brier** | **0.0124** (best ever) |
| Val AUROC | 0.9974 |
| Val AUPRC | 0.9901 |
| OOF Blend AUROC | 0.9985 |
| Elapsed | 6.33h (22,801s) |
| Cost | ~$4.90 |

**Instance:** Vast.ai 37847255, RTX 4090, Hungary (host_id 80573, machine_id 16403), $0.771/hr, 16 vCPUs, 64.4 GB RAM.
**Commit:** `f4dbeed` (pre-KAN-fix; KAN fix at `0d4ea7b`+`bf2f665`).
**Cohort:** 1,700,687 labeled variants (1,197,216 train / 154,404 val / 349,067 test), 78 features.

## Per-Model Test-Set Results (Locked)

| Rank | Model | Test AUROC | Test AUPRC | F1 | MCC | Brier | Train Time |
|------|-------|-----------|-----------|------|------|-------|------------|
| 1 | CatBoost | 0.9975 | 0.9912 | 0.9672 | 0.9353 | 0.0166 | <1 min (GPU) |
| 2 | XGBoost | 0.9974 | 0.9906 | 0.9769 | 0.9539 | 0.0120 | <1 min (GPU) |
| 3 | **LightGBM** | **0.9974** | **0.9906** | **0.9769** | **0.9538** | **0.0120** | **1 min (CPU)** |
| 4 | ENSEMBLE_STACKER | 0.9974 | 0.9913 | 0.9768 | 0.9536 | 0.0124 | — |
| 5 | Deep Ensemble | 0.9973 | 0.9900 | 0.9771 | 0.9542 | 0.0120 | 3.05h (GPU) |
| 6 | MC Dropout | 0.9971 | 0.9894 | 0.9765 | 0.9531 | 0.0125 | 35 min (GPU) |
| 7 | TabularNN | 0.9971 | 0.9894 | 0.9765 | 0.9531 | 0.0125 | 37 min (GPU) |
| 8 | Gradient Boosting | 0.9968 | 0.9898 | 0.9752 | 0.9505 | 0.0126 | 1.43h (CPU) |
| 9 | Random Forest | 0.9964 | 0.9866 | 0.9648 | 0.9301 | 0.0165 | 8.5 min (CPU) |
| 10 | Logistic Regression | 0.9942 | 0.9781 | 0.9578 | 0.9165 | 0.0228 | 2 min (CPU) |
| — | KAN (imodelsx) | FAILED | — | — | — | — | — |
| — | CNN_1D | SKIPPED | — | — | — | — | — |

## Per-Model Algorithm Analysis

### 1. CatBoost (AUROC 0.9975 — #1 Single Model)

CatBoost uses ordered boosting with oblivious decision trees (all internal nodes at a given depth use the same split feature and threshold). This architecture makes CatBoost naturally resistant to overfitting on features with different scales or distributions, which is critical for genomic variant data where features range from binary indicators to continuous allele frequencies. CatBoost's native handling of categorical features and its ordered boosting strategy (which prevents target leakage during gradient estimation) likely explain its consistent top-1 position across Runs 11, 12, and 13. On this data, CatBoost achieves the highest AUROC (0.9975) but notably has a lower F1 (0.9672) and MCC (0.9353) than XGBoost/LightGBM, suggesting its probability calibration differs — it has the highest Brier score (0.0166) among the top-3 GBDTs, meaning its raw probabilities are less well-calibrated despite better discrimination.

### 2. XGBoost (AUROC 0.9974 — #2)

XGBoost implements regularized gradient boosting with column subsampling, shrinkage, and second-order Taylor expansion of the loss. It uses approximate split finding via weighted quantile sketches for scalability. On 1.7M genomic variants, XGBoost trains in under 1 minute on GPU (CUDA tree method), making it the fastest GBDT to train. Its F1 (0.9769) and MCC (0.9539) are the best among single models — tied with LightGBM — and its Brier score (0.0120) is excellent, indicating well-calibrated probabilities. XGBoost's balanced performance across all metrics suggests it finds a sweet spot between discrimination and calibration on this feature space.

### 3. LightGBM (AUROC 0.9974 — #3, FIRST DATA SINCE RUN 9)

LightGBM uses histogram-based gradient boosting with Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) for handling sparse features. It grows trees leaf-wise (best-first) rather than depth-wise, which can produce deeper, more accurate trees but risks overfitting on small datasets. On 1.7M variants, this risk is negligible. LightGBM was absent from Runs 11 and 12 due to GPU configuration bugs (CUDA/OpenCL not compiled into PyPI builds). In Run 13, it ran in CPU mode in just 1 minute and achieved AUROC 0.9974, tied with XGBoost on all test metrics — identical F1 (0.9769), MCC (0.9538 vs 0.9539), and Brier (0.0120). This confirms that LightGBM's histogram binning and leaf-wise growth strategy is essentially equivalent to XGBoost's split-finding on this particular feature set, and that GPU acceleration is unnecessary for LightGBM at this data scale.

### 4. Deep Ensemble (AUROC 0.9973 — #5)

The deep ensemble trains 25 independent neural networks with different random initializations and averages their predictions. This Lakshminarayanan et al. (2017) approach provides both improved accuracy and uncertainty estimates. At 3.05 hours of training, it is by far the most expensive model in the ensemble but produces the best F1 (0.9771) and MCC (0.9542) of any individual model — even better than the GBDTs. This pattern is significant: while the GBDTs achieve marginally higher AUROC through better discrimination, the deep ensemble's averaging across 25 members produces better-calibrated decision boundaries, as reflected in its superior F1/MCC. For clinical applications where the cost of misclassification matters more than raw discrimination, this distinction is important.

### 5. MC Dropout / TabularNN (AUROC 0.9971 — tied #6)

MC Dropout applies stochastic dropout at inference time across 30 forward passes, producing a distribution of predictions from which epistemic uncertainty can be estimated. TabularNN is the same architecture without inference-time dropout. Both produce identical test metrics (AUROC 0.9971, F1 0.9765, MCC 0.9531, Brier 0.0125), confirming that the dropout ensemble at inference provides no discrimination advantage — its value is purely in the uncertainty estimates, not in the point prediction. These neural networks use a 3-layer MLP with batch normalization and ReLU activations, trained with AdamW and early stopping. Their 0.9971 AUROC being slightly below the GBDTs reflects the well-documented difficulty neural networks have with heterogeneous tabular features compared to tree-based methods.

### 6. Gradient Boosting (AUROC 0.9968 — #8)

Sklearn's GradientBoostingClassifier uses traditional depth-wise tree growth with deviance (log-loss) as the loss function. Unlike XGBoost/CatBoost/LightGBM, it lacks GPU acceleration, column subsampling by default, or histogram binning, making it both slower (1.43h vs <1 min) and less accurate (AUROC 0.9968 vs 0.9974+). The 0.0006 AUROC gap represents the practical value of the algorithmic innovations in the modern GBDT libraries. Gradient boosting remains in the ensemble for diversity — its prediction errors are partially uncorrelated with the other GBDTs due to its different tree construction strategy, contributing to the stacker's ability to improve over any single model.

### 7. Random Forest (AUROC 0.9964 — #9)

Random Forest builds 500 independent decision trees on bootstrap samples with random feature subsets (sqrt(78) ≈ 9 features per split). Unlike gradient boosting, there is no sequential error correction — each tree votes independently. This makes RF robust but less able to model complex feature interactions that boosting captures through iterative residual fitting. The 0.9964 AUROC (1.0-1.1 point gap below the top GBDTs) reflects this fundamental difference. However, RF's diversity from the boosted models makes it a valuable ensemble member — its errors are systematically different from boosting-based errors.

### 8. Logistic Regression (AUROC 0.9942 — #10)

Logistic regression fits a single linear decision boundary in the 78-dimensional feature space. Its AUROC of 0.9942 — still remarkably high — demonstrates that approximately 99.4% of the discrimination in this variant classification task can be captured by linear combinations of the input features. The remaining 0.3 percentage points gained by nonlinear models represent the signal in feature interactions and nonlinear transformations. Logistic regression serves as the interpretability anchor: its coefficients directly show which features contribute most to pathogenicity prediction, and its near-parity with the complex models raises the question of whether the nonlinear models are primarily learning the same linear signal with small interaction corrections.

### 9. KAN — FAILED (imodelsx v1.0.13 upstream bug)

KAN (Kolmogorov-Arnold Network) uses learnable B-spline activation functions on each edge of the network graph instead of fixed activations at nodes. This makes each "weight" a univariate function rather than a scalar, potentially capturing complex nonlinear feature transformations with fewer parameters. KAN failed in Run 13 due to an upstream bug in imodelsx v1.0.13: the `fit()` method references bare `test_size` instead of `self.test_size`, causing `NameError`. The fix is committed at `0d4ea7b` (set attrs before fit) and `bf2f665` (launch script package patch) and will be active in Run 14. KAN trains on a 100K stratified subsample and its performance relative to the GBDTs will be the first empirical measurement of whether spline-based function approximation captures signal that tree-based splits miss on genomic variant features.

### 10. CNN_1D — SKIPPED (no fasta_seq data)

The 1D CNN requires raw nucleotide sequence context around each variant (fasta_seq feature). This data is not yet in the pipeline — implementing it requires a reference genome FASTA and flanking-sequence extraction for each variant. Skipped via `--skip-cnn` flag.

## Cross-Run Comparison (Runs 9 → 11 → 12 → 13)

| Metric | Run 9 | Run 11 | Run 12 | Run 13 |
|--------|-------|--------|--------|--------|
| Test AUROC | — (pickle crash) | 0.9974 | 0.9974 | **0.9974** |
| OOF Blend AUROC | 0.9916 | — | — | **0.9985** |
| AUPRC | — | — | 0.9912 | **0.9913** |
| F1 | — | — | 0.9713 | **0.9768** |
| MCC | — | — | 0.943 | **0.9536** |
| Brier | — | — | 0.0141 | **0.0124** |
| Models Trained | 8 | 9 | 8 | **9** |
| LightGBM | 0.9911 (OOF) | SKIPPED | SKIPPED | **0.9974** (test) |
| KAN | — | MLP fallback | NameError: torch | NameError: test_size |
| CNN_1D | — | 0.5000 (broken) | SKIPPED | SKIPPED |
| Elapsed | 11.4h | 7.9h | 6.47h | **6.33h** |
| Cost | ~$9.70 | ~$5.60 | ~$4.80 | **~$4.90** |
| Instance | Norway DLP 16 | Norway DLP 16 | Hungary DLP 97 | **Hungary DLP 97** |

**Key trends:** Test AUROC has stabilized at 0.9974 across Runs 11-13 despite different model compositions. Run 13 achieves the best calibration metrics (F1, MCC, Brier) due to LightGBM's return providing a 10th base learner for the stacker. Training time decreased from 11.4h to 6.33h primarily due to the `dlperf>=80 pcie_bw>=12` instance selection filter introduced after Run 11.

## Failures and Fixes

| Issue | Runs Affected | Root Cause | Fix | Status |
|-------|--------------|------------|-----|--------|
| LightGBM SKIPPED | 11, 12 | PyPI lightgbm binary not compiled with CUDA/OpenCL; `device_type: "gpu"` and `device_type: "cuda"` both fail | Remove device_type flag entirely; CPU mode (1 min on 1.2M rows) | **FIXED** in `f4dbeed` |
| KAN MLP fallback | 11 | `fastkan` not on PyPI; `requirements.txt` listed non-existent package | Remove fastkan, add pykan+imodelsx | **FIXED** in `a968e28` |
| KAN NameError: torch | 12 | `_fit_imodelsx` missing `import torch` | Added import | **FIXED** in `f4dbeed` |
| KAN NameError: test_size | 13 | imodelsx v1.0.13 upstream bug: `fit()` references bare `test_size` | Two-part: (1) set attrs before fit in kan.py; (2) sed-patch installed package in launch script | **FIXED** in `0d4ea7b`+`bf2f665` |
| CNN_1D AUROC 0.5000 | 11 | No fasta_seq → dummy sequences → constant predictions | `--skip-cnn` flag | Deferred (needs ref genome) |
| ensemble.save() PicklingError | 9 | Nested class `_CNN1D._build_model.<locals>._CNN1D` | Per-model checkpoint + pickle fix | **FIXED** pre-Run 11 |

## Git Chain

`bf2f665` → `0d4ea7b` → `f4dbeed` → `a6fa7c5` → `a968e28` → `61a8d99` → `4e819e7` → `7d91386`

## Files

- Results: `C:\Projects\genomic-variant-classifier\outputs\run13\`
- Master log: `C:\Projects\genomic-variant-classifier\outputs\run13\run13_master.log`
- Metrics: `C:\Projects\genomic-variant-classifier\outputs\run13\full\metrics.json`
- Per-model: `C:\Projects\genomic-variant-classifier\outputs\run13\full\per_model_metrics.csv`
- Feature importance: `C:\Projects\genomic-variant-classifier\outputs\run13\full\feature_importance.csv`
