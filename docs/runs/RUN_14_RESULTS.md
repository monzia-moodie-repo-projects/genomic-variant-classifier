# RUN 14 — Results

> **Status:** TEMPLATE. Fill in TBD fields after `Run14_Postflight.ps1` completes.

**Commit:** `bf2f665`
**Launch date:** 2026-05-26
**Completion date:** TBD
**Instance:** Vast.ai `<id>`, RTX 4090, `<region>`, `$<rate>/hr`
**Elapsed:** TBD h
**Cost:** ~$TBD

---

## 1. Headline numbers

| Metric | Value | vs Run 13 |
|---|---|---|
| Test AUROC      | TBD | TBD |
| Test AUPRC      | TBD | TBD (Run 13: 0.9913) |
| Test F1         | TBD | TBD (Run 13: 0.9768) |
| Test MCC        | TBD | TBD (Run 13: 0.9536) |
| Test Brier      | TBD | TBD (Run 13: 0.0124) |
| Models trained  | TBD / 10 | Run 13: 9 / 10 |
| KAN status      | TBD | Run 13: failed |

---

## 2. Per-model OOF + test metrics

> Fill from `run14_observability.md` table.

| Model | OOF AUROC | Test AUROC | AUPRC | F1 | MCC | Brier | Train time |
|---|---|---|---|---|---|---|---|
| catboost            | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| xgboost             | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| lightgbm            | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| gradient_boosting   | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| random_forest       | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| logistic_regression | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| tabular_nn          | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| mc_dropout          | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| deep_ensemble       | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **kan**             | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| cnn_1d              | SKIPPED | — | — | — | — | — | — |

---

## 3. Hypothesis test results

| # | Hypothesis | Pass/Fail | Evidence |
|---|---|---|---|
| H1 | KAN trains successfully and produces stable OOF AUROC | TBD | TBD |
| H2 | KAN OOF AUROC in [0.996, 0.998] | TBD | TBD |
| H3 | Adding KAN does not move blend AUROC | TBD | TBD |
| H4 | LightGBM stays in CPU mode at 0.9974 ± 0.0002 | TBD | TBD |
| H5 | Dead-feature count matches prior estimate (25-35 of 78) | TBD | TBD |

---

## 4. Per-model algorithm analysis

> Write 2-3 sentences per model on (a) how it performed on this dataset, (b) how it differs from neighboring models in the table, and (c) what (if anything) we can interpret about the data from this performance. This section is the reusable documentation that informs future projects.

### CatBoost — TBD
Performance: TBD. Algorithm: Ordered boosting with categorical feature handling natively encoded; uses oblivious decision trees which fit each level with the same split. Differs from XGBoost mainly in the ordered-boosting trick (reduces target leakage when categorical features overlap with the target) and from LightGBM in its tree structure (oblivious vs leaf-wise). On this dataset: TBD.

### XGBoost — TBD
Performance: TBD. Algorithm: Standard gradient boosting on regression trees with second-order Taylor expansion and L1/L2 regularization on leaf weights. Differs from CatBoost in tree growth (depth-wise vs oblivious) and from LightGBM in split-finding (exact / hist vs histogram with goss/efb). On this dataset: TBD.

### LightGBM — TBD
Performance: TBD. Algorithm: Histogram-based gradient boosting with leaf-wise tree growth, gradient-based one-sided sampling (GOSS), and exclusive feature bundling (EFB). Differs from XGBoost by growing the leaf with maximum loss reduction rather than depth-wise. **CPU vs GPU note for this dataset:** TBD. On this dataset: TBD.

### Gradient Boosting (sklearn) — TBD
Performance: TBD. Algorithm: Reference implementation of GBM with depth-wise tree growth, no histogram binning, no GPU support. Differs from XGBoost/LightGBM mainly in speed (10-100× slower) and lack of regularization tuning options. Included primarily as a baseline for the modern GBDTs. On this dataset: TBD.

### Random Forest — TBD
Performance: TBD. Algorithm: Bagging of decision trees with feature subsampling at each split. Differs from GBDTs by training trees in parallel (no sequential residual fitting) and averaging rather than summing predictions. Tends to underfit signal but produces well-calibrated probabilities natively. On this dataset: TBD.

### Logistic Regression — TBD
Performance: TBD. Algorithm: Linear model fit with L2 regularization. Establishes the linear baseline — any model materially above this line is exploiting feature interactions or nonlinear thresholds. On this dataset: TBD.

### TabularNN — TBD
Performance: TBD. Algorithm: Feedforward neural network on tabular features. Differs from KAN by using fixed activation functions (ReLU) and learned weights at edges, vs KAN which learns the activation functions themselves. Differs from MC Dropout / Deep Ensemble by being a single deterministic forward pass. On this dataset: TBD.

### MC Dropout — TBD
Performance: TBD. Algorithm: Tabular NN with dropout kept active at inference; predictive uncertainty estimated by averaging T stochastic forward passes. Captures epistemic uncertainty cheaply. Differs from Deep Ensemble by sharing weights across the T passes. On this dataset: TBD.

### Deep Ensemble — TBD
Performance: TBD. Algorithm: 25 independently-initialized TabularNNs trained on the same data; predictions averaged. Captures epistemic uncertainty by genuine model diversity rather than dropout noise. Differs from MC Dropout by training 25 distinct networks. Wall-clock cost is the dominant neural-model line item. On this dataset: TBD.

### KAN — TBD
Performance: TBD. Algorithm: Kolmogorov-Arnold Network — learns the activation functions themselves (parameterized as B-splines), placing them on edges rather than nodes. Theoretical interest is whether learnable activations can capture nonlinear thresholds that ReLU networks must approximate piecewise. Backend used: TBD (imodelsx wraps efficient-kan with sklearn fit/predict_proba; pykan is the older, slower reference implementation; MLP is a fallback). Differs from TabularNN by every-edge spline rather than every-node ReLU. On this dataset: TBD.

### CNN_1D — SKIPPED
Skipped intentionally via `--skip-cnn`. The training pipeline does not populate `fasta_seq` (the connector would otherwise feed dummy sequences and the model predicts 0.5000 for every variant — observed in Run 11). Will be re-enabled once the sequence-extraction path is wired in a future run.

---

## 5. KAN deep-dive (this run's signature event)

> The Run 14 narrative is whether the 4-bug KAN remediation chain held. Document in detail:

- **Package patch applied?** TBD (`imodelsx_patch: fixed 3 bare-name refs in <path>` should appear in log)
- **Backend used:** TBD (imodelsx / pykan / mlp_fallback)
- **Fit success/failure:** TBD
- **Training time:** TBD (compare to Run 10a's 19h pykan runaway)
- **Subsample size:** TBD (expect 100,000 rows)
- **OOF AUROC:** TBD
- **Test AUROC:** TBD
- **If failed — new error or known error?** TBD

---

## 6. Feature signal coverage

> From `run14_observability.md` — populate from the structured JSON.

- **Total features:** 78
- **Dead features (non-zero rate < 0.001):** TBD
- **Top 10 populated features by non-zero rate:** TBD
- **Notable changes vs Run 13:** TBD

---

## 7. Errors and warnings observed

> List from log scan. Filter out benign UserWarning / FutureWarning / DeprecationWarning.

- TBD

---

## 8. Operational notes

- **Time-to-first-AUROC:** TBD (target < 30 min — random forest first model trained)
- **Symlink bootstrap:** TBD (any anomalies)
- **SCP transfer time:** TBD (data up + report down)
- **Idle minutes post-completion (before destroy):** TBD (target < 10)

---

## 9. Standing rules updated (if any)

- TBD

---

## 10. Items deferred to next run

- TBD
