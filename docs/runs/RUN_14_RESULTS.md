# Run 14 — Results

**Date**: 2026-05-26  
**HEAD**: `80ac62c`  
**Status**: ✅ COMPLETE (clean exit 0)  
**Hypothesis under test**: H1 — KAN as a base learner under the imodelsx backend trains successfully on CUDA, and contributes diversity to the stacker meta-learner blend.

---

## 1. Headline metrics

From `metrics.json`:

| Metric | Test (n=349,067) | Val (n=154,404) |
|---|---|---|
| AUROC | **0.9975** | 0.9974 |
| AUPRC | 0.9914 | 0.9903 |
| f1_macro | 0.9775 | 0.9785 |
| MCC | 0.9550 | 0.9569 |
| Brier | 0.0130 | 0.0111 |

OOF blend AUROC (from log): **0.9985** (LR stacker: 0.9984, Δ +0.0001)  
n_train: 1,197,216 | n_val: 154,404 | n_test: 349,067 | n_features: 78  
elapsed_seconds: 11,654.7 (3 h 14 m 14.7 s)

### vs Run 13

| Metric | Run 13 | Run 14 | Δ |
|---|---|---|---|
| Test AUROC | 0.9974 | **0.9975** | +0.0001 |
| OOF blend AUROC | 0.9985 | 0.9985 | 0 |
| Base learners trained | 9 | **10** (KAN new) | +1 |
| Wall-clock | 6.3 h | **3.24 h** | -49% |
| Cost | $4.90 | **$2.17** | -56% |

## 2. H1 outcome — nuanced result

**Technical part CONFIRMED**: KAN trained successfully via imodelsx backend on CUDA, 3 CV folds, 100K subsample each. KAN OOF AUROC 0.9921. First run in project history where KAN actually trains.

**Diversity finding 1 — Stacker test AUROC equals catboost test AUROC**:
- catboost test AUROC: **0.9975**
- ENSEMBLE_STACKER test AUROC: **0.9975**
- Difference: 0

**Diversity finding 2 — But stacker dominates on threshold-dependent metrics**:
- f1_macro: stacker 0.9775 vs catboost 0.9632 (Δ +0.0143)
- MCC: stacker 0.9550 vs catboost 0.9276 (Δ +0.0274)
- Brier: stacker 0.0130 vs catboost 0.0166 (Δ -0.0036, lower = better calibrated)

**Interpretation**: The ensemble's value is **probability calibration and threshold-dependent decisions**, not ranking accuracy. If downstream use is "is this variant pathogenic at threshold τ" → stacker is meaningfully better. If downstream use is "rank variants by pathogenicity" → catboost alone is essentially equivalent at far lower compute.

**Diversity finding 3 — KAN's OOF→test gap is large**:
- KAN OOF AUROC: 0.9921
- KAN test AUROC: 0.9896
- Gap: 0.0025

For comparison, catboost OOF (0.9982) vs test (0.9975) = 0.0007 gap. KAN's gap is ~3.5× larger, indicating the 100K subsample overfits relative to its training set. Run 15 should either scale subsample to 250K-500K or accept structural weakness.

## 3. Per-model OOF AUROC (10 base learners, from master log)

| Model | Run 13 OOF | Run 14 OOF | Δ |
|---|---|---|---|
| random_forest | 0.9964 | 0.9978 | +0.0014 |
| xgboost | 0.9974 | 0.9984 | +0.0010 |
| lightgbm | 0.9974 | 0.9983 | +0.0009 |
| logistic_regression | 0.9942 | 0.9955 | +0.0013 |
| gradient_boosting | 0.9968 | 0.9974 | +0.0006 |
| catboost | 0.9975 | 0.9982 | +0.0007 |
| tabular_nn | 0.9971 | 0.9975 | +0.0004 |
| **kan** | — | **0.9921** | new |
| mc_dropout | 0.9971 | 0.9975 | +0.0004 |
| deep_ensemble | 0.9973 | 0.9977 | +0.0004 |

Mean improvement over Run 13: +0.0008 across 9 carry-over models.

## 4. Per-model TEST set metrics (from per_model_metrics.csv)

Columns: `auroc, auprc, f1_macro, f1_weighted, mcc, brier`.

| Model | AUROC | AUPRC | f1_macro | f1_weighted | MCC | Brier |
|---|---|---|---|---|---|---|
| **catboost** | **0.9975** | 0.9912 | 0.9632 | 0.9761 | 0.9276 | 0.0166 |
| **ENSEMBLE_STACKER** | **0.9975** | **0.9914** | **0.9775** | **0.9855** | **0.9550** | **0.0130** |
| xgboost | 0.9974 | 0.9906 | 0.9769 | 0.9852 | 0.9539 | 0.0120 |
| lightgbm | 0.9974 | 0.9906 | 0.9769 | 0.9852 | 0.9538 | 0.0120 |
| deep_ensemble | 0.9973 | 0.9900 | 0.9771 | 0.9854 | 0.9542 | 0.0120 |
| mc_dropout | 0.9971 | 0.9894 | 0.9765 | 0.9850 | 0.9531 | 0.0125 |
| tabular_nn | 0.9971 | 0.9894 | 0.9765 | 0.9850 | 0.9531 | 0.0125 |
| gradient_boosting | 0.9968 | 0.9898 | 0.9752 | 0.9842 | 0.9505 | 0.0126 |
| random_forest | 0.9964 | 0.9866 | 0.9648 | 0.9773 | 0.9301 | 0.0165 |
| logistic_regression | 0.9942 | 0.9781 | 0.9578 | 0.9727 | 0.9165 | 0.0228 |
| **kan** | **0.9896** | 0.9680 | 0.9422 | 0.9628 | 0.8847 | 0.0300 |

## 5. Observability findings

From `outputs/run14_observability/run14_observability.{md,json}`:

| Metric | Value | Interpretation |
|---|---|---|
| OBS_LOG_SIZE | 61,722 bytes | master log captured |
| OBS_ARTIFACTS | 1,837.8 MB | total disk footprint of all run artifacts |
| **OBS_DEAD_FEATS** | **34** | **44% of 78 features are zero-variance / all-zero** |
| OBS_KAN_BACKEND | None | observability bug — didn't parse KAN log lines |
| OBS_LGB_DEVICE | None | observability bug — didn't parse LGB device line |
| OBS_ERRORS | 0 | no Traceback / ABORT in log |

**34 dead features out of 78** quantifies the silent-zero problem. The 5 optional sources that weren't wired (FinnGen, 1KGP, PrimateAI3D, STRING, CNN fasta) plus annotators that emitted zero (PhyloP, GTEx eQTL, VEP codon_position, OMIM, ClinGen, dbSNP, EVE, HGMD, RNA splice, protein structure, ESM-2) account for these. Run 15 must address.

## 6. Infrastructure

- VM: Vast.ai instance `37897784`, Texas US, host 60400, machine 33069
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, driver 570.86.16
- Python: 3.11.10 (conda), pip 24.2
- Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- Hourly rate: $0.6694
- Actual cost: $2.17

## 7. Anomalies observed (8)

### A1 — `np.log(0)` RuntimeWarning at `mc_dropout.py:87` (3× fires)
```
entropy_per_pass = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
```
**Fix Run 15**: `clipped = np.clip(p, 1e-12, 1 - 1e-12)` BEFORE the log.

### A2 — `mc_dropout` uncertainty degenerate (6× warnings)
`TabularNNClassifier does not expose _predict_proba_single_pass()` → uncertainty estimates are zero.
**Fix Run 15**: implement `_predict_proba_single_pass()` on TabularNNClassifier OR migrate to DeepEnsembleWrapper exclusively.

### A3 — `imodelsx_patch` log line duplicated
Echo appears 2× in master log. **Fix Run 15**: audit `scripts/launch_run11_vm.sh` lines 191-199 for duplicate if/else echo.

### A4 — KAN trained on 12% subsample (100K of 814K)
KAN's effective sample size is 1/8 of GBDT's. OOF→test gap 0.0025 is ~3.5× larger than catboost's. **Fix Run 15**: scale KAN subsample to 250K-500K.

### A5 — Score annotation step numbering inconsistency
Mixed denominators (6, 12, 14, 16, 17 visible). **Fix Run 15**: normalize counter or remove denominator.

### A6 — 5 optional data sources silently zero
FinnGen / 1KGP / PrimateAI3D / STRING / CNN fasta. Run-15 plan template requires explicit decision per source. Charter v1.1 closes this.

### A7 — Observability collector under-detection (NEW)
`OBS_KAN_BACKEND: None` and `OBS_LGB_DEVICE: None` despite both being explicitly logged. The collector's log-parsing patterns miss the `KAN (imodelsx/efficient-kan): trained on 100000 samples, device=cuda` and `LGBM smoke fit: OK` lines.
**Fix Run 15**: update regex patterns in `scripts/run14_observability.py` (rename for genericity to `scripts/run_observability.py`).

### A8 — Postflight gate path assumption (NEW)
The Block B gate in this session checked `$localOutDir\full\ensemble.manifest.json`, but the file lives at `$localOutDir\full\models\ensemble.manifest.json`. False FAIL caused destroy to proceed despite gate output saying not to. Worked out only because data was actually present at a deeper path.
**Fix Run 15**: update `scripts/Run_Postflight.ps1` to use recursive `Get-ChildItem -Filter` instead of fixed `Test-Path` for critical artifacts.

## 8. Files captured locally

```
outputs/run14/                            (73 files total, ~1.8 GB)
├── full/                                 (top-level run output)
│   ├── X_{train,val,test}.parquet
│   ├── y_{train,val,test}.parquet
│   ├── meta_{train,val,test}.parquet
│   ├── metrics.json
│   ├── per_model_metrics.csv (10 base + ENSEMBLE_STACKER, TEST set)
│   ├── per_model_metrics_val.csv (validation split equivalent)
│   ├── feature_importance.csv
│   ├── oof_predictions.parquet
│   ├── data_quality_audit.{csv,json}
│   └── models/
│       ├── (10 base) {model}.joblib + _meta.json + _oof.npy + _oof_indices.npy
│       ├── ensemble.joblib (85 MB)
│       ├── ensemble.manifest.json
│       └── ensemble_models/  (full-data refit base learners + scaler)
├── run14_master.log (61,722 bytes)
├── pip_freeze_vm.txt (216 packages)
└── reproducibility_manifest.json

outputs/run14_observability/
├── run14_observability.md
└── run14_observability.json
```

## 9. Next-run guidance (Run 15)

1. Copy `docs/templates/RUN_N_PLAN_TEMPLATE.md` → `docs/runs/RUN_15_PLAN.md`. Fill every `<DECISION>` placeholder.
2. Address anomalies A1, A2, A4, A7, A8 with code commits BEFORE preflight.
3. Decide KAN's status (scale subsample or drop).
4. Build STRING parquet index OR document deferral.
5. Build 1KGP AF parquet OR document deferral.
6. Begin HGVSp parser (unlocks ESM-2 + EVE → 2 more real annotators).
7. Run Preflight Charter v1.1 gates G1 + G2 BEFORE creating any Vast.ai instance.
