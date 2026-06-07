# GenAssoc v1 — Metrics Glossary (LIVING DOCUMENT)

Every metric used anywhere in the project, fully defined, with where/how it is
applied and how its value has moved across runs. Append a new row after every run.
Last updated: Run 15 (2026-06-06). Run 9 row corrected 2026-06-06 (prior 0.9814/0.9850
was wrong; see note under the matrix).

Conventions: y = true label (1 = pathogenic, 0 = benign); p = model score / probability
of pathogenicity; positives = pathogenic; Run 15 class balance = 210,549 pathogenic /
1,279,465 benign (~1:6).

---

## I. Discrimination / ranking

### AUROC — Area Under the ROC curve
- **Construction:** sweep the threshold; plot TPR = TP/(TP+FN) vs FPR = FP/(FP+TN); area under it.
- **Equivalent:** P(p(random positive) > p(random negative)).
- **Range:** 0.5 (random) -> 1.0 (perfect); <0.5 = inverted.
- **Captures:** pure ranking; threshold- and calibration-independent.
- **Limits:** optimistic under imbalance (read with AUPRC); says nothing about calibration.
- **Why here:** headline metric + PASS gate (>=0.9); cross-run comparator.
- **Where:** per base model (OOF), ENSEMBLE_STACKER (dev/test, holdout/val), unseen-gene.

### AUPRC — Area Under the Precision-Recall curve
- **Construction:** area under Precision = TP/(TP+FP) vs Recall = TP/(TP+FN).
- **Range:** baseline = positive prevalence (~0.14) -> 1.0.
- **Captures:** positive-(pathogenic-)class performance specifically.
- **Where:** beside AUROC (Run 15 ensemble 0.9936 main / 0.9945 UGH).

---

## II. Threshold classification

### F1 (macro / weighted)
- **Formula:** 2*P*R/(P+R) at threshold 0.5. Macro = unweighted per-class mean;
  weighted = support-weighted (benign-dominated). Macro exposes minority weakness.
- **Range:** 0 -> 1. **Where:** per model/ensemble (Run 15 f1_macro 0.9826, weighted 0.9913).

### MCC — Matthews Correlation Coefficient
- **Formula:** (TP*TN - FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)).
- **Range:** -1 -> +1 (0 = random). Balanced single summary over all four cells.
- **Where:** per model/ensemble (Run 15 0.9652 main / 0.9695 UGH).

---

## III. Calibration

### Brier score
- **Formula:** mean((p - y)^2). **Range:** 0 (best) -> 1. Joint accuracy + calibration.
- **Where:** per model/ensemble (Run 15 0.0069 main / 0.0059 UGH).

### Calibration (reliability) curve
- Bin predictions (10 bins); mean predicted vs observed fraction positive; diagonal =
  perfect. Diagnostic plot in the HTML report, not a scalar.

### Isotonic calibration (method)
- Monotonic post-hoc score remap; applied to xgboost/lightgbm (and random_forest in some
  runs) before their OOF metrics.

---

## IV. Cross-validation protocol

### OOF — Out-Of-Fold predictions
- **Protocol:** k-fold CV (k=5 in Run 15); for each fold i train on the other k-1, predict
  fold i; concatenate so every row has a prediction from a model that never trained on it.
- **Why:** near-unbiased generalization estimate, and load-bearing for stacking - the
  meta-learner trains on base-model OOF (not in-sample) predictions (Wolpert stacked
  generalization), preventing it from overfitting base memorization.
- **"OOF AUROC"** = AUROC on OOF predictions = the per-model line each run.
- **Caveat:** with random folds the same gene spans train/held-out folds, so OOF AUROC can
  be inflated by gene-level memorization - the reason the unseen-gene holdout exists.

---

## V. Model-specific diagnostics

- **best_iteration** (boosting): early-stopping iteration count; bookkeeping.
- **GNN Best Val AUC:** the GNN's own validation AUROC (Run 15: 0.6509, early stop ep 92).
- **gnn_score dist (mean/std/nonzero_frac):** injected GNN feature spread - the degeneracy
  detector. Run 14 bug -> all-zero (std 0). Run 15: std~0.099, nonzero_frac 1.0,
  range [0.0012, 0.5000] - gate PASS on all three splits.
- **Feature importance:** model-reported per-feature contribution ("Top 10 features").
  Run 15 top = n_pathogenic_in_gene (391) - the standing signal-vs-memorization concern.

---

## VI. Statistical association (report_generator.py)

- **Odds ratio:** (a*d)/(b*c) on the 2x2 class x phenotype table; >1 positive; 0 -> inf.
- **p-value:** association significance (chi-square / Fisher); 0 -> 1.
- **Cramer's V:** categorical effect size; 0 (none) -> 1 (perfect).
- **significant:** boolean from the p-value threshold.
- **bootstrap CI (bootstrap_metric):** resample-with-replacement [lo, hi] on any metric.

---

## VII. Decision gates

- **AUROC >= 0.9** holdout PASS gate. Run 15: 0.9984 PASS.
- **C3 falsifier >= 0.95** on unseen-gene holdout AUROC. Run 15: 0.9988 PASS.
- **GNN std>0 / nonzero_frac=1.0** non-degeneracy gate. Run 15: PASS.
- **Seq-window coverage <0.5% unmapped** abort gate (returns 2) - guards cnn_1d.

---

## Per-run matrix (rows = runs)

Ensemble figures unless noted. "Test" = locked-test AUROC. "—" = not produced / not in the
run's results doc (backfill from docs/CHANGELOG.md if found).

| Run | Date | Test AUROC | OOF blend | AUPRC | F1 | MCC | Brier | Base models | KAN | CNN_1D | Test n | Time | Cost |
|----|------|-----------|-----------|-------|----|----|-------|-------------|-----|--------|--------|------|------|
| 9   | 05-13 | — (save crash) | 0.9916 | — | — | — | — | 11 (8 saved) | 0.9855 | — | n/a | 11.4 h | ~$9.70 |
| 10b | 05-?? | 0.9970 (8-avg) | — | — | — | — | — | 8 | — | — | 349,067 | — | — |
| 11  | 05-?? | 0.9974 | — | — | — | — | — | 9 | MLP fallback | 0.5000 (broken) | 349,067 | 7.9 h | ~$5.60 |
| 12  | 05-?? | 0.9974 | — | 0.9912 | 0.9713 | 0.943 | 0.0141 | 8 | NameError torch | skipped | 349,067 | 6.47 h | ~$4.80 |
| 13  | 05-25 | 0.9974 | 0.9985 | 0.9913 | 0.9768 | 0.9536 | 0.0124 | 9 | NameError test_size | skipped | 349,067 | 6.33 h | ~$4.90 |
| 14  | 05-26 | 0.9975 | 0.9985 | — | — | — | — | 10 | 0.9921 | skipped | 349,067 | 3 h 14 m | $2.17 |
| 15  | 06-06 | **0.9984** | 0.9984 (stacker) | 0.9936 | 0.9826 (macro) | 0.9652 | 0.0069 | **13** | 0.9968 | **0.8536** | 304,711 | ~11.2 h | ~$6.3 |
| 15-UGH | 06-06 | 0.9988 (unseen-gene, C3 PASS) | — | 0.9945 | 0.9847 | 0.9695 | 0.0059 | 13 | 0.9970 | 0.8448 | 213,436 / 2,407 genes | — | — |

**Comparability caveat (read before trending the AUROC column).** Runs 10b-14 scored on a
locked 349,067-variant test set. Run 15's test set is 304,711 because the review-tier <=3
filter changed the cohort (1,686,333 -> 1,490,014; 88% retained). So the Run 14->15
"+0.0009" is NOT on an identical test set, and the tier-filter semantics are themselves a
standing audit item. Run 15 also expanded the roster to 13 (added svm_bagged_rbf, activated
cnn_1d, GNN as a feature) and added the unseen-gene holdout.

**Run 9 correction note.** The earlier 0.9814/0.9850 was incorrect: Run 9 crashed at
ensemble.save() (unpicklable nested _CNN1D) and the instance was lost before test scoring,
so it has no locked-test number; its recorded results are the OOF Nelder-Mead blend 0.9916
and LR-stacker 0.9911 (per-model OOF 0.9855 KAN -> 0.9911 LightGBM).

---

## Variation narrative

**AUROC trajectory.** OOF/blend rose 0.9916 (Run 9) -> 0.9985 (Runs 13-14) -> 0.9984
stacker / 0.9988 unseen-gene (Run 15). The locked-test line sat at 0.9974 (Runs 11-13) ->
0.9975 (Run 14) -> 0.9984 (Run 15), but the Run 15 jump coincides with the cohort change
(tier filter) and dbNSFP coming online (0 -> 188,023 real SIFT), so treat it as a
new-baseline rather than a clean delta off Run 14. The unseen-gene 0.9988 ~= in-distribution
0.9984 = no generalization collapse on 2,407 unseen genes, but NOT proof of leakage-free:
gene-level features (n_pathogenic_in_gene 391 top, pLI, LOEUF) may inform holdout genes;
attribution requires the n_pathogenic_in_gene ablation. Do not record 0.9988 as
"leakage disproven."

**KAN.** Failed every run from introduction through Run 13 (OOM -> MLP fallback -> NameError
torch -> NameError test_size), first trained Run 14 (0.9921), Run 15 0.9968/0.9970 -
competitive with the linear/kernel tier. The newer-architecture comparison goal now
produces real measurements, not failures.

**cnn_1d.** 0.5000 (broken, Run 11) -> skipped (12-14) -> 0.8536 main / 0.8448 UGH (Run 15).
The 0.52->0.85 jump from 3 k smoke to 1.49 M full is a pure data-scale effect; weakest base
learner but a genuine, non-degenerate contributor at scale.

**GNN.** Crashed (Run 9 era) -> all-zero (Run 14) -> real non-degenerate gnn_score (Run 15:
std 0.099, nonzero 1.0), though weak standalone (val AUC 0.65). Value is as an ensemble
feature; the non-degeneracy gate now guards the Run-14 regression permanently.

**Calibration.** Brier 0.0141 (Run 12) -> 0.0124 (Run 13) -> 0.0069 main / 0.0059 UGH
(Run 15); MCC 0.943 -> 0.9536 -> 0.9652/0.9695. Both track roster growth and the
isotonic-calibrated tree learners.

**n_pathogenic_in_gene.** Top feature every run (3.3x next feature historically; 1.3x in
Run 14 after the rich config; importance 391 in Run 15). **Adjudicated 2026-06-07 by
ablation** (`run9_ablations.py --ablation no_gene_prevalence` vs `full`, Run 15 splits,
150k/3-fold/7-model tree subset, seed 42): zeroing this single top feature moved
ENSEMBLE_STACKER test AUROC 0.99820 -> 0.99800 (Δ −0.0002, bootstrap 95% CIs overlapping).
High split-usage importance but negligible marginal contribution -> the ~0.998 is NOT
gene-prevalence memorization through this feature (effect concentrates in the LoF subgroup,
−0.0291, ~5% of test). See `docs/ABLATION_n_pathogenic_in_gene.md`. Residual channel
`gene_has_known_disease` untested (`no_gene_level`, optional).
