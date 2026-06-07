# Ablation: `n_pathogenic_in_gene` (gene-prevalence memorization) — 2026-06-07

**Verdict: the headline ~0.998 AUROC does NOT depend on `n_pathogenic_in_gene`.**
Zeroing the #1-importance feature costs 0.0002 test AUROC (within the bootstrap 95%
CIs' overlap). The long-standing "gene-prevalence memorization inflates AUROC" concern,
*for this feature*, is substantially falsified. Reclassify from standing concern to
addressed (redundant, not load-bearing). One residual channel (`gene_has_known_disease`)
was subsequently tested too (the `no_gene_level` third arm below, 2026-06-07): likewise not load-bearing.

## Hypothesis
`n_pathogenic_in_gene` is computed from a gene's ClinVar pathogenic counts, so it is
mechanically circular with the label and has been the top feature every run (importance
391 in Run 15). Concern: the headline AUROC could be an artifact of memorizing gene-level
pathogenic prevalence rather than learning per-variant biology.

## Design
- Harness: `scripts/run9_ablations.py` (extended this session with a `no_gene_prevalence`
  mask + `--max-train`). Ablation = zero the column in the StandardScaler-transformed
  splits (zero = training mean = "ignore feature"); trees see zero information gain.
- Splits: the Run 15 de-leaked splits at `outputs/run15_rerun_report/full/splits/`
  (78 cols; `n_pathogenic_in_gene` non-zero rate 1.0; `gnn_score` non-zero rate 1.0).
- Controlled comparison: `full` (zero columns) vs `no_gene_prevalence` (zeros only
  `n_pathogenic_in_gene`). Identical otherwise: `--max-train 150000 --n-folds 3 --seed 42`,
  tree subset (`--skip-nn --skip-svm --skip-mc-dropout --skip-kan --skip-shap
  --skip-permutation`) -> 7 base models: random_forest, xgboost, lightgbm,
  svm_bagged_rbf, logistic_regression, gradient_boosting, catboost (Nyström `svm`
  auto-skips at n=150000).
- Mask validation: harness logged `Ablation 'no_gene_prevalence' zeroed 1 columns:
  ['n_pathogenic_in_gene']` on train/val/test, and its `assert zeroed_train == zeroed_val
  == zeroed_test` passed.

## Result (ENSEMBLE_STACKER on the 304,711-row test set)

| Metric | full | no_gene_prevalence | Δ (ablated − full) |
|---|---|---|---|
| Test AUROC | 0.99817 [0.99797–0.99836] | 0.99802 [0.99782–0.99823] | −0.0002 |
| Test AUPRC | 0.9930 | 0.9921 | −0.0009 |
| Test MCC | 0.9630 | 0.9613 | −0.0017 |
| Test Brier | 0.0071 | 0.0076 | +0.0005 |
| Val AUROC | 0.9980 | 0.9979 | −0.0001 |

Per-consequence test AUROC (where the effect concentrates):

| Consequence | N | %Path | full | no_gene_prevalence | Δ |
|---|---|---|---|---|---|
| loss_of_function | 15,170 | 98.6% | 0.9447 | 0.9156 | −0.0291 |
| missense | 43,861 | 33.1% | 0.9921 | 0.9918 | −0.0003 |
| synonymous | 103,320 | 0.1% | 0.9449 | 0.9467 | +0.0018 |
| other | 142,099 | 10.6% | 0.9969 | 0.9965 | −0.0004 |
| inframe_indel | 261 | 41.0% | 0.9462 | 0.9387 | −0.0075 |

Blend-weight rebalancing when the feature is removed: catboost 0.336 -> 0.526,
xgboost 0.433 -> 0.172 — the ensemble compensates through correlated learners.

## Interpretation
- **Importance ≠ marginal contribution.** `n_pathogenic_in_gene` is heavily *used* by the
  trees (top split-importance) but is *redundant*: removing it costs 0.0002 overall AUROC
  because correlated features recover the signal. This is the cleanest possible answer to
  the standing concern — the model does not *need* gene-prevalence to hit ~0.998.
- **Consistent with the unseen-gene holdout (0.9988).** For genes absent from train,
  `n_pathogenic_in_gene` is ~uninformative anyway, yet UGH AUROC matched in-distribution —
  the ablation now confirms the same on the in-distribution test directly.
- **Where it does matter:** the loss-of-function subgroup (−0.0291), where a gene-level
  pathogenic count is most informative for the rare benign-LoF discrimination. Small
  subgroup (~5% of test), so it does not move the headline.

## Caveats
1. **Does not rule out other gene-level channels.** `gene_has_known_disease` (alive) could
   still carry gene-identity signal. The `no_gene_level` mask (adds `gene_has_known_disease`
   + the dead `gene_constraint_oe`/`gene_is_constrained`) was tested in the third arm below (2026-06-07); even so, the
   strict result is already decisive.
2. **Subsampled tree-subset.** 150 k train / 3-fold / 7 models gives full=0.9982 vs the
   real run's 0.9984; the interpretable quantity is the Δ, not the absolute. More data and
   models add redundancy -> the marginal contribution would only shrink, so −0.0002 is an
   upper bound on the effect.
3. **Wall-clock contaminated (AUROCs are not).** Reported elapsed (full 324 min,
   no_gene_prevalence 200 min for identical config; `gradient_boosting` 4h15m in one arm
   vs 9 min in the other) reflects the laptop sleeping/throttling overnight, not compute.
   The metrics are deterministic (seed + data) and unaffected. Lesson: do not time
   multi-hour runs on the CPU laptop without disabling sleep (`powercfg`), or time on GPU.

## Optional confirmation
- **`no_gene_level`** (DONE 2026-06-07; see the Update section below — was a one-liner the harness already supported):
  `--ablation no_gene_level` against the same splits — tests proxy-recovery via
  `gene_has_known_disease`.
- **Definitive full-roster/full-data** on GPU (13 models, 1.04 M train, 5-fold) if a
  publication-grade number is wanted; expected to show ≤ 0.0002 by the redundancy argument.

## Artifacts
`outputs/ablation_run15/full/`, `outputs/ablation_run15/no_gene_prevalence/`, and `outputs/ablation_run15/no_gene_level/`
(each: eval_report.json, per_model_metrics.csv, test_predictions.parquet, oof_predictions.parquet,
models/); aggregator `outputs/ablation_run15/ablation_results.parquet` (3 rows).

## Code-hygiene item observed
`variant_ensemble.py` previously used deprecated `datetime.utcnow()` (DeprecationWarning);
RESOLVED 2026-06-07 — now `datetime.now(timezone.utc)` (added `from datetime import timezone`); full suite green.

---

## Update 2026-06-07 -- third arm: `no_gene_level` (gene-channel question closed)

The first two arms (`full`, `no_gene_prevalence`) isolated `n_pathogenic_in_gene` alone. To test the
broader concern -- that the *entire* gene-level channel (not just prevalence) might memorise rather than
generalise -- a third arm zeroes all four gene-level features, including `gene_has_known_disease`.

Same protocol as the prior arms: 150k-row train subsample, 3-fold OOF, seed 42, heavy models skipped
for cost. Completed only after the ScalableSVM full-test crash was fixed (the arm reached evaluation
with `GVC_SVM_NJOBS=2`).

| Arm | Description | Test AUROC [95% CI] | Delta vs full |
|-----|-------------|-----------:|--------------:|
| `full` | all features | 0.99817 [0.99797-0.99836] | -- |
| `no_gene_prevalence` | zero `n_pathogenic_in_gene` only | 0.99802 [0.99782-0.99823] | -0.0002 |
| `no_gene_level` | zero all 4 gene-level features (incl. `gene_has_known_disease`) | 0.99783 [0.99761-0.99804] | -0.0003 |

### Per-consequence localisation

The effect is not spread across the cohort -- it concentrates entirely in the loss-of-function subgroup:

| Subgroup | full | no_gene_prevalence | no_gene_level |
|----------|-----:|-------------------:|--------------:|
| loss_of_function (AUROC) | 0.9447 | 0.9156 | 0.9096 |

Other consequence subgroups are essentially unchanged across the three arms.

### Verdict

Removing the entire gene-level channel costs 0.0003 AUROC overall (0.00034 at full precision), and even the most affected subgroup
(LoF) remains above 0.90. The gene channel is therefore **not** the driver of the headline ~0.998, and
the gene-prevalence memorisation concern is resolved: the gene signal is a small, interpretable
contribution localised to loss-of-function variants (where gene-level constraint is genuinely
informative biology), not a global inflation of the metric. No feature is dropped -- the channel stays
in the model; the ablation simply bounds its contribution. Results in
`outputs/ablation_run15/ablation_results.parquet`.
