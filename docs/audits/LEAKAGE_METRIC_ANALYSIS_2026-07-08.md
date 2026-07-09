# LEAKAGE & METRIC ANALYSIS — 2026-07-08 (resolved from artifacts)

**Author:** Monzia Moodie
**Data sources (gitignored — multi-GB parquet, on disk only):** `outputs/run14/full/splits/`
(regime v0), `outputs/ablation_run15/*/`, `outputs/run10/full/`, `outputs/run16/` — read via
`scripts/probe_run14_univariate_leakage.py`, `scripts/evaluate_predictions.py`,
`scripts/read_run_artifacts.py`.
**Frozen console evidence (version-controlled):** the exact output backing every number below is
committed under `docs/audits/evidence/2026-07-08/` —
`run14_univariate_leakage_test.txt`, `run14_univariate_leakage_train.txt`,
`eval_predictions_run15_by_representation.txt`, `run_artifacts_summary.txt`. The parquet inputs
are not tracked (large and regenerable); the evidence of what they produced is.
**Status of the investigation opened over the last six turns:** largely RESOLVED. Three of my
own hypotheses are REFUTED by this evidence and are retracted below.

---

## 1. What is now settled

### 1.1 There is NO univariate leak. (refutes my "near-label features" alarm)

The single most predictive feature in Run 14, taken alone, is `is_loss_of_function` at
**AUROC 0.7603** (test) / 0.7503 (train). Nothing reaches the 0.90 leak threshold. The full
top-of-table, test split:

| feature | standalone AUROC | nunique |
|---|---:|---:|
| is_loss_of_function | 0.7603 | 2 |
| af_log10 / af_raw | 0.7587 (inverted) | 114,315 |
| cadd_phred | 0.7496 | 7,447 |
| n_tools_pathogenic | 0.7439 | 5 |
| af_is_absent | 0.7318 | 2 |
| consequence_severity | 0.7218 | 7 |
| n_pathogenic_in_gene | 0.6902 | 215 |

**The HGMD / LOVD / ClinGen near-label features I raised as the leading explanation for AUROC
0.998 are ALL DEAD** — `hgmd_is_disease_mutation`, `hgmd_n_reports`,
`n_known_pathogenic_protein_variants`, `clingen_validity_score` all have nunique == 1.
`lovd_variant_class` is dead on test and nunique 4 / power 0.5004 on train (i.e. noise). **I was
wrong about this for three turns. There is no curated disease register doing any work in this
model.** Retracted.

### 1.2 The `n_pathogenic_in_gene` leak was LIVE in Run 14, but is not load-bearing

`nunique = 215` on the gene-disjoint TEST split (min/max −0.437 / 4.886), 100% finite. Post-fix
it would be identically zero on test. It is not. **So Run 14's held-out genes did carry counts
derived from their own held-out labels — the INCIDENT_2026-06-13 leak was live 18 days before it
was fixed.**

But its standalone power is only **0.6902**, and — decisively — the Run-15 gene-prevalence
ablation, which refits the *entire ensemble* (see 1.3), costs:

| ablation | test AUROC | delta vs full | delins AUROC delta |
|---|---:|---:|---:|
| full | 0.99817 | — | — |
| no_gene_prevalence | 0.99802 | **−0.00015** | −0.00602 |
| no_gene_level (4 cols) | 0.99783 | −0.00034 | −0.00709 |

Removing the leaked feature and retraining changes AUROC in the fourth decimal. **The leak was
real but not what produces the 0.998.** (Caveat: the ablation is regime-v1, post-fix, so it
measures the feature's *predictive* contribution, not the *leak's* magnitude in v0. The v0 leak's
magnitude cannot be measured from a v1 ablation; it would need a v0 refit with the column zeroed.
But the univariate power of 0.69 caps how much a single column can contribute either way.)

### 1.3 The ablation is VALID — it refits the whole ensemble. (refutes my "vacuous ablation" doubt)

`scripts/run9_ablations.py:666`: `ensemble.fit(X_train_abl, seq_tr, y_train)` — a full refit of
every base model and the stacker on the ablated matrix, not a stacker-only re-fit over cached
base predictions. I doubted this last turn on the strength of the `fit_seconds` gap (16,233 vs
2,293 vs 1,552). **The doubt was unfounded; the ablation is a legitimate leave-one-out retrain.**
The time gap is unexplained but does not invalidate the comparison — likely gradient-boosting
early-stopping or a warm cache on the later arms. `split_hashes["X_train_full"]` is recorded in
each manifest precisely to guarantee the arms share a training set (`run9_ablations.py:608-611`).

Retracted: "the ablation may be vacuous."

### 1.4 The ensemble genuinely discriminates within EVERY stratum. (refutes my Simpson's-paradox story)

I argued the 0.998 decomposes into "CADD-on-SNVs plus is_indel". That is true of `cadd_phred`
**alone** — it scores AUROC 0.9761 on SNVs and exactly 0.5000 (constant) on every indel class,
because CADD does not score indels. **But the ensemble does not behave that way.** On the
run15-full test predictions, stratified by representation:

| stratum | n | pos_rate | ensemble AUROC | auprc_lift |
|---|---:|---:|---:|---:|
| SNV | 290,019 | 0.1243 | 0.9980 | 7.98 |
| padded_insertion | 13,033 | 0.5773 | 0.9965 | 1.73 |
| delins | 1,659 | 0.7589 | 0.9736 | 1.31 |

The ensemble is strong on SNVs **and** on insertions **and** on delins. It is not merely reading
`is_indel`. My decomposition was wrong at the ensemble level. Retracted.

Note the `auprc_lift` column still does its job: padded_insertion's raw AUPRC of 0.9955 looks
triumphant, but at pos_rate 0.577 the lift is only 1.73. The lift is the honest number.

### 1.5 Per-base-model: the ensemble is not carried by one model

run15-full test, every base model scores AUROC 0.996-0.998 independently: catboost 0.99805,
lightgbm 0.99795, xgboost 0.99785, random_forest 0.99771, gradient_boosting 0.99769, svm 0.99675,
logistic_regression 0.99622. Seven models, all strong, all agreeing. That is not the signature of
a single leaked column; it is the signature of a genuinely separable (or genuinely circular —
see 2.1) problem.

---

## 2. What remains a live concern

### 2.1 Type-1 circularity is the surviving explanation for AUROC 0.998

With no univariate leak and no single load-bearing feature, the most likely reason a
gene-disjoint ClinVar AUROC reaches 0.998 — far above published metapredictors at 0.85-0.95 — is
**Type-1 circularity** (Grimm et al. 2015). The features `cadd_phred`, `revel_score`,
`sift_score`, `polyphen2_score`, `alphamissense_score`, `n_tools_pathogenic` were themselves
trained or calibrated on ClinVar. Predicting ClinVar labels from them means the test variants sat
in the *features'* training data. `n_tools_pathogenic` (an aggregate of exactly these tools)
scores 0.744 standalone; the tools collectively, in an ensemble, plausibly reach 0.99 on the
variants where they and ClinVar agree — which is exactly the tier-3-filtered population.

**This is not a bug and not a leak in the code.** It is a study-design limitation shared with much
of the variant-effect-prediction literature. It cannot be "fixed"; it must be MEASURED (hold out
variants absent from the tools' training sets, or report on a ClinVar-independent benchmark such
as a MAVE/DMS set) and DISCLOSED. It is the single most important caveat on every number this
project reports, and it belongs in METHODS, not in an incident log.

### 2.2 The coordinate bug is invisible to all of this

Every metric above is computed on splits where train and test share the padded-deletion
coordinate error identically (INCIDENT_2026-07-08 sec 3). `cadd_phred`, `af_log10`, and every
positional feature are CONSTANT (AUROC exactly 0.5000) on padded_deletion in both the univariate
panels and the ensemble strata — because those rows never received an annotation. The model
scores padded deletions using only the non-positional features (is_deletion, is_loss_of_function,
consequence). It cannot be doing variant-level pathogenicity prediction on deletions; it is doing
class-prior prediction. AUROC on that stratum (ensemble, full arm) is not even reported separately
in v1 because the tier filter removed 99% of them — the 3,384 padded deletions in the v1 test set
are the ~1% delins-and-survivors. cohort-v2 is still required.

### 2.3 Provenance defects, now confirmed in the artifacts

* **`scikit-learn: not_installed`** in every manifest, while the run computed `roc_auc_score`.
  The version capture queries the wrong distribution name. A falsehood in the provenance record.
* **`min_review_tier` absent from every manifest config.** The config carries only
  `{n_folds, seed}` (the ablation knobs), not the `DataPrepConfig`. So no manifest can pin its own
  cohort regime — the regime had to be inferred from `n_samples` (349,067 => v0; 304,711 => v1).
  `save_manifest(config=...)` must receive the data config, cohort MD5, and schema fingerprint.
* **`f1: None`** in all six reports — predates the PHASE5 addition; dates the artifacts, not a bug.
* **`run16/eval_report.json` has empty consequence AND gene breakdowns** (n_consequence_rows 0,
  n_gene_error_rows 0) — the silent-empty trap, in production.
* **35 of 78 features are dead** (constant) in Run 14: all AlphaFold structural, ESM-2, EVE, GNN,
  GTEx, OMIM, 1000G, dbSNP, phyloP, FinnGen, and the gnomAD constraint metrics. The model that
  reports 0.998 uses ~40 live features; the other 35 are plumbed but unpopulated. `gnn_score` is
  dead BY DESIGN (splits written before the GNN trains; run9_ablations.py:174-177).

### 2.4 The ECE is under-reported in every eval_report.json

`evaluator.py::_calibration_error` bins `[0.9, 1.0)`, dropping every `p == 1.0`. With Brier 0.007
and AUROC 0.998 this model emits `p == 1.0` constantly. `run15-full` reports ECE 0.00283; the
metric stack, closing the top bin, reports 0.0028 on the same predictions (they agree here because
the ensemble_prob rarely hits exactly 1.0 after stacking) — but the MCE of 0.1466 shows the worst
bin is real. Fix `_calibration_error` regardless; the divergence is data-dependent and will bite
on a run whose base models dominate.

---

## 3. Corrections to my own prior claims (this session)

| turn | claim | status |
|---|---|---|
| −6 | "SHORTCUT PRESENT" from the all-defaults signature | RETRACTED (probe rule measured indel-ness) |
| −3 | HGMD/LOVD/ClinGen near-label features likely explain 0.998 | REFUTED — all dead |
| −2 | the 0.998 decomposes into CADD-on-SNVs + is_indel (Simpson) | REFUTED at ensemble level; true only of CADD alone |
| −1 | the gene-prevalence ablation may be vacuous (stacker-only refit) | REFUTED — full ensemble refit, run9_ablations.py:666 |
| −1 | "no run artifact records prevalence" | REFUTED — EvaluationReport.prevalence exists for run10/15/16 |
| ongoing | AUROC 0.998 is "not credible" | REFINED — credible IF Type-1 circularity is the cause; still not a real-world generalization estimate |

The pattern: I repeatedly reached for a *leak* or a *bug* to explain an implausibly high number,
and the evidence keeps saying the number is a genuine-but-circular ensemble effect on a
tool-agreement-filtered cohort. That is a subtler and more important finding than a leak: the
model is real, and it is measuring the wrong thing.

---

## 4. What this means for the launch (item 4)

A smoke test confirms the pipeline runs. It cannot address 2.1 (circularity), 2.2 (coordinate
bug), or 2.3 (provenance). The metric stack now exists and is applied; the honest pre-launch
sequence is:

1. **cohort-v2**: fix padded-deletion coordinates, rebuild variant_id, rebuild NT windows.
   Re-annotate — the positional joins will now hit the 189,468 deletions.
2. Fix `_calibration_error` (p==1.0), `_bootstrap_ci` (stratify), manifest version+config capture.
3. Wire `run_phase2_eval.py` to `RunArtifactWriter` so run14/run17-class runs get real provenance.
4. Re-baseline Run 14's config on cohort-v2, and report metrics **stratified by representation**
   and with an explicit **Type-1 circularity disclosure** and, if possible, a ClinVar-independent
   validation stratum.
5. Then local smoke -> VM smoke -> launch.

The launch is not blocked by a bug anymore. It is blocked by the fact that the headline metric,
as currently produced, measures agreement with tools that were trained on the same labels — and
until that is disclosed and bounded, a higher number is not better science.
