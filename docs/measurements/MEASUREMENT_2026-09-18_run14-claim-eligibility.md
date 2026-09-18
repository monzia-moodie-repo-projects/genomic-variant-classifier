# Run 14 Claim-Eligibility Record

**Date:** 2026-09-18 (updated same day — see §5 and the revised blocker list)
**Subject:** `outputs/run14/full/models/ensemble.joblib` and its 10-model bundle
**Purpose:** Record, per claim type, what evidence has actually been established for Run 14 — not a single validity flag, but eligibility scoped to specific claims, per the six-claim framework below.

This record supersedes any prior blanket statement that "Run 14 is verified" or that its reported `0.9974` validation AUROC is confirmed. It also supersedes the claim that Run 14's launch-record cohort path (`clinvar_grch38.parquet`) is definitively "the same file" later found defective — an identical pathname establishes a common locator, not identical historical bytes.

**This revision adds a confirmed finding materially more serious than anything previously recorded here: `n_pathogenic_in_gene`, Run 14's single highest-importance feature, was almost certainly computed with direct label leakage into the validation and test partitions, per a documented incident fix committed 18 days after Run 14 trained. See §5.**

---

## Claim table and current status

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | These historical bytes are preserved | **Partially established** | See §1 |
| 2 | This configuration executes | **Established, with caveats** | See §2 |
| 3 | This reproduces historical predictions | **Not established; a specific confound is now confirmed present** | See §3 |
| 4 | This model is sensitive to annotation policy | **Not yet attempted** | See §4 |
| 5 | This model generalizes | **Not established — confirmed label leakage in the top feature** | See §5 |
| 6 | This replacement is better | **Not applicable — no replacement built yet** | — |

---

## §1 — Artifact identity and custody

**Established:**
- `ensemble.joblib` SHA-256 matches Run 14's own `reproducibility_manifest.json` exactly: `82eed09bdedc945d92bb69c1764598b989a9274cdbe069098da2fd2aa96d6314`.
- `ensemble.manifest.json` SHA-256 matches exactly: `be818fed4c8cea08b284a490aa719eeec97eb8fc690c489a4b87ffab86d47aaf`.
- This is a meaningful check given documented history: the original training session's own `session_notes` record that the VM was destroyed while a save-verification gate was still showing FAIL on `ensemble.*` files, with a locator later confirming file *presence* (not byte-identity) at `outputs/run14/full/models/`.

**Not established:**
- **`ensemble.manifest.json` records only environment metadata** (Python version, platform, library versions) — confirmed directly by reading its complete contents. It contains **zero digests for the 10 individually-loaded model files** under `ensemble_models/` (`random_forest.joblib`, `xgboost.joblib`, `lightgbm.joblib`, `logistic_regression.joblib`, `gradient_boosting.joblib`, `catboost.joblib`, `tabular_nn.joblib`, `kan.joblib`, `mc_dropout.joblib`, `deep_ensemble.joblib`). The SHA-256 verification above authenticates only the small orchestrator file (config, blend weights, `saved_model_paths` metadata) — **not the fitted model weights themselves.**
- No historical digest exists for the training cohort file. Recovering a candidate file today and hashing it would not retroactively prove it is the historical training input.

**Blocker:** no per-model digest exists anywhere to check against. This would need to be established from a source outside the current repository (an archived training-session record, cloud storage metadata from the original `vast.ai` instance, or similar) if it can be recovered at all.

---

## §2 — Complete load and valid inference

**Established, by direct, repeated measurement on this machine:**
- All 10 base models load successfully under three combined, documented interventions:
  1. `pathlib.PosixPath = pathlib.WindowsPath` (cross-platform path unpickling).
  2. A scoped `torch.storage._load_from_bytes` monkey-patch forcing CUDA-tensor storages to deserialize onto CPU (verified against PyTorch's own `validate_cuda_device()` source and the maintainer-documented pattern in `pytorch/pytorch#16797`/`#43369`).
  3. Explicit post-load correction of `kan_model._imodelsx_model.device` and `.model.to("cpu")` — confirmed necessary by direct object-graph inspection, which found a real `KANModule` (`torch.nn.Module`) requiring explicit relocation, not just a string attribute.
- `predict_proba()` executes end-to-end and returns valid `(n, 2)` probability rows summing to 1.0.
- **Repeatability confirmed empirically:** identical input, called twice, produces bit-for-bit identical output (`max diff = 0.0`). This rules out MC-Dropout's own stochasticity as a confound in the column-order finding below — `MCDropoutWrapper.predict_with_uncertainty()` reseeds `np.random.default_rng(self.random_state)` from a fixed, persisted seed on every call.
- Two operational bugs found and fixed independently of the above: `random_forest.joblib` was absent from this local checkout (confirmed present, uncorrupted-by-size at 1,159,721,122 bytes, on the project's Google Drive archive — `.gitignore` deliberately excludes all `outputs/**/*.joblib`, so this is expected, not a data-loss signal); `config.model_dir` records a stale `\workspace\outputs\run11\full\models` path, now explained directly by `run14_master.log`'s recorded CLI invocation (`--output /workspace/outputs/run11/full`) — confirmed inert for inference (used only for training-time checkpointing and save-path defaults, never in `load()` or `predict_proba()`).

**Column-order finding, now more precisely scoped:**
- `predict_proba()`'s internal loop passes `X_tab.values` (a bare array) to every model except `catboost`, meaning correctness depends entirely on column *position* matching training order, with no runtime verification.
- Only `catboost._feature_names` (78 entries) retained a recorded column order anywhere in the loaded object graph — confirmed via a full recursive search across all 10 models' object trees, not a partial or assumed check.
- Using `catboost`'s recorded order vs. an importance-sorted order, on identical synthetic values, produced positive-class probability differences of `[0.452, 0.119, 0.169]` — large enough to flip a predicted class at a 0.5 threshold.
- **What this does and does not establish**, per the ruling's caveats, all of which still hold: `catboost`'s order is direct evidence about `catboost` specifically, not proven identical to what the other 9 models were trained on (they share no recorded order to cross-check against); the input was uniform random noise, not a representative genomic feature distribution; a threshold crossing on synthetic data is not a clinically meaningful misclassification. The repeatability check above removes stochasticity as an alternative explanation for the *magnitude* observed, but does not by itself prove `catboost`'s order is the historically correct one for the ensemble as a whole.

**Executable-code identity — checked directly for the two methods that matter most, not assumed:**
- `VariantEnsemble.load()` constructs an instance of the *currently installed* class and assigns historical state to it. Run 14's recorded `git_head` (`80ac62ca7e83d35638274a01170d4c8f4f62c418`) exists as a real commit object but is **not an ancestor of current `main`** — a genuine history divergence, not simple "N commits behind."
- Despite that divergence, a direct content comparison of `predict_proba()` between Run 14's commit and current `main` found the core prediction loop — `base_preds` construction, the `cnn_1d`/`catboost`/else branching, the blend formula, even the exact code comment about "Nelder-Mead convex blend" — **identical, line for line**. The one substantive addition, a call to `self._require_sequence_windows(...)`, was verified (by reading its implementation and confirming the early-return path) to be a genuine no-op for any roster without `cnn_1d` — which Run 14's is, per its recorded `--skip-cnn` flag. `load()` shows the same pattern: same format-version-2 handling, same per-model loading loop, same error-catching structure, no substantive change found.
- This is real, positive evidence specifically for these two methods. It does not extend to the individual base-model wrapper classes (`CatBoostVariantClassifier`, `MCDropoutWrapper`, etc.), which have not been diffed against Run 14's commit.

---

## §3 — Reproduces historical predictions

**Not established, and a specific confound in the bundle's own saved data is now confirmed.**

- Run 14's committed outputs (`per_model_metrics.csv`, `per_model_metrics_val.csv`) contain only aggregate summary metrics — no per-row predictions are committed anywhere in this repository.
- No cohort file path or digest is recorded in `reproducibility_manifest.json`'s `dataset` section (only row counts: `n_train: 1197216, n_val: 154404, n_test: 349067, n_features: 78`).
- `run14_master.log` records the actual CLI invocation: `--clinvar /workspace/data/processed/clinvar_grch38.parquet`. This filename is identical to the one a separate, later measurement found `_assert_clean_cohort` refuses today. An identical filename establishes a common locator, not identical historical bytes.
- **Positive consistency signal, independently obtained:** today's `data/processed/clinvar_grch38.parquet` has exactly 4,420,180 rows, matching `run14_master.log`'s recorded load count exactly. Replicating Run 14's own historical `_load_and_label` logic (confirmed against its training-time commit, not assumed from current code — see §5) on this file produces exactly 1,700,687 labeled rows, matching a separately-logged figure (`n_pathogenic_in_gene ... nonzero=1,700,687`) from the same training run. Two independent row-count matches on a 4.4-million-row file is meaningful, though not proof of byte-identity.
- **The bundle itself carries real out-of-fold (OOF) data:** `ens.oof_predictions_` (1,017,633 × 10), `ens.oof_fit_indices_` (max value 1,197,215, matching `n_train` exactly), and `ens.oof_model_names_` are all populated, not `None` — confirmed by direct inspection, not assumed from the loader's field list. These are training-fold cross-validation predictions, not the separate validation/test predictions behind `0.9974`/`0.9975` — useful provenance, not a substitute for those.
- **A first attempt to use this OOF data for a consistency check failed for a mechanical reason, now understood precisely, not worth pursuing further given §5:** `oof_fit_indices_` indexes into the *post-split, `reset_index`-ed* training partition produced by `_gene_aware_split` (`GroupShuffleSplit`, gene-disjoint), not the full 1,700,687-row post-labeling cohort. Indexing into the wrong frame produced a uniform ~0.48 AUROC across all 10 independently-trained models — the signature of row misalignment, not genuine non-generalization. The fix (replicating the gene-aware split itself, including its exact `random_state`) was identified but not completed, because §5's finding makes a "successful" reproduction of this kind less informative than it would otherwise be: it would only confirm the presence of the same confirmed leakage, not genuine historical parity.

**Required evidence chain, still substantially unresolved:**

| Question | Status |
|---|---|
| Which source bytes did Run 14 read? | Row-count-consistent with today's file; not digest-confirmed |
| What did historical preprocessing retain? | Label-filtering logic now confirmed via direct historical-commit comparison |
| Which rows entered each partition? | Split mechanism identified (`_gene_aware_split`, `GroupShuffleSplit`); exact reproduction not completed |
| Which defects reached those partitions? | `n_pathogenic_in_gene` leakage into val/test now confirmed — see §5 |
| Did identity failures cross partition boundaries? | Unresolved |

---

## §4 — Sensitivity to annotation policy (the gnomAD v4.1 → v4.1.1 question this investigation started from)

**Not yet attempted**, and correctly blocked from being attempted validly until §3's cohort-identity question is further resolved and §5's leakage finding is accounted for — an annotation-policy comparison on a model whose top feature is confirmed leaky would conflate the annotation effect with an artifact that has nothing to do with gnomAD at all.

**What is independently established and remains valid regardless:** the gnomAD-side work (the 178-gene canonical-tier discrepancy census, the MANE-tier verification across 18,394 dual-namespace pairs confirming `mane_pair_disagreement = 0`, the mechanistic trace to `syn.possible` differing by exactly 1 across a confirmed-identical set of 21 genes across all four affected metrics) is upstream of and independent from Run 14's own eligibility questions, and can proceed on its own track.

`run14_master.log` confirms Run 14's own constraint features were computed from `gnomad.v4.1.constraint_metrics.tsv` (v4.1, not v4.1.1) via `--gnomad-constraint`, and allele-frequency features from `gnomad_v4_exomes.parquet` via `--gnomad`. `--skip-cnn` confirms `cnn_1d`'s absence from the 10-model roster was deliberate, not an omission.

---

## §5 — Generalizes

**Not established. A specific, high-confidence lineage defect in Run 14's single highest-importance feature is now confirmed, with an exact mechanism, exact dates, and a measured effect size — this is the most consequential finding in this record.**

**The finding, precisely:**

`feature_importance.csv` ranks `n_pathogenic_in_gene` as Run 14's highest-importance feature by a wide margin (`mean_importance ≈ 464.3`, against `loeuf` at `≈ 273.5` for second place). Tracing its computation:

- Run 14 trained at commit `80ac62ca7e83d35638274a01170d4c8f4f62c418`, `2026-05-26 05:55:05 -0400`.
- Commit `070ea735e7c2056172211a9bba4c680cf7adf1b1`, `2026-06-13 20:29:08 -0400` — **18 days after Run 14 trained** — is titled `fix(leakage): train-only n_pathogenic_in_gene post-split`.
- Reading `enrich_gene_counts()` at Run 14's own training commit (not current code) confirms the pre-fix implementation directly: it computes `n_pathogenic_in_gene` via `df[df["label"] == 1].groupby("gene_symbol").size()` on the **full corpus, before the gene-disjoint train/val/test split**, then merges the result onto every row by gene symbol. The function's own docstring states its (incorrect) justification: *"Must be computed on the FULL labeled dataset BEFORE splitting to avoid information leakage (the count uses only labeled rows, not the test set)."*
- Because the split (`_gene_aware_split`, `GroupShuffleSplit`, gene-disjoint) happens *after* this computation, a held-out gene's count is built from that same gene's own labels — including its own validation- or test-set rows. A pathogenic validation variant contributes directly to the feature value later used to help classify that same variant.
- The fix commit's own comment records the measured magnitude directly, not as a re-derived estimate here: *"probe 2026-06-13: lone-feature test AUROC 0.7181 corpus vs 0.5000 train-only."* Computed with the leak, this one feature alone reaches 0.72 AUROC in isolation. Computed correctly, it carries no signal at all.
- Because the leaky computation is pre-split and corpus-wide, this directly implicates **the headline validation and test AUROC figures themselves** (`0.9974` / `0.9975`), not only the OOF training-fold data — a validation-set gene's feature value is built in part from that gene's own validation-set labels.

**What this does and does not establish:** this confirms a mechanism and a measured effect size for this one feature, evaluated alone. It does not by itself quantify how much of Run 14's full 78-feature, 10-model ensemble's `0.9974` figure is attributable to this leakage specifically, versus genuine signal from the other 77 features. That decomposition — training an equivalent model with the corrected, train-only feature and measuring the actual delta — has not been done and is the natural next step if Run 14's generalization is to be assessed further.

**Also still open, from before, now secondary to the above but not resolved:** whether evaluation labels contributed to any *other* fitted transformation (not just this one feature), whether the calibration split (row-based vs. gene-based, per prior repository documentation naming random_forest/xgboost/lightgbm and the Run 14–17 launch path) affects Run 14's specific partitions, and whether `n_pathogenic_in_gene`'s temporal availability (was this count known at the claimed prediction time, independent of the leakage question) was ever assessed.

---

## §6 — This replacement is better

Not applicable. No replacement model or corrected bundle has been built. This claim requires a prespecified comparison against an eligible baseline, which does not yet exist.

---

## Overall disposition

**Run 14: retained as a historical research candidate, with a confirmed, material defect in its highest-importance feature.** Artifact identity partially confirmed (orchestrator file only). Execution and inference are directly demonstrated under three documented, necessary compatibility interventions, with the core prediction logic verified unchanged against Run 14's own training-time commit. The column-order hazard is measured with the stochasticity confound ruled out but not yet confirmed applicable beyond `catboost`. Cohort identity has two independent positive row-count consistency signals but no digest-level confirmation. **Most significantly: `n_pathogenic_in_gene`, the model's single most important feature, was computed with a documented, dated, measured label-leakage mechanism that predates its own fix by 18 days relative to Run 14's training — directly implicating the reported `0.9974`/`0.9975` AUROC figures, not just internal training-fold statistics.**

This supersedes both "Run 14 is fully verified" and "Run 14's launch record proves its AUROC used the confirmed-defective cohort." It also supersedes any reading of the earlier row-count consistency signals as evidence that Run 14's reported performance is trustworthy — that performance now has a specific, confirmed, non-hypothetical reason to be inflated, independent of the cohort-identity question.

## Explicit blockers, ranked by severity and what they gate

1. **CONFIRMED: `n_pathogenic_in_gene` label leakage into validation and test partitions**, per the exact mechanism, dates, and measured magnitude in §5. This is no longer a hypothetical audit target — it is a dated, documented, measured defect in the training-time code, directly implicating the headline AUROC. Resolving this requires retraining with the corrected (post-`070ea735`) feature computation and measuring the actual performance delta; a repaired reproduction of Run 14's exact historical bytes would not resolve this, since the leakage is in the feature-generation logic, not the model weights.
2. **No historical cohort digest** — blocks confirming byte-identity of the training cohort beyond the two independent row-count consistency signals already obtained.
3. **No per-model digests** — blocks any claim about the 10 individual fitted models' integrity beyond the small orchestrator file.
4. **Gene-aware split not yet reproduced exactly** — blocks completing a corrected OOF consistency check; lower priority now given finding 1 makes a "successful" reproduction less informative than it would otherwise be.
5. **Row-vs-gene calibration split (per prior repository documentation) unverified against Run 14's actual partitions** — blocks treating Run 14's calibration metrics as valid for gene-disjoint evaluation, independent of finding 1.

## Recommended stopping rule for further recovery effort (per the three-outcome framework)

Given finding 1, the priority ordering changes: **historical replay of Run 14's exact bytes is no longer the most valuable next step**, since even a perfect reproduction would faithfully reproduce a confirmed leakage, not establish generalization. The more valuable next step is measuring the leakage's actual contribution to the reported AUROC — retraining with the corrected feature computation on an attributable cohort and comparing directly against Run 14's `0.9974`/`0.9975`. Continue bounded recovery on cohort and per-model digests in parallel, toward the same three outcomes as before, but do not treat their resolution as a prerequisite for beginning the corrected-feature retraining comparison. Do not select a feature order, scaler, or cohort variant because it makes recomputed AUROC approach `0.9974` — that number is now known to be partly an artifact, not a target worth approaching.
