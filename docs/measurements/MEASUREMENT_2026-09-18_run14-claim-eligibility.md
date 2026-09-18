# Run 14 Claim-Eligibility Record

**Date:** 2026-09-18
**Subject:** `outputs/run14/full/models/ensemble.joblib` and its 10-model bundle
**Purpose:** Record, per claim type, what evidence has actually been established for Run 14 — not a single validity flag, but eligibility scoped to specific claims, per the six-claim framework below.

This record supersedes any prior blanket statement that "Run 14 is verified" or that its reported `0.9974` validation AUROC is confirmed. It also supersedes the claim that Run 14's launch-record cohort path (`clinvar_grch38.parquet`) is definitively "the same file" later found defective — an identical pathname establishes a common locator, not identical historical bytes.

---

## Claim table and current status

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | These historical bytes are preserved | **Partially established** | See §1 |
| 2 | This configuration executes | **Established, with caveats** | See §2 |
| 3 | This reproduces historical predictions | **Not established** | See §3 |
| 4 | This model is sensitive to annotation policy | **Not yet attempted** | See §4 |
| 5 | This model generalizes | **Not established** | See §5 |
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

**Executable-code identity — not yet addressed at all:**
- `VariantEnsemble.load()` constructs an instance of the *currently installed* class and assigns historical state to it; `predict_proba()`'s behavior comes from currently-imported code, not code frozen at Run 14's `git_head` (`80ac62ca7e83d35638274a01170d4c8f4f62c418`, per `reproducibility_manifest.json`). No check has been performed confirming today's `variant_ensemble.py` behaves identically to that commit's version for this bundle. "Loads and predicts on this machine" and "faithfully reproduces Run 14's original behavior" are, at present, different, unequated claims.

---

## §3 — Reproduces historical predictions

**Not established.**
- Run 14's committed outputs (`per_model_metrics.csv`, `per_model_metrics_val.csv`) contain only aggregate summary metrics (AUROC, AUPRC, F1, MCC, Brier — one row per model, 11 rows total) — confirmed by reading their complete contents. **No per-row predictions are committed anywhere in this repository.**
- No cohort file path or digest is recorded in `reproducibility_manifest.json`'s `dataset` section (only row counts: `n_train: 1197216, n_val: 154404, n_test: 349067, n_features: 78`).
- `run14_master.log` records the actual CLI invocation: `--clinvar /workspace/data/processed/clinvar_grch38.parquet`. This filename is identical to the one a separate, later measurement (`SESSION_2026-09-12_measurement-overturns-the-claim.md`) found `_assert_clean_cohort` refuses today, citing 13,295 rows ending `:na:na` and 1,311 identifiers spanning more than one accession. **An identical filename establishes a common locator, not identical historical bytes**, and even confirmed identical bytes would not by itself establish that flagged records reached the fitted model or its validation partition — historical preprocessing may have filtered, deduplicated, or transformed them before the split.
- Whether `_assert_clean_cohort` existed as an enforced check at Run 14's training time (2026-05-26) has not been determined.
- Multiple ClinVar accessions per identifier are not inherently biological duplication or corruption; ClinVar's own identifier scheme distinguishes submission, variant–condition, and variant-level records, and which type(s) were merged in the 1,311-count has not been determined.

**Required evidence chain, none of it yet obtained:**

| Question | Status |
|---|---|
| Which source bytes did Run 14 read? | Unresolved — no historical digest recorded |
| What did historical preprocessing retain? | Unresolved |
| Which rows entered each partition? | Unresolved |
| Which defects reached those partitions? | Unresolved |
| Did identity failures cross partition boundaries? | Unresolved |

**Blocker:** full feature-matrix reconstruction, or recovery of an attributable archived prediction table, out-of-fold prediction set, or preprocessing cache, none of which have yet been searched for outside this repository (the original `vast.ai` instance's storage, if retained, is the most likely remaining source).

---

## §4 — Sensitivity to annotation policy (the gnomAD v4.1 → v4.1.1 question this investigation started from)

**Not yet attempted**, and correctly blocked from being attempted validly until §2's executable-code identity question and §3's cohort-identity question are further resolved — an annotation-policy comparison run through unverified inference code, on an unresolved cohort, would conflate at least three effects (runtime migration, code drift, annotation change) that the ruling's factorial design (`source_runtime_effects()`) is specifically built to separate.

**What is independently established and remains valid regardless:** the gnomAD-side work (the 178-gene canonical-tier discrepancy census, the MANE-tier verification across 18,394 dual-namespace pairs confirming `mane_pair_disagreement = 0`, the mechanistic trace to `syn.possible` differing by exactly 1 across a confirmed-identical set of 21 genes across all four affected metrics) is upstream of and independent from Run 14's own eligibility questions, and can proceed on its own track.

`run14_master.log` confirms Run 14's own constraint features were computed from `gnomad.v4.1.constraint_metrics.tsv` (v4.1, not v4.1.1) via `--gnomad-constraint`, and allele-frequency features from `gnomad_v4_exomes.parquet` via `--gnomad`. `--skip-cnn` confirms `cnn_1d`'s absence from the 10-model roster was deliberate, not an omission.

---

## §5 — Generalizes

**Not established**, and not attempted. Requires an independent, identity-resolved, leakage-audited evaluation population — none has been built. `n_pathogenic_in_gene` (the single highest-importance feature at `mean_importance ≈ 464`) has an unaudited lineage: whether evaluation labels were excluded from its computation, whether counts were recomputed within inner training folds, and whether the feature used information available at the intended prediction time are all open, high-priority questions distinct from the cohort-defect question in §3.

**Repository documentation, not yet cross-checked against Run 14's actual partitions:** `EnsembleConfig`'s own documentation reportedly records a historical calibration split by row rather than by gene, naming random_forest, xgboost, and lightgbm as affected, and identifying the Run 14–17 launch path as using it. This is prior repository documentation of a known issue, not a fresh measurement against Run 14's exact partition membership — it needs direct verification before Run 14's calibration metrics are treated as valid for genes unseen during training.

---

## §6 — This replacement is better

Not applicable. No replacement model or corrected bundle has been built. This claim requires a prespecified comparison against an eligible baseline, which does not yet exist.

---

## Overall disposition

**Run 14: retained as a historical research candidate.** Artifact identity partially confirmed (orchestrator file only). Execution and inference are directly demonstrated under three documented, necessary compatibility interventions, with the column-order hazard now measured with the stochasticity confound ruled out but not yet confirmed applicable beyond `catboost`. Historical prediction reproduction, cohort identity, and scientific generalization remain unestablished. Executable-code identity relative to Run 14's own recorded git commit has not been checked at all.

This supersedes both "Run 14 is fully verified" and "Run 14's launch record proves its AUROC used the confirmed-defective cohort."

## Explicit blockers, ranked by what they gate

1. **No per-model digests** — blocks any claim about the 10 individual fitted models' integrity beyond the small orchestrator file.
2. **No historical cohort digest or archived snapshot search performed** — blocks §3 entirely; also blocks a valid §4 (annotation-sensitivity) experiment until either resolved or explicitly bypassed via a new, independently-attributable input set.
3. **No executable-code equivalence check against Run 14's recorded `git_head`** — blocks distinguishing "this machine's current code reproduces Run 14" from "this machine's current code executes Run 14's weights under a new, uncharacterized configuration."
4. **`n_pathogenic_in_gene` lineage unaudited** — blocks §5 and materially affects confidence in the headline AUROC regardless of cohort-defect status.
5. **Row-vs-gene calibration split (per prior repository documentation) unverified against Run 14's actual partitions** — blocks treating Run 14's calibration metrics as valid for gene-disjoint evaluation.

## Recommended stopping rule for further recovery effort (per the three-outcome framework)

Continue bounded recovery — a defined search of the original `vast.ai` instance's retained storage (if any), attributable archives, and preprocessing caches — toward one of three outcomes: (a) original input contract and execution identity recovered → complete historical replay becomes possible; (b) inference contract recovered but cohort remains unresolved → Run 14 is retained for scoped *new* sensitivity experiments only, never for a historical-performance claim; (c) input semantics or fitted preprocessing prove irrecoverable → retire Run 14 as a quantitative baseline, preserve it for forensic reference only. Do not select a feature order, scaler, or cohort variant because it makes recomputed AUROC approach `0.9974` — that would convert the historical metric into a tuning target rather than a genuine reconstruction check.
