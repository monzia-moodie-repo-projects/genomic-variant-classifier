# Run 14 Claim-Eligibility Record

**Date:** 2026-09-18 (twice-updated same day — see revision note below, §5, and the revised blocker list)
**Subject:** `outputs/run14/full/models/ensemble.joblib` and its 10-model bundle
**Purpose:** Record, per claim type, what evidence has actually been established for Run 14 — not a single validity flag, but eligibility scoped to specific claims, per the six-claim framework below.

This record supersedes any prior blanket statement that "Run 14 is verified" or that its reported `0.9974` validation AUROC is confirmed. It also supersedes the claim that Run 14's launch-record cohort path (`clinvar_grch38.parquet`) is definitively "the same file" later found defective — an identical pathname establishes a common locator, not identical historical bytes.

**Revision note (second same-day update):** the previous revision of this record overstated the practical significance of the `n_pathogenic_in_gene` leak by describing it as "directly implicating" the headline AUROC. `docs/audits/LEAKAGE_METRIC_ANALYSIS_2026-07-08.md` — a prior, rigorous, self-correcting investigation already committed to this repository — ran the actual ablation this record should have checked for first: a full ensemble refit with the leaked feature removed changes AUROC by **−0.00015**, not a material amount. That correction is made in §5 below, alongside a materially more important finding from the same document that this record previously missed entirely: **Type-1 circularity** — several of Run 14's features were themselves trained or calibrated on ClinVar by their own creators, making prediction of ClinVar labels from them circular by study design, independent of any code defect. This, not the leak, is the leading explanation for the 0.998-range AUROC, per the cited document's own analysis.

---

## Claim table and current status

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | These historical bytes are preserved | **Partially established** | See §1 |
| 2 | This configuration executes | **Established, with caveats** | See §2 |
| 3 | This reproduces historical predictions | **Not established; a specific confound is now confirmed present** | See §3 |
| 4 | This model is sensitive to annotation policy | **Not yet attempted** | See §4 |
| 5 | This model generalizes | **Not established — real leak confirmed but minor; Type-1 circularity is the leading, undisclosed explanation for the headline metric** | See §5 |
| 6 | This replacement is better | **Not applicable — no replacement built yet** | — |

---

## §1 — Artifact identity and custody

**Established:**
- `ensemble.joblib` SHA-256 matches Run 14's own `reproducibility_manifest.json` exactly: `82eed09bdedc945d92bb69c1764598b989a9274cdbe069098da2fd2aa96d6314`.
- `ensemble.manifest.json` SHA-256 matches exactly: `be818fed4c8cea08b284a490aa719eeec97eb8fc690c489a4b87ffab86d47aaf`.
- This is a meaningful check given documented history: the original training session's own `session_notes` record that the VM was destroyed while a save-verification gate was still showing FAIL on `ensemble.*` files, with a locator later confirming file *presence* (not byte-identity) at `outputs/run14/full/models/`.

**Not established:**
- **`ensemble.manifest.json` records only environment metadata** (Python version, platform, library versions) — confirmed directly by reading its complete contents. It contains **zero digests for the 10 individually-loaded model files** under `ensemble_models/`. The SHA-256 verification above authenticates only the small orchestrator file — **not the fitted model weights themselves.**
- No historical digest exists for the training cohort file.
- **New: this environment metadata is independently confirmed unreliable.** `docs/audits/LEAKAGE_METRIC_ANALYSIS_2026-07-08.md` documents `scikit-learn: not_installed` recorded in every manifest it examined, despite `roc_auc_score` (a scikit-learn function) actually running during those same runs — a confirmed falsehood in the provenance capture mechanism itself, not limited to Run 14's specific manifest but structural to how these manifests are generated.

**Blocker:** no per-model digest exists anywhere to check against, and the environment-capture mechanism itself is documented as producing false entries.

---

## §2 — Complete load and valid inference

**Established, by direct, repeated measurement on this machine:**
- All 10 base models load successfully under three combined, documented interventions:
  1. `pathlib.PosixPath = pathlib.WindowsPath` (cross-platform path unpickling).
  2. A scoped `torch.storage._load_from_bytes` monkey-patch forcing CUDA-tensor storages to deserialize onto CPU (verified against PyTorch's own `validate_cuda_device()` source and the maintainer-documented pattern in `pytorch/pytorch#16797`/`#43369`).
  3. Explicit post-load correction of `kan_model._imodelsx_model.device` and `.model.to("cpu")` — confirmed necessary by direct object-graph inspection, which found a real `KANModule` (`torch.nn.Module`) requiring explicit relocation, not just a string attribute.
- `predict_proba()` executes end-to-end and returns valid `(n, 2)` probability rows summing to 1.0.
- **Repeatability confirmed empirically:** identical input, called twice, produces bit-for-bit identical output (`max diff = 0.0`). This rules out MC-Dropout's own stochasticity as a confound in the column-order finding below.
- Two operational bugs found and fixed independently: `random_forest.joblib` was absent from this local checkout (confirmed present on the project's Google Drive archive; `.gitignore` deliberately excludes all `outputs/**/*.joblib`, expected, not a data-loss signal); `config.model_dir` records a stale `\workspace\outputs\run11\full\models` path, explained by `run14_master.log`'s recorded CLI invocation (`--output /workspace/outputs/run11/full`) — confirmed inert for inference.

**Column-order finding, precisely scoped:**
- `predict_proba()`'s internal loop passes `X_tab.values` (a bare array) to every model except `catboost`, meaning correctness depends entirely on column *position* matching training order, with no runtime verification.
- Only `catboost._feature_names` (78 entries) retained a recorded column order anywhere in the loaded object graph — confirmed via a full recursive search across all 10 models' object trees.
- Using `catboost`'s recorded order vs. an importance-sorted order, on identical synthetic values, produced positive-class probability differences of `[0.452, 0.119, 0.169]`.
- `catboost`'s order is direct evidence about `catboost` specifically, not proven identical to what the other 9 models were trained on; the input was synthetic, not representative genomic data; a threshold crossing on synthetic data is not a clinically meaningful misclassification. Repeatability rules out stochasticity as an alternative explanation for the magnitude, but not that `catboost`'s order is universally correct.

**Executable-code identity — checked directly for the two methods that matter most:**
- Run 14's recorded `git_head` (`80ac62ca7e83d35638274a01170d4c8f4f62c418`) exists as a real commit object but is **not an ancestor of current `main`** — a genuine history divergence.
- Despite that, a direct content comparison of `predict_proba()` between Run 14's commit and current `main` found the core prediction loop identical, line for line, including the exact code comment about "Nelder-Mead convex blend." The one substantive addition, `self._require_sequence_windows(...)`, was verified by reading its implementation to be a genuine no-op for any roster without `cnn_1d` — which Run 14's is, per its recorded `--skip-cnn` flag. `load()` shows the same pattern.
- This evidence is specific to these two methods; it does not extend to the individual base-model wrapper classes, which have not been diffed against Run 14's commit.

---

## §3 — Reproduces historical predictions

**Not established, and a specific confound in the bundle's own saved data is confirmed.**

- No per-row predictions are committed anywhere in this repository — only aggregate summary metrics.
- No cohort file path or digest is recorded in `reproducibility_manifest.json`.
- **Positive consistency signal, independently obtained:** today's `data/processed/clinvar_grch38.parquet` has exactly 4,420,180 rows, matching `run14_master.log`'s recorded load count exactly. Replicating Run 14's own historical `_load_and_label` logic (confirmed against its training-time commit, not current code) on this file produces exactly 1,700,687 labeled rows, matching a separately-logged figure from the same training run — and independently matching the row count in Run 14's own committed `data_quality_audit.csv` (`n_total: 1700687` throughout). Three independent matches now, not two.
- **The bundle carries real OOF data** (`oof_predictions_`: 1,017,633 × 10; `oof_fit_indices_`: max 1,197,215, matching `n_train` exactly; `oof_model_names_` populated) — training-fold cross-validation predictions, not the separate validation/test predictions behind `0.9974`/`0.9975`.
- **A first attempt to use this OOF data for a consistency check failed for a mechanical, now-understood reason:** `oof_fit_indices_` indexes into the post-split, `reset_index`-ed training partition produced by `_gene_aware_split` (`GroupShuffleSplit`, gene-disjoint), not the full 1,700,687-row cohort. Indexing into the wrong frame produced a uniform ~0.48 AUROC across all 10 models — the signature of row misalignment. Not pursued further, since §5 makes a successful reproduction of this specific check less informative than it would otherwise be.

---

## §4 — Sensitivity to annotation policy (the gnomAD v4.1 → v4.1.1 question this investigation started from)

**Not yet attempted**, blocked pending §3 and §5.

**Directly relevant new finding:** Run 14's own committed `data_quality_audit.csv` confirms, per-column, which gnomAD-sourced features are genuinely live in this exact model: `pli_score` (11,906 unique values), `loeuf` (1,910 unique values), `syn_z` (15,296 unique values), and `mis_z` (14,960 unique values) are all real, non-degenerate, standardized (`std ≈ 1.0`) features. **`gene_constraint_oe` and `gene_is_constrained` are confirmed dead** — constant at exactly `0.0` across all 1,700,687 rows, `nunique=1`. This is a genuinely different value from `loeuf`'s own distribution (range −1.9 to 3.3), ruling out the historical "silently defaults to loeuf" bug documented elsewhere in this codebase as the cause — the zero-fill mechanism here is a separate, unresolved question. Practically: `loeuf` — the exact metric this session's entire gnomAD v4.1.1 investigation concerns — is confirmed to be a real, substantial input to Run 14's actual predictions, giving that upstream work genuine bearing on this model, once §3/§5 permit a valid sensitivity experiment.

What remains independently established: the 178-gene canonical-tier discrepancy census, the MANE-tier verification across 18,394 dual-namespace pairs (`mane_pair_disagreement = 0`), and the `syn.possible` off-by-one trace across a confirmed-identical set of 21 genes. `run14_master.log` confirms constraint features came from `gnomad.v4.1.constraint_metrics.tsv` (v4.1) via `--gnomad-constraint`.

---

## §5 — Generalizes

**Not established. Two distinct findings, corrected and added this revision — one real but minor, one likely major and previously undisclosed.**

### 5.1 — `n_pathogenic_in_gene`: leak confirmed live in Run 14, but its measured contribution is small

- `feature_importance.csv` ranks this feature highest by a wide margin (`mean_importance ≈ 464.3` vs. `loeuf` at `≈ 273.5`).
- Run 14 trained at commit `80ac62ca...`, `2026-05-26 05:55:05 -0400`. Commit `070ea735...`, `2026-06-13 20:29:08 -0400` — 18 days later — is titled `fix(leakage): train-only n_pathogenic_in_gene post-split`, confirmed by directly reading `enrich_gene_counts()` at Run 14's own training commit: it computed this count corpus-wide, pre-split, merged onto every row by gene symbol, on the (incorrect) stated reasoning that using only labels made it leakage-safe.
- **Corrected assessment, per `docs/audits/LEAKAGE_METRIC_ANALYSIS_2026-07-08.md`, a full-ensemble-refit ablation already run and committed to this repository:** removing this feature and retraining (`ensemble.fit(...)` on the ablated matrix, confirmed not a stacker-only shortcut) moves AUROC from `0.99817` to `0.99802` — **a delta of −0.00015**. Standalone univariate power for this feature is `0.6902`, not enough to be "the" explanation for a 0.998-range model. The leak was genuinely live in Run 14 (`nunique=215` on the gene-disjoint test split — a fixed feature would show zero there) but is not what produces the headline number. The previous revision of this record overstated this.

### 5.2 — Type-1 circularity: the leading, undisclosed explanation for the headline AUROC

Per the same document, section 2.1, not previously reflected in this record:

- No single feature reaches a leak threshold (0.90 univariate AUROC); the top standalone feature, `is_loss_of_function`, reaches only `0.7603`.
- Several of the model's live features — `cadd_phred`, `revel_score`, `sift_score`, `polyphen2_score`, `alphamissense_score` — were themselves trained or calibrated by their own creators on ClinVar, the same label source Run 14 is evaluated against. Predicting ClinVar labels using tools trained on ClinVar labels is circular by study design, independent of any code defect: the evaluated variants sat inside the *features'* own training data.
- Quoting directly: *"This is not a bug and not a leak in the code. It is a study-design limitation shared with much of the variant-effect-prediction literature. It cannot be 'fixed'; it must be MEASURED... and DISCLOSED. It is the single most important caveat on every number this project reports."*
- Every base model independently scores AUROC 0.996–0.998 (catboost 0.99805 through logistic_regression 0.99622, per the same document's run15-full analysis) — consistent with a genuinely separable-looking but circular problem, not a single leaked column carrying seven independently-trained architectures.

### 5.3 — A separate, unaddressed defect: the padded-deletion coordinate bug

Same document, section 2.2: positional features (`cadd_phred`, `af_log10`, and others) are constant (AUROC exactly 0.5000) on padded-deletion variants specifically, because those rows never received a real annotation — a coordinate bug, not a modeling limitation. On this stratum, the model can only be doing class-prior prediction, not genuine variant-level pathogenicity assessment. Not yet resolved (`cohort-v2`, per the same document's own recommended sequence, is required first).

**What none of this establishes:** how much of the 0.998-range AUROC is attributable to circularity specifically versus genuine, disclosed-worthy predictive signal from features not implicated in it. That decomposition — evaluating on a ClinVar-independent benchmark (a MAVE/DMS set, as the source document suggests) — has not been done.

---

## §6 — This replacement is better

Not applicable. No replacement model or corrected bundle has been built.

---

## Overall disposition

**Run 14: retained as a historical research candidate, with two confirmed generalization concerns of materially different severity.** Artifact identity partially confirmed (orchestrator file only, and the environment-metadata mechanism is independently documented as producing false entries). Execution and inference are directly demonstrated under three necessary compatibility interventions, with core prediction logic verified unchanged against Run 14's own training-time commit. Cohort identity now has three independent row-count consistency signals but no digest-level confirmation.

On generalization: the `n_pathogenic_in_gene` leak was genuinely live but, per an already-completed full-ensemble-refit ablation, contributes only ~0.00015 AUROC — real, but not the explanation for the headline number. **The actual leading explanation, previously undisclosed in this record, is Type-1 circularity: several input features were trained on the same label source Run 14 is evaluated against, by design, independent of any code defect** — described in the source document as the single most important caveat on every metric this project reports. A separate coordinate bug additionally invalidates variant-level scoring specifically for padded deletions.

## Explicit blockers, ranked by severity and what they gate

1. **Type-1 circularity, undisclosed and unmeasured** — the leading explanation for the 0.998-range AUROC; requires a ClinVar-independent evaluation stratum (e.g., a MAVE/DMS benchmark) to bound, per the source document's own recommendation. Not addressed by retraining on cleaner ClinVar data alone, since the circularity is in the *features*, not the labels.
2. **Padded-deletion coordinate bug** — invalidates variant-level scoring for that stratum specifically; requires the `cohort-v2` fix described in the source document before any stratum-level claim about deletions is trustworthy.
3. **No historical cohort digest** — three independent row-count consistency signals now obtained, still no byte-level confirmation.
4. **No per-model digests, and the environment-metadata capture mechanism is independently confirmed to produce false entries** (`scikit-learn: not_installed` while scikit-learn demonstrably ran).
5. **`n_pathogenic_in_gene` leak** — confirmed live, now correctly scoped as minor (~0.00015 AUROC) rather than headline-explaining. Still worth fixing in any retrain, but not a priority driver on its own.
6. **Gene-aware split not yet reproduced exactly** — blocks completing the corrected OOF consistency check; low priority given findings 1–2 make that check less informative regardless of outcome.

## Recommended stopping rule for further recovery effort

Given 5.2, the priority changes again from the previous revision: **neither historical byte-level replay nor a corrected-feature retrain is the most valuable next step.** Both would still be evaluated against ClinVar, and the circularity concern applies regardless of which specific ClinVar snapshot or feature-leakage state is used. The genuinely load-bearing next step is constructing or locating a ClinVar-independent evaluation population (a MAVE/DMS-derived benchmark, or another source not used to calibrate the input tools) and measuring Run 14 — or any successor — against it. Cohort-digest and per-model-digest recovery can continue in parallel but should not be treated as blocking this. Do not select a feature order, scaler, or cohort variant because it makes recomputed AUROC approach `0.9974` — that number is now understood to reflect, in significant and currently unmeasured part, agreement with tools evaluated on their own training population, not a target worth approaching on its own terms.
