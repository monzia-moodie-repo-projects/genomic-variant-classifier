# Run 14 Claim-Eligibility Record

**Date:** 2026-09-18 (third same-day revision — see revision note, §3, §5, and the revised blocker list)
**Subject:** `outputs/run14/full/models/ensemble.joblib` and its 10-model bundle
**Purpose:** Record, per claim type, what evidence has actually been established for Run 14 — not a single validity flag, but eligibility scoped to specific claims, per the six-claim framework below.

**Revision note (third same-day revision, supersedes the second):** `docs/incidents/INCIDENT_2026-05-31_run14-split-leakage.md` and `docs/incidents/INCIDENT_2026-05-31_null-key-leak.md` — both committed to this repository, both written 5 days after Run 14 trained, both about Run 14 specifically, neither previously read for this record — establish a more severe and more direct finding than anything in the prior two revisions: **measured, literal variant-ID overlap between Run 14's own train and test/val splits (247 train↔test, 115 train↔val, 46 val↔test), plus 2,125+129+409 within-split duplicates, plus 11,320 structural/null-key rows that should have been quarantined and were not.** This is direct data contamination, not gene-level feature leakage (§5.1, retained, still minor) or study-design circularity (§5.2, retained, still real). It is now the most severe confirmed finding in this record. Two further corrections from these documents: Run 14's ensemble is missing **two** modalities against its intended architecture (cnn_1d and a GNN/STRING-DB component), not one as previously recorded; and the run11/run14 provenance question, previously treated as resolved by confirming `config.model_dir` is functionally inert during inference, has a separate, unresolved dimension — the project's own contemporaneous incident report explicitly did not consider data attribution settled.

This record continues to supersede any blanket statement that "Run 14 is verified," and the claim that Run 14's cohort filename alone establishes identical historical bytes.

---

## Claim table and current status

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | These historical bytes are preserved | **Partially established** | See §1 |
| 2 | This configuration executes | **Established, with caveats; ensemble confirmed missing 2 of 11 intended modalities** | See §2 |
| 3 | This reproduces historical predictions | **Not established; run/data provenance has an unresolved dimension** | See §3 |
| 4 | This model is sensitive to annotation policy | **Not yet attempted** | See §4 |
| 5 | This model generalizes | **Not established — direct variant-level train/test contamination confirmed, in addition to a minor feature leak and unmeasured Type-1 circularity** | See §5 |
| 6 | This replacement is better | **Not applicable — no replacement built yet** | — |

---

## §1 — Artifact identity and custody

**Established:**
- `ensemble.joblib` SHA-256 matches Run 14's own `reproducibility_manifest.json` exactly: `82eed09bdedc945d92bb69c1764598b989a9274cdbe069098da2fd2aa96d6314`.
- `ensemble.manifest.json` SHA-256 matches exactly: `be818fed4c8cea08b284a490aa719eeec97eb8fc690c489a4b87ffab86d47aaf`.

**Not established:**
- Zero digests exist for the 10 individually-loaded model files under `ensemble_models/`. The SHA-256 verification above authenticates only the small orchestrator file.
- No historical digest exists for the training cohort file.
- The environment-metadata capture mechanism is independently documented as producing false entries (`scikit-learn: not_installed` while scikit-learn demonstrably ran, per `docs/audits/LEAKAGE_METRIC_ANALYSIS_2026-07-08.md`).

**Blocker:** no per-model digest exists anywhere to check against, and the environment-capture mechanism itself is documented as producing false entries.

---

## §2 — Complete load and valid inference

**Established, by direct, repeated measurement on this machine:**
- All 10 base models load successfully under three combined, documented interventions: `pathlib.PosixPath = pathlib.WindowsPath`; a scoped `torch.storage._load_from_bytes` monkey-patch forcing CUDA-tensor storages to deserialize onto CPU; explicit post-load correction of `kan_model._imodelsx_model.device` and `.model.to("cpu")`.
- `predict_proba()` executes end-to-end and returns valid `(n, 2)` probability rows summing to 1.0.
- Repeatability confirmed empirically: identical input, called twice, produces bit-for-bit identical output.
- `random_forest.joblib`'s absence from this local checkout and `config.model_dir`'s stale `run11` path were both diagnosed and explained (Drive archive; training-time checkpoint default respectively).

**Corrected: the ensemble is missing two modalities, not one.** `INCIDENT_2026-05-31_run14-split-leakage.md`, item 2: *"Reduced ensemble: skip_cnn=True (cnn_1d closure bug, B.D6) and string_db=None (GNN off). The headline came from ~9 of 11 models with two modalities dead."* The 10 base models confirmed loadable and functional this session (random_forest through deep_ensemble) are real and correctly enumerated — but the *intended* architecture apparently includes 11 components, and a GNN/STRING-DB-based component, separate from `cnn_1d`, was also disabled for Run 14. This session had only identified `cnn_1d`'s absence (via `--skip-cnn`) prior to reading this incident.

**Column-order finding, precisely scoped:** `predict_proba()` passes `X_tab.values` to every model except `catboost`; only `catboost._feature_names` retained a recorded column order. Using it vs. an importance-sorted order on identical synthetic values produced probability differences of `[0.452, 0.119, 0.169]`. Not proven to be the correct order for the other 9 models; repeatability rules out stochasticity as an alternative explanation for the magnitude observed.

**Executable-code identity:** `predict_proba()`/`load()` core logic verified identical, line for line, between Run 14's own training commit and current `main`, despite the two not being in a linear ancestor relationship. Evidence specific to these two methods; not extended to individual base-model wrapper classes.

---

## §3 — Reproduces historical predictions

**Not established. A provenance question previously treated as resolved has a dimension this record had not addressed.**

- No per-row predictions are committed anywhere in this repository.
- **Three independent row-count consistency signals** now support that today's `clinvar_grch38.parquet` matches what Run 14 loaded: 4,420,180 raw rows (matches `run14_master.log`); 1,700,687 post-label-filter rows (matches both a separately-logged figure and Run 14's own committed `data_quality_audit.csv`).
- **The bundle carries real OOF data** (`oof_predictions_`: 1,017,633 × 10; `oof_fit_indices_`: max 1,197,215; `oof_model_names_` populated) — training-fold data, not the headline val/test predictions. A first attempt to use it for a consistency check failed mechanically (indexed into the wrong, pre-split frame, producing a uniform ~0.48 AUROC across all 10 models) and was not pursued further, since §5's findings make a "successful" reproduction of this specific check substantially less informative regardless of outcome.
- **The run11/run14 provenance question is not fully resolved, correcting this record's own prior treatment of it.** This record previously stated `config.model_dir`'s stale `run11` path was "confirmed inert for inference" and treated that as closing the question. That conclusion is still true narrowly — the field is genuinely never read during `load()` or `predict_proba()` — but `INCIDENT_2026-05-31_run14-split-leakage.md`, written by the project's own contemporaneous investigation, raises a different, unresolved question: *"Confirm which run produced `outputs/run14/` before trusting run labels."* That is a question about data attribution — whether the artifacts under `outputs/run14/` are genuinely what a run labeled "14" produced, independent of whether any loaded config field is functionally used — and this record has not answered it. Row-count consistency (above) is suggestive but not dispositive on this specific question.

**Required evidence chain, still substantially unresolved:**

| Question | Status |
|---|---|
| Which source bytes did Run 14 read? | Three independent row-count matches; not digest-confirmed |
| What did historical preprocessing retain? | Label-filtering logic confirmed via direct historical-commit comparison |
| Which rows entered each partition? | **Now measured directly and found contaminated — see §5.3** |
| Is `outputs/run14/` genuinely Run 14's own output? | **Explicitly flagged unresolved by the project's own incident report; not settled here** |

---

## §4 — Sensitivity to annotation policy

**Not yet attempted**, blocked pending §3 and §5. `loeuf`, `pli_score`, `syn_z`, `mis_z` confirmed genuinely live (non-degenerate, real variance) in Run 14's actual feature matrix via its own committed `data_quality_audit.csv`; only `gene_constraint_oe`/`gene_is_constrained` are dead (constant zero, `nunique=1`) — a separate, unresolved zero-fill question, not the historical "defaults to loeuf" bug (ruled out directly: `loeuf`'s own distribution is non-zero, range −1.9 to 3.3). The upstream gnomAD census work (178-gene discrepancy census, MANE-tier verification, `syn.possible` trace) remains independently valid and has genuine bearing on this model once §3/§5 permit a valid experiment.

---

## §5 — Generalizes

**Not established. Three distinct, independently-confirmed findings, in order of severity.**

### 5.1 — NEW, MOST SEVERE: direct, measured variant-level train/test/val contamination

From `INCIDENT_2026-05-31_run14-split-leakage.md`, measured directly against `outputs/run14/full/splits`:

```
within-split duplicate variant_id:    train 2,125 / val 129 / test 409
cross-split variant_id overlap:       train&test 247 / train&val 115 / val&test 46
structural (null-key) variant_ids
  present in splits despite quarantine intent: 11,320 of 21,091
gene_symbol overlap train&test:       0  (gene-disjoint split mechanism itself worked)
```

The incident's own severity assessment: *"HIGH — the headline ~0.9974 test AUROC is inflated."*

**Mechanism, from `INCIDENT_2026-05-31_null-key-leak.md`:** the source cohort had 19,988 rows with null `ref`/`alt` alleles and 1,103 non-allele tokens (21,091 "structural" rows total), plus 4,203 duplicate `variant_id`s concentrated entirely within that structural bucket. The gnomAD allele-frequency join in `real_data_prep.py` builds its join key via `astype(str)`, which collapses a null allele to the literal string `"None"`/`"nan"` — distinct records then collide onto a shared key, and colliding keys could land on either side of the train/test boundary independent of the gene-disjoint split logic (which itself worked correctly — the contamination bypasses it rather than defeating it).

**Status of the fix, and why it does not resolve Run 14 specifically:** `scripts/clean_cohort.py` (Phase 0) resolved this at the data layer, producing `clinvar_grch38_clean.parquet` (4,399,089 rows, 0 null, 0 duplicate variant_id) — verified 2026-05-31, 5 days *after* Run 14 trained (2026-05-26). Run 14 necessarily trained on the pre-fix cohort. The same incident's own "Residual / follow-on (OPEN)" section states explicitly: *"Regenerate splits from `clinvar_grch38_clean.parquet`... the current splits derive the ~1.7M labeled subset from the pre-clean cohort"* — meaning even after the data-layer fix landed, split regeneration was a separate, explicitly open action item, not assumed to have happened automatically for any specific run without direct verification.

### 5.2 — `n_pathogenic_in_gene`: leak confirmed live, contribution measured as small

Retained from the prior revision. Commit `070ea735` (2026-06-13, 18 days after Run 14 trained) fixed a corpus-wide, pre-split computation of this feature. A full-ensemble-refit ablation already run and committed (`docs/audits/LEAKAGE_METRIC_ANALYSIS_2026-07-08.md`) moves AUROC by **−0.00015** when the feature is removed — real, confirmed live in Run 14 (`nunique=215` on the gene-disjoint test split), but not what produces the headline number.

### 5.3 — Type-1 circularity: unmeasured, still potentially the largest single factor

Retained from the prior revision. Several live features (`cadd_phred`, `revel_score`, `sift_score`, `polyphen2_score`, `alphamissense_score`) were trained or calibrated on ClinVar by their own creators — circular by study design, independent of any code defect. No single feature reaches a leak threshold alone (top univariate: `is_loss_of_function` at 0.76); all base models score in the 0.996–0.998 range independently, consistent with a genuinely-separable-looking but circular problem rather than one carried feature. Per the source document: *"the single most important caveat on every number this project reports."* Unmeasured against a ClinVar-independent benchmark.

**Relationship between the three findings:** 5.1 is direct data contamination — the model may have literally seen some test-set variants during training, independent of any feature-level question. 5.2 is a feature-level leak with a measured, small effect. 5.3 is a study-design limitation affecting the entire evaluation paradigm, unmeasured. All three are real; 5.1 is the most severe because it is the least ambiguous and most directly explains an inflated number, though its own precise magnitude (how much AUROC inflation 247+115+46 contaminated pairs and 11,320 misclassified structural rows actually produce, out of 1.7M total rows) has not itself been measured — the incident report states the concern and the mechanism but does not report a re-run AUROC on corrected splits for Run 14 specifically.

---

## §6 — This replacement is better

Not applicable.

---

## Overall disposition

**Run 14: retained as a historical research candidate, with three confirmed generalization concerns of decreasing but each independently real severity, plus an unresolved data-attribution question.** Execution and inference are directly demonstrated, with 2 of 11 intended architectural modalities confirmed absent, not 1 as previously recorded. Cohort byte-identity has three independent row-count consistency signals but no digest confirmation, and the deeper question of whether `outputs/run14/` is genuinely and exclusively Run 14's own output remains explicitly unresolved per the project's own contemporaneous investigation, not merely by omission in this record.

Most significantly: Run 14's own split files show **direct, measured variant-ID overlap between train and test/val** — not inferred, not theoretical, measured against the actual splits, with a confirmed mechanism (null-allele join-key collapse) and a fix that landed 5 days after Run 14 trained, too late to have applied to it. This is more severe than either the feature-level leak or the circularity concern, both of which remain independently real.

## Explicit blockers, ranked by severity and what they gate

1. **CONFIRMED: direct variant-level train/test/val split contamination** (§5.1) — measured against Run 14's own splits; mechanism confirmed (null-allele join-key collapse); the data-layer fix postdates Run 14's training by 5 days; split regeneration was separately flagged open and not confirmed for Run 14. Resolving this requires either locating regenerated, clean splits specifically used for Run 14 (unlikely to exist, given Run 14 predates the fix) or retraining on `clinvar_grch38_clean.parquet`-derived splits.
2. **Type-1 circularity, undisclosed and unmeasured** (§5.3) — requires a ClinVar-independent evaluation stratum to bound; not resolved by fixing splits or features, since it concerns the input tools' own training data.
3. **Padded-deletion coordinate bug** (prior revision, `docs/audits/LEAKAGE_METRIC_ANALYSIS_2026-07-08.md` §2.2) — invalidates variant-level scoring for that stratum specifically.
4. **`outputs/run14/` data attribution unresolved** — the project's own contemporaneous incident report explicitly did not consider this settled; this record's earlier "inert config field" finding addresses a narrower, different question and does not resolve it.
5. **No historical cohort digest; no per-model digests; environment-metadata mechanism confirmed to produce false entries.**
6. **`n_pathogenic_in_gene` leak** — confirmed live, correctly scoped as minor (~0.00015 AUROC).
7. **Gene-aware split not yet reproduced exactly** for the OOF consistency check — low priority given findings 1–3 make that check substantially less informative regardless of outcome.

## Recommended stopping rule for further recovery effort

Given §5.1, historical replay of Run 14's exact bytes is now actively counter-indicated as a goal in itself — a faithful reproduction would faithfully reproduce confirmed train/test contamination. The genuinely load-bearing next steps, in order: (a) determine whether any *later* run (15, 16, 17 — launch scripts exist, and `docs/audits/LEAKAGE_METRIC_ANALYSIS_2026-07-08.md` already references substantial "run15-full" evaluation data) was trained on `clinvar_grch38_clean.parquet`-derived, properly regenerated splits, since such a run — if it exists and is otherwise eligible — may already be a materially better candidate than Run 14 for any claim requiring generalization; (b) independent of that, construct or locate a ClinVar-independent evaluation population to bound §5.3, since that concern applies to any ClinVar-trained model regardless of split hygiene. Cohort-digest and per-model-digest recovery, and the `outputs/run14/` attribution question, can continue in parallel. Do not select a feature order, scaler, split, or cohort variant because it makes recomputed AUROC approach `0.9974` — that number is now understood to reflect, in an amount not yet decomposed, direct data contamination, a minor confirmed feature leak, and likely study-design circularity, none of which make it a target worth approaching on its own terms.


---

## Correction addendum (2026-09-19)

Per external review, three prior statements in this record require correction, and one new, historically-attributed finding is added. Per the same review's guidance, this record is not being rewritten; corrections are appended in place, preserving the original text and its basis above.

**1. Contamination-arithmetic units were mixed (originally in the section 5 cross-reference note).** The prior text combined "247+115+46 contaminated pairs" and "11,320... structural rows" over a shared 1.7M-row denominator without establishing whether these are the same kind of quantity, whether they overlap, or whether the full source cohort is even the correct evaluation denominator. No percentage or combined figure from that passage should be treated as a computed magnitude. The underlying counts from INCIDENT_2026-05-31_run14-split-leakage.md remain accurate; only the arithmetic built on top of them is withdrawn.

**2. The claim that CADD and REVEL were "trained... on ClinVar" is corrected.** Verified directly against both primary methods papers, not merely re-cited: CADD (Rentzsch et al., NAR 2019) trains on a binary distinction between simulated de novo variants and human/chimpanzee-fixed variants -- proxy-neutral vs. proxy-deleterious -- explicitly not ClinVar pathogenic/benign labels; the same paper states CADD's own tuning instead relies on curated disease-related datasets, which may include ClinVar-derived comparisons for evaluation, not training. REVEL (Ioannidis et al., AJHG 2016) trains on a separate collection of pathogenic and rare-neutral missense variants, explicitly excluding variants used to train its constituent tools, and is evaluated against ClinVar-derived test sets described as independent of its training data. Neither paper supports "trained on ClinVar" as a blanket description. What remains a genuine, unresolved question -- not settled by this correction -- is version-specific exposure: whether variants in REVEL's circa-2016 training set, or in any constituent tool's training data, overlap with the much larger and more recent ClinVar snapshot this project evaluates against. That requires a dedicated exposure audit, not assumed from either the original blanket claim or this correction.

**3. New finding, historically attributed to Run 14 specifically: the combiner was fit on one prediction representation and served a different one at inference, for random_forest, xgboost, and lightgbm.** Confirmed by direct inspection of `variant_ensemble.py` at both current `main` and Run 14's own training commit (80ac62ca7e83d35638274a01170d4c8f4f62c418): out-of-fold predictions feeding the blend-weight search are generated from the raw, uncalibrated base model (`cross_val_predict(model, ...)` at Run 14's commit; `self._leakfree_oof(name, model, ...)` in current code, added after Run 14 trained). Only afterward, for `{"xgboost", "lightgbm", "random_forest"}` specifically, is the model wrapped in `_IsotonicCalibrator` and that wrapper stored in `self.trained_models_` -- which is what `predict_proba()` actually calls at inference. This exact sequence, including the same three-model `_RECALIBRATE` set, is confirmed present at Run 14's own training commit, not only in later code. This means the blend weights Run 14's bundle carries were optimized against a prediction representation different from what those three models actually produce at inference time -- a real architectural property of the examined implementation, not yet quantified in its effect on the reported AUROC.

**4. Correction: there is no Run 16-successor "Run 17."** Per direct confirmation: Run 16 is the last actual run. Prior references in this record's stopping-rule section to "later run (15, 16, 17...)" should be read as "15, 16" only. Launch scripts named `launch_run17_*.sh` exist in the repository, but no execution evidence for a completed Run 17 has been found or should be assumed; those scripts appear to represent prepared, unexecuted configurations.

**Per the same external review, this record's own recommended next step is superseded:** rather than continuing recovery work on Run 14 itself, open a separate, dedicated Run 15 evidence record, and treat Run 14's record here as a stable historical comparator going forward, corrected as above but not further rewritten.
