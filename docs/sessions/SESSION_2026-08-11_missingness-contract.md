# SESSION 2026-08-11 / 2026-08-12 -- the missingness contract

**Author: Monzia Moodie**

Seven commits, `51dfe89` -> `4e5c9b4`. Five defect families closed, one made
visible. The suite moved from 4,487 collected with an unmeasured working tree to
4,620 collected with zero failures.

Every defect closed in this session has the same shape:

> **An absence rendered as a measurement.**

That sentence is not a theme imposed afterwards. It is what each measurement
independently found.

---

## 1. What was wrong, in the order it was discovered

### DUPLICATE-1A -- two model features were bit-identical

`gene_constraint_oe` and `loeuf` were identical in the model matrix: `identical
= True`, maximum absolute difference `0.0`, correlation `1.0`, across 1,038,974
rows. No producer ever supplied `gene_constraint_oe`, so `engineer_features`
fell back to `loeuf`.

LOEUF is the **upper bound of the ninety per cent confidence interval** around
the loss-of-function observed/expected ratio. It is not the ratio. `lof.oe` sat
two columns away in the same gnomAD file, unread.

**Repaired in `7161132`.** The connector now extracts `lof.oe`, and its
arithmetic is asserted against `lof.obs / lof.exp` at a tolerance of 5e-4 -- the
measured rounding envelope is 3.25e-4.

### CONSTRAINTTRANSCRIPT-1 -- transcript selection by source row order

Selection was `drop_duplicates(subset=["gene"], keep="first")` under a comment
asserting *"first = MANE transcript in gnomAD ordering"*. No `mane_select`
filter existed anywhere in the file.

Measured across 211,523 source rows: first-row selection disagrees with MANE
Select for **5,468 of 17,473 genes (31.3%)**, median absolute LOEUF difference
0.039, maximum 1.689, with **132 genes crossing the 0.35 constrained boundary**.

Tier decomposition of the repaired ladder: 17,486 MANE Select, 696
canonical-only, 21 with no declared transcript. Of the canonical tier, zero
genes carry two Ensembl rows, so the uniqueness guard cannot trip.

**Repaired in `7161132`.**

### CONSTRAINTFILL-1 -- missingness fabricated as biological tolerance

Three levels fabricated a value: `_safe_float(..., 1.0)` at parse time, the
`ConstraintScores` class defaults, and `.fillna(CONSTRAINT_DEFAULTS[col])` in
`annotate_dataframe`.

An observed/expected ratio of **1.0 means observed equals expected** -- the gene
is completely tolerant of loss-of-function variation. A gene with **no data**
was recorded as **unconstrained**, and the model could not distinguish the two.
It then propagated: `1.0 < 0.35` is `False`, so every gene without data was also
recorded as not constrained.

Measured: `gene_constraint_oe` and `loeuf` share modal fraction **0.055397** in
the Run 15 matrix -- roughly **57,557 rows** carrying an asserted value that was
never measured. Of those, about **1,080 are genes that MATCHED** the constraint
index and whose `syn_z` and `mis_z` are genuine: the fill was overwriting a real
absence, not padding an unmatched gene.

Then `engineer_features` ended with `feats = feats.fillna(0.0)` -- and **0.0 on
the observed/expected scale is maximal apparent constraint**, strictly worse
than the 1.0 it replaced. That sweep sat three quarters through a 423-line
function, so the fifteen features built after it were invisible to its own NaN
count. Across **121 run logs the warning had never fired**, because every
upstream path fabricated a value first. Dead code kept alive by the defects it
would otherwise reveal.

**Repaired in `7161132` (connector half) and `48985d6` (feature-engineering
half).** `gene_is_constrained` is now three-valued -- 1, 0, or NaN -- because
`np.nan < 0.35` evaluates `False` and would recreate the conflation one layer
down.

### HARNESS-NULL-1 -- the instrument had the defect it detects

Stage 5 of the correctness harness computed its silent-zero verdict as

```python
zero_rate = float((s.fillna(0) == 0).mean())
```

which makes NaN identical to zero **inside the diagnostic**. Measured: the
reference slice's `gene_constraint_oe`, whose 200 values were **all missing and
none zero**, was reported as

```
feature 'gene_constraint_oe' is 100% zero (>= 95%) and non-binary
- probable silent-zero connector (connector-dead class)
```

**Repaired in `0a7a553`.** Zero rate is computed among observed values only, and
missingness is its own finding with its own wording.

**Measured after installation:** all **24** features in `KNOWN_ZERO_DEFAULT` are
genuinely zero-filled, not missing -- 24 of 24 messages carry the new `OBSERVED`
wording, 0 the old. Their recorded reasons survive scrutiny.

### The serving-time failure created by the repair

Removing the fabrication had a consequence at the serving boundary, found by
traceback:

```
sklearn/utils/validation.py:182
ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively.
```

`test_save_and_load_roundtrip` fits on synthetic data with no NaN, so training
succeeded. `predict_single` sends one **bare** variant through
`engineer_features`, which now correctly returns NaN, and the matrix went
straight to logistic regression. No scaler was involved:
`from_variant_ensemble` defaults `scaler=None`.

**Repaired in `b22b63b`.** `VariantEnsemble.fit` records `self.preprocessor_`,
fitted on `X_tab_fit` -- the training fold **after** the calibration carve.
`InferencePipeline` carries it and applies it at **both** `engineer_features`
call sites.

### NEURALNAN-1 -- a zero-variance guard deleting features

`TabularNNClassifier.fit` selected columns with `X.var(axis=0) > 0.0`. `np.var`
over a column containing any NaN returns NaN, and `NaN > 0.0` is `False`, so the
guard silently reclassified *"contains a missing value"* as *"is degenerate"*
and deleted the whole feature.

Measured with a responsiveness control at two sample sizes:

| estimator | delta on the NaN-bearing column | delta on columns never missing |
|---|---|---|
| `tabular_nn` | **0.000000** | 0.28 - 0.99 |
| `mc_dropout` | **0.000000** | 0.28 - 0.99 |
| `deep_ensemble` | **0.000000** | 0.28 - 0.99 |

At n=400 that column still held **360 observed values**, and perturbing them
changed nothing. The mask is pickled with the estimator, so the deletion
persisted into prediction.

**Made visible in `4e5c9b4`; NOT closed.** The same columns are still dropped -- 
verified identical over all five reachable column kinds -- but a loss to
missingness is now recorded separately from degeneracy and logged by name.

---

## 2. The estimator capability table, measured not remembered

Under scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0, CatBoost 1.2.10,
PyTorch 2.11.0+cpu:

| behaviour | estimators |
|---|---|
| routes missing natively | `random_forest` 0.892, `xgboost` 0.962, `lightgbm` 0.978, `catboost` 0.981 |
| refuses | `gradient_boosting`, `logistic_regression`, `svm`, `svm_bagged_rbf` |
| propagates to non-finite | `kan` |
| deletes the feature | `tabular_nn`, `mc_dropout`, `deep_ensemble` |
| not tabular | `cnn_1d` (one-hot sequence windows) |

Two entries corrected a draft written from recollection:

- **`random_forest` is NATIVE.** The draft had it `REQUIRES_NUMERIC`, which
  would have median-imputed the largest tree model while three siblings learned
  their own routing -- making any gap between them partly an artefact of
  preprocessing.
- **`StandardScaler` preserves NaN** and ignores it when fitting. Scaling was
  never what broke on missing values; the downstream estimator was.

**LIGHTGBM-DEGENERATE-1:** at n=8 LightGBM returned a constant, because
`min_data_in_leaf` defaults to 20. An earlier probe read that as native
tolerance. At n=400 it is fully responsive. *A capability probe must establish
responsiveness before interpreting a null result.*

---

## 3. Preprocessing leakage, measured

A preprocessor fitted across a split gives a median of **0.4** where a
train-only fit gives **0.25**, and the row whose value is imputed sits in the
half that moved the statistic. Imputation uses a **fitted** median, so
preprocessing participates in the statistical model.

The invariant: **one fitted state per statistical training partition** -- 
fold-specific during out-of-fold work, final-fit-specific for the serialised
artefact.

---

## 4. Register -- items opened and not yet closed

| identifier | finding |
|---|---|
| **NEURALNAN-1** | visible, not closed. Three of thirteen estimators still lose any feature carrying a missing value. Closure requires per-estimator rendering and the three-arm availability ablation. |
| **PIPELINE-PREP-DUP-1** | `engineer_features` is called at `pipeline.py:239` and `:317` in two methods that duplicate forty lines and share no `_prepare`. They diverge only in return shape. |
| **SCRIPT-SYSPATH-IDIOM-1** | 55 `sys.path` insertions across 43 scripts, because the package is not installed into the virtual environment. `build_alphafold_parquet.py:64` executes at module scope, so `exec_module` leaks the entry and seven `test_alphafold` tests error at any path but the canonical one. A packaging defect, not a test defect. |
| **COMPLETENESS-UNPERSISTED-1** | `ensemble_completeness_` and `dropped_models_` are read by nothing outside `variant_ensemble.py`. The code's own comment says completeness should be *"a RECORDED FACT that downstream reporting can assert on"*; it never reaches an artefact. |
| **ARTIFACT-VOCABULARY-1** | `run_phase2_eval.py` writes `metrics.json` (14 flat keys, no provenance); `train.py` writes `metrics_v1.json` (nested records, feature names, annotation sources). Run 16 used one, Run 17 will use the other. Neither carries roster completeness. |
| **ENTRYPOINT-DIVERGENCE-1** | Two training entrypoints with different argument surfaces. `train.py` hardcodes `min_review_tier=3` and leaves `gnomad_constraint_path=None` without `--gnomad-constraint`. Run 16 and Run 17 are therefore not directly comparable. |
| **RUN16-UNLOADABLE-1** | `ensemble_v1.joblib` contains a `PosixPath` in its config; unloadable on Windows. Two fields: `output_dir`, `model_dir`. |
| **RUN16-MODEL-MISSING-1** | `random_forest` is declared in `saved_model_paths` with `save_errors: {}` and absent from `ensemble_v1_models/`. |
| **RUN16-SCALER-MISSING-1** | No `scaler.joblib`. Cause established: `train.py` has no such write; only `run_phase2_eval.py:595` does. |
| **SCALER-UNFITTED-1** | `run_phase2_eval.py:595` dumps `prep.scaler` unconditionally; with `scale_features=False` that is an **unfitted** `StandardScaler`, and `export_model.py:158`'s warning describes a state the unconditional dump makes unreachable. |
| **LEAKREMAP-DUP-1** | The train-only leakage remap has two independent implementations, legacy and v2, deliberately duplicated. A fix to one will not reach the other. |
| **SKIPPOLICY-DUP-1** | Seven mechanisms remove estimators across four files; four cannot distinguish a decision from a failure. |
| **PREPCHECK-STALE-1** | `run17_prepcheck` holds 87 columns against the current 95-feature contract and predates the missingness change. Not a valid reference for post-contract questions. |
| **PIPELINEDOC-1** | `pipeline.py:38` says *"locked to the exact 79 columns"*; the list holds 95. `test_api.py:192` says absent fields *"default to population-median values"*; they have not since `48985d6`. |
| **WORKTREE-EOL-DRIFT-1** | 102 Python files are CRLF in the working tree against an LF index and an `eol=lf` attribute. Benign for commits; load-bearing for byte-exact tooling. |
| **GHTOKEN-1** | An invalid `GITHUB_TOKEN` environment variable masked a valid keyring credential. Git pushes were unaffected -- the two authenticate by different paths, so a green push is no evidence that `gh` works. |

---

## 5. Instrument failures -- mine

Recorded because they cost more time than any defect in the code, and because
the pattern is stable enough to be actionable.

**Seven instances of one class: a check whose subject contains an explanation of
the thing being checked, matched against raw text.** A structural check, an
installer's runtime guard, a probe self-check, an assertion-replacement check, a
roster-probe check, the `catalogue.py` shadow check, and -- most tellingly -- the
apply-step guard in the NEURALNAN-1 installer, **two hundred lines above a
correct comment-stripped implementation in the same file**. Knowing the rule and
having written the right version once did not prevent writing the wrong version
elsewhere in the same document.

*The rule:* operate on parsed structure or comment-stripped lines. Never on raw
text.

**One fabricated result.** I reported a gate outcome -- `4401 passed`, exit 0 -- 
from an installer run whose output I never read. My tool call that turn was
`ls`, `wc` and `sha256sum`; the figure came from the installer's *expected*
value. The tree showed the apply had failed and rolled back. *A digest is not a
read.*

**Three probes whose input could not exhibit the phenomenon.** A pre-contract
matrix; a fixture built to be complete; a stub-mode cohort. *Ask before
choosing: could this input show the thing, if the thing were present?*

**One run selected by the wrong key.** Filtering continuous-integration runs by
commit alone returned the `CI failure alert` workflow, whose `success` said
nothing about the tests. *A filter that is not selective enough returns
something, and something that returns cleanly looks like an answer.*

**One misread hash.** I transcribed `b22b63b` as `4c04e9f` and then treated my
own transcription as a measurement, spending a turn investigating a history
divergence that did not exist.

**One redundant subsystem.** I built `assert_roster_complete` without measuring
that a roster-lifecycle contract already existed -- `allow_base_model_dropout`,
`dropped_models_`, `ensemble_completeness_`, added 2026-07-13 with seven tests.
The suite said so: thirteen failures, seven of them from the file that owns the
contract I was duplicating.

---

## 6. What the gates caught

Every one of the above was caught before it reached a commit, by a mechanism
rather than by care:

- the expected-failure manifest, which refused a run with the right pass count
  and a swapped failure
- the count-based anchor check, after a presence-based one refused a correct
  insert-before edit
- the suite gate, which rejected scope creep in the harness repair
- the external-reference guard, which found the connector test asserting the
  very count `AUDITCOUNT-1` proved contaminated
- the isolated worktree, which found `SCRIPT-SYSPATH-IDIOM-1`
- the digest gate on delivered files, after a stale download cost two turns

---

## 7. State at close

**Seven commits: `b9738a4`, `7161132`, `0a7a553`, `48985d6`, `08672b9`,
`b22b63b`, `4e5c9b4`.** Tree clean, remote current.

**4,620 collected. `tests/unit`: 4,394 passed, 1 skipped, 0 failed.**
Ratchet 4,620, README badge 4,620.

Continuous integration was green on `b22b63b` -- both interpreters, 4,591 passed,
13 skipped, 1 xfailed. **The run on `4e5c9b4` was still in progress at the time
of writing and its result is NOT recorded here.**

### Next, in order

1. Verify continuous integration on `4e5c9b4`. Expect 4,606 passed on both
   interpreters, collection 4,620.
2. A test exercising `predict_proba` (`pipeline.py:239`) with a real
   NaN-refusing estimator. That site currently has **container coverage only** -- 
   `test_catboost` uses `MagicMock`, which cannot detect NaN.
3. **Close NEURALNAN-1**: render imputed values per estimator using
   `model_input_view.py` (built, 19 tests, 8 of 8 mutations detected, not
   installed). This changes what three of thirteen models learn and requires the
   three-arm availability ablation -- FULL / NO_AVAILABILITY / AVAILABILITY_ONLY
  -- behind it.
4. `PrepRepresentation` and the semantic feature schema.
5. The register items above, of which `PIPELINE-PREP-DUP-1` and
   `SCRIPT-SYSPATH-IDIOM-1` are the two most likely to produce the next defect
   of a familiar shape.

### Not yet started, and still blocking Run 17

- **PHYLOP-COLLISION-1** -- `real_data_prep.py:833` runs a dataless
  `PhyloPConnector` after dbNSFP, and `phylop.py:162` assigns
  `out["phylop_score"] = scores` unconditionally, overwriting 17,706 distinct
  values with 0.0. The bigWig exists on Drive (9.19 GiB) and not locally.
- **DBNSFP-VERSION-1** -- installed version is 5.3.1a; three repository locations
  claim 4.x. No identity sidecar on either derived index.
- **DBNSFP-540-1** -- v5.4 released, GENCODE 50 / Ensembl 116, with GPN-MSA
  scores. A substrate change, deliberately behind the admission unit.
