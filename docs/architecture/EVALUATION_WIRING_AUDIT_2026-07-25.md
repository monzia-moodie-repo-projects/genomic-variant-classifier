# Evaluation-Layer Wiring Audit -- 2026-07-25

Option C, step 5 preparation: cohort-agnostic metric-stack wiring.

This document is the source-grounded audit that must precede any seam or
orchestrator code. Every claim below was read from the tree at
`origin/main` HEAD `90ae44e` ("fix(tests): freeze the 8 new probe tier-maps in
the inventory ratchet"), verified clean, on 2026-07-25. Line numbers are as of
that commit and will drift; the file/function/class identities are the durable
part.

The audit exists because the evaluation layer is already large -- 23 modules,
roughly 8,700 lines -- and partially implemented across metric panels A through
R. Building a canonical seam or an orchestrator on assumptions about that layer
would risk duplicating existing abstractions, bypassing panel-specific
contracts, wiring stale or placeholder paths, or forcing incompatible input
shapes through one object. The purpose here is to establish, from source rather
than from the roadmap narrative, exactly what exists, what consumes it, what its
real input and return contracts are, and which parts can be made cohort-agnostic
now versus which are blocked behind cohort v2.


## 1. Exact file / function / class inventory

The evaluation package is `src/genomic_variant_classifier/evaluation/`. Twenty-
three Python modules, grouped below by role. Line counts are from `wc -l` on
2026-07-25.

### 1.1 The package entry file

`__init__.py` (53 lines) re-exports only the classical clinical surface --
`ClinicalEvaluator`, `ConsequenceBreakdown`, `EvaluationReport`,
`GeneErrorAnalysis`, `OperatingPoint`, `compare_models` -- from `evaluator.py`,
plus `RunArtifactWriter` from `prediction_artifacts.py`.

It carries a locked import contract stated in its own docstring: **the package
must import cleanly with scikit-learn absent**, and therefore `__init__.py` MUST
NOT import `metrics.py`, which imports scikit-learn at module level. This is
enforced by two tests:

- `tests/unit/test_evaluator_phase5.py::test_module_imports_without_sklearn`
- `tests/unit/test_evaluation_metrics.py::test_package_imports_without_sklearn`

The history recorded in the docstring is directly relevant to this audit: commit
`015ff94` once added `from ... import metrics` to `__init__.py`, which pulled
scikit-learn eagerly and broke the Phase-5 contract. Any registry or orchestrator
built in step 5 must not reintroduce that import through the package root. The
metric kernel is imported directly by consumers that expect scikit-learn present.

Two pre-existing, deliberately-unchanged anomalies are noted in the docstring:
`RunArtifactWriter` is imported but absent from `__all__`, and its import sits
below the `__all__` assignment. Correcting them changes `import *` behaviour and
belongs in its own commit, not in step 5.

### 1.2 The classical clinical / metric stack

- `metrics.py` (761 lines) -- the fail-closed metric kernel. Public surface
  (from `__all__`): `auroc`, `auprc`, `auprc_gain`, `no_skill_auprc`,
  `brier_score`, `log_loss`, `expected_calibration_error`,
  `calibration_slope_intercept`, `bootstrap_ci`, `cluster_bootstrap_ci`,
  `evaluate`, `stratified_evaluate`, `is_probability`, `clean_arrays`,
  `CleanArrays`, `CalibrationFit`. This is the kernel hardened at ratchet entry
  2055 (six defects, fail-closed). `compute_classification_metrics` and
  `ModelEvaluator` were removed on 2026-07-21 (ratchet 2446); the `__all__`
  comment records this.
- `evaluator.py` (640 lines) -- `ClinicalEvaluator` and `compare_models`, the
  higher-level report builder. Dataclasses `OperatingPoint`,
  `ConsequenceBreakdown`, `GeneErrorAnalysis`, `EvaluationReport`.
- `benchmark.py` (517 lines) -- benchmark harness. Note: it imports itself in the
  grep graph, meaning it references its own module name; no external eval-layer
  dependency.
- `prediction_artifacts.py` (460 lines) -- `RunArtifactWriter`, the persistence
  path (atomic .tmp-then-rename + fsync, JSON manifest with git SHA, parquet not
  pickle). Imports scikit-learn / shap inside functions only.
- `ntqr_evaluator.py` (238 lines), `moe_identity.py` (328 lines),
  `model_introspect.py` (61 lines) -- specialised evaluators, not yet traced to a
  live consumer in this audit (see section 3).

### 1.3 The Panel Q / R representation-geometry stack

A self-contained sub-graph, the product of the 2026-07-20/21/22 ratchet work
(entries 2055 through 2846):

- `capabilities.py` (276 lines) -- `CapabilityEvidence`, `CapabilityState`,
  `MetricStatus`. The construction-time invariant that an OK capability must be
  VALIDATED with an admissible target and a named artifact.
- `capability_lifecycle.py` (239 lines) -- the state ladder
  `NOT_IMPLEMENTED -> IMPLEMENTED_NO_OUTPUT -> OUTPUT_AVAILABLE -> VALIDATED`,
  one-rung-forward transitions, backward moves legal at any distance,
  `DEPRECATED` reachable from anywhere.
- `clustering_metrics.py` (1,328 lines) -- Panel Q. Silhouette, Davies-Bouldin
  (Euclidean and spherical), Calinski-Harabasz, agreement metrics, the confounder
  gate, gene-block permutation null.
- `representation_geometry.py` (581 lines) -- Panel R stages R1/R2, plus the
  `panel_r_capabilities` registrations for R3-R7.
- `representation_artifact.py` (302 lines) -- `RepresentationArtifact`, the frozen
  persisted embedding with row-order hash and partition role.
- `norm_angle_probe.py` (265 lines) -- Panel R stage R3, the norm-angle probe with
  the train-only whitening leakage guard.
- `null_family.py` (213 lines) -- the matched-spectrum null family.
- `alignment_recovery.py` (413 lines) -- the whitening-alignment recovery estimand.
- `r3_validation.py` (463 lines) -- the held-out transfer validation of R3.
- `r3_capability.py` (181 lines) -- R3 capability registration.
- `recovery_protocol.py` (169 lines) -- the two-null intersection admissibility
  rule.

### 1.4 Agent-facing detectors (not metric panels)

- `agent_ops_detector.py` (186), `data_readiness_detector.py` (100),
  `finops_detector.py` (75), `model_insights_detector.py` (130). These back the
  agent layer, not the clinical metric report. Out of scope for the metric-stack
  seam but listed for completeness because they live in `evaluation/`.


## 2. Panel-to-input dependency matrix

The metric layer has TWO live entry surfaces with DIFFERENT input contracts.
This is the single most important finding for seam design.

### 2.1 The low-level kernel (`metrics.py`)

Every kernel function takes parallel sequences:

- `auroc(y, score)`, `auprc(y, score)`, `brier_score(y, prob)`,
  `log_loss(y, prob)`, `expected_calibration_error(y, prob, n_bins)`,
  `calibration_slope_intercept(y, prob)` -- `(y, score-or-prob)` as `Sequence`.
- `evaluate(y, score, *, ...)` -- the omnibus, `(y, score)` plus keyword options.
- `stratified_evaluate(y, score, groups, *, ...)` -- adds `groups: Iterable`, one
  label per row, for per-subgroup evaluation.
- `bootstrap_ci(fn, y, score, *, ...)` -- variant-level resampling.
- `cluster_bootstrap_ci(fn, y, score, clusters, *, ...)` -- gene-cluster
  resampling; reports the design effect (ratio of clustered to naive CI width).
- `CleanArrays` / `clean_arrays(y, score, ...)` -- the joint-mask cleaner that
  exposes `.mask`, so callers do not reconstruct alignment (defect A, ratchet
  2055).

Row-level, array-in, scalar-or-dataclass-out. No DataFrame, no schema, no
partition concept, no cohort version.

### 2.2 The high-level evaluator (`evaluator.py`)

`ClinicalEvaluator.evaluate(y_true, y_proba, meta=None, model_name="model")`
returns an `EvaluationReport`. Its docstring describes `meta` verbatim as:

> "Canonical variant DataFrame aligned with y_true/y_proba. Required for per-gene
> and per-consequence analysis."

`compare_models(y_true, model_probas, meta=None, n_bootstrap=500, output_csv=...)`
iterates models, calls `evaluator.evaluate` per model, and writes a comparison
CSV.

The evaluator therefore ALREADY has an informal notion of a "canonical variant"
metadata frame -- but it is a bare `pd.DataFrame` with:

- no declared schema,
- no partition / split field,
- no cohort-version field,
- no gene-disjointness or leakage guard,
- no distinction between `y_true` provenance (adjudicated vs raw).

**This is exactly the gap the step-5 seam should formalise.** The seam is not a
new idea grafted onto the layer; it is the schema the layer already gestures at
in a docstring and never enforced.

### 2.3 The Panel Q / R stack

Different input entirely. It consumes a `RepresentationArtifact` (a frozen
embedding matrix with a bound `partition_role` and a row-order hash), not
`(y, score)`. Its contracts are already typed and already partition-aware -- this
is the recent, well-wired work. It does NOT need the classical seam and must not
be forced through it.


## 3. Current execution graph

Traced from the real consumer call sites (`grep` across `src/` and `scripts/`
outside `evaluation/`).

### 3.1 Intra-evaluation import edges

```
__init__            -> evaluator, metrics, prediction_artifacts
alignment_recovery  -> norm_angle_probe, null_family, representation_artifact
capability_lifecycle-> capabilities
clustering_metrics  -> capabilities
norm_angle_probe    -> capabilities, clustering_metrics, representation_artifact
null_family         -> norm_angle_probe
r3_validation       -> norm_angle_probe, null_family, representation_artifact
recovery_protocol   -> alignment_recovery, null_family
representation_geometry -> capabilities, clustering_metrics
```

Two disjoint clusters are visible: the classical stack
(`evaluator`/`metrics`/`prediction_artifacts`) and the Panel Q/R stack (rooted at
`capabilities` / `representation_artifact`). They do not import each other today.

### 3.2 External consumers

- `scripts/train.py` -- the TRAINING DRIVER and the live production evaluation
  path. Line 567 `ensemble.evaluate(X_test, X_seq_test, y_test)`; line 570-571
  imports and constructs `ClinicalEvaluator`; line 587 calls
  `evaluator.evaluate(y_test, ..., meta=...)`. Line 670 uses
  `model_introspect.model_input_width`.
- `scripts/evaluate_predictions.py` -- the POST-HOC path. Line 68 imports the
  low-level `metrics.py` kernel directly and evaluates predictions that
  `RunArtifactWriter.save_test_predictions` persisted.
- `scripts/run9_ablations.py` -- ablation harness, consumes the eval layer.
- `scripts/verify_ece_fix.py`, `scripts/probe_run14_univariate_leakage.py`,
  `scripts/patch_orchestrator_lazy_registration.py` -- verify/probe scripts.
- Four agents: `model_insights_agent.py`, `data_readiness_agent.py`,
  `agent_ops_monitor_agent.py`, `finops_advisor_agent.py` -- consume the
  agent-facing detectors (section 1.4), not the clinical report.

### 3.3 The two live metric entry points

1. Training-time: `train.py` -> `ClinicalEvaluator.evaluate(y_true, y_proba,
   meta=DataFrame)` -> report + (via `RunArtifactWriter`) persisted artifacts.
2. Post-hoc: `evaluate_predictions.py` -> `metrics.py` kernel functions over
   arrays read back from persisted predictions.

The seam must serve BOTH without forcing them into one call path. A single
"canonical table" contract can feed both -- arrays are a projection of the table
-- but the orchestrator must not require the training path to persist-then-reload.


## 4. Duplicate or conflicting dispatch paths

- **Two calibration-error lineages.** `metrics.expected_calibration_error` is the
  hardened kernel version (closed-top bin, aligned counts; ratchet 2060 fixed ten
  divergent implementations). `ClinicalEvaluator._calibration_error`
  (evaluator.py:305) is a SEPARATE method. Ratchet 2060 recorded that
  `evaluator.py`'s open-top defect was repaired 2026-07-10 (its own dated comment
  at lines 321-323), so the two should now AGREE, but this audit has not
  re-measured them against each other on a shared fixture. Flagged for the seam
  work: the canonical path should route calibration through ONE implementation,
  and a test should pin agreement.
- **Two bootstrap lineages.** `metrics.bootstrap_ci` / `cluster_bootstrap_ci`
  (kernel, gene-cluster aware, reports design effect) versus
  `ClinicalEvaluator._bootstrap_ci` (evaluator.py:284). Ratchet 2060's "WHAT IS
  NOT DONE (d)" explicitly lists three unreconciled bootstrap implementations:
  `evaluator.py:284`, `report_generator.py:85`, and the kernel's. The kernel's is
  the only one that respects gene clustering; the gene-cluster design effect was
  measured at 2.935x (ratchet 2055), meaning the variant-level bootstraps are
  anti-conservative in a known direction. The seam should prefer the kernel's
  clustered bootstrap and the audit should measure whether the other two share the
  independence assumption before either is trusted for a clinical CI.
- **No conflicting REGISTRY exists yet.** There is no metric registry or dispatch
  table today; the "dispatch" is direct function calls from two scripts. This is
  good news -- the orchestrator in step-5-part-2 will be the first, not a
  reconciliation of several.


## 5. Required canonical contracts

Determined by the input contracts in section 2, NOT assumed in advance.

The kernel wants `(y, score, groups?, clusters?)` as aligned sequences. The
evaluator wants `(y_true, y_proba, meta: DataFrame)`. Both are projections of one
aligned, per-variant table. The correct seam is therefore a TABLE contract with a
typed schema, from which arrays are trivially projected -- not a single scalar
record type, and not a bare untyped DataFrame.

Proposed (HYPOTHESIS, to be validated against the two entry points before it is
frozen):

```
CanonicalVariantTable  (the aligned evaluation input; a typed wrapper over a frame)
  variant_id              str            stable identity, one per row
  y_true                  int | None     canonical binary label (None = withheld)
  y_score                 float | None   model score / probability, if scoring
  gene_id                 str | None     for per-gene analysis + cluster bootstrap
  group_id                str | None     generic stratum for stratified_evaluate
  partition               str            train/tune/calib/conformal/structure/test
  cohort_version          str            e.g. "v1", "v2-<hash>" -- provenance
  adjudication_reason     str | None     why this label (P6 state), free-form
```

Projections:

- kernel `evaluate(y, score)`  <- `table.y_true`, `table.y_score` over a chosen
  partition and a not-null-label mask.
- `stratified_evaluate(y, score, groups)`  <- add `table.group_id`.
- `cluster_bootstrap_ci(..., clusters)`  <- `table.gene_id`.
- `ClinicalEvaluator.evaluate(y_true, y_proba, meta)`  <- the whole table as
  `meta`, with `y_true`/`y_proba` projected.

Key design constraints the contract must enforce, each traceable to a recorded
defect:

- ALIGNMENT is structural: one row, one variant; projections cannot desync
  (defect A, ratchet 2055 -- two separate masks misaligned score and prob).
- MISSING labels are representable and never coerced (defect B, ratchet 2055 --
  `y[ok].astype(int)` produced a signed AUROC on `[0,1,3]`). `y_true = None` is a
  first-class value, distinct from 0.
- PARTITION is mandatory, because probability calibration must be fitted on data
  untouched by model/method/alpha selection (Panel finding 2; ratchet 2192/2408
  -- the isotonic calibrator was fitted on trained-on genes in every production
  run).
- COHORT_VERSION is mandatory, because Option C forbids certifying any production
  metric against the superseded v1 cohort (`BLOCKED_BY_COHORT_V2`).
- The seam imports NOTHING from the probe / P6 / clean_cohort code. It is a
  generic contract; the cohort builder produces instances of it, but the metric
  stack never reaches back into cohort construction.

Whether `CanonicalVariantTable` should decompose further into a small family
(`PredictionBundle` for `y_score` per model, `PartitionManifest`,
`GroupMembership`) is deferred to the seam commit's own design step, and will be
decided by whether `compare_models` (multiple `y_score` columns per variant)
reads cleanly from one wide table or from a table + a per-model bundle. Both
entry points must be fed by the chosen shape before it is frozen.


## 6. Cohort-v2 dependencies versus cohort-agnostic components

| Component | Cohort-agnostic now | Blocked by v2 |
|---|---|---|
| Binary discrimination (auroc/auprc/mcc/f1) | yes | metric BASELINE blocked |
| Calibration (brier/log_loss/ece/slope) | yes | metric BASELINE blocked |
| Gene-cluster bootstrap (design effect) | yes | metric BASELINE blocked |
| Stratified / subgroup evaluate | yes | metric BASELINE blocked |
| Panel Q clustering + confounder gate | yes | no (representation input) |
| Panel R geometry (R1/R2) | yes | no (representation input) |
| Panel R R3 (norm-angle, validated negative) | yes | no |
| Multi-label disease head | no | yes -- no target exists |
| Regression / conformal-quantile head | no | yes -- no target exists |

Distinction that Option C makes and this table encodes: the metric CODE
(discrimination, calibration, bootstrap) is cohort-agnostic and may be WIRED now;
what is blocked is publishing any metric BASELINE, comparison, or clinical
conclusion computed against the superseded v1 cohort. The seam carries
`cohort_version` precisely so a v1-derived result is machine-refusable at the
certification boundary, not merely by convention.


## 7. Leakage and partition hazards

- **Probability calibration fitted on selection / trained-on data.** Ratchet 2192
  (calibrator on the `tune` selection set) and 2408 (calibrator on trained-on
  genes in `run_phase2_eval.py:590`, every Run 14-17). Both are repaired in the
  split/calibration layers; the seam must carry `partition` so a metric consumer
  cannot re-open the hole by evaluating calibration on the wrong partition.
- **Gene-cluster non-independence in bootstrap.** Variant-level resampling
  understates variance; measured design effect 2.935x (ratchet 2055). The
  canonical table's `gene_id` makes `cluster_bootstrap_ci` the default-reachable
  path.
- **`clingen_validity_score` circularity for gene ranking** (ratchet 2544): a
  model input used as a ranking target is inadmissible. The seam does not fix
  this, but `adjudication_reason` / capability integration must let the
  certification layer see it.
- **Whitening leakage in Panel R** is already guarded (train-only fit,
  `LeakageError`); the seam does not touch that path.


## 8. Missing implementations and stubs

- `ntqr_evaluator.py`, `moe_identity.py`, `model_introspect.py`: present but no
  live clinical-report consumer traced in this audit. `model_introspect` IS used
  by `train.py:670` for `model_input_width`. The other two are unproven; the seam
  work should not assume they are on the metric path.
- Multi-label disease head and regression/CQR head: NOT_IMPLEMENTED, no target
  column exists in the cohort. Blocked by v2.
- Panel R R4-R7: registered NOT_IMPLEMENTED / IMPLEMENTED_NO_OUTPUT; R3 is
  OUTPUT_AVAILABLE with a validated NEGATIVE transfer finding (ratchet 2800). None
  are release-admissible; the capability contract already refuses them.
- The legacy `compute_classification_metrics` / `ModelEvaluator` are GONE (ratchet
  2446); a test pins their absence. The seam must not resurrect a bare-float
  metrics dict.


## 9. Proposed adapter boundary

A single module, `evaluation/canonical.py` (name provisional), that:

1. Declares `CanonicalVariantTable` (the section-5 contract) as a typed wrapper
   over a validated frame -- schema, dtypes, partition membership, cohort version
   checked at construction, fail-closed.
2. Provides projections: `.arrays(partition, drop_missing_labels=True) ->
   (y, score)`, `.groups(...)`, `.gene_clusters(...)`, and `.as_meta()` (the frame
   `ClinicalEvaluator` already expects).
3. Imports scikit-learn NOWHERE at module level (so it can live in the package
   without breaking the no-sklearn import contract), and imports nothing from
   `data/clean_cohort.py`, the probes, or P6.
4. Does NOT change any metric mathematics. It only converts an aligned,
   provenance-carrying table into the exact inputs the kernel and the evaluator
   already accept.

The adapter is deliberately thin: it is a schema + projections, not a metric
framework. It feeds `metrics.py` and `ClinicalEvaluator.evaluate` unchanged.


## 10. Proposed registry / orchestrator design (part 2, after the seam is proven)

Only after the seam feeds several existing metrics end-to-end:

- A `MetricRegistry` mapping metric name -> callable + declared input requirement
  (needs score? needs gene_id? needs partition X?) + capability id.
- An orchestrator that takes ONE `CanonicalVariantTable`, runs the registered
  metrics whose requirements the table satisfies, SKIPS (with a recorded reason,
  never a silent zero) those it cannot satisfy, refuses to certify anything whose
  `cohort_version` is v1, and emits a typed report consulting `CapabilityEvidence`
  so a NOT_IMPLEMENTED / blocked panel cannot report as complete.
- It must NOT import `metrics.py` through the package root (section 1.1), must NOT
  build a second calibration or bootstrap implementation (section 4), and must be
  high-blast-radius-aware: it is built last, over a proven seam, as thin
  orchestration, not a new metric implementation layer.


## 11. Ordered implementation commits

1. **Seam (this is step-5-part-1).** `evaluation/canonical.py`:
   `CanonicalVariantTable` + projections + construction-time validation. Tests:
   projection round-trips, missing-label representability, partition enforcement,
   cohort-version refusal, and a proof it feeds BOTH `metrics.evaluate` and
   `ClinicalEvaluator.evaluate` on a synthetic aligned table. No cohort data, no
   probe imports. Ratchet moves by the number of new tests, measured on the
   staged tree.
2. **Reconcile the duplicate calibration + bootstrap paths (section 4)** behind
   the seam, with agreement tests, BEFORE the orchestrator can prefer one. This is
   its own commit because it may change numbers `ClinicalEvaluator` emits.
3. **Registry + orchestrator (step-5-part-2).** Thin, capability-aware, over the
   proven seam.
4. **Wire `train.py` / `evaluate_predictions.py` onto the seam** last, so the
   production paths change only after the contract is proven.


## 12. Sabotage tests and acceptance criteria

Every guard proven falsifiable, per the project pattern (a guard never observed
to fire is not known to work):

- Construct a `CanonicalVariantTable` with `y_true` containing a value outside
  {0, 1, None} -> refused.
- Misalign `y_score` length vs rows -> refused at construction, not at metric time.
- Project a partition that does not exist -> refused with the partition named.
- Certify a metric whose `cohort_version == "v1"` -> refused
  (`BLOCKED_BY_COHORT_V2`).
- Feed the same table to the kernel and to `ClinicalEvaluator`; assert identical
  `y`/`score` reach both (the seam does not desync them).
- Remove the missing-label handling -> a withheld row coerces to 0 and the AUROC
  moves; the test catches it.
- Orchestrator (part 2): a metric requiring `gene_id` on a table without it is
  SKIPPED with a recorded reason, never silently zeroed.

Acceptance: the seam feeds both live entry points on a synthetic table with zero
change to metric mathematics; all sabotage tests fire; the ratchet is updated in
the same commit as the tests; both Python 3.11 and 3.12 CI legs are green.


## Decision table (summary)

| Component | Build now | Blocked by v2 | Reuse | Refactor |
|---|---|---|---|---|
| Binary discrimination | yes | baseline only | yes | minor |
| Calibration | yes | baseline only | yes | reconcile 2 paths |
| Gene-cluster bootstrap | yes | baseline only | yes (kernel) | reconcile 3 paths |
| Stratified / subgroup | yes | baseline only | yes | adapter |
| Panel Q clustering | yes | no | yes | none |
| Panel R geometry (R1-R3) | yes | no | yes | separate repr. input |
| Multi-label disease | no | yes | no | blocked |
| Regression / CQR | no | yes | partial | blocked |
| Canonical seam | yes (part 1) | no | n/a (new) | formalises evaluator `meta` |
| Registry / orchestrator | yes (part 2) | no | n/a (new) | thin, over proven seam |

**Final call, matching the agreed sequence: read-and-report (this document)
first; then the seam (part 1); then the registry / orchestrator (part 2), with
the calibration/bootstrap reconciliation slotted between them because it can
change emitted numbers.**
