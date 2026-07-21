# HANDOFF -- 2026-07-20, end of session

**Repository:** `github.com/monzia-moodie-repo-projects/genomic-variant-classifier`
**Local:** `C:\Projects\genomic-variant-classifier`
**HEAD at handoff:** `0208998`, pushed, Continuous Integration #551 green.
**Suite:** 2060 collected (2053 passed, 7 skipped locally on Python 3.12).
**Environment:** Windows, PowerShell 5.1, Python 3.12.10, virtual environment `.venv312`.

---

# PART ONE -- THE DIRECT ANSWER TO "WERE THEY DELIVERED"

Monzia asked, at the close of this session, whether **JEPA (Joint Embedding Predictive
Architecture)**, **conformal prediction**, **the RNA infrastructure**, **the full metric
stack**, and **conformal quantile regression** were fully implemented, verified and wired.

**NO. None of the five. This section is the measured state of each, so the next session does
not have to rediscover it.**

## 1.1 JEPA (Joint Embedding Predictive Architecture) -- ABSENT, HARD-BLOCKED

- **Code written: zero.** No module, no stub, no test. Searching the repository for `jepa`
  returns nothing under `src/`.
- **Hard blocker: disk.** Measured this session: **10.91 GB free** against an estimated
  **~14.7 GB minimum** for the embedding cache. Shortfall **3.79 GB**. This is not a
  scheduling problem; the artifact does not fit.
- **Second gate: ESM-2 coverage.** JEPA V1 was scoped to consume protein embeddings. See
  section 1.3 below -- the HGVSp parser exists and is wired, but coverage on the 4.4M-row
  cohort has not been measured since.
- **What to do first:** measure actual free space and the actual embedding-cache size
  requirement rather than reusing these figures; they were taken on 2026-07-20 and disk moves.

## 1.2 CONFORMAL PREDICTION -- PARTIAL. SIX MODULES GENUINELY ABSENT

`src/genomic_variant_classifier/conformal/` **EXISTS** and contains, verified by directory
listing this session:

    calibrate.py    coverage.py    grouped.py    mondrian.py    scores.py    split.py

Tests exist at `tests/conformal/`: `test_calibrate.py`, `test_grouped.py`,
`test_mapie_crosscheck.py`, `test_mondrian.py`, `test_split.py`.

**ABSENT -- six modules, and this list is the scoped expansion, not a wish list:**

1. `artifacts.py` -- persisting conformal sets alongside predictions so a run's coverage claim
   is reproducible from disk rather than recomputed.
2. `ordinal.py` -- the ACMG/AMP five-tier outcome is ORDINAL (Pathogenic, Likely Pathogenic,
   Variant of Uncertain Significance, Likely Benign, Benign). Nothing currently respects that
   ordering; a prediction set of {Pathogenic, Benign} is currently as admissible as
   {Pathogenic, Likely Pathogenic}, which is clinically absurd.
3. `multilabel.py` -- a variant can implicate more than one disease.
4. `gene_ranking.py` -- set-valued output over ranked genes rather than over classes.
5. `risk_control.py` -- distribution-free risk control (Learn-then-Test style), which is what
   a clinical false-negative budget actually needs. Coverage alone does not bound it.
6. `monitoring.py` -- coverage drift over time, which is the conformal analogue of the
   calibration-drift agent.

**Also absent:** the fourth partition, `split_protocol_v2`. The current split protocol has
train / calibration / test. Conformal calibration and probability calibration are currently
drawing from partitions whose independence has NOT been established -- this is recorded as an
unresolved finding in `project_metrics.txt` (see section 5) and it is a **correctness**
question, not a tidiness one.

## 1.3 RNA INFRASTRUCTURE -- NOT VERIFIED

- The **RNA-seq differential-expression leakage check** was flagged by
  `preflight_stage1_paths.py` (from its `--help` output) and has **never been run**. The
  question it answers: are the differential-expression statistics used as features computed on
  a cohort that OVERLAPS the training cohort? If so, that is label leakage of the most direct
  kind.
- GTEx is wired as a connector (`local_path=data/raw/cache`) but its **coverage on the
  4,399,089-row cohort is unmeasured**.
- Ratchet entry 1924 records that in Run 15, **36 of 78 features were constant zero**, GTEx
  among them. Nothing since has confirmed GTEx now varies on the full cohort.

## 1.4 THE FULL METRIC STACK -- PARTIAL, AND NOW MEASURED

`project_metrics.txt` (see section 5 for the path) specifies **sixteen panels, A through P**,
an 18-row dashboard, and five release-gate tiers.

**Present:** Panel B and most of Panel D, delivered by commit `5615cd0` on 2026-07-20.

**Absent: A, C, E, F, G, H, I, J, K, L, M, N, O, P.** Fourteen of sixteen.

**`METRIC_REGISTRY` is not built.** This was designated Priority 1 by the independent audit
and by my own plan, and commit 2 has not started. Its scope, unchanged:

- One entry per metric: name, callable, panel letter, formula, range, direction, required
  context, and an `IMPLEMENTED` / `NOT_IMPLEMENTED` status -- so Panels I, J and L register
  as unimplemented rather than being silently absent.
- A typed `BinaryEvaluationReport` that REFUSES to serialise without prevalence, sample count
  and split identity.
- Missing metrics: balanced accuracy, Matthews correlation coefficient, sensitivity,
  specificity, positive predictive value, negative predictive value, positive and negative
  likelihood ratios, partial area under the receiver operating characteristic curve,
  integrated calibration index, adaptive expected calibration error, maximum calibration
  error, and the Brier decomposition into reliability / resolution / uncertainty.
- `OperatingPointMetrics` with threshold provenance.
- Clinical panels: decision-curve net benefit, selective-prediction risk-at-coverage.
- A living glossary GENERATED from the registry and pinned by a test.

## 1.5 CONFORMAL QUANTILE REGRESSION -- NOT STARTED

Not among the six existing conformal modules. No code, no test, no stub.

## 1.6 WHY THIS SESSION WENT ELSEWHERE -- STATED SO IT CAN BE REVERSED

The metric work began as commit 1 of the metric programme and its stated precondition for
commit 2 was a **measured call-site census**. That census (section 3.2) found that the
canonical kernel has **no production caller at all**, and that ten independent
implementations of expected calibration error existed across the repository. Probing them
found **two distinct defects across six files**, both of which halve or worse the reported
calibration error on saturated predictions.

Building fourteen more panels on top of that surface would have produced fourteen more
numbers of unknown correctness. Repairing it first is defensible -- but the decision was the
assistant's, taken while following evidence, and it is not what was asked for. **The next
session should decide explicitly whether to continue the metric registry or to go straight at
the five deliverables above.**

---

# PART TWO -- WHAT THIS SESSION ACTUALLY DID

Eight commits, all pushed, Continuous Integration green throughout. Suite 1985 -> 2060.

| commit | Continuous Integration | what |
|---|---|---|
| `fb23543` | green | `WindowAttachment` derives its counts from two masks (ratchet 1999) |
| `106d107` | green | sequence-provenance gate (ratchet 2017) |
| `3bba87e` | green | session record: window attachment + sequence gate |
| `bd4d223` | green | roadmap delta: provenance, monitoring, JEPA readiness, conformal gap |
| `5615cd0` | #548 | metric kernel becomes fail-closed, six defects (ratchet 2055) |
| `fd85f0f` | #549 | session record: metric kernel |
| `44511fa` | #550 | **two calibration defects across six files** (ratchet 2060) |
| `0208998` | #551 | session record + roadmap: the calibration surface |

---

# PART THREE -- THE CALIBRATION WORK IN DETAIL

## 3.1 Where the artifacts are

Committed:

- `docs/sessions/SESSION_2026-07-20_calibration-defect-repair.md` -- **227 lines, seven
  sections. Read this first in the next session.** It contains the full narrative,
  every measurement, the five probe versions, and the retraction.
- `docs/sessions/SESSION_2026-07-20_metric-kernel-fail-closed.md` -- 260 lines, the earlier
  part of the day.
- `docs/ROADMAP.md` -- three entries appended today. The part-three entry is at the end.
- `tests/EXPECTED_SUITE_SIZE` -- the 2060 entry is 154 lines and is the densest single record
  of this work. **That file's history section is the best documentation in the project;
  read the 2055 and 2060 entries.**
- `tests/unit/test_calibration_implementations_agree.py` -- 134 lines, 5 tests.

Delivered to `C:\Users\monzi\Downloads\` (NOT committed; they are instruments, not code):

| file | SHA-256 (first 8) | purpose |
|---|---|---|
| `census_metric_callsites_2026-07-20.py` | `83994ebe` | the 813-file census |
| `probe_calibration_divergence_2026-07-20.py` | `5d69ed63` | v1 |
| `probe_calibration_divergence_v2_2026-07-20.py` | `bfb18d94` | v2 |
| `probe_calibration_divergence_v3_2026-07-20.py` | `9541e229` | v3 |
| `probe_calibration_divergence_v4_2026-07-20.py` | `481ede66` | v4 |
| `probe_calibration_divergence_v5_2026-07-20.py` | `59a9265e` | **v5 -- the working one** |
| `install_calibration_defect_repair_2026-07-20.py` | `7439381d` | the repair, already applied |
| `install_ratchet_bump_calibration_2026-07-20.py` | `eae4b741` | ratchet, already applied |
| `install_session_record_calibration_2026-07-20.py` | `5143b0c6` | record, already applied |

**Re-run `probe_calibration_divergence_v5_2026-07-20.py` at the start of the next session.**
It is read-only, takes seconds, and immediately shows whether the repair still holds. Expected:
all nine loaded implementations return 0.250000 on fixture D, and "matches its own OPEN-top
reference" reads `none` on all four fixtures.

## 3.2 The census -- the finding that reframes everything

`census_metric_callsites_2026-07-20.py`, 813 Python files under `src/`, `scripts/`, `tests/`,
parsed by abstract syntax tree, zero parse failures.

**Only four modules import `evaluation.metrics`, and two are tests:**

    scripts/evaluate_predictions.py
    scripts/probe_run14_univariate_leakage.py
    tests/unit/test_evaluation_metrics.py
    tests/unit/test_metric_kernel_is_fail_closed.py

**NO MODULE UNDER `src/` IMPORTS THE CANONICAL KERNEL.** The hardening in `5615cd0` improved
a module with no production caller.

**The legacy interface has zero external callers.** `compute_classification_metrics` and
`ModelEvaluator` appear only at `metrics.py:73`, `metrics.py:721` (its own `__main__`), and
`tests/unit/test_evaluation_metrics.py:226,230`. Removal is a live scope decision. NOTE the
history: `metrics.py`'s docstring records they were "restored verbatim from commit 87e32ad^,
after 87e32ad overwrote them", so a previous loss was deliberately reversed.

**Duplication found:** 10 implementations of expected calibration error, 3 of bootstrap
confidence intervals, 2 of rank-based area under the receiver operating characteristic curve,
3 of coverage. 27 production files compute metrics with no kernel import.

## 3.3 The two defects, repaired in `44511fa`

**Defect 1 -- open top bin, three files.** `(p >= lo) & (p < hi)` with `hi == 1.0` drops every
prediction of exactly 1.0 -- a pure decision-tree or ensemble leaf.

    scripts/calibrate_thresholds.py:167     <- SELECTS OPERATING THRESHOLDS
    scripts/validate_external.py:88
    scripts/calibration_analysis.py:75      (_calibration_summary inherits by delegation)

Under-report on a 20%-pure-leaf fixture: **86.7%**, independently reproducing the 86.5%
measured on 2026-07-08.

**Defect 2 -- counts misaligned with the bins they weight, three files, previously
undocumented anywhere.**

    scripts/run_benchmark.py:65-74
    scripts/validate_clinvar_temporal.py:235-242
    src/genomic_variant_classifier/evaluation/benchmark.py:119-132

`calibration_curve` returns only NON-EMPTY bins; `np.histogram` returns ALL of them; `zip`
truncates and pairs each non-empty bin's statistics with the wrong bin's count.

| fixture | as written | aligned | empty bins |
|---|---|---|---|
| perfectly calibrated | 0.024937 | 0.024937 | 0 |
| overconfident | 0.162413 | 0.162413 | 0 |
| 20% at p == 1.0 | 0.118207 | 0.118207 | 0 |
| saturated | 0.125000 | **0.250000** | 13 |
| sparse + saturated | 0.004825 | **0.309788** | 9 |

**64x under-reported on sparse saturated data** -- which is what a well-separated classifier's
output looks like. Correct whenever every bin is occupied, which is why it survived review.

**Both defects reached 0.125000 against a true 0.250000 by different routes.** Unanimity read
as correctness.

## 3.4 THE RETRACTION -- do not re-propagate this

`src/genomic_variant_classifier/evaluation/evaluator.py:305` **DOES NOT** carry the open-top
defect. It was **repaired on 2026-07-10** and carries its own dated comment at lines 321-323.

The false claim came from `metrics.py`'s docstring, which stated in the present tense
something true on 2026-07-08 and false from 2026-07-10. **A document was quoted instead of the
code being read.** The docstring is corrected in `44511fa`, with its dates preserved.

---

# PART FOUR -- OPEN ITEMS, IN PRIORITY ORDER

## 4.1 The five deliverables of Part One

Sections 1.1 to 1.5. **These are the owner's stated priorities and they outrank everything
below unless the owner says otherwise.**

## 4.2 The tenth calibration implementation -- UNEVALUATED, NOT ASSUMED CLEAN

`src/genomic_variant_classifier/agent_layer/agents/calibration_drift_agent.py:45`,
`_binned_calibration`. The v5 probe could not reach it:

    CalibrationDriftAgent.__init__() missing 3 required positional arguments:
    'classes', 'baseline_ece', and 'output_dir'

**It is the agent that monitors calibration drift in production.** Which of the two defects it
carries is UNKNOWN. Two ways to settle it: construct a real instance with those three
arguments, or read its binning line directly with `Get-Content` and compare against the two
patterns in section 3.3. **The second takes one minute and should be done first.**

## 4.3 Dead code

`src/genomic_variant_classifier/evaluation/benchmark.py:125` computes `bin_midpoints` and
never uses it.

## 4.4 Three unreconciled bootstrap implementations

`evaluator.py:284` (`_bootstrap_ci`), `reports/report_generator.py:85` (`bootstrap_metric`),
and the kernel's `bootstrap_ci` / `cluster_bootstrap_ci`.

Commit `5615cd0` measured a **gene-cluster design effect of 2.935x** on a fixture where six of
thirty genes carry inverted discrimination -- meaning every confidence interval this project
has published understated its uncertainty by roughly that factor. **Whether the other two
implementations share the variant-level independence assumption has NOT been measured.**
The same probe technique (sanitised-module isolation, v5) will answer it.

## 4.5 METHODS.md section 3.1 -- stale in three ways

File is at **repository root**: `C:\Projects\genomic-variant-classifier\METHODS.md`
(12,462 bytes), NOT under `docs/`. Section 3.1 is at lines 137-169.

- Line 139: "Four tabular base models were trained on the 64-feature matrix" -- against a
  roster of **thirteen** and a contract of **95**. Nine models absent.
- Line 152: says the one-dimensional sequence convolutional network is excluded from the
  inference pipeline -- written before its 2026-07-05 Tier-1 re-architecture (`57cf459d`).
- Line 164: states STRING combined score >= 500 while the registry caches
  `string_graph_700.pkl`. **FLAGGED UNVERIFIED -- verify, do not assert.**

`tests/unit/test_methods_feature_count.py` passes throughout because it checks the count
sentence, the group-table sum, and HGMD's absence -- never the roster. `test_readme_claims.py:375`
has read the roster from a live ensemble since 2026-07-14; **widen the METHODS gate the same
way.** Generate section 3.1 from `VariantEnsemble.base_estimators`.

## 4.6 Monitoring remediation -- three fixes, none started

- `src/genomic_variant_classifier/monitoring/registry.py:137-140` -- stale AlphaFold `.cif`
  path. (Retraction on record: the DATA is present, 107.1 MB at `data/external/alphafold/`.
  Only the path is stale.)
- `scripts/run_data_freshness.py` -- `main()` returns 0 unconditionally.
- `agent_layer/orchestrator.py:261` -- dry-run never writes telemetry.
- `monitoring/database_freshness_detector.py:109` parent-directory cruft and `:107`
  directory-total, where `gtex` and `esm2` both have `local_path=data/raw/cache`.

## 4.7 Monthly Drift Monitor never dispatched

`drift_monitor.yml`, 34,876 bytes, repaired 2026-07-14 (`4528414` / `68d8321` / `69b9f01`),
**never executed since**.

## 4.8 ESM-2 coverage on the 4.4M cohort

`hgvsp_parser.py` **EXISTS, IS WIRED, IS TESTED** -- it is NOT a stub, and any memory saying
otherwise is stale. What is unmeasured is coverage on the full cohort.

---

# PART FIVE -- REFERENCE MATERIAL AND EXACT PATHS

## 5.1 The specification documents

Both were uploaded during this session and live in the transcript, not the repository:

- `project_metrics.txt` -- 34,678 bytes, SHA-256 `db987039`. Sixteen panels A-P, 18-row
  dashboard, five release-gate tiers, ten priorities. **Priority 1 is the canonical registry.**
  Its findings: not yet a true five-class model (labels collapse Pathogenic/Likely Pathogenic
  to 1, Benign/Likely Benign to 0, Variant of Uncertain Significance excluded);
  calibration/conformal partition independence unresolved; gene-level leakage vectors;
  disease-informed graph circularity; headline results unrepresentative; METHODS.md
  contradiction; insufficient clinical-utility metrics.
- `Independent_audit_of_metric_stack.txt` -- 21,353 bytes, SHA-256 `184adc61`. The seven
  defects A-G, six of which were repaired in `5615cd0`.

**Re-upload both at the start of the next session** if the metric programme continues.

## 5.2 Prior conversation

The full uncompacted conversation for this session is preserved in the assistant's own
transcript store, which is NOT a path on Monzia's machine and cannot be opened from
PowerShell. A new session reaches it with the past-conversation search tools; useful queries
are "calibration divergence probe", "metric kernel fail-closed", "call-site census", and
"sequence provenance gate".

It preserves every installer source in full, all five probe versions, every measurement, and
all pytest and Continuous Integration output. Nothing in it is needed to CONTINUE the work --
the committed session records and the ratchet history carry the findings -- but it is where to
look if a specific number or a discarded intermediate needs recovering.

## 5.3 Key file paths, verbatim

    Repository root       C:\Projects\genomic-variant-classifier
    Metrics kernel        src/genomic_variant_classifier/evaluation/metrics.py  (722+ lines;
                          legacy at :47-93, banner ~:96, new stack below, expected_calibration_error :361)
    Evaluator             src/genomic_variant_classifier/evaluation/evaluator.py  (:305 correct since 2026-07-10)
    Benchmark             src/genomic_variant_classifier/evaluation/benchmark.py  (:119 repaired, :125 dead code)
    Ensemble              src/genomic_variant_classifier/models/variant_ensemble.py
                          (SEQUENCE_MODELS :1963, roster NOT enumerated in docstring, telemetry roster :2762-2765)
    Conformal package     src/genomic_variant_classifier/conformal/
    METHODS               C:\Projects\genomic-variant-classifier\METHODS.md   (ROOT, not docs/)
    Ratchet               tests/EXPECTED_SUITE_SIZE   (= 2060)
    Continuous Integration  .github/workflows/ci.yml
                          (push-ghcr :554-558 is RELEASE-GATED -- skipping is correct, not a failure)
    Sessions              docs/sessions/
    Roadmap               docs/ROADMAP.md
    Downloads             C:\Users\monzi\Downloads\

## 5.4 The roster -- authoritative, derived from live `VariantEnsemble().base_estimators`

Thirteen base models: `catboost`, `cnn_1d`, `deep_ensemble`, `gradient_boosting`, `kan`,
`lightgbm`, `logistic_regression`, `mc_dropout`, `random_forest`, `svm`, `svm_bagged_rbf`,
`tabular_nn`, `xgboost`. Plus a logistic-regression stacking meta-learner, plus a STRING-DB
Graph Attention Network branch (`gnn.py`), plus a hetero-GNN.

`EXPECTED_TABULAR_FEATURE_COUNT = 95 = len(TABULAR_FEATURES)`.
`SEQUENCE_MODELS = frozenset({cnn_1d})`.

---

# PART SIX -- PROCEDURAL RULES LEARNED OR RE-LEARNED TODAY

## 6.1 The ratchet is an EQUALITY check and it only runs under a flag

`tests/EXPECTED_SUITE_SIZE` is compared against the COLLECTED count by `tests/conftest.py`
**only under the explicit `--assert-suite-size` flag**, in BOTH directions. A plain
`pytest tests/ -q` does NOT consult it. Continuous Integration DOES pass the flag.

**A green plain run is not evidence the ratchet agrees.** Always run:

    python -m pytest tests/ --assert-suite-size -q

## 6.2 Measure the count on the STAGED tree

Ratchet entry 1962 records a count measured on a working tree holding an untracked file, then
written into a file guarding a committed tree. Procedure:

    git -C $Repo add -A
    git -C $Repo status --short      # verify NO '??' lines
    python -m pytest tests/ --collect-only -q | Select-Object -Last 1

`*.bak_*` is gitignored (`.gitignore:262`), so installer backups do not pollute `git add -A`.

## 6.3 The README states the test count in exactly ONE place

`README.md` line 8, the badge. `tests/unit/test_readme_claims.py:222` asserts it EQUALS the
ratchet with **no tolerance**. Both must move in the same commit.

## 6.4 `Path.write_text` on Windows translates line endings

It opens in text mode and converts `\n` to `\r\n`. Git reported CRLF normalisation on all
eight files of `44511fa`. Cosmetic, because Git normalises the index -- but **always pass
`newline="\n"`** in installers.

## 6.5 The methodology ledger -- ~28 instances this session

A checker that string-matches or name-matches fires on prose describing its own rule, or on
something that merely resembles its target. New instances today: the probe binding arguments
by position (manufactured a false 20x finding); `eps` classified as a probability because it
contains "p"; a 1e-9 tolerance calling a rounded value "a different quantity"; a single
bin-count reference reporting a parameter choice as a defect; and a validation check that
re-indented already-indented blocks and reported six false failures.

**The durable lesson, now very well evidenced: outcome-asserting checks catch what careful
reading does not.** Every defect this session -- in the codebase and in the instruments -- was
found by running something against a fixture.

## 6.6 Retractions on record -- do not re-propagate

- `evaluator.py` open-top defect: **repaired 2026-07-10**, not live.
- Roadmap 6.29a's 19-column claim: superseded by measurement (21 columns including `ok`).
- Roadmap 6.23's "WITHDRAWN 2026-07-14": stale, the ban was deleted deliberately 2026-07-15
  per `test_readme_claims.py:698`.
- AlphaFold "[missing] = data loss": wrong, 107.1 MB present.
- `check_agents_active.py` broken: wrong, STALE is warning-only by design.
- push-ghcr skipping: correct, it is release-gated.

---

# PART SEVEN -- SUGGESTED FIRST FIFTEEN MINUTES OF THE NEXT SESSION

1. `git -C C:\Projects\genomic-variant-classifier log --oneline -10` and
   `git -C C:\Projects\genomic-variant-classifier status --short` -- confirm HEAD is `0208998`
   and the tree is clean.
2. Read `docs/sessions/SESSION_2026-07-20_calibration-defect-repair.md` in full.
3. Read the 2055 and 2060 entries in `tests/EXPECTED_SUITE_SIZE`.
4. Run `probe_calibration_divergence_v5_2026-07-20.py` -- read-only, seconds, confirms the
   repair holds.
5. Settle section 4.2 in one minute: read the binning line of
   `calibration_drift_agent.py:45` and compare against the two defect patterns.
6. **Then ask Monzia explicitly** whether to continue the metric registry (commit 2) or to go
   straight at the five deliverables of Part One. That decision was made implicitly today and
   should not be made implicitly twice.
