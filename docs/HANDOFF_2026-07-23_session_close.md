# Session handoff -- 2026-07-23

Written at the close of the 2026-07-22/23 session so the next session resumes with
no guesswork and no relitigation. Read this top to bottom before touching anything.

## 1. Where the repository stands right now

- **Remote:** `github.com/monzia-moodie-repo-projects/genomic-variant-classifier`
- **`main` HEAD:** `644a184` -- "fix(scripts): read parquet natively to remove the
  teardown abort at its root"
- **Suite-size ratchet:** **2874** (`tests/EXPECTED_SUITE_SIZE`, README badge agrees)
- **Full suite last measured on Windows:** 2867 passed, 7 skipped = 2874 collected
- **The seven skips (unchanged all session, none new):** four in
  `tests/integration/test_mc_dropout_calibration.py`, one in
  `tests/unit/test_preflight_data_paths.py:45`, one in
  `tests/unit/test_tabular_nn_mc_dropout.py:232`
- **Local:** `C:\Projects\genomic-variant-classifier`, Python 3.12.10, venv
  `.venv312`, PowerShell 5.1. Downloads always land in `C:\Users\monzi\Downloads\`.

## 2. What was completed this session

### 2.1 The Panel R matched-null sequence (seven commits, all on main)

| # | SHA | What |
| --- | --- | --- |
| 1 | `2aabf12` | record the negative R3 transfer-validation result |
| 2 | `789d225` | codify the orthogonal invariance of angular recovery |
| 3 | `5088b5a` | add the eigenvalue-assignment matched null |
| 4 | `fea98e4` | wire the matched null family into a two-null intersection |
| 5 | `199b813` | calibrate alignment-sensitive whitening recovery |
| 6 | `d605dc9` | record the R3a/R3b split in the panel capability reason |
| 7 | `3117205` | reuse the validated null protocol across the R4 ladder |

Scientific result: angular concentration is **alignment-blind** -- it responds to
whitening gain-spectrum magnitude, which the matched null preserves by construction,
not to gain-to-direction alignment, which the null scrambles. It therefore cannot
beat the matched null even in principle. The alignment-**sensitive** estimand is
trace-normalised covariance-shape recovery, which was calibrated (Type-I 0/40, power
40/40) and is the estimand the matched null can test. R3 split into R3a (angular,
non-admissible, negative finding preserved) and R3b (alignment, method-validated on
synthetic only, scientific validation NOT run). R4 inherits the validated protocol by
contract without being promoted.

### 2.2 Panel S0 / Mixture-of-Experts (two commits on main, one built and waiting)

| SHA | What |
| --- | --- |
| `20289c8` | docs(moe): specify Panel S0 expert identity and interpretive admissibility |
| `4251834` | docs(moe): revise Panel S0 with the causal boundary and four-quantity routing |

`docs/specifications/PANEL_S0_ROUTING_IDENTIFIABILITY.md`, 22 sections
(S0.0-S0.21), ~6,600 words. The decisive correction, contributed by Monzia and
adopted: **model-component interventions establish model reliance, not biological
causation.** Removing an expert intervenes on a computational component, not on
splicing or protein stability. So `BIOLOGICAL_MEDIATION` is permanently inadmissible
through Panel S0 and is deferred to a future **Panel T (Causal Mechanism and
Mediation)**. Routing is decomposed into four quantities -- allocation `a_e`
(simplex-constrained, engineering), mechanism evidence `m_e` (the only quantity
eligible for a mechanistic reading, and barred from entering the allocation), expert
utility `u_e`, reliability `q_e` (runtime only, must exclude anchor availability).

### 2.3 Hygiene and infrastructure (three commits on main)

| SHA | What |
| --- | --- |
| `821a990` | test(hygiene): message-pin the scikit-learn parallel.delayed warning |
| `8f0b452` | ci(diagnostic): probe the native teardown abort before changing any code |
| `c7809dc` | ci(diagnostic): keep core dumps out of the artifact, add an evidence consolidator |
| `3685292` | ci(diagnostic): target the Arrow-to-pandas conversion, and stop over-claiming |

`.gitignore` gained coverage artifacts (`.coverage`, `.coverage.*`, `coverage.xml`,
`htmlcov/`) and `diagnostics_out/`. Before that, running the exact Continuous
Integration command locally left an untracked `.coverage` that `git add -A` would
have committed.

### 2.4 The teardown-abort investigation, closed (`644a184`)

**Symptom (2026-07-22).** Continuous Integration run 29962715186 (run number 585,
commit `821a990`) failed one test, on Python 3.12 only. A child process of
`tests/unit/test_rnaseq_ablation_tools.py::test_full_is_unchanged` returned -6
(SIGABRT) with `terminate called without an active exception`, **after** printing its
own success line.

**Method.** A dispatch-only diagnostic workflow ran two rounds -- 9 then 14 arms, 5000
child executions each, 115,000 executions total -- bisecting by construction and by
mitigation, with core dumps captured and backtraced by the GNU Debugger.

**Root cause, proven from 27 identical core-dump frame chains:**

```
arrow::py::PyReadableFile::~PyReadableFile()   libarrow_python.so.2300
arrow::py::OwnedRefNoGIL::~OwnedRefNoGIL()     libarrow_python.so.2300
PyGILState_Ensure()                            Python/pystate.c:2240
take_gil()                                     Python/ceval_gil.c:353
PyThread_exit_thread()                         Python/thread_pthread.h:370
_Unwind_ForcedUnwind()                         libgcc_s.so.1
std::terminate()                               libstdc++.so.6  -> SIGABRT
```

`pandas.read_parquet` hands Arrow a **Python file handle**, wrapped in
`arrow::py::PyReadableFile`. That wrapper holds a Python object reference, so its
destructor calls `PyGILState_Ensure`. Running on an Arrow background thread after
interpreter finalisation has begun, CPython's `take_gil` kills the thread with
`pthread_exit`; the forced unwind crosses C++ destructor frames and libstdc++ calls
`std::terminate`.

**Fix.** `scripts/make_rnaseq_ablation_parquet.py` now reads through
`pq.read_table(args.src).to_pandas()`, which opens the file natively in C++ so the
faulting object is never constructed. Measured in ONE run at 5000 executions per arm:
pandas read **27 aborts**, native read **0**, twice, in two independent arms.

**Two hypotheses that were wrong, recorded so they are not re-formed:** OpenBLAS
(threads appear parked on `thread_status` futex waits in every core, but **zero**
OpenBLAS frames appear on any abort path -- bystanders), and the Arrow-to-pandas
conversion (`to_pandas_explicit` returned 0/5000 -- refuted).

**A statistical over-claim, corrected:** round one's three thread-constraint arms were
referenced against the real script at 1/5000, so a zero result was ~37% likely even if
the constraint did nothing (Fisher exact p = 1.000). The consolidator now refuses to
call suppression when fewer than three events are expected. Also measured:
`ARROW_IO_THREADS=1` does **not** suppress (22/5000), and `read_cpu_count_1` is
confounded (it changes import order as well as thread counts).

Full record: `docs/INCIDENT_2026-07-23_rnaseq_ablation_teardown_abort.md`.

## 3. ITEM 3 -- S0 Commit 2, built and verified, READY TO PUSH

This is the immediate next action. All four files are already in
`C:\Users\monzi\Downloads\`.

**IMPORTANT:** an earlier installer named `install_ratchet_bump_2879_2026-07-22.py`
is **STALE** (it targets 2860 -> 2879 and its pre-check will fail against the current
ratchet of 2874). Use `install_ratchet_bump_2893_2026-07-23.py` instead.

| File | Destination | SHA-256 |
| --- | --- | --- |
| `moe_identity.py` | `src\genomic_variant_classifier\evaluation\` | `0ede6db0e8372c9721afb42f4f23725b0138e342e573ee4d2a179e2cbe948621` |
| `test_moe_identity.py` | `tests\unit\` | `ff6eaf2212ab60c39104690339cffa83066af0c4b24a108f43b34ce26a45bf8c` |
| `install_ratchet_bump_2893_2026-07-23.py` | Downloads (run from there) | `04d340f7e358b48f121adc4dffc542ebdcc3b2a655da1b66bdd02050dc3a9129` |
| `COMMIT_MSG_2026-07-22_moe_identity_contracts.txt` | Downloads | `a3207de4d3b44e206b3c6015233584656aabdd69e3d52c33cdfa34bb4e09a533` |

Net **+19**, ratchet **2874 -> 2893**, four files, purely additive (reuses the two
validation axes from `r3_capability.py`, modifies no existing module). Eight sabotages
verified. A dead report-label assertion was found and removed during that pass.

### Exact command sequence

```powershell
$Repo = "C:\Projects\genomic-variant-classifier"
git -C $Repo status --short   # MUST be clean; if not, stop

Copy-Item "C:\Users\monzi\Downloads\moe_identity.py" "$Repo\src\genomic_variant_classifier\evaluation\" -Force
Copy-Item "C:\Users\monzi\Downloads\test_moe_identity.py" "$Repo\tests\unit\" -Force
```

```powershell
Get-FileHash "$Repo\src\genomic_variant_classifier\evaluation\moe_identity.py" -Algorithm SHA256
# expect 0EDE6DB0E8372C9721AFB42F4F23725B0138E342E573EE4D2A179E2CBE948621

Get-FileHash "$Repo\tests\unit\test_moe_identity.py" -Algorithm SHA256
# expect FF6EAF2212AB60C39104690339CFFA83066AF0C4B24A108F43B34CE26A45BF8C
```

```powershell
python -m pytest tests\unit\test_moe_identity.py tests\unit\test_r3_validation.py -q
```
Expect **43 passed**.

```powershell
python "C:\Users\monzi\Downloads\install_ratchet_bump_2893_2026-07-23.py"
```
Pre-check reads **2874**, all ten POST checks OK, -> 2893.

```powershell
python -m pytest tests/ -q -rs --assert-suite-size
```
Expect **2886 passed, 7 skipped (2893 collected)**, same seven skips.

```powershell
git -C $Repo add -A
git -C $Repo status --short
```
Expect exactly four paths:
```
M  README.md
A  src/genomic_variant_classifier/evaluation/moe_identity.py
M  tests/EXPECTED_SUITE_SIZE
A  tests/unit/test_moe_identity.py
```

```powershell
git -C $Repo commit -F "C:\Users\monzi\Downloads\COMMIT_MSG_2026-07-22_moe_identity_contracts.txt"
git -C $Repo push
```

## 4. ITEM 1 -- repository-wide Python-handle-into-Arrow audit (NOT STARTED)

**The finding that scopes it:** `pandas.read_parquet` is called at **328 sites**
across `src/genomic_variant_classifier/` and `scripts/` (counted by syntax-tree walk,
not grep). Tonight's fix corrected exactly one.

**Do NOT impose a blanket ban.** A guard forbidding `pandas.read_parquet`
repository-wide would turn 328 sites red at once and demand a rewrite justified by a
fault observed at ~0.5% in one narrow condition.

**Exposure is not uniform and has NOT been measured.** The abort needs three things
together: Arrow constructing a `PyReadableFile` from a Python handle, that destructor
still pending at interpreter finalisation, and the process exiting promptly
afterwards. The last matters enormously -- the same read aborted 27/5000 when the
process exited immediately but 1/5000 and then 0/5000 inside the real script, which
does more work first. Long-running processes (the application-programming-interface,
the agent layer, training) may be effectively immune; short-lived scripts that read a
parquet and exit are the exposed shape.

**Protocol for the next session:**
1. Enumerate all 328 sites by syntax-tree walk (the method that found them).
2. Classify each by process lifetime: short-lived entry point that reads and exits,
   versus long-running service.
3. Measure the exposed shapes on the Continuous Integration runner using the existing
   diagnostic (`scripts/diagnostics/probe_teardown_abort.py`, dispatch-only workflow
   `.github/workflows/teardown_abort_diagnostic.yml`, 14 arms, `iterations` input).
   Add an arm per candidate shape; reference against `pyarrow_read` (0.54-0.90%),
   never against a low-rate arm.
4. Remediate only the exposed class with the native read, and pin THAT class with a
   guard test modelled on `tests/unit/test_rnaseq_ablation_native_read.py`, which
   walks the syntax tree rather than grepping.
5. Record the criterion in the incident note so the classification can be continued.

## 5. ITEM 2 -- data-source freshness failures (NOT STARTED)

From `FRESHNESS_2026-07-20.md`, 24 sources scanned:
- **alphafold: `[unreachable] HTTPError: HTTP Error 404: Not Found`**
- **lovd: `[unreachable] HTTPError: HTTP Error 400: Bad Request`**
- 5 sources reported `[changed] first observation` (clinvar, alphamissense, gnomad,
  gnomad_constraint, esm2) -- expected on a first probe, not defects.
- 19 local assets `[missing] absent on disk` -- expected because `data/external` is
  empty, but the report does not say so, making 19 expected states look like 19
  defects.
- 4 sources `[present] no local_path declared` (phylop, vep, kgp_1000, reactome,
  cosmic, tcga) -- an inconsistency worth closing.

Treat each unreachable source as its own root-cause investigation: endpoint drift
versus genuine outage versus malformed request. Do not assume drift. The AlphaFold
connector lives in `src/genomic_variant_classifier/data/alphafold.py`, LOVD in
`src/genomic_variant_classifier/data/lovd.py`.

## 6. ITEM 4 -- environment reproducibility (NOT STARTED)

`requirements.lock` (162 exact pins) and `requirements-dev.lock` (174 exact pins)
**exist but are referenced ZERO times** in `.github/workflows/ci.yml`. The pytest job
installs `requirements-api.lock` (pinned, 18 references) and then the **unpinned**
`requirements.txt` and `requirements-dev.txt`, which carry 15 unpinned entries between
them (`optuna>=4.0`, `polars>=1.0`, `duckdb>=1.0`, `pykan>=0.2.0`, `gudhi` with no
constraint at all, `pybigtools>=0.3.0`, `pytest>=8.0,<10.0`, `pytest-cov>=5.0`, and
others).

Consequence: the test environment is re-resolved from the package index on every run,
so Continuous Integration can go red with zero code changes and Python 3.11 and 3.12
can resolve different versions of the same package. **This did not cause the teardown
abort** (numpy, pandas and pyarrow are all exact-pinned) but it is real.

Fix: install the lock files in the pytest job and extend the existing lockfile-drift
check to cover `requirements.lock` and `requirements-dev.lock` so they cannot rot
unused again. Adding no `uses:` line means no ratchet change.

## 7. Status of the larger deliverables (asked 2026-07-23)

**None of the following were worked on in this session.** This session covered the
Panel R matched-null sequence, the Panel S0 specification and typed contracts, three
hygiene commits, and the teardown-abort investigation. What follows is the status as
observed in the code, with an explicit note where I did not verify.

- **JEPA (Joint-Embedding Predictive Architecture): NOT IMPLEMENTED.** No module
  exists. It was identified during the Mixture-of-Experts evaluation as a
  prerequisite that does not yet exist, alongside activated ESM-2 and Nucleotide
  Transformer encoders and exported graph-neural-network embeddings.

- **Conformal prediction: IMPLEMENTED, not touched this session.** Modules present
  under `src/genomic_variant_classifier/conformal/`: `calibrate.py` (95% covered),
  `coverage.py` (94%), `grouped.py` (94%), `mondrian.py` (63%), `ordinal.py` (96%),
  `scores.py` (66%), `split.py` (83%). Tests: `tests/unit/test_ordinal_conformal.py`,
  `test_conformal_package_exports.py`, `test_calibration_baseline.py`,
  `test_calibration_carve.py`, `test_calibration_implementations_agree.py`,
  `test_calibration_validity_contract.py`, plus `tests/conformal/` (7 files).
  Note the low coverage on `mondrian.py` (63%) and `scores.py` (66%) -- worth an audit.

- **Conformal quantile regression: NOT VERIFIED.** I did not confirm whether this
  specific method is implemented. Do not assume either way; check
  `src/genomic_variant_classifier/conformal/` for a quantile-regression score
  function before planning work.

- **Mixture-of-Experts: SPECIFICATION AND TYPED CONTRACTS ONLY. No model exists.**
  The Panel S0 specification is on main (`4251834`). The typed contracts
  (`moe_identity.py`, +19 tests) are built and waiting to be pushed -- see section 3.
  No router, no experts, no training. The build order agreed with Monzia is: Phase S0
  (spec plus contracts, everything NOT_ADMISSIBLE) is buildable now; S1 an anonymous
  predictive MoE on real tabular and graph-neural-network features once data is clean;
  S2 anchor acquisition; S3 real representations; S4 anchored dense MoE; S5 Panel S0
  validation; S6 name promotion; S7 sparse routing.

- **RNA infrastructure: PARTIALLY IMPLEMENTED, not touched this session except the
  ablation script fix.** `src/genomic_variant_classifier/pipelines/rna_pipeline.py`
  (56% covered), `src/genomic_variant_classifier/data/rnaseq.py` (89% covered),
  `scripts/make_rnaseq_ablation_parquet.py` (fixed this session),
  `scripts/launch_run17_rnaseq_ablation.sh`. A carried-over item from earlier
  sessions remains open: the **RNA-sequencing differential-expression leakage check**
  (cohort-independence), flagged by preflight and never completed.

- **The full metric stack: IMPLEMENTED for the predictive metrics; the
  representation-geometry ladder is deliberately incomplete.** Present and tested:
  AUROC, AUPRC and no-skill baseline, Brier, log loss, Expected Calibration Error,
  calibration slope and intercept, bootstrap and gene-cluster confidence intervals,
  the conformal coverage family, odds ratio, Fisher exact, Cramer's V, the Panel Q
  clustering metrics (both Davies-Bouldin forms, Calinski-Harabasz, estimated
  silhouette, Adjusted Mutual Information with a permutation null, the confounder
  gate), and drift detection. Panel R: R1 and R2 have output; **R3 is
  OUTPUT_AVAILABLE but explicitly NON-ADMISSIBLE** (R3a alignment-blind, R3b
  method-validated on synthetic only); **R4 and R5 are IMPLEMENTED_NO_OUTPUT**;
  **R6 and R7 are NOT_IMPLEMENTED**. `docs/METRICS.md` (180 lines) has **not** been
  audited against the code since the R3a/R3b split and the protocol-inheritance work
  landed -- that audit is outstanding.

## 8. Session-durable lessons (added this session)

- **Compute sums in a tool call, never from memory.** Prose arithmetic drifted three
  times. Monzia's `CalibrationSummary`, which enforces `observed_rate ==
  n_admitted / n_simulations` at construction, structurally kills the class.
- **Never state a rate without its denominator.** "0.067" was 2/30 reported as if
  from 40 trials.
- **A verdict that over-claims is worse than no verdict** -- it ends an investigation
  in the wrong place. The consolidator's power gate exists because of this.
- **Zero events is a bound, not proof.** Report the rule-of-three upper bound.
- **Hashes, not test counts, prove a copy landed.** Three times this session the suite
  was green while the working tree held stale files.
- **Every step the user must perform gets a runnable command.** Prose steps do not get
  run; this cost a full cycle when `Copy-Item` blocks were omitted.
- **Never paste angle-bracket placeholders into PowerShell** -- `<` is a reserved
  operator and the parser dies before anything runs.
- **A shallow clone lies about dates.** `git log -1 --format=%ad` on a `--depth N`
  clone reports the shallow boundary. Use a full clone for history.
- **`--noconftest` is invalid for tests that exercise plugin registration.** It
  produced a false failure in `test_suite_size_ratchet.py`.
- **Check an installer's prose against its own arithmetic.** The 2874 installer
  printed a stale "Expect 2860 passed" because a build step failed silently.
- **Adding a workflow changes the collected test count.**
  `tests/unit/test_workflow_action_pins.py` is parametrized over workflow files and
  over every `uses:` line: a new workflow with K action references adds 1 + 2K tests.
  Reuse commit hashes already in `EXPECTED_PINS` or the count grows further.

## 9. Key file locations

- Diagnostic probe: `scripts/diagnostics/probe_teardown_abort.py` (14 arms)
- Evidence consolidator: `scripts/diagnostics/collect_teardown_evidence.py`
- Diagnostic workflow: `.github/workflows/teardown_abort_diagnostic.yml`
  (**dispatch-only**; Actions -> Teardown abort diagnostic -> Run workflow;
  inputs `iterations` and `python_version`)
- Panel S0 spec: `docs/specifications/PANEL_S0_ROUTING_IDENTIFIABILITY.md`
- Incident: `docs/INCIDENT_2026-07-23_rnaseq_ablation_teardown_abort.md`
- Panel R evaluation modules: `src/genomic_variant_classifier/evaluation/`
  -- `r3_validation.py`, `norm_angle_probe.py`, `null_family.py`,
  `alignment_recovery.py`, `r3_capability.py`, `recovery_protocol.py`,
  `representation_geometry.py`, `capabilities.py`, `capability_lifecycle.py`
- Ratchet: `tests/EXPECTED_SUITE_SIZE` (number plus a dated prose ledger)
- `gh` is UNAUTHENTICATED -- the pasted token returns HTTP 401. Use the browser, or
  `gh auth login -h github.com` and choose "Login with a web browser" rather than
  pasting a token. `GITHUB_TOKEN` at Machine scope is empty; User scope unchecked.
