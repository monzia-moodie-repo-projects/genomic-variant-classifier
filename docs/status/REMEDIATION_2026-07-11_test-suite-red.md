# REMEDIATION 2026-07-11 — the red test suite, all four clusters

## FINAL RESULT — SUITE GREEN (2026-07-11, `outputs/fullsuite_2026-07-11c.log`)

```
1814 passed, 8 skipped, 41 warnings in 508.52s (0:08:28)
```

**Zero failures.** Trajectory across the session:

| run | log | result |
|---|---|---|
| baseline (2026-07-08 triage) | — | **24 failed**, 1,585 passed |
| after clusters A–D fixed | `fullsuite_2026-07-11.log` (11:23) | **11 failed**, 1,802 passed |
| after the cluster-A′ regression fixed | `fullsuite_2026-07-11b.log` (11:39) | **5 failed**, 1,809 passed |
| after cluster E (sys.path leak) fixed | `fullsuite_2026-07-11c.log` | **0 failed**, 1,814 passed |

The `sys.path` leak guard added to `tests/conftest.py` tripped **no other test**, which is itself
a result: `test_rekey_seq_windows_v2.py` was the suite's only path-leaker.

**NOT YET COMMITTED.** Every change below is still an uncommitted working-tree modification.

**Remaining (none are test failures):** 41 warnings in three classes (§4b.3), the dual
feature-engineering implementations (§5b — the most consequential finding of the session), the
absent test gate (§5), and the repository hygiene debt (§6).

---

**Status:** all four triaged clusters plus one regression and one newly-discovered cluster are
FIXED and verified green against the full suite in `.venv312`.
**Supersedes:** `TRIAGE_2026-07-08_test-suite-red.md` (which remains the record of discovery).
**HEAD at time of work:** `e3e422e` on `main`.
**Author:** Monzia Moodie. **Session:** 2026-07-11.

---

## 0. The correction that matters most

Sessions between 2026-07-09 and 2026-07-11 were operating on a **two-class model of the
failure surface** — "Class A: the ESM-2 `cache_path` AttributeError" and "Class B: the
stage-5 allowlist" — totalling **7 failures**, and quoting suite sizes of *596*, *654*, and
*128 passed* at various points.

Every one of those numbers is wrong. The project's own triage of **2026-07-08** had already
established the truth, and it was never carried forward:

| | two-class model (stale) | reality (`TRIAGE_2026-07-08`, re-confirmed 2026-07-11) |
|---|---|---|
| suite size | "596" / "654" | **1,616 collected** |
| failures | 7 | **24 failed, 1,585 passed, 7 skipped** (365 s) |
| clusters | 2 | **4** (A=12, B=6, C=1, D=5) |

The two "classes" under discussion were clusters **B (6)** and **C (1)** — **7 of 24**. The
largest cluster, **A (12)**, and cluster **D (5)** were absent from the working model
entirely. Any plan built on the two-class model would have declared victory with **17
failures still red**.

**Root lesson, recorded so it is not relearned:** a handoff that carries forward a *summary*
rather than the *authoritative artefact* silently loses the artefact's contents. The triage
document existed in `docs/incidents/` the whole time.

---

## 1. Cluster A (12 tests) — the suite's verdict depended on which files sat on the disk

### 1.1 What the triage said, and where it was wrong

The 2026-07-08 triage localised this correctly in spirit but **misidentified the arming
path**. It stated:

> "The failing tests construct `AnnotationConfig()` with no AlphaMissense path, so `_am_tsv`
> is `None`. The gate can only return `True` through `Path(str(pc.cache_path)).exists()`."

Reading the source (2026-07-11) shows `_am_tsv` was **never `None`**:

```python
# real_data_prep.py:940 (before)
_am_tsv = ac.alphamissense_tsv_path or Path(r"data/external/alphamissense/AlphaMissense_hg38.tsv.gz")
```

`ac.alphamissense_tsv_path` defaults to `None` (line 308), so in unit tests `_am_tsv` became
the **hard-coded 613 MB TSV path**. That file is present on this box
(`AlphaMissense_hg38.tsv.gz`, 642,961,469 bytes). The gate therefore armed through the
*first* branch of `_protein_coord_source_present` (`am_path is not None and
Path(am_path).exists()`), not the `cache_path` branch the triage blamed.

The consequence is the same and the principle is the same — but a fix aimed only at the
`cache_path` branch would have left the defect fully intact.

### 1.2 The mechanism

A library silently substituted a filesystem default for a configuration the caller never
declared. "The caller declared nothing" became "the caller declared the TSV." Because that
TSV happens to exist here, the coverage gate armed against 2- and 5-row unit fixtures whose
`protein_pos` is legitimately all-NA, computed coverage `0.0000 < 0.50`, and raised — in
direct contradiction of its own docstring, which promises stub mode "must never raise."

**These 12 tests are green on a clean machine and red on a populated one.** A suite whose
outcome is a function of untracked filesystem state is not a suite.

### 1.3 The fix — source presence is a property of declared configuration

- `_protein_coord_source_present(am_path)` now returns `True` **iff the caller explicitly
  declared a source and that declared path exists**. The `cache_path` filesystem branch is
  removed: a coord cache built by a *previous run against a previous cohort* is not a source
  wired into *this* run (the gate's own error text already concedes this — "stale or
  mismatched for THIS cohort/box").
- The hard-coded fallback at the call site is **deleted**: `_am_tsv = ac.alphamissense_tsv_path`.

**Production is provably unaffected.** A repository-wide search for `alphamissense_tsv_path`
returns exactly three hits: the dataclass default, the call site, and
`scripts/run_phase2_eval.py:356` — **the only production caller, and it declares the path
explicitly** (supplying the same default itself). The gate therefore stays armed exactly
where it must be, and callers that declare nothing get stub mode, which never raises.

---

## 2. Cluster B (6 tests) — a fixture that constructed an object whose `__init__` never ran

### 2.1 Root cause (hypothesis in the triage; now **confirmed** by reading the fixture)

```python
# tests/unit/test_esm2_llr.py:50 (before)
c = E.ESM2Connector.__new__(E.ESM2Connector)   # bypass __init__ / index plumbing
c.model_name = "esm2_t33_650M_UR50D"
c.device = "cpu"
c._missing_genes = set()
c._get_sequence = _get_seq
```

Four attributes on a half-built object. `cache_path` is assigned **only** in `__init__`
(`esm2.py:571`). When the parquet score-cache later landed in `annotate_llr`, the new path
`annotate_llr:947 → _score_cache_load:657 → _score_cache_path:643 → self.cache_path` reached
an attribute the fixture never created:

```
E  AttributeError: 'ESM2Connector' object has no attribute 'cache_path'
```

Confirmed against the live traceback. Six tests. Production was never affected — no
production path constructs via `__new__`, and ESM-2 scored 176/181 variants on the VM on
2026-07-06.

### 2.2 The fix I explicitly did **not** make, and why

The obvious one-liner — `getattr(self, "cache_path", None)` in `_score_cache_path` — is
**actively dangerous** and was rejected. That method's existing fallback is
`Path("data/raw/cache")`, i.e. the **real repository cache**. Made defensive, the unit tests
would have:

1. **read** any real `data/raw/cache/esm2_scores.parquet` — non-hermetic, order-dependent tests; and
2. via `_score_cache_append`, **written fake stub scores into the production ESM-2 score cache** — silent data corruption of a real artefact by a unit test.

It would also have papered over a fixture defect by teaching production code to tolerate
malformed objects, in a codebase whose stated rule is that nothing fails silently. The
2026-07-08 triage reached the same conclusion independently ("Not prescribing
`getattr(...)`").

### 2.3 The fix made instead — construct the object properly

The fixture now goes through the real `__init__`, with `cache_path` pinned under `tmp_path`.
This is safe and stays torch-free, verified against source:

- `device="cpu"` is passed explicitly, so `_resolve_device` short-circuits at `esm2.py:226`
  and **never touches `torch.cuda`**;
- `__init__` performs **no I/O** — the sqlite handle is lazy (`_conn = None`);
- the model loader remains monkeypatched to the fake tokenizer/model;
- `allow_network=False` is explicit: no test can reach UniProt.

Because `_score_cache_path` derives the score cache from `cache_path.parent`, pinning
`cache_path` to `tmp_path` makes **both** caches hermetic.

This also removes a standing fragility: the `__new__` fixture was guaranteed to break again
the next time `annotate_llr` touched any new `__init__` attribute. It was the **only**
`__new__` construction in the suite — every other ESM-2 test file already constructs
normally, most with `cache_path=tmp_path / ...`.

**New guard:** `test_fixture_is_fully_constructed_and_hermetic` asserts every `__init__`
attribute exists *and* that the score cache resolves inside `tmp_path` and never into the
repository `data/` tree.

---

## 3. Cluster C (1 test) — the stage-5 zero-audit allowlist

### 3.1 Ground truth (computed against live source, 2026-07-11, not inferred)

`engineer_features(build_reference_slice())` → 97 numeric columns, 200 rows. Stage 5 flags
30 non-binary columns with zero-rate ≥ 95%. Against `KNOWN_ZERO_DEFAULT`:

**Flagged − allowlist (the CI failure) — all six at 100% zero:**

```
cosmic_recurrence   cosmic_sig_tier   genomiclm_delta_norm
genomiclm_llr       kegg_pathway_count   kegg_disease_pathway_flag
```

These are the six features added by the 91→97 work in **`80eb9c8` (2026-07-06)**. All six are
plain `df.get(col, 0.0)` passthroughs (`variant_ensemble.py:657-695`), and
`build_reference_slice` supplied **none** of them — so they came out 100% zero for exactly
the same structural reason as the allowlisted columns. The harness file's last commit was
`e6447fb` (2026-06-27): **the feature contract advanced and the fixture did not follow.**

### 3.2 The disposition — Option B, per the project's own committed precedent

Allowlisting the six would have been the patchwork answer, and it is **wrong**: they are
*live* connectors (Run-17 real-data smoke shows them populated). Allowlisting a live feature
permanently blinds stage 5 to a real regression in it.

The repository already settled this question. Commit **`e6447fb` (2026-06-27)** —
*"feed FinnGen R12+R13 in reference slice (Option B), allowlist 29→25"* — establishes the rule,
and the surrounding code states it explicitly (`clingen_validity_score` "is a live feature,
not a dead connector, so it must stay outside the allowlist"; `esm2_llr` "live feature, NOT
allowlisted"):

> **live connector → FEED it in the fixture, keep it OUT of the allowlist.**
> **dead connector → allowlist it, with the reason it cannot yet populate.**

The six are therefore **fed** in `build_reference_slice`, not allowlisted. This also restores
the fixture's own docstring contract — "every input `engineer_features` consumes is supplied…
the only ~all-zero columns are exactly `KNOWN_ZERO_DEFAULT`" — which had been **false** since
2026-07-06.

### 3.3 Two further defects found by auditing the allowlist in **both** directions

Nobody had checked the reverse direction — allowlisted columns that are *no longer* flagged.

- **`gene_is_constrained` is a stale allowlist entry — REMOVED (25 → 24).** It is not a
  connector at all: it is a *derived binary indicator*,
  `(gene_constraint_oe < 0.35).astype(int)` (`variant_ensemble.py:439`). On the reference
  slice it takes both 0 and 1 (zero-rate 83.5%), so stage 5's binary exemption skips it and
  it **can never be flagged** — the entry was dead weight. Worse, it was *harmful*: if the
  constraint connector ever went dead, the column would collapse to `{0}` → non-binary →
  flagged, and the allowlist would have **silently swallowed that regression**. Outside the
  allowlist, stage 5 now catches it. Removal strictly increases coverage and changes no
  current verdict.

- **The count comment was stale by two.** The header claimed "27 columns"; the literal held
  **25**, and has since `e6447fb` (2026-06-27). Corrected to 24, with the full count history
  recorded inline (21 → 22 → 27 → 29 → 25 → 24) so each change reads as an audit, not a bump.

### 3.4 Lockstep guard — caught before it could fail

`tests/unit/test_harness_fixture_omim_molecular.py:55` hard-asserts
`len(KNOWN_ZERO_DEFAULT) == 25`. Removing `gene_is_constrained` would have broken it.
Updated to `== 24` with the reasoning, in the same change.

**New guard:** `test_live_connectors_are_fed_not_allowlisted` makes THE RULE executable — every
live connector input must be present in `build_reference_slice` **and** absent from
`KNOWN_ZERO_DEFAULT`. This is precisely the guard whose absence let `80eb9c8` land six live
connectors the fixture never fed.

### 3.5 Verification (run against the edited source before it was committed)

```
KNOWN_ZERO_DEFAULT            = 24        (was 25)
engineered numeric columns    = 97
flagged − allowlist           = {}   ← the CI assertion now passes
allowlist − flagged           = {}   ← no stale entries remain
allowlist ⊆ engineered matrix = PASS ← no phantom entries
build_reference_slice         = deterministic across builds
```

---

## 4. Cluster D (5 tests) — the Run-17 audit split fixtures

`audit_smoke_feature_population.py --run17` grades `genomiclm_delta_norm` and
`kegg_pathway_count` as **FAIL-severity when absent, all-null, all-default, or constant**. The
split fixtures `_cols()` (`test_run17_audit_persplit.py`) and `_write_splits()`
(`test_run17_fullflag_smoke.py`) never emitted them, so the audit returned 1 while the tests
asserted 0. Red since the connectors landed. `_write_splits`' docstring — *"EVERY run17
feature is populated"* — had been false the whole time: the identical contract-drift as
cluster C, in a second fixture.

**Fix:** both fixtures now emit all six new connector columns with **varying** values (the
audit's constant-check would fail a constant column).

---

## 5. The meta-finding — CORRECTED 2026-07-12. I had this badly wrong.

**Retraction.** Throughout 2026-07-11 I asserted, repeatedly and with confidence, that *"no
gate runs the test suite."* **That is false.** This repository has had GitHub Actions
Continuous Integration all along — `.github/workflows/ci.yml`, **508 workflow runs**, running
`pytest` against a matrix of Python 3.11 and Python 3.12, and *blocking* the container build
and the push to the GitHub Container Registry (`docker-build` has `needs: test`).

I took the 2026-07-08 triage's true statement — that `Run_Preflight_VM.sh` and
`vm_bootstrap_run.sh` do not run `pytest` — and generalised it into "no Continuous Integration
exists," then repeated it as established fact for a full day without once opening
`.github/workflows/`. That is exactly the unverified-stale-assumption failure this document
exists to catalogue, committed by its author.

### 5.1 What is ACTUALLY wrong — and it is worse in two ways, better in one

**Continuous Integration exists, runs, blocks deployment of the container — and has been RED,
and was merged past anyway.** Run **#499**, commit `e3e422e`, 2026-07-11: FAILED. The failures
are precisely cluster B (`AttributeError: 'ESM2Connector' object has no attribute 'cache_path'`)
and cluster C (the six stage-5 columns `cosmic_recurrence, cosmic_sig_tier,
genomiclm_delta_norm, genomiclm_llr, kegg_disease_pathway_flag, kegg_pathway_count`).

Three defects in the gate itself:

| # | defect | consequence |
|---|---|---|
| **5.1a** | `pytest tests/unit/ --maxfail=5` — **stops after 5 failures** | The gate is configured never to tell you the truth. It reported `5 failed, 649 passed`. The real number was **24**. Nobody knew the scale because the gate cannot report it. |
| **5.1b** | `pytest tests/unit/` — **`tests/unit/` only** | `tests/conformal/` (7 files), `tests/integration/` (1 file), and **22 root-level `tests/test_*.py` files** — 30 test files — **never run in Continuous Integration**. This is why the clean-clone breakage (tracked tests importing untracked scripts, fixed in `343cc66`) could persist indefinitely: Continuous Integration never executes those tests. |
| **5.1c** | the rented-GPU path bypasses Continuous Integration entirely | `Run_Preflight_VM.sh` / `vm_bootstrap_run.sh` check GPU, CUDA, VRAM, dependencies, disk, RAM and git HEAD — and **do not run `pytest`**. This part of the 2026-07-08 triage was correct. On 2026-07-06 code shipped to a rented GPU with 24 red tests. Continuous Integration would have blocked the *container*; nothing blocked the *VM*. |

### 5.2 Required

1. **Remove `--maxfail=5`.** A gate that hides the size of the failure is worse than a gate
   that fails loudly. Use `--maxfail=0` (report everything) or a large bound.
2. **Widen the scope to `tests/`**, not `tests/unit/`. Thirty test files are currently
   unguarded, including every alleleless-recovery test.
3. **Run `pytest` in the VM preflight**, before any paid compute — the fast subset at minimum.
4. **Derive the expected feature set from one source of truth** (done, 2026-07-11: §5b).
5. **Do not merge past a red Continuous Integration run.** The gate worked. It was ignored.

Six of the 24 (clusters C and D) were introduced by this project's own 91→97 feature work and
went unnoticed for **two days of paid compute**. Twelve more (cluster A) had a verdict that
depended on the contents of the developer's disk. A guard nobody exercises is a comment.

**Required (not yet done):**
1. `pytest -q` as a **commit gate** (pre-commit hook or CI). The full suite is 365 s — too slow
   to boot-gate, fast enough to commit-gate.
2. A fast subset in `vm_bootstrap_run.sh` Phase E, **before any paid compute**.
3. Derive the expected feature set from **one** source of truth. It is currently restated in
   `TABULAR_FEATURES`, `KNOWN_ZERO_DEFAULT`, `build_reference_slice`, the audit's `expect`
   dict, and two split fixtures. Clusters C and D are both drift between these copies.

---

## 4b. RESULT OF THE FULL-SUITE RUN (2026-07-11 11:23, `outputs/fullsuite_2026-07-11.log`)

```
11 failed, 1802 passed, 8 skipped, 47 warnings in 392.69s (0:06:32)
```

**24 → 11.** Suite grew 1,616 → 1,821 collected (the conformal-prediction suite, `2903bee`,
plus two guards added here).

| cluster | before | after | status |
|---|---:|---:|---|
| A — protein-coord gate keyed off the filesystem | 12 | **0** | **FIXED** |
| B — ESM-2 `cache_path` / `__new__` fixture | 6 | **0** | **FIXED** |
| C — stage-5 zero-audit allowlist | 1 | **0** | **FIXED** |
| D — Run-17 audit split fixtures | 5 | **0** | **FIXED** |
| **A′ — regression introduced by the cluster-A fix** | — | **6** | **fixed 2026-07-11, re-run pending** |
| **E — subprocess cannot import the package (NEW)** | — | **5** | **OPEN, root cause not yet established** |

Positive confirmation for cluster B: `test_esm2_llr` tests now appear in the *warnings*
summary against `esm2.py:684` (the score-cache append). They could not have reached that
line before — they died at `_score_cache_path`. The fixture now exercises the score cache
end to end.

### 4b.1 Cluster A′ — a regression I introduced, and owned

Changing `_protein_coord_source_present` from two arguments to one broke six tests in
`tests/unit/test_protein_coord_coverage_gate.py`:

```
TypeError: _protein_coord_source_present() takes 1 positional argument but 2 were given
```

**Cause:** I searched for *production* callers (`alphamissense_tsv_path`, three hits) and did
not search for callers of the function itself. The gate has a dedicated unit-test file. That
is the precise failure mode this project's standing rules exist to prevent, and it was mine.

**Not a mechanical signature patch.** Reading the file showed that
`test_source_present_when_cache_exists` **asserted the defect**: it locked in "a coord cache
file on disk ⇒ a source is present", which is exactly what made twelve tests depend on the
contents of the developer's disk. Mechanically adapting it would have re-enshrined the bug in
a passing test.

The file is rewritten to the corrected contract, and the defect-asserting test is replaced by
its **inverse**, which is now the cluster-A regression guard:

- `test_stale_coord_cache_on_disk_is_NOT_a_source` — a cache exists; still not a source.
- `test_stub_mode_holds_even_when_the_am_tsv_exists_but_was_not_declared` — the 613 MB TSV
  exists on this box; declaring nothing must still mean stub mode.
- `test_declared_am_tsv_that_exists_is_a_source` — the production path still arms the gate.
- plus stub-mode and missing-declared-path cases.

A repository-wide search now confirms exactly two code callers, both single-argument:
`real_data_prep.py:983` and this test file.

### 4b.2 Cluster E (5, NEW) — subprocess cannot import the package

```
tests/unit/test_evaluation_metrics.py::test_package_imports_without_sklearn
tests/unit/test_evaluator_phase5.py::test_module_imports_without_sklearn
tests/unit/test_orchestrator_lazy_registry.py::test_orchestrator_constructs_without_sklearn
tests/unit/test_orchestrator_lazy_registry.py::test_orchestrator_constructs_without_sklearn_or_torch
tests/unit/test_orchestrator_lazy_registry.py::test_ci_data_freshness_pipeline_runs_without_sklearn

ModuleNotFoundError: No module named 'genomic_variant_classifier.evaluation'
ModuleNotFoundError: No module named 'genomic_variant_classifier.agent_layer'
```

These were **not** in the 2026-07-08 triage (whose 24 failures sat in six other files), so
they broke between 2026-07-08 and 2026-07-11. They are **not** caused by the fixes above —
none of those files touches evaluation, the agent layer, or the import machinery.

**Established:**
- The failure is always in a **child process**; the parent imports fine (1,802 tests pass).
- The failure is on the **subpackage**, not the top package — so the child resolves *a*
  `genomic_variant_classifier` that has **no subpackages beneath it**.
- The editable install is a **simple `.pth`** containing `C:\Projects\genomic-variant-classifier\src`,
  so `src` should land on `sys.path` at startup in parent *and* child.
- **Every** subpackage `__init__.py` exists and is tracked. The layout is intact.
- The top-level `__init__.py` is an inert 250-byte docstring that imports nothing — so
  `import genomic_variant_classifier` succeeds from *any* directory of that name, **including
  an implicit namespace package**.

**Leading hypothesis (NOT yet established):** something on the child's path supplies a bare
`genomic_variant_classifier` name that **shadows** the real package in `src`. An implicit
namespace package would produce exactly this signature — top-level resolves, subpackages do
not. A second candidate is truncation of the very long `PYTHONPATH` the tests synthesise from
`sys.path`.

**This is not asserted as fact.** `scripts/diagnose_subprocess_import.py` measures it: it
prints the child's `sys.path`, the resolved top-level spec `origin` (a `None` origin with a
`__path__` is the namespace-shadowing signature), and each subpackage's origin. Run it before
any fix is attempted.

### 4b.3 Warnings (47) — every distinct class, dispositioned

Nothing here is cosmetic; each is recorded rather than tolerated.

| warning | where | disposition |
|---|---|---|
| `FutureWarning: DataFrame concatenation with empty or all-NA entries is deprecated` | `esm2.py:684`, `_score_cache_append` | **FIXED 2026-07-11.** On a cold cache, `existing` is the empty dtype-less frame from `_score_cache_load`; concatenating it will **silently change score-column dtypes** in a future pandas. Empty frames are now dropped before `concat`, so `fresh` keeps dtype authority. Fixed at source, not suppressed. |
| `ConvergenceWarning: lbfgs failed to converge` (max_iter **1000** *and* **200**) | correctness-harness stages 1/3/5 | **OPEN — quality signal.** The harness's own sanity/smoke logistic regression does **not converge**. The harness passes, but its reference model is unconverged, which weakens stage-3 sanity as evidence. Needs scaling or a solver/iteration change — not a warnings filter. |
| `UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names` | `test_catboost.py`, `test_ensemble_persistence.py` | **OPEN — potential silent misalignment.** The model is fitted on a named DataFrame and predicted on a bare array, so column order is trusted implicitly. If the caller's column order ever drifts from training order, predictions are silently wrong with no error. Should pass a DataFrame at predict, or assert the feature order. |
| `UserWarning: n_components > n_samples ... set to n_samples` | `test_catboost.py` (kernel approximation, `svm_bagged_rbf`) | **OPEN — inefficiency, test-scale only.** Confirm it cannot occur at production cohort size. |

---

## 4c. SECOND FULL-SUITE RUN (2026-07-11 11:39, `outputs/fullsuite_2026-07-11b.log`)

```
5 failed, 1809 passed, 8 skipped, 41 warnings in 413.82s (0:06:53)
```

**Trajectory: 24 → 11 → 5.**

- Cluster A′ (my six regressions in `test_protein_coord_coverage_gate.py`): **CLEARED.**
- The `esm2.py:684` `FutureWarning` is **absent from the warnings summary** — warnings fell
  47 → 41. The pandas concatenation deprecation is fixed at source, confirmed empirically.
- Remaining: **cluster E only, 5 tests.**

### 4c.1 Cluster E — investigation status (root cause NOT yet established)

A repository-wide search for every directory named `genomic_variant_classifier` returns three:

```
./src/genomic_variant_classifier          <- the real package
./.mypy_cache/3.12/genomic_variant_classifier   <- mypy cache, not importable
./notebooks/genomic_variant_classifier    <- ???
```

**`notebooks/genomic_variant_classifier/` is not a package at all.** It contains
`data/{raw,processed,features,images}`, `logs/`, `models/`, `reports/` — it is the project's
standard *output* tree, accidentally created by a notebook that ran with its working directory
set to `notebooks/`. It has **no `__init__.py`**, and it is **untracked by git**.

A directory with the package's exact name and no `__init__.py` is, under Python 3, an
**implicit namespace package**: if it is ever reached first on `sys.path`, `import
genomic_variant_classifier` *succeeds* and every subpackage import then *fails* — which is
precisely the observed signature (top-level resolves, `.evaluation` / `.agent_layer` do not).

**However — this is a CANDIDATE, not a conclusion.** For it to shadow, `notebooks/` itself
must reach `sys.path`, and I have not shown that it does. `tests/conftest.py` inserts only
`<repo>/scripts` and `<repo>` at position 0; neither contains a `genomic_variant_classifier`
directory. So the shadow mechanism is **not yet proven**, and the alternative — that
`<repo>/src` is somehow absent from the `sys.path` the tests copy into `PYTHONPATH` — remains
open. All three failing files use the identical idiom
`os.pathsep.join(p for p in sys.path if p)`, so a single broken assumption explains all five.

**Do not fix on this hypothesis.** `scripts/diagnose_subprocess_import.py` prints the child's
`sys.path`, the resolved top-level spec `origin` (a `None` origin plus a `__path__` is the
namespace-shadow signature), and each subpackage's origin. One run settles it.

Regardless of the outcome, `notebooks/genomic_variant_classifier/` is untracked output debris
sitting in the source tree under a name that collides with the package, and should be removed
or relocated on hygiene grounds alone.

### 4c.1b CLUSTER E — ROOT CAUSE ESTABLISHED AND FIXED (2026-07-11)

**The hypotheses I floated were all wrong, and the isolation experiment killed them.**

```
python -m pytest tests/unit/test_evaluation_metrics.py tests/unit/test_evaluator_phase5.py \
                 tests/unit/test_orchestrator_lazy_registry.py -q
-> 38 passed in 30.02s
```

Green alone, red in the full suite ⇒ **order-dependent global-state pollution**. Not a
namespace shadow (`notebooks/genomic_variant_classifier` is innocent), not a broken editable
install, not `PYTHONPATH` truncation.

**The polluter: `tests/test_rekey_seq_windows_v2.py::_install_real_join`.**

It writes a **counterfeit package** into a temp directory —

```
tmp_path/gvc_join/genomic_variant_classifier/__init__.py        <- a REAL __init__.py
tmp_path/gvc_join/genomic_variant_classifier/data/__init__.py   <- only `data` beneath it
tmp_path/gvc_join/genomic_variant_classifier/data/seq_window_join.py
```

— and published it with a bare, unscoped, never-reverted

```python
sys.path.insert(0, str(tmp_path / "gvc_join"))
```

**Why nothing broke in-process.** The real `genomic_variant_classifier` was already in
`sys.modules`, so it was never re-resolved. The counterfeit sat at `sys.path[0]`, inert.

**Why the child died.** The five failing tests launch a subprocess with

```python
PYTHONPATH = os.pathsep.join(p for p in sys.path if p)
```

The child starts with an **empty `sys.modules`** and therefore resolves
`genomic_variant_classifier` from the counterfeit, which is *first* on the path. It has an
`__init__.py`, so the **top-level import succeeds** — and then `evaluation` and `agent_layer`
do not exist beneath it:

```
ModuleNotFoundError: No module named 'genomic_variant_classifier.evaluation'
ModuleNotFoundError: No module named 'genomic_variant_classifier.agent_layer'
```

This accounts for **every** observed fact: the exact error text, the top-resolves /
subpackage-fails signature, the order dependence, and the clean result from a standalone
diagnostic (which never ran the polluter).

**Fix, in two layers.**

1. *The polluter.* `_install_real_join` now publishes the counterfeit via
   `monkeypatch.syspath_prepend(...)`, which pytest **reverts at teardown**. Both call sites
   pass `monkeypatch` through. The counterfeit can no longer outlive its own test.

2. *The class of bug.* A new **autouse `sys.path` leak guard** in `tests/conftest.py` snapshots
   `sys.path` around every test, **restores it**, and **fails the offending test by name**,
   listing exactly what it added or removed. Design points that matter:
   - it compares **sets**, not lists — several tests legitimately re-insert the already-present
     `scripts` or `src` directory, and a strict list comparison would fail them for a harmless
     duplicate. Only a genuinely new or vanished entry can change what an import resolves to.
   - it **restores before it raises** — a leaking test fails alone and cannot cascade.

Cluster E was invisible for three days precisely because it was silent in isolation. The guard
converts that silence into an immediate, named failure.

### 4c.2 Related find — a test module shipped *inside* the package

`pyproject.toml`'s pytest configuration carries this comment:

> *"Restrict pytest auto-discovery to `tests/`. Without this, pytest walks the entire rootdir
> and imports `src/genomic_variant_classifier/agent_layer/test_message_bus.py` during
> collection. That file's module-level code stubs `sys.modules["torch"] = MagicMock()` … which
> pollutes torch for the remainder of the collection."*

So a **test file lives inside the shipped package** and, at module level, replaces `torch`
with a `MagicMock`. The current mitigation is to narrow pytest's discovery so it is never
imported — the hazard is *avoided*, not *removed*. Any tool that imports the package tree
(mypy, a packaging step, a different test runner, an IDE) can still trip it. It belongs in
`tests/`, not in `src/`.

---

## 5b. NEW FINDING (2026-07-11) — the feature matrix has **two** implementations

Discovered while verifying the fixes above. This outranks the 24 test failures, because it
means **the correctness gate is pointed at a code path the training pipeline does not use.**

### 5b.1 The fact

There are two independent, hand-maintained implementations of feature engineering:

| implementation | location | who calls it |
|---|---|---|
| `engineer_features` | `models/variant_ensemble.py` (~line 340) | the **correctness harness** (`correctness_harness.py:59`), `api/pipeline.py`, the unit tests |
| `DataPrepPipeline._engineer_features` | `data/real_data_prep.py:1260` (~440 lines) | the **data-prep / training pipeline** |

`DataPrepPipeline._engineer_features` does **not** delegate — it rebuilds `feats` from scratch,
column by column, duplicating the same feature names (`af_log10`, `is_snv`,
`gene_is_constrained`, `genomiclm_delta_norm`, `kegg_pathway_count`, …).

This is not an inference. The codebase says so, repeatedly and in its own words:

- `variant_ensemble.py:121` — *"Feature definitions (65 features -- **must match**
  `DataPrepPipeline._engineer_features`)"*
- `variant_ensemble.py:340` — *"**Mirrors** `DataPrepPipeline._engineer_features()` in
  `src/genomic_variant_classifier/data/real_data_prep.py`."*
- `scripts/install_docs_close_cnn_rna.py:34` — *"maxentscan_delta registered in
  TABULAR_FEATURES, **BOTH `_engineer_features` blocks**"*
- `features/topological_ph.py:85`, `api/schemas.py:13`, `api/main.py:48` — all pin their
  contract to `_engineer_features`.

The two copies are kept in sync **by hand**. That is a permanent drift generator.

### 5b.2 Why it is worse than a style problem

**The five-stage correctness harness imports `engineer_features` from `variant_ensemble`**
(`correctness_harness.py:59`). Therefore stage 5's zero-audit — and the entire G1 pre-flight
gate that depends on it — validates **only** the `variant_ensemble` copy.
`DataPrepPipeline._engineer_features`, the code that actually builds the training matrix, is
**never exercised by the harness at all**.

A silent-zero, a wrong clip, a truncating cast, or a missing connector introduced in the
pipeline's own engineering block is **structurally invisible** to the gate built to catch
exactly that class of defect. The gate can be green while the training matrix is wrong. This
is the same species as the Run-15 silent zero the coverage gate was written to prevent.

### 5b.3 Drift already demonstrable

- The header comment claims **65 features**. `TABULAR_FEATURES` actually holds **97**
  (verified by import, 2026-07-11). The comment is stale by **32 features** — so the
  "must match" contract has demonstrably not been maintained as documentation.
- Raw `feats["…"]` occurrence counts differ: **71** in `real_data_prep.py` vs **70** in
  `variant_ensemble.py`. *This is a smell, not a proof* — the pattern matches reads as well as
  writes. It is **not** yet established that the two produce different matrices. §5b.4 settles it.
- A third, older generation is still on disk: `variant_ensemble_cff925c.py` at the repository
  **root** (35 KB, untracked, dated 2026-04-01), whose header reads *"55 features — must match
  DataPrepPipeline._engineer_features"*. Three generations of the same contract — 55, 65, 97 —
  coexist in the tree.

### 5b.4 The decisive test (must run in `.venv312`; not yet run)

Run both implementations on the identical reference slice and diff the outputs. Until this is
run, **divergence is unproven and must not be asserted either way.**

```powershell
python - <<'PY'
import numpy as np, pandas as pd
from genomic_variant_classifier.agent_layer.harness.correctness_harness import build_reference_slice
from genomic_variant_classifier.models.variant_ensemble import engineer_features, TABULAR_FEATURES
from genomic_variant_classifier.data.real_data_prep import DataPrepPipeline, DataPrepConfig

df = build_reference_slice()
A = engineer_features(df)                                        # what the HARNESS validates
B = DataPrepPipeline(DataPrepConfig())._engineer_features(df)    # what the PIPELINE trains on

a, b = set(A.columns), set(B.columns)
print("TABULAR_FEATURES     :", len(TABULAR_FEATURES))
print("variant_ensemble cols:", len(a))
print("real_data_prep   cols:", len(b))
print("ONLY in variant_ensemble :", sorted(a - b))
print("ONLY in real_data_prep   :", sorted(b - a))
shared = sorted(a & b)
diff = [c for c in shared
        if not np.allclose(pd.to_numeric(A[c], errors="coerce").fillna(0),
                           pd.to_numeric(B[c], errors="coerce").fillna(0), equal_nan=True)]
print("SHARED but NUMERICALLY DIFFERENT:", diff)
print("VERDICT:", "IDENTICAL" if not (a ^ b) and not diff else "DIVERGED  <-- silent-drift confirmed")
PY
```

### 5b.4b RESOLVED — proved equivalent, then collapsed (2026-07-11)

**Step 1 — the first audit was true but weak.** `scripts/diff_engineer_features.py` compared
the two on `build_reference_slice` and reported IDENTICAL (97/97, no set difference, no
numeric difference, same order). Two blind spots made that insufficient to delete 435 lines:

- The reference slice supplies **every** input column by contract, so **not one `df.get`
  default is ever exercised**. Default drift was structurally invisible to it.
- The script coerced through `pd.to_numeric(...).astype(float)`, so **dtype differences were
  invisible** — precisely the class of `INCIDENT_2026-05-30_clingen-int-truncation`, and the
  fixture feeds `clingen_validity_score` as an *integer*, so an int-vs-float cast difference
  would have produced identical values and passed.

**Step 2 — the real proof.** `scripts/prove_engineer_features_equivalence.py`:
**117 comparisons, zero divergences**, exact on column set, column **order**, **dtype**, and
values (NaN positions preserved, no float coercion):

| block | cases | what it forces |
|---|---:|---|
| C1 | 9 | reference slice × seeds × sizes |
| C2 | 1 | **minimal frame — every connector column absent → forces EVERY `df.get` default** |
| C3 | 43 | single-column dropout, one default at a time |
| C4 | 16 | integral inputs made fractional — **incl. `clingen_validity_score`**, the exact column of the 2026-05-30 truncation incident |
| C5 | 41 | 20% NaN injection per column |
| C6 | 6 | extremes (0, negative, ±inf, 1e300, 1e-300) |
| C7 | 1 | empty frame |

The script hard-fails (exit 2) if any block executes zero cases — an earlier version could
have claimed "EQUIVALENT ... including the int-truncation trap" while testing nothing there,
and that was unfalsifiable from its output. Log: `outputs/engineer_equiv_2026-07-11b.log`.

**Step 3 — the collapse.** `scripts/collapse_engineer_features.py` (guarded, reversible,
idempotent, AST-located, auto-restores on any parse failure) replaced the 435-line
`DataPrepPipeline._engineer_features` with a delegation. **Net −376 lines.** The pipeline now
trains on exactly the matrix the harness validates.

Two things deliberately preserved, both of which the equivalence proof would NOT have caught:

- **The import stays local.** `real_data_prep` must not pull the heavy ML stack in at module
  import; the suite asserts the package imports with `sklearn`/`torch` blocked
  (`test_orchestrator_lazy_registry`, `test_evaluation_metrics`).
- **`.reset_index(drop=True)` is kept.** The proof compared values via `.to_numpy()`, which is
  **index-agnostic** — it never checked the index. Dropping the reset could have silently
  misaligned downstream joins and splits with the proof saying nothing.

**The guard was upgraded, not merely relocated.** The old fail-loud check compared only the
feature **count** against `EXPECTED_TABULAR_FEATURE_COUNT`, and its own comment conceded that
is how "the 88-vs-91 R13 drift went unnoticed for a full 13-hour run". A count cannot catch
different names at the same count, different values, or a different column **order** — and
order is the quiet one, because an estimator fitted on a named DataFrame and then fed a bare
`ndarray` trusts position implicitly (cf. the standing LightGBM feature-name warning). The new
guard asserts **name and order** against `TABULAR_FEATURES`.

**Fallout, and what it exposed.** The full suite came back `1 failed, 1813 passed`. The single
failure was `test_core.py::test_sift_score_fill_is_not_threshold`, which asserted on the
**source text** of `real_data_prep.py` (`'"sift_score":' in src`). It broke because the code
*moved*, not because behaviour changed. It was testing *where* the code lived rather than
*what it did* — and it would have passed had someone changed the default in the other module.

The invariant it guarded is real and important: SIFT semantics are *score < 0.05 == deleterious*,
so an absent SIFT score filled at or below the threshold would make **every unannotated variant
silently deleterious** — a silent-pathogenic default, the same species as the Run-15 silent zero.
It is now a **behavioural** test asserting the invariant through *both* entry points: the fill
equals the neutral `DEFAULT_SIFT` (0.5), sits above the deleterious threshold, and yields
`sift_deleterious == 0`. A repository-wide sweep confirms it was the **only** source-grep test
in the suite.

The obsolete hand-sync comments (`"must match"`, `"Mirrors ..."`, and the stale `"65 features"`
header) are deleted rather than corrected: a count written into a comment is a fact that rots.

### 5b.5 The correct fix (ground-up, not patchwork)

**One implementation, one source of truth.** `DataPrepPipeline._engineer_features` should be
deleted and replaced by a call to `variant_ensemble.engineer_features` — the copy the harness,
the API, and the unit tests already validate. The pipeline then inherits every guard the
harness applies, and the "must match" comments become unnecessary because nothing can diverge.

Sequencing matters: run §5b.4 **first**. If the two have already diverged, every diverging
column is a live defect in either the training matrix or the harness's model of it, and each
must be adjudicated **before** the merge — otherwise the merge silently changes the training
matrix underneath the trained models. Do not merge blind.

---

## 6. Still open (carried forward, do not lose again)

| # | item | source | status |
|---|---|---|---|
| 6.1 | **No test gate** (§5) | TRIAGE 2026-07-08 §6.5 | OPEN — the headline item |
| 6.2 | Inverted retention guidance: `real_data_prep.py:388`, `:1684` advise *"Lower `min_review_tier`"* when a split lacks a class. The filter is `<=`, so lowering keeps **fewer** rows; recovery requires **raising** it. The message sends the operator the wrong way. | TRIAGE §5.1 | OPEN |
| 6.3 | Latent `TypeError`: `real_data_prep.py:474-481` maps `ReviewStatus` with `k in s`; `metadata.review_status` contains real nulls. Blocks the deletion-incident source decision. | TRIAGE §5.2 | OPEN |
| 6.4 | Duplicate test basename: `tests/test_clean_cohort.py` vs `tests/unit/test_clean_cohort.py`. Fragile under pytest's default import mode. | TRIAGE §5.3 | OPEN |
| 6.5 | **85 untracked files** in the working tree (≈60 `scripts/probe_*.py` / `diagnose_*.py` / `verify_*.py`, 5 `tests/test_*.py` at the wrong level, `docs/status/ALLELELESS_PROVENANCE_2026-07-09.md`). Each needs disposition: commit, `.gitignore`, or delete. | this session | OPEN |
| 6.6 | **`docs/ROADMAP.md` is stale** — last modified **2026-07-01**, ten days ago. It predates cohort-v3, the alleleless-variant recovery, the history rewrite, and every finding in this document. The project brief requires it to be current. | this session | OPEN |
| 6.7 | Open **pull request #1** on the remote (`refs/pull/1/head` → `run9a-prep`). | ls-remote, 2026-07-11 | OPEN — needs a decision |
| 6.8 | Repo authority: `monzia-moodie` and `monzia-moodie-repo-projects` resolve to the **same** repository (identical refs). Not fixable with git — a GitHub-side transfer/rename. | 2026-07-11 | OPEN — GitHub UI action |
| 6.9 | `_score_cache_load` catches bare `Exception` and warns "corrupt cache → recompute". Not silent, but broad enough to swallow a genuine schema bug. Narrow it. | this session | OPEN (minor) |

**Settled — do not reopen:** `run9a-prep` holds 60+ unique commits, carries no AlphaFold blob,
and is referenced by tag `run9a-baseline` and PR #1. **Keep it.** It is a curation choice, not
a hygiene emergency.

---

## 7. Files changed, 2026-07-11

| file | change |
|---|---|
| `src/.../data/real_data_prep.py` | Cluster A: `_protein_coord_source_present` keys off declared config only; hard-coded TSV fallback deleted. |
| `src/.../agent_layer/harness/correctness_harness.py` | Cluster C: six live connectors FED; `gene_is_constrained` removed (25→24); count comment corrected; THE RULE + count history documented. |
| `tests/unit/test_esm2_llr.py` | Cluster B: `__new__` bypass replaced with real construction under `tmp_path`; hermeticity guard added. |
| `tests/unit/test_harness_fixture_omim_molecular.py` | Lockstep guard 25→24; new executable RULE guard. |
| `tests/unit/test_run17_audit_persplit.py` | Cluster D: `_cols()` emits the six new columns. |
| `tests/unit/test_run17_fullflag_smoke.py` | Cluster D: `_write_splits()` emits the six new columns. |
