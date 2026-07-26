# SESSION 2026-07-26 -- Option C commit 2: bootstrap inference reconciliation

**Branch point:** `origin/main` at `2e04bd9` ("feat(evaluation): CanonicalVariantTable seam
for the metric stack"), committed 2026-07-26T02:53:49-04:00. Tree clean, zero untracked
files, suite-size ratchet 2991, README badge `tests-2991-success`. Status re-verified by
`git fetch origin main` at the start of every working segment of this session; `origin/main`
never moved.

**Deliverable:** one atomic commit reconciling the three bootstrap implementations this
project carried into a single canonical engine, with the resampling unit an explicit typed
part of every confidence interval.

---

## 1. WHY THIS COMMIT EXISTS

Three bootstrap implementations existed before today:

| implementation | design | correct? |
|---|---|---|
| `evaluation/metrics.py` `bootstrap_ci` / `cluster_bootstrap_ci` | row-level (stratified) and gene-cluster | the gene-cluster one is the correct estimator |
| `evaluation/evaluator.py:284` `ClinicalEvaluator._bootstrap_ci` | row-level, unstratified | anti-conservative |
| `reports/report_generator.py:85` `bootstrap_metric` | row-level, unstratified | anti-conservative |

Only the kernel's `cluster_bootstrap_ci` respected the fact that variants cluster within
genes. The gene-cluster design effect was measured at **2.935 times** on the real cohort
(suite-size ratchet entry 2055) and at **2.807 times** on a synthetic fixture. Every
confidence interval this project published through the other two paths was therefore
narrower than the data support by roughly that factor.

The agreed design, accepted before implementation: one canonical kernel implementation,
gene-cluster resampling REQUIRED for certified intervals, variant resampling available
ONLY as an explicit exploratory mode, and all evaluator and report paths delegated to that
implementation. The governing principle is that the resampling UNIT is an explicit, typed
part of every interval and never an accidental consequence of which caller produced it or
whether metadata happened to be present. **No silent fallback.**

Excluded by decision, and unchanged: calibration (already reconciled by the 2026-07-20
census and pinned by `test_calibration_implementations_agree.py`), the metric registry and
orchestrator, production-script rewiring beyond the evaluator and report bootstrap paths,
cohort version 2, and the P6 provenance correction.

---

## 2. DEFECTS FOUND AND RESOLVED

### 2.1 The reference dispatcher failed an existing repository guard

The dispatcher built in the prior session was delivered as
`metrics_with_bootstrap_dispatcher_REFERENCE.py`, SHA-256
`624C36CFE9FC54746F8A7C11AA17218DCDFE912E17196CA194F54282621F9F2E`. That hash was
independently recorded in the prior session's transcript and matched on receipt, so the
artifact was genuine. Diffed against `metrics.py` at `2e04bd9` it was strictly additive:
three hunks, 183 lines added, **zero** removed.

It nevertheless **fails `test_capability_contract.py::test_no_module_uses_strenum_which_
would_break_the_declared_python_floor`**. That test walks every `.py` file under the
package and flags any unbackticked occurrence of `StrEnum`, because `StrEnum` arrived in
Python 3.11 while `pyproject.toml` declares `requires-python = ">=3.10"`. The dispatcher's
own docstring read:

```
str-Enum (not StrEnum) to keep the 3.10 floor declared in pyproject.toml.
```

Committing the reference as delivered would have failed continuous integration on both the
3.11 and 3.12 legs. The prior session's regression list -- `test_evaluator_phase5`,
`test_evaluation_metrics`, `test_calibration_implementations_agree`,
`test_metric_kernel_is_fail_closed` -- did not include the one test guarding the enum floor
the dispatcher touches.

**Resolved** by moving `BootstrapUnit` to `capabilities.py`, where the surrounding
docstrings already use the backticked form. See section 3.2 for why that move was required
independently.

### 2.2 The replicate accounting modelled a sampling scheme the kernel does not use

`_count_valid_replicates` in the reference dispatcher drew, for the variant path, two
strata from `{positive, negative}` **with replacement**. That yields a single-class
resample half the time:

```
P(both draws land on the same stratum) = 2 x (1/2)^2 = 0.500
measured on a 1,200-row fixture         =              0.506
VARIANT accounting: n_valid = 494, n_degenerate = 506
```

`bootstrap_ci` never draws that way. At `metrics.py:467-469` it **always takes both strata**
and resamples within each. The accounting was describing a sampling scheme that does not
exist in this codebase, and it reported roughly half of all sound replicates as degenerate.

**Why the prior proof could not see it.** The old code hard-coded
`status=MetricStatus.INSUFFICIENT_SUPPORT` for every variant result. The prior session's
check -- "variant yields `INSUFFICIENT_SUPPORT`" -- therefore passed *for the wrong reason*,
and the broken accounting was invisible behind a constant. It surfaced only once status
became a computed quantity. A check that cannot fail is worse than no check.

**Resolved.** `_count_valid_replicates` now takes the design's own index generator:
`_cluster_draw` mirrors `cluster_bootstrap_ci`, `_stratified_row_draw` mirrors
`bootstrap_ci(stratified=True)`.

```
before: n_valid = 494,  n_degenerate = 506   (0.506 degenerate)
after:  n_valid = 1000, n_degenerate = 0     (0.000 degenerate)
```

### 2.3 Status and certification were conflated

The reference marked every variant-level result `INSUFFICIENT_SUPPORT` even when a
perfectly good interval had been produced, making a produced interval indistinguishable
from a withheld one. The two questions are now separate axes:

```
status                  was an interval successfully produced?
certification_eligible  is that interval admissible for a certified claim?
```

| situation | `status` | `certification_eligible` | `stratified` |
|---|---|---|---|
| gene clusters, replicate floor cleared | `OK` | `True` | `False` |
| variant, replicate floor cleared | `OK` | `False` | `True` |
| too few valid replicates | `INSUFFICIENT_DATA` | `False` | as run |
| one class present | `UNDEFINED` | `False` | as run |
| gene design requested with no clusters | raises `InsufficientSupportError` | -- | -- |

`INSUFFICIENT_DATA` rather than `INSUFFICIENT_SUPPORT` for the thin case, matching
`capabilities.py:99-103` exactly: the machinery is ready and the cohort admissible, there
is simply too little to estimate from.

### 2.4 The evaluator held one mutable generator shared by both intervals

`ClinicalEvaluator.__init__` created `self.rng = np.random.default_rng(random_state)` and
both bootstrap calls drew from it in sequence. Three measured consequences:

* the precision-recall interval depended on the receiver-operating-characteristic interval
  having been computed first, because it inherited the advanced stream;
* **calling `evaluate()` twice on one evaluator returned different intervals for identical
  inputs**;
* adding, removing or reordering any bootstrap silently changed every interval after it.

**Resolved** by `derive_seed(base_seed, namespace)`, which returns a reproducible seed per
named quantity. It uses `hashlib.blake2b`, not the builtin `hash()`: `PYTHONHASHSEED`
randomises string hashing per process, so the builtin would have made a "reproducible" seed
vary between interpreter invocations -- the exact defect being removed. A subprocess test
runs the derivation under three different `PYTHONHASHSEED` values and asserts one result.

### 2.5 `_bootstrap_ci` crashed on an all-degenerate input

If every replicate was degenerate, `scores` was empty and
`np.percentile(np.array([]), 2.5)` raised `IndexError: index -1 is out of bounds for axis 0
with size 0`. Measured, not inferred. The reconciled path returns typed
`UNDEFINED` instead, and the report-layer shim returns `(nan, nan)`.

### 2.6 The evaluator's row bootstrap was unstratified; the kernel's is stratified

`evaluator.py:294` drew `rng.integers(0, n, n)`; `metrics.py:447` declares
`stratified: bool = True`. They were different estimators. Delegation therefore **changes
emitted numbers**, which is why audit section 11 made this its own commit. The distinction
is now persisted per interval as `stratified`.

### 2.7 Both artifact writers silently coerced numbers into strings

```
json.dumps({"auroc": np.float64(0.9123), "n": np.int64(7)}, default=str)
    ->  {"auroc": 0.9123, "n": "7"}
```

The float survived as a number because NumPy floats subclass `float`; the integer came back
as the **string** `"7"`, because `np.int64` does not subclass `int`. Present at
`evaluator.py:589` and `prediction_artifacts.py:219`. Neither writer set `allow_nan=False`,
so both could also emit bare `NaN` literals, which are not valid JavaScript Object Notation
number literals however leniently Python's own parser reads them back.

**Resolved** by `evaluation/serialization.py`: explicit NumPy normalisation, refusal of
unrecognised types rather than stringification, a finite-value audit that names every
offending field path, and `allow_nan=False` as a backstop.

### 2.8 `gene_id` and `gene_symbol` are different namespaces

The canonical seam exposes `gene_id`; the evaluator's gene-error analysis reads
`gene_symbol`. Establishing which to use for clusters required knowing whether they name
the same things. Three pieces of in-repository evidence say they do not:

* `scripts/build_gtex_de_features.py:100-101` writes `gene_id` from the Genotype-Tissue
  Expression matrix `Name` column (Ensembl gene identifier) and `gene_symbol` from its
  `Description` column (HUGO Gene Nomenclature Committee symbol), in the same constructor.
* `scripts/build_rnaseq_canonical_real.py:82` explicitly excludes rows whose `gene_symbol`
  begins with `ENSG` -- a filter that exists only because the project knows Ensembl
  identifiers must not leak into the symbol column.
* `src/genomic_variant_classifier/data/database_connectors.py:328` passes `gene_id` as the
  Genome Aggregation Database GraphQL variable, which is an Ensembl identifier.

Census: `gene_symbol` appears **413 times across roughly 110 files** and is the project's
universal gene key, used by gene-stratified splitting and the graph neural network;
`gene_id` appears **19 times**, confined to the seam and the Ensembl-sourced paths.

Raw string equality between the two would therefore be false on every row.

### 2.9 Documentation found stale

* `docs/architecture/EVALUATION_WIRING_AUDIT_2026-07-25.md:271` describes `gene_id` as
  serving "per-gene analysis + cluster bootstrap". Per-gene analysis is
  `_gene_error_analysis`, which reads `gene_symbol` and returns `[]` for a `gene_id`-only
  frame.
* `evaluation/canonical.py`'s `as_meta()` docstring calls its output "The aligned metadata
  frame `ClinicalEvaluator.evaluate` expects as `meta`", but a seam-produced frame carries
  no `gene_symbol`, so `gene_errors` comes back **silently empty**. Pre-existing at
  `2e04bd9`. Recorded, not fixed here -- it is evaluator breakdown logic, not bootstrap, and
  the work-item preemption rule schedules it at the next clean boundary.

---

## 3. WHAT WAS BUILT

### 3.1 `evaluation/cluster_resolution.py` (new, 326 lines, 67 tests)

Resolves per-row gene-cluster labels from an evaluator metadata frame.

```
cluster_id present            -> used directly; source "cluster_id"
exactly one legacy column     -> used; source names it; partition_verified False
both legacy columns           -> induced PARTITIONS compared, never raw strings
                                 equivalent -> source "gene_id+gene_symbol",
                                               partition_verified True
                                 divergent  -> MetricStatus.FAILED,
                                               "gene_cluster_partitions_disagree"
neither                       -> MetricStatus.INSUFFICIENT_SUPPORT,
                                 "gene_cluster_identifier_required"
```

**Partition equivalence rather than string equality.** The bootstrap consumes only the
grouping -- `cluster_bootstrap_ci` builds `{label: row_positions}` and never interprets a
label. Two labelings are interchangeable if and only if they induce the same partition of
rows, which is namespace-free. So `ENSG00000012048 ENSG00000012048 ENSG00000141510` and
`BRCA1 BRCA1 TP53` are accepted as equivalent though they share no characters, while
`ENSG1 ENSG1` against `BRCA1 TP53` is refused.

The shipped implementation factorises both vectors, encodes each pair as one integer and
counts distinct pairs: a bijection exists exactly when the pair count equals the distinct
label count on both sides. That is O(n) plus one sort, against O(n log n) for the
`groupby(...).nunique()` formulation, which matters at 1.5 million rows. The `groupby`
version is retained verbatim in the test file as a reference, and **40 randomised cases
with injected asymmetric missingness assert the two agree**, so the optimisation cannot
drift from the definition.

Three decisions beyond the specification, each with a falsifiable test: the empty string
counts as missing (`pandas.isna("")` is `False`, so an unlabelled row would otherwise
become a legitimate cluster and pool every unlabelled variant into one enormous
pseudo-gene); partial missingness is refused rather than pooled or made singletons (pooling
resamples unlabelled rows as one gene, singletons resample them individually, and both are
silent changes to the inferential design); and two blanket invariants assert that every
refusal carries no values and a machine-readable finding while every success carries values,
no finding and zero missing.

### 3.2 `BootstrapUnit` moved to `capabilities.py`

Not cosmetic. `evaluator.py` must record the resampling unit as a typed field and is
contractually barred from importing scikit-learn -- locked by
`test_evaluator_phase5.py::test_module_imports_without_sklearn`, which runs in a subprocess
with the package blocked. `metrics.py` imports scikit-learn at module level. The enum was
therefore unreachable from where it was needed.

It now sits beside `MetricStatus`, which is what that module's docstring argues for:
*"Status vocabulary is more foundational than any panel, so it lives at the bottom of the
layering and panels import upward."* `metrics.py` re-exports it, so
`from ...metrics import BootstrapUnit` keeps working, and
`metrics.BootstrapUnit is capabilities.BootstrapUnit` is `True` -- the same single-definition
property the project already pins for `MetricStatus`. `from enum import Enum` became dead in
`metrics.py` once its only user moved and was removed.

`MetricStatus` was reused rather than a parallel `CIStatus` introduced, after verifying all
six preconditions against a census of **183 references across 10 files**: values are stable
and pinned by `test_the_original_status_values_are_frozen`; `OK` is the success state;
`INSUFFICIENT_SUPPORT` is cleanly distinct from `DEPENDENCY_UNAVAILABLE`; three containers
already carry a `MetricStatus` under three different invariants, so sub-result use is
established; and `release_gate_satisfied` raises `TypeError` on non-`CapabilityEvidence`
input, so a report field can never be silently read as a release gate. The `FAILED` comment
was widened to cover prerequisite-validation failure; its value is frozen and its pinning
test still passes.

### 3.3 Kernel dispatcher extension

* `DEFAULT_MIN_VALID_FRACTION = 0.5` and `_effective_min_valid`, giving
  `n_valid >= max(100, ceil(0.5 * n_boot))`. An absolute floor alone is satisfied by 100
  valid replicates out of 100,000 -- a run in which 99.9 per cent of resamples were
  degenerate and the survivors are a biased subsample.

```
n_boot =     20  ->  floor    100   (unreachable; correctly withheld)
n_boot =   1000  ->  floor    500
n_boot = 100000  ->  floor  50000
```

* `stratified` and `min_valid_effective` recorded on every result.
* `status` always populated; annotation tightened from `MetricStatus | None` to
  `MetricStatus`.
* `math` was not imported in `metrics.py`; caught before running rather than after.

### 3.4 `evaluator.py` rewiring

Schema version 2. Endpoints `Optional[float]`; twenty new provenance fields, ten per
metric, deliberately not shared because the two metrics can fail differently on the same
cohort. `__post_init__` invariants make an impossible artifact unconstructable -- an
available interval with no endpoints, a null interval that still claims certification, a
variant-level interval marked certifiable. Placement in the constructor rather than at read
time follows the argument `capabilities.py` makes for `CapabilityEvidence`: a check at
decision time can be skipped by a caller who forgets to consult it.

`format_ci` replaces all three formatting sites, so an unavailable interval renders as
`unavailable (gene_cluster_identifier_required)` rather than `[nan, nan]`.

`n_requested` contract, stated explicitly: **0** when no bootstrap was attempted, the
configured count when one was attempted whatever its outcome. A reader can distinguish "we
never asked" from "we asked and got too few back" without parsing a finding string.

`_nan_safe` wraps the metric because `roc_auc_score` **raises** `ValueError` on a
single-class resample while the kernel's loops test `np.isfinite` and never catch.
Gene-cluster resampling makes all-one-class draws entirely reachable -- thirty genes can
easily draw thirty of one class -- so without this the certified path would have crashed on
exactly the cohorts it exists to serve. It wraps the same function the point estimate uses,
so the interval bounds precisely the quantity the report states.

### 3.5 `evaluation/serialization.py` (new) and the two writers

`to_json_compatible`, `validate_json_finite`, `dump_strict_json`. Both
`ClinicalEvaluator.save_report` and `RunArtifactWriter.save_eval_report` now produce
byte-identical encodings of the same report, which they previously did not guarantee.

### 3.6 `report_generator.bootstrap_metric` is a delegate

No longer implements resampling; forwards to the kernel with an explicit variant unit. The
`(lo, hi)` signature and tuple return are preserved for `test_core.py:988`. Proven to
return endpoints byte-identical to a direct kernel call.

### 3.7 `scripts/read_run_artifacts.py` reads both schema versions

A version-1 interval is normalised to `legacy_unknown` provenance and
`certified = False`. **Never retroactively certified**: those endpoints came from a
row-level bootstrap and are too narrow by roughly the 2.935 design effect.

### 3.8 Deferred, with evidence

`ValidationMetrics.auroc_ci` and `auprc_ci` default to `(0.0, 0.0)` -- a zero-width interval
at zero is fabricated evidence. It is **constructed nowhere in the repository**, so
correcting it would broaden this commit for no behavioural gain.
`test_validation_metrics_is_dead_and_therefore_deferred` scans the package and `scripts/`
for any construction site and fails if one appears, which is the evidence for the deferral
rather than an assertion of it.

---

## 4. VERIFICATION

```
tests/unit/test_bootstrap_reconciliation.py    60 passed   (3.11 and 3.12)
tests/unit/test_cluster_resolution.py          67 passed   (3.11 and 3.12)
dispatcher re-proof probe                      43/43 checks passed (3.11 and 3.12)

regression, unchanged by design:
  test_capability_contract.py                  56 passed   <- was FAILING against the reference
  test_calibration_implementations_agree.py     5 passed
  test_evaluator_phase5.py                      4 passed   <- the no-scikit-learn contract
  test_evaluation_metrics.py, test_metric_kernel_is_fail_closed.py,
  test_evaluator_meta.py, test_canonical_variant_table.py,
  test_clustering_metrics.py, test_representation_geometry.py,
  test_norm_angle_probe.py, test_capability_lifecycle.py     all passed

Python 3.10.20 floor: capabilities.py, cluster_resolution.py and metrics.py
  IMPORT AND EXECUTE, not merely parse. StrEnum confirmed absent on 3.10.
```

**`test_core.py` shows 32 failures in the sandbox. All 32 are pre-existing.** Verified by
cloning `2e04bd9` into a separate directory and running the same suite: identical count, and
comparing the sorted failing test identifiers gives an empty set difference in both
directions -- zero introduced, zero fixed. Root cause `ModuleNotFoundError: No module named
'xgboost'`, an environment gap the handoff anticipated.

**Ratchet delta, measured not computed.** Full collection needs `xgboost` and cannot
complete in the sandbox, so the authoritative total must be taken on the development
machine. The delta is exact:

```
collected, pristine 2e04bd9 : 1762
collected, this tree        : 1889
delta                       : +127   ( = 67 + 60 )
```

Expected ratchet and badge after this commit: **2991 + 127 = 3118**, to be **copied from
`pytest tests/ --collect-only -q` on the staged tree**, never typed by hand.

---

## 5. LIVING METRICS GLOSSARY -- terms this commit introduced or changed

| term | definition | range | why it matters here |
|---|---|---|---|
| **Bootstrap confidence interval** | Interval formed from the percentiles of a metric recomputed over many resamples of the data | metric-dependent | The output being reconciled |
| **Resampling unit** | The thing drawn with replacement: a whole gene, or one variant row | `gene` / `variant` | Now typed and persisted; previously implicit in which caller ran |
| **Gene-cluster bootstrap** | Resamples whole genes, preserving within-gene dependence | -- | The only certifiable design for a gene-disjoint claim |
| **Design effect (confidence-interval width ratio)** | Cluster interval width divided by naive row interval width | >= 0; 1.0 means clustering carried no information | 2.935 measured on the real cohort; quantifies how much every prior interval understated uncertainty |
| **Variance ratio versus row** | The square of the width ratio; the classical survey design effect | >= 0 | Reported separately so "design effect" is never ambiguous between the two |
| **Valid replicate** | A resample on which the metric returned a finite value | 0 to `n_requested` | Percentiles taken from too few are not percentiles of the sampling distribution |
| **Degenerate replicate** | A resample that was empty or single-class, or on which the metric was non-finite | 0 to `n_requested` | `n_valid + n_degenerate == n_requested` is asserted |
| **Effective minimum valid** | `max(absolute_floor, ceil(fraction x n_requested))` | >= 1 | The binding floor; the relative term catches 100-of-100,000 |
| **Certification eligibility** | Whether an interval may back a certified claim | boolean | Independent of status; only gene-unit intervals qualify |
| **Partition equivalence** | Two labelings induce identical row groupings | boolean | Namespace-free test replacing invalid string equality |
| **Schema version** | Version of the persisted evaluation report | 1 or 2 | Version 1 is never read as certified |

---

## 6. OPEN ITEMS RECORDED, NOT FIXED HERE

1. **`as_meta()` produces no `gene_symbol`**, so a seam-produced frame yields an empty
   `gene_errors` list while its docstring claims to be the frame the evaluator expects.
   Pre-existing at `2e04bd9`. Evaluator breakdown logic, not bootstrap.
2. **Audit line 271** attributes per-gene analysis to `gene_id`.
3. **`ValidationMetrics` zero-defaults**, deferred with a proof test (section 3.8).
4. **`default=str` remains at `prediction_artifacts.py:125` (manifest) and `:416`
   (statistics)**. The same silent-coercion defect class; the eval-report writer was in
   scope and is fixed, these two are adjacent and would widen the blast radius.
5. **`metrics.py` still lacks `from __future__ import annotations`** though the project
   convention requires it and `evaluator.py:38` carries it. Unrelated to bootstrap.
6. **`frozen=True` on `EvaluationReport`** -- inventory proves it safe (one constructor,
   zero mutations) but it is an orthogonal mutability change.
7. **A generic `cluster_id` projection in the seam.** The resolver accepts one today; moving
   normalisation into the seam adapter would stop `ClinicalEvaluator` embedding
   schema-discovery policy at all.
8. **The continuous-integration matrix does not exercise the declared 3.10 floor.** Only
   3.11 and 3.12 run. `test_no_module_uses_strenum...` greps for the one known hazard, but
   an import-level 3.10 test would be stronger.

---

---

## 7. LANDING RECORD (2026-07-26)

Recorded because a session document that stops at "ready to commit" leaves the
next reader unable to tell whether the work landed, and under what evidence.

**Commit.** `2e04bd9..eca534e`, "feat(evaluation): one bootstrap engine, explicit
resampling unit", pushed 2026-07-26T09:35:45-04:00. Fifteen files, 2,943
insertions, 56 deletions. No stray artifacts: no `.bak_` backups, no
`__pycache__`, no `.pyc`, nothing under `outputs/`.

**Local full suite, Windows, Python 3.12 in `.venv312`:**

```
3111 passed, 7 skipped in 832.20s (0:13:52)
3111 + 7 = 3118 = tests/EXPECTED_SUITE_SIZE = README badge
```

**Continuous Integration run #617 -- SUCCESS, total 15m 20s.** Rule D16 satisfied
explicitly, per job rather than per aggregate:

| job | result | duration |
|---|---|---|
| lockfile drift check | pass | 13s |
| pytest (3.11) | pass | 12m 32s |
| pytest (3.12) | pass | 13m 40s |
| drift monitor (isolated env) (3.11) | pass | 1m 23s |
| drift monitor (isolated env) (3.12) | pass | 2m 22s |
| Docker build smoke test | pass | 1m 34s |
| Push image to GHCR | SKIPPED BY DESIGN | -- |

`Push image to GHCR` is the `push-ghcr` job, gated on
`github.event_name == 'release' && github.event.action == 'published'`. This was a
push, so skipping is correct. Note that the 2026-07-24 changelog entry cites this
gate as "ci.yml:558"; it is now line 565 because the file grew. A line number in
prose is a fragile citation, and the JOB NAME is the durable reference.

Artifact produced: `coverage-report`, 44.3 kilobytes, digest
`sha256:46fbe3d86e8cc7940740701296ec88b530de1285777762a9f17adc3ae40e3cf2`.

**The skip-cascade failure mode did NOT recur.** The comment at ci.yml:174-190
records run 29374485597, where a spurious lockfile-gate failure caused `pytest`,
`drift monitor`, `Docker build smoke test` and `Push image to GHCR` to ALL report
"Skipped after 0s" and **1,936 tests never ran on Linux**. Every job in run #617
reports a real, non-zero duration, and the two pytest legs report 12m 32s and
13m 40s against a local 13m 52s for the same 3,118 tests -- proportionate, and
therefore genuine execution rather than a silent skip.

**Skip surface, unchanged.**

```
previous run (changelog 2026-07-24) : 2886 passed + 7 skipped = 2893 = ratchet 2893
this run     (2026-07-26)           : 3111 passed + 7 skipped = 3118 = ratchet 3118

tests added by this commit : +127
skips added by this commit :    0
```

All 127 new tests EXECUTE. Neither new test file uses a module-level
`pytest.importorskip`, which is the construct that collapses an entire file into
one skip entry and is how the graph-neural-network branch went untested for 508
Continuous Integration runs (roadmap 6.17).

**SKIP CENSUS (measured 2026-07-26, both platforms).** Item (i) below asked for
the seven skips to be named; they now are, and the answer is worse than
bookkeeping.

Windows, Monzia's machine, `pytest tests/ -q -rs` -- 7 skipped:

| n | location | mechanism |
|---|---|---|
| 5 | tests/integration/test_mc_dropout_calibration.py | unconditional `@pytest.mark.skip` |
| 1 | tests/unit/test_preflight_data_paths.py:45 | `skipif os.name == "nt"` |
| 1 | tests/unit/test_tabular_nn_mc_dropout.py:232 | corpus-conditional `pytest.skip()` |

Linux, the platform Continuous Integration runs, measured directly -- 9 confirmed:

| n | location | mechanism |
|---|---|---|
| 5 | tests/integration/test_mc_dropout_calibration.py | unconditional `@pytest.mark.skip` |
| 3 | tests/unit/test_run17_postflight_paths.py:130 | `skipif sys.platform != "win32"` |
| 1 | tests/unit/test_run17_postflight_paths.py:143 | `skipif sys.platform != "win32"` |
| ? | tests/unit/test_tabular_nn_mc_dropout.py:232 | corpus-conditional; not measured here |

Three consequences, all previously unrecorded:

  1. **FIVE MONTE CARLO DROPOUT TESTS RUN NOWHERE.** They are unconditional stubs
     carrying TODO markers, dormant since 2026-05-27, and they skip on BOTH
     platforms. `mc_dropout` is one of the thirteen permanent base models, and
     these five are the tests of its epistemic-uncertainty claims:
     `test_held_out_gene_families_have_higher_epistemic` (the core
     out-of-distribution claim: epistemic uncertainty should be higher on
     held-out gene families than on in-distribution variants),
     `test_spearman_correlation_between_epistemic_and_error_positive`,
     `test_accuracy_decreases_monotonically_across_epistemic_quartiles`,
     `test_ece_lower_with_mc_dropout_vs_single_pass` and
     `test_epistemic_estimate_converges_with_k`. The uncertainty quantification
     of a permanent model is therefore asserted by nothing. They await the Run 15
     cohort plus gene-family-disjoint splits and expected-calibration-error
     infrastructure.
  2. **FOUR TESTS HAVE NO CONTINUOUS-INTEGRATION COVERAGE AT ALL.** The four in
     `test_run17_postflight_paths.py` execute only on Windows. If they regress,
     nothing in the pipeline notices; only a local run does.
  3. **ONE TEST NEVER RUNS ON THE DEVELOPMENT MACHINE.**
     `test_preflight_data_paths.py:45` runs only on Linux, where it and its five
     siblings pass.

The skip COUNT was assumed stable at 7. It is not a single number: the two
environments skip DIFFERENT tests, and Linux skips more. Neither figure was ever
recorded alongside the other until today.

**INSTALLER DEFECT, third of the session.** The landing-record installer aborted
with "Method invocation failed because [System.Char] does not contain a method
named 'Trim'." `Get-Content | Where-Object` returned ONE line, and PowerShell
unwraps a single-element collection to a SCALAR, so the variable held the STRING
"3118" rather than an array of one string. Indexing a string yields a
`[System.Char]`, which has no `Trim`. The same hazard was latent in the
bootstrap installer -- `Get-BareRatchetLines` returned an ArrayList that also
unwrapped -- and survived only because indexing an INT returns the int while
indexing a STRING returns a char. Same defect, different type, different outcome.
Both are now coerced with `@( ... )`. Rollback behaved correctly and restored all
three files.

---

*Written 2026-07-26.*
