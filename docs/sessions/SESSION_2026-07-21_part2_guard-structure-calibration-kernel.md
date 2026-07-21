# SESSION 2026-07-21 (ADDENDUM) -- The Storage Guard, the STRUCTURE Partition, the Calibration Carve, and the Metric Kernel

**Date:** 2026-07-21
**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `0021a72`, 09:04:51, "fix(forensics): repair the disk census and give it a home and tests"
**Ending HEAD:** `b8275a0`, 15:01:32, "fix(metrics): one evaluation contract, and a validity flag that checks"
**Suite at start:** 2283 collected (2276 passed, 7 skipped)
**Suite at end:** 2446 collected (2439 passed, 7 skipped)
**Net test change:** +163
**Skips:** 7 at start, 7 at end, identical set

This document is an ADDENDUM. `SESSION_2026-07-21_conformal-split-panelq-census.md`
(commit `ea81fa7`) covered five commits through `0021a72` and stopped there. Four
commits landed afterwards with no session record, and this closes that gap.

| Commit | Time | Subject | Files | Insertions | Deletions | Tests |
|---|---|---|---|---|---|---|
| `2df15a1` | 09:51:04 | the data guard checks free space, and is finally called | 6 | 802 | 13 | +45 |
| `c68934d` | 10:31:50 | a STRUCTURE partition, and a role cap that cannot go stale | 4 | 523 | 5 | +50 |
| `623c98c` | 14:59:25 | carve the isotonic fold by gene, not by label | 4 | 781 | 9 | +39 |
| `b8275a0` | 15:01:32 | one evaluation contract, and a validity flag that checks | 5 | 347 | 77 | +29 |

Ratchet arithmetic: 2283 + 45 + 50 + 39 + 29 = 2446. Verified against the remote
in a FULL clone on 2026-07-21. An earlier reconciliation in a `--depth 4` clone
reported `2df15a1` as 1,251 files and 772,082 insertions; that was an artifact of
the shallow graft boundary, since the commit's parent lay outside the clone. The
figures above come from a complete clone and match each commit's own output.

---

## PART ZERO -- THE THREAD THROUGH ALL FOUR

Every commit in this addendum repairs the same shape of defect, and it is the
same shape the earlier five repaired: **a check that could not fail, or a claim
nothing verified.**

- `preflight_data_guard.py` never checked free space, and its entry point
  `assert_data_usable` was called from nowhere in the repository.
- `PartitionSchema` capped role repetition by ENUMERATING three roles. The list
  was already stale on the morning it was written.
- The production isotonic calibrator was fitted on a label-stratified carve with
  no gene grouping, so 100.0 per cent of calibration rows came from genes the
  models had trained on -- while the cross-validation twenty lines below was
  carefully gene-disjoint.
- `calibration_valid` asserted that calibration numbers were sound after checking
  only that the values lay between zero and one.

The lesson was stated five times in one day and earned twice by my own work:

> **A check that cannot fail is worse than no check, because it manufactures
> confidence.**

The second half of the day added a corollary, learned by breaking the suite:

> **A new guard that refuses what its predecessor accepted is a regression, not a
> stricter standard.**

---

## PART ONE -- THE GUARD THAT NEVER RAN (`2df15a1`, 09:51:04)

### The two defects

`scripts/maintenance/preflight_data_guard.py` existed to protect a run from
starting without usable data. It had two independent failures.

**It never checked free space.** The module validated that expected data paths
existed and were readable. It did not ask whether the volume had room for the
outputs a run would produce. A Run 17 launch on a volume at 8.9 per cent free
would pass the guard and then fail partway through, after hours of compute.

**Its entry point was called from nowhere.** A repository-wide search for
`assert_data_usable` returned **zero** invocations outside its own definition and
its own tests. The guard was a well-tested function that no code path reached.

### The repair

`preflight_data_guard.py` was rewritten to 287 lines around a `StoragePolicy`
dataclass loaded from `configs/data_manifest.yaml`, with three severities: OK,
WARN and FAIL. The required figure, 61.48 gibibytes, reproduces the disk census
measurement from `0021a72` exactly rather than being an independently chosen
round number.

`storage_gate()` was wired into `scripts/preflight_run17.py::run_all()`, so the
guard now runs where a run actually begins.

### An implementation detail worth recording

A `@dataclass` defined in a module carrying `from __future__ import annotations`
cannot be constructed by a loader that executes the module before registering it
in `sys.modules`. Annotations are strings under the future import, and
`dataclasses` resolves them by module lookup. The test harness had to register
the module BEFORE calling `exec_module`. This was found by the failure, not by
reading, and it will recur in any future test that loads a dataclass module by
path.

### Verification

45 tests in `tests/unit/test_storage_guard.py`. Four skipped in the authoring
sandbox for want of xgboost; all 45 ran on the development machine. Suite moved
2283 to 2328. That run recorded 664.35 seconds for 2328 tests, which was at the
time the fastest per-test figure the project had measured.

---

## PART TWO -- THE STRUCTURE PARTITION (`c68934d`, 10:31:50)

### Why the role exists

Panel Q, shipped that morning in `5dcb932`, had no partition to run on.

Its specification is explicit that unsupervised structure analysis is a
MODEL-SELECTION activity. It chooses a representation, a preprocessing scheme, a
dimensionality, a distance geometry, a clustering algorithm, a cluster count, a
noise-handling rule, stability thresholds and a biological interpretation.
Performing those choices on the locked test partition is selection on test.

The specification therefore requires a dedicated STRUCTURE partition
"gene-disjoint from train, tune, probability calibration, conformal calibration,
and test", with test admitting only a predeclared replication of a solution
frozen on STRUCTURE.

### A stale list, found by probing rather than reading

`PartitionSchema.__post_init__` capped role repetition by enumerating three
roles: SELECT, CALIBRATE_CONFORMAL and TEST.

That list was ALREADY STALE. Commit `5b1c82b`, earlier the same morning, had
added `CALIBRATE_PROBABILITY` and had not added it here. Two partitions could
both declare that role, the schema would be ACCEPTED, and `name_for_role()` would
silently return whichever was declared first. `train.py` would fit the
probability calibrator on one partition while a second believed it held the role
-- a silent ambiguity in precisely the role the five-way schema exists to serve.

It was found by constructing a six-partition schema and watching what the
validator accepted. Not by review.

**The repair removes the list rather than extending it.** Every role except TRAIN
is capped, so a future enum member cannot be forgotten. TRAIN remains constrained
more strictly, at exactly one, because the train-only leakage remap derives every
partition's counts from it. The test is parametrized over `PartitionRole` ITSELF,
so a role added tomorrow is covered without anyone editing the test.

### The migration guarantee, and the test failure that earned it

Hash intervals are assigned in `hash_order`, so a partition's genes depend on the
cumulative fraction of everything ordered ahead of it.

```
FIVE_WAY   test [0.00,0.12)  conformal [0.12,0.20)  calib [0.20,0.30)
           tune [0.30,0.45)  train     [0.45,1.00)

SIX_WAY    ... identical ...  structure [0.45,0.52)  train [0.52,1.00)
```

Verified across seeds 42, 7, 123 and 2026, migrating FIVE_WAY to SIX_WAY moves
EXACTLY ONE set of genes, out of train and into structure:

- test, conformal, calib and tune are **byte-identical**
- structure is **precisely the genes train gave up**
- train loses those and gains nothing, dropping 0.55 to 0.48

So the locked test set is untouched, a fitted probability calibrator stays valid,
the conformal calibration set is unchanged, and model selection does not have to
be repeated. Only the training set shrinks, which is the honest cost of funding a
structure partition. No evaluation partition was narrowed to pay for exploratory
analysis.

**The first hash order tried placed structure between calib and tune.** That
shifted tune by 0.07 and would have silently invalidated every model-selection
decision made under FIVE_WAY, for no benefit. A test asserting the OPPOSITE
property -- that genes get reassigned -- FAILED, and that failure is what found
it. The guarantee is now four parametrized tests rather than a remark.

### What did not change

`split()`, the invariant checker and the train-only leakage remap needed NO
changes at all. Six partitions worked on the first probe. That is the payoff of
the schema-as-data rewrite in `5b1c82b`: adding a partition is now a data change,
not six coordinated code edits.

### A skip avoided

An early draft skipped the TRAIN case inside the parametrized role test. That
would have added a permanent EIGHTH skip to every run forever, and the suite's
skip count is a monitored signal. Filtering the parametrize list instead keeps
the enum-driven property at zero skip cost.

### Verification

50 tests, 104 passing with the existing 54, deterministic across three runs. Five
sabotages, all detected: role cap reverted (4 failed), structure placed before
tune (6 failed), structure funded from test (13 failed), STRUCTURE removed from
the enum (3 errors), `name_for_role` substituting instead of returning None (5
failed).

---

## PART THREE -- THE CALIBRATION CARVE (`623c98c`, 14:59:25)

This is the most consequential commit of the day and the one I got most wrong
before getting it right.

### An error I made three times

I stated, repeatedly and confidently, that wiring `scripts/train.py` to
`rows_for_role(CALIBRATE_PROBABILITY)` would be "the commit that actually
corrects every calibrated number the project reports."

**`train.py` is not the production entry point.**
`scripts/preflight_run17.py::emit_command()` builds
`python scripts/run_phase2_eval.py ...`, and no launcher passes
`--split-protocol` at all. The `v2_conformal` path -- and the `tune`-calibration
defect I had described -- has NEVER executed in a production run.

Checking `emit_command()` before designing the fix is the only reason this commit
touches `variant_ensemble.py` instead. The rule this reinforces:

> **Check the entry point before asserting what a change will affect.**

### The real defect

`scripts/run_phase2_eval.py:590` calls `ensemble.fit(...)` with no `*_cal_ext`
argument, so `variant_ensemble.fit()` takes the self-carve branch:

```python
idx_fit, idx_cal = _tts(idx, test_size=0.15, stratify=y_arr,
                        random_state=self.config.random_state)
```

`stratify=`, and no `groups=`. Measured 2026-07-21:

| Cohort | Calibration-fold genes | Also in fit fold | Calibration rows from trained genes |
|---|---|---|---|
| 500 genes | 319 | 319 | **100.0 %** |
| ClinVar-like, 8,000 genes | 7,864 | 7,856 | **100.0 %** |

Twenty lines below the carve, the inner cross-validation is carefully
gene-disjoint via `GroupKFold`, citing INCIDENT_2026-06-13 by name. Out-of-fold
PREDICTIONS were gene-disjoint; the fold the CALIBRATOR was fitted on was not.
One function, two standards. It affects xgboost, lightgbm and random_forest --
the `_RECALIBRATE` set -- and through them the stacking meta-learner.

### The fix

`EnsembleConfig.calibration_carve`, defaulting to `"gene_disjoint"`: carve whole
genes with `GroupShuffleSplit` so the calibrator never sees a gene the models
were fitted on. 100.0 per cent to 0.0 per cent, both cohorts.

`"legacy_stratified"` reproduces the previous behaviour BYTE-FOR-BYTE -- identical
indices at seeds 42, 7 and 2026 -- so historical runs remain reproducible. The
decision to default to gene-disjoint, with legacy reachable by explicit flag, was
made by Monzia after being shown the measurement below.

`calibration_carve_used_` records what actually happened -- `gene_disjoint:seed=N`,
`legacy_stratified:configured`, `legacy_stratified:no_gene_labels`, or
`external_partition` -- so a fallback can never be silent. `run_phase2_eval` reads
`gene_symbol` from `splits/meta_train.parquet`, which can be absent; that path now
warns loudly instead of degrading quietly.

### The measurement contradicted my prediction

I predicted the fix would change reported numbers and that current values were
optimistic. Five independent synthetic cohorts, Brier score on a locked set of
ENTIRELY UNSEEN genes:

| Carve | Brier on the calibration fold | Brier on unseen genes | Optimism |
|---|---|---|---|
| legacy | -- | 0.18063 | **+0.00415 mean, 5 of 5 cohorts** |
| gene-disjoint | -- | 0.18064 | +0.00048 mean |

**Difference in the reported number: +0.00001, or +0.00 per cent.**

The reported numbers do not move. What was optimistic is the INTERNAL estimate:
the legacy calibration fold overstated its own quality in every cohort tested,
while the gene-disjoint fold is essentially unbiased.

**CONSEQUENCE: the Run 14 through 17 reported test metrics do NOT need
retracting.** The defect corrupted a diagnostic, not the headline. These are
synthetic cohorts; the magnitude on the real cohort may differ, and the first
real run under this change should re-measure rather than assume.

### A regression I shipped, and the repair

The first version used a single absolute floor, `calibration_min_rows = 200`. It
BROKE `test_level2_leakfree_oof::test_fit_accepts_gene_symbol_and_runs`. Full
suite at that point: **1 failed, 2400 passed, 7 skipped in 914.61 seconds.**

That cohort is 427 rows across 30 genes, where a 15 per cent carve of ANY kind
yields about 64 rows. The floor refused gene-disjoint folds of 63 to 76 rows --
while the legacy stratified carve it replaces had been using a 62-row fold on
that exact cohort for the entire history of the project, without complaint.

No test caught it because every cohort in the original 30 tests had 500 or more
genes. The guard was right in intent and wrong in instrument. The question it
should ask is not "is this fold small?" but "did this carve deliver roughly what
was ASKED for?" -- which is what the row-versus-group trap actually poses.

**The trap runs both ways**, found by probing the lower bound rather than by
review. A cohort of 11,000 rows whose largest gene held 9,000 produced a
9,300-row calibration fold against a 1,650-row target: 85 per cent of the data,
leaving the models 1,700 rows to fit on. A fold far larger than requested starves
the fit exactly as a smaller one starves the calibrator.

Four thresholds replaced the single floor:

| Setting | Value | Role |
|---|---|---|
| `calibration_min_rows` | 25 | absolute floor |
| `calibration_min_fraction_of_target` | 0.5 | catches an undersized carve |
| `calibration_max_fraction_of_target` | 2.0 | catches an oversized one |
| `calibration_advisory_rows` | 200 | WARNS, does not refuse |

Verified across three regimes: 427 rows / 30 genes gives a 76-row fold that warns
and proceeds; 11,000 rows with a dominant gene is REFUSED with both verdicts
observed; 1,201,942 rows / 8,000 genes gives 180,458 rows, exactly 15.0 per cent.

### A sabotage that failed to fail

Removing the both-classes guard left all 28 tests passing, because every cohort
used happened to yield both classes. A cohort where the label is a property of
the GENE -- 4 of 40 genes pathogenic -- makes `GroupShuffleSplit` at
`random_state=42` return an 89-row ALL-BENIGN calibration fold. Isotonic
regression fitted on one class returns a constant and raises nothing, so the
calibrator would silently become the class base rate.

Two tests now pin it, and the fixture ASSERTS it still forces the condition, so
the test cannot quietly stop testing.

### The withdrawn ratchet, recorded honestly

`install_ratchet_bump_2408` ran on the development machine BEFORE the suite
failed, so the ratchet read 2408 while the change was being repaired. The
replacement bump was therefore 2408 to 2417, not 2378 to 2417. **Both ledger
entries remain in `tests/EXPECTED_SUITE_SIZE`** -- the 2408 entry describing the
original 30 tests, and the 2417 entry stating plainly that the 2408 bump was
withdrawn and why. The history of a mistake is more useful than a tidy ledger.

### Verification

39 tests, 43 passing with `test_level2_leakfree_oof`, deterministic across three
runs. Six sabotages, all detected: default flipped to legacy (5 failed), disjoint
branch using the stratified split (8), both-classes guard removed (1), min-rows
guard removed (3), missing gene labels falling back mutely (1), legacy test_size
drift (4).

---

## PART FOUR -- THE METRIC KERNEL (`b8275a0`, 15:01:32)

### Origin

Monzia produced a written evaluation of the metric stack. I verified every
falsifiable claim in it against the code rather than agreeing on plausibility.
Three results:

**Two of its three implementation defects were already fixed.** The claim that
`evaluate()` cleaned score and probability independently, and that `_clean()` cast
labels without validating them, described the pre-2026-07-20 module. `clean_arrays`
now names them explicitly, calling the label coercion "DEFECT B" and explaining
that `astype(int)` truncated 0.9 to 0 and left 2.0 as 2, making AUROC's
denominator negative. Measured: one joint mask drops rows missing in ANY array,
and `[0,1,2,1]` and `[0,0.9,1,1]` are both rejected by name.

**Its permutation-null correction was right and my description was wrong twice
over.** The code permutes whole-gene covariate values ACROSS genes --
`group_values = cov[first_pos]; rng.permutation(group_values)[inv]` -- so every
variant in a gene keeps one shared value. That is gene-block permutation. I had
described it as "labels shuffled within genes": wrong exchangeability unit AND
wrong quantity, since it permutes a covariate against a fixed clustering, not
labels.

**Its third defect was real and worse than described.** The legacy evaluator did
not merely return a misleading `specificity: 0` for an undefined quantity -- on a
single-class cohort it CRASHED, raising "not enough values to unpack (expected 4,
got 1)" because `confusion_matrix(...).ravel()` yields one cell, not four.

### A defect the evaluation did not list

`calibration_valid` gated on `is_probability(p)` alone. Measured on
`y = [1,1,1,1]`, `p = [.9,.8,.85,.95]`:

```
auroc  NaN   auprc  NaN          correct, ranking is undefined
cal_slope NaN  cal_intercept NaN
brier  0.01875   ece  0.125      NUMBERS
calibration_valid  True          asserting those numbers are sound
```

That ECE is `1 - 0.875`: the gap between the mean prediction and the only label
present. The reliability diagram has a single occupied row, so it says nothing
about calibration across the probability range.

The flag also broke its own documented invariant in the OTHER direction. The
comment promised that False implies the calibration metrics are NaN by design;
here `cal_slope` and `cal_intercept` were NaN while the flag read True, so a
reader would take an undefined estimand for a failed computation.

### The repair

Both classes present is a HARD requirement, since without it the quantity is not
calibration. Thin support is REPORTED, not refused -- `DEFAULT_MIN_POS` and
`DEFAULT_MIN_NEG` are the SAME floors `stratified_evaluate` already applied per
subgroup, so identical data was being called insufficient as a stratum and sound
on its own. Refusing it outright would have repeated the 427-row regression made
three hours earlier.

A new field, `calibration_support`, carries the reason in machine-readable form:
`sufficient`, `thin:...`, `single_class:...`, `not_probabilities`, or
`insufficient_rows`.

**The invariant is now enforced in code, not promised in a comment.** Fixing the
flag alone was only half the defect: `calibration_valid=False` still returned
`ece=0.125`, because `brier_score` and the ECE estimator self-guard on
`is_probability` only. A flag that does not enforce what it asserts is the same
defect it exists to prevent.

**Check order matters.** An empty cohort first reported `not_probabilities` --
true, since `is_probability([])` is correctly False, but the less useful of two
true reasons. Checks now run most-specific first and it reports
`insufficient_rows`. A test found that, not review.

### The legacy evaluator removed, not wrapped

`compute_classification_metrics` and `ModelEvaluator` are DELETED. The
2026-07-20 banner in the module had called them "unsafe in ways this stack
explicitly rejects" and deferred unification to "a separate, separately-measured
commit". This is that commit.

Removal rather than delegation, because delegation preserves the contract that is
the actual problem: a dict of bare floats cannot express undefined, insufficient
support, dependency unavailable, or computationally deferred -- all first-class
scientific states here.

Neither had a production caller. One test pinned them; one notebook imported them
from `src.evaluation.metrics`, a package path that has not existed since the
rename. The `__all__` note reading "original API -- do not remove" was itself
stale. `test_legacy_metrics_api_is_preserved` is replaced one-for-one by
`test_the_legacy_metrics_api_is_gone`, which pins their ABSENCE, so reintroducing
a second contract requires deleting a test explaining why there is one.

### Verification

29 tests, 175 passing across the four metric suites, deterministic across three
runs. Three sabotages, all detected: `calibration_valid` reverted to
`is_probability` (18 failed), invariant left unenforced (5 failed), legacy
evaluator restored (1 failed).

---

## PART FIVE -- A COMMIT THAT WAS SPLIT AFTER PUSHING

### What happened

At 15:00 the metric-kernel files were copied into the repository and verified
(175 passed) while the calibration-carve change was still staged but uncommitted.
A `git add -A` then swept up both changes, and `git commit -F` applied the
CALIBRATION CARVE message to all seven paths.

Commit `d7c4d35` -- "fix(calibration): carve the isotonic fold by gene, not by
label", 7 files, 1,057 insertions, 84 deletions -- was pushed.

### Why this mattered, beyond tidiness

**The commit message described half its content.** It said nothing about removing
the legacy evaluator, nothing about hardening `calibration_valid`, nothing about
the enforced NaN invariant. Anyone reading the history would find
`compute_classification_metrics` deleted with no explanation anywhere.

**More seriously, the ratchet was wrong.** It had been bumped to 2417 for the
calibration carve. The metric kernel adds 29 tests, so the suite now collected
2446 while `EXPECTED_SUITE_SIZE` and the README badge both read 2417.
`install_ratchet_bump_2446` had never been run. **The next
`pytest --assert-suite-size` would have FAILED the size gate on `main`.** Loudly,
which is the gate working -- but broken all the same.

### The repair

A tag `pre-split-2026-07-21` was placed on `d7c4d35` first, making the rewrite
fully reversible. Then `git reset --soft HEAD~1`, `git reset`, selective staging
of the four carve paths, commit; `install_ratchet_bump_2446` (pre-check read
2417); `git add -A`, commit the five kernel paths; full suite; and
`git push --force-with-lease`.

Result: `+ d7c4d35...b8275a0 main -> main (forced update)`. The lease held,
confirming nothing had moved underneath. The tag was deleted after verification.

### The arithmetic, reconciled

The split does not sum to the original, and that is correct:

- **Files:** 4 + 5 - 2 = 7. `README.md` and `tests/EXPECTED_SUITE_SIZE` appear in
  both commits.
- **Insertions:** 781 + 347 = 1,128, exceeding 1,057 by **71**.
- **Deletions:** 9 + 77 = 86, exceeding 84 by **2**.

The excess is precisely the second ratchet bump. `d7c4d35` carried ONE ledger
entry (2378 to 2417); the split carries TWO, adding 2417 to 2446 in `b8275a0`.
That second entry is +70/-1 in `EXPECTED_SUITE_SIZE` and +1/-1 in `README.md`:
**+71 insertions and +2 deletions, matching the excess exactly.**

That difference IS the point of the split. Under `d7c4d35` the 29 kernel tests
were riding along uncounted.

### The rule this produces

> **Never `git add -A` with two logical changes in the tree.** Stage by path, and
> read `git status --short` against the commit message's own file list before
> committing.

---

## PART SIX -- THE CAPABILITY-VALIDATION PRINCIPLE (ACCEPTED, NOT YET BUILT)

Monzia's evaluation proposed ten release-blocking metric additions. Seven
evaluate outputs that exist today. Three do not: gene-attribution validation,
multi-label disease metrics, and regression / conformal-quantile-regression
coverage.

I argued that a gate over a nonexistent output does not block -- **it passes
vacuously**, because there is nothing to check and therefore nothing to fail.
That is the same pattern this session caught five times. A green Panel H would
then be cited as evidence a disease head was validated.

Monzia accepted the refinement and elevated it into an architectural law:

> **Every evaluation panel must correspond to an implemented capability. Panels
> evaluating absent capabilities must return `MetricStatus.INSUFFICIENT_SUPPORT`
> with a machine-readable reason. No absent capability may satisfy a release
> gate.**

The codebase already has the vocabulary: `MetricStatus.UNDEFINED`,
`INSUFFICIENT_SUPPORT`, `DEPENDENCY_UNAVAILABLE`, with named `REASON_` constants,
and `decide_confounder_gate` already treats a non-OK status as unsatisfied.
Monzia proposed expanding the enum to separate `NOT_IMPLEMENTED`,
`NOT_APPLICABLE`, `INSUFFICIENT_DATA` and `INSUFFICIENT_SUPPORT`, plus a
`CapabilityState` model and a typed `EvaluationRegistry`. **None of this is built
yet.**

---

## PART SEVEN -- FEASIBILITY OF THE THREE UNBUILT HEADS (MEASURED)

Asked whether the three are impossible to build, I checked the data rather than
estimating. None are impossible; they are at very different distances.

**Regression and conformal quantile regression -- genuinely blocked.**
`data/external/gtex`, `data/rnaseq` and `data/external/functional_assays` are all
ABSENT. A regression panel needs quantitative targets: change in
percent-spliced-in, change in Gibbs free energy of folding, expression change,
assay activity. SpliceAI and AlphaMissense supply scores, but those are another
model's predictions used as INPUT FEATURES, not ground truth. Blocked on data
acquisition, not engineering.

**Multi-label disease -- tractable, but a data-ingestion project.** ClinVar
carries a disease field (CLNDN) and phenotype identifiers, and the pipeline reads
NEITHER. It ingests clinical significance, review status, gene symbol and
coordinates. OMIM contributes `omim_n_diseases`, a gene-level COUNT, not a
per-variant disease label. The head needs CLNDN ingested and normalised to MONDO
or MedGen first.

**Gene ranking -- buildable now, but the obvious validation is contaminated.**
Gene-level ground truth exists: `clingen_validity_score` (0 to 5) and
`omim_n_diseases`. **But all four gene-disease annotations are INPUT FEATURES**,
listed in the 91-feature contract under "Gene-disease annotation (4)". Ranking
genes with a model handed ClinGen's verdict, then scoring that ranking against
ClinGen, is circular. It is the `n_pathogenic_in_gene` concern in a new costume,
and that one needed a permutation ablation to settle. A valid design needs either
a temporal holdout -- genes upgraded after a cutoff date -- or those four features
ablated for the ranking experiment.

**Consequence for the architectural law:** `NOT_IMPLEMENTED` is the wrong status
for gene ranking. The capability is present; what is missing is an uncontaminated
TARGET. That is `INSUFFICIENT_SUPPORT` with a reason such as
`REASON_TARGET_IS_AN_INPUT_FEATURE`. The distinction is exactly why the enum
expansion earns its place.

---

## PART EIGHT -- RUNTIME, FOURTEEN MEASUREMENTS

| Tests | Seconds | Milliseconds per test |
|---|---|---|
| 2283 | 728.76 | 319.2 |
| 2328 | 664.35 | 285.4 |
| 2378 | 746.52 | 313.9 |
| 2417 | 648.54 | 268.3 |
| 2417 | 961.88 | 398.0 |
| 2446 | 680.18 | 278.1 |

**The two 2417 runs are the cleanest evidence yet.** Identical tree, identical
test count, one session apart: 648.54 seconds then 961.88 seconds, a **1.48-fold
spread**. Across all fourteen measurements the range is 605.08 to 1131.67
seconds, 1.87-fold, with NO growth trend against test count.

This settles a claim retracted earlier in the day. Suite runtime is dominated by
host contention on the development machine, not by suite size. A single timing is
not evidence of anything.

---

## PART NINE -- SKIPS

Seven throughout, unchanged in membership and count across every run:

| Count | Location | Reason |
|---|---|---|
| 5 | `tests/integration/test_mc_dropout_calibration.py` | Run 15 cohort, Spearman infrastructure, ECE reliability bins, multiple K-value runs |
| 1 | `tests/unit/test_preflight_data_paths.py:45` | POSIX symlink standing in for a Windows dangling junction |
| 1 | `tests/unit/test_tabular_nn_mc_dropout.py:232` | Test corpus does not span both boundary and extreme prediction regions |

Skip POSITIONS were verified by character offset on every run, not just counted.
Across `c68934d` both tail skips shifted +50; across `623c98c`, +39; across
`b8275a0`, +29 -- each matching the new file's test count exactly, because each
new file sorts alphabetically before both. The five integration skips never moved,
correctly, since `tests/integration/` sorts before `tests/unit/`.

**One arithmetic error of mine, caught by its own check.** A first pass summed
2,460 progress characters against a reported 2,417 and concluded both skips had
shifted +171. It had swallowed the 43-character progress row from an earlier
two-file pytest command in the same transcript. Reconciling the character count
against the reported total is exactly what caught it, which is why that check
exists rather than reading positions directly.

---

## PART TEN -- REPOSITORY HYGIENE OBSERVED IN PASSING

Found while reconciling the diffstats, recorded but NOT acted on:

- **Four tracked `.parquet` files** and four tracked files over one megabyte,
  totalling 20.2 mebibytes: `experiments/2026-04-04_03-39/eval_report.json`
  (11.46), `data/external/1kgp/kg_grch38_af.parquet` (6.37),
  `notebooks/genomic_variant_classifier_setup.ipynb` (1.30),
  `data/reference/drift/run15_reference_profile.json` (1.04). Modest, but the
  April experiment artifact is 3.5 months old.
- **`notebooks/genomic_variant_classifier_setup.ipynb` is now definitively
  broken.** It imports from `src.evaluation.metrics`, a package path that has not
  existed since the rename, and the symbols it imported were deleted in
  `b8275a0`. It is dead code that looks live.

---

## PART ELEVEN -- OPEN ITEMS

**Immediately next, in order:**

1. Gene-block permutation rename in `clustering_metrics.py`, with full provenance
   recorded: permutation unit, strata, count, seed, null mean, null 95th
   percentile, empirical p-value, AND the representative-value rule (where a gene
   carries several covariate values the first is taken, which silently defines
   what was permuted).
2. Panel O, decision-curve net benefit -- the largest genuine gap, evaluating
   outputs that already exist.
3. Panels E and F, risk-coverage and class-conditional conformal coverage, which
   connect directly to the measured Benign 0.882 / Pathogenic 0.880 under-coverage
   against roughly 0.92 in the middle tiers.
4. The `MetricStatus` expansion, `CapabilityState` model and `EvaluationRegistry`,
   before Panels H, J and L are specified.
5. Panels H, J and L specified and fail-closed, armed when their heads land.

**Carried forward, unchanged:**

- `train.py` and `run_v2` wiring to `rows_for_role(CALIBRATE_PROBABILITY)`, and
  flipping `--split-protocol` off its legacy default at `train.py:126`. SMALLER
  than I claimed: it fixes the DORMANT tune-calibration path and gives conformal a
  clean partition. `run_v2` at `real_data_prep.py:1322` builds a
  `SplitProtocolV2Config` with NO schema, so it silently gets FOUR_WAY, which has
  no CALIBRATE_PROBABILITY partition at all.
- Panel Q orchestrator (`evaluate_panel_q` and `StructureEvaluationMode`), now
  unblocked by the STRUCTURE partition.
- Class-conditional (Mondrian) ordinal conformal for extreme-tier under-coverage.
- Storage decisions, no longer urgent now the guard reports OK with an advisory at
  8.9 per cent free: OneDrive Files On-Demand (31 gibibytes, reversible) plus a
  pagefile cap (~15) is roughly 46 gibibytes with nothing deleted. Docker (73.49)
  and FinnGen (57.64) remain judgement calls; FinnGen is
  `public_redownloadable`, `sync:false`, with ZERO test dependencies.
- `conftest.py` cache comment says 36,202 files / 8.77 gigabytes; measured 41 /
  19.62 gibibytes. `STORAGE_ACTION_LEDGER_2026-07-03.md` is stale.
- Full metric stack, Panels A through P plus Q, re-arms the five skipped tests in
  `tests/integration/test_mc_dropout_calibration.py`.
- Own rclone Drive `client_id` (the shared one is retired during 2026).
- `data/external/gtex` and `data/rnaseq` EMPTY -- a hard blocker on RNA
  infrastructure. `phylop_score`, `esm2_llr` and `eve_score` remain stub features
  pending the HGVSp parser.
- The first REAL run under `623c98c` should re-measure the calibration optimism,
  since all five cohorts above were synthetic.
- `notebooks/genomic_variant_classifier_setup.ipynb` is broken and should be
  repaired or removed.
