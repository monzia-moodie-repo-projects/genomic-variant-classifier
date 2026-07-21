# SESSION 2026-07-21 -- Ordinal Conformal, the Partition Schema, Panel Q, and the Instrument That Measured the Disk

**Date:** 2026-07-21
**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `9772794`, 2026-07-21 02:13:55, "docs(session): part nine -- EVE restore, new baseline, drive settled"
**Ending HEAD:** `0021a72`, 2026-07-21 09:04:51, "fix(forensics): repair the disk census and give it a home and tests"
**Suite at start:** 2071 collected (2064 passed, 7 skipped)
**Suite at end:** 2283 collected (2276 passed, 7 skipped)
**Net test change:** +212
**Free space at start:** 82.85 gibibytes (8.86 per cent)
**Free space at end:** 83.50 gibibytes (8.925 per cent)

Five commits, none of which had a session record before this document.

| Commit | Time | Subject | Tests | Insertions |
|---|---|---|---|---|
| `c663c89` | 05:16:14 | ordinal prediction sets with contiguity by construction | +57 | 1,235 |
| `5120dd6` | 05:43:53 | export every submodule, and stop the list going stale | +20 | 249 |
| `5b1c82b` | 06:25:54 | partition schema as data, and a five-way protocol | +44 | 1,109 |
| `5dcb932` | 08:11:11 | Panel Q structure metrics and fail-closed guards | +78 | 2,463 |
| `0021a72` | 09:04:51 | repair the disk census and give it a home and tests | +13 | 943 |

---

## PART ZERO -- THE THREAD THAT RUNS THROUGH ALL FIVE

Four of the five commits repair the same shape of defect: **a literal kept
correct by memory, with no mechanism to notice when it stops being correct.**

- The conformal package's `__init__.py` imported five of seven submodules. The
  list was correct when written and silently went stale.
- `split_protocol_v2.py` held `PARTITIONS` as a hard-coded module constant
  referenced in five places, beside four separately-named fraction fields and a
  hand-written sum check.
- `scripts/forensics/verify_dtype.py` asserted the literal source text
  `"for p in PARTITIONS:"`, which verified that code *looked* a certain way
  rather than that it *worked*.
- `audit_disk_reclaim_v3` reused one walker for two incompatible purposes,
  because nothing named the difference between them.

This is the same family as the Run-16 gate's `EXPECTED_COUNT = 81` repaired the
previous day. In each case the repair was not to update the literal but to
remove the need for one.

The fifth commit, Panel Q, is new capability rather than repair -- but it was
built to the same rule: every metric it produces carries a status and a reason,
and it refuses rather than guesses.

**A durable lesson was reinforced four separate times today: a check that cannot
fail is worse than no check, because it manufactures confidence.** Four checks
were found this session that could not fail, three of them mine.

---

## PART ONE -- ORDINAL CONFORMAL PREDICTION (`c663c89`)

### 1.1 What was built

`src/genomic_variant_classifier/conformal/ordinal.py`, 530 lines, plus
`tests/unit/test_ordinal_conformal.py`, 57 tests.

The American College of Medical Genetics and Genomics / Association for
Molecular Pathology (ACMG/AMP) five-tier scale is ordered: Benign, Likely
benign, Variant of uncertain significance, Likely pathogenic, Pathogenic. A
conformal prediction set over an ordered scale that is allowed to be
non-contiguous -- {Benign, Pathogenic} but not the three tiers between -- is
clinically meaningless.

**Contiguity here is by construction, not by repair.** The non-conformity score
is built so its level sets are nested intervals:

    s(k) = u * M_{t-1} + (1 - u) * M_t

which is monotone in the tier index, so thresholding it can only ever produce a
contiguous run. No post-hoc interval-filling step exists, because none is needed.

### 1.2 Two defects caught during construction

**Empty-set repair inflated coverage.** The first version silently replaced an
empty prediction set with the single most likely tier. That raises measured
coverage above the true conformal guarantee -- the set is no longer the set the
calibration justified. Made opt-in (`force_nonempty`) and counted in
`n_forced_nonempty_`, so the inflation is visible rather than absorbed.

**`ordinal_report` counted abstentions as catastrophic errors.** An abstention
and a confidently wrong call are different failures with different clinical
consequences. Separated.

### 1.3 Deliberate behaviour that looks like a bug

`fit()` raises `OrdinalLabelError` on fewer than three distinct labels. On
today's binary cohort that fires immediately. **This is correct and specified**
(specification Panel C): a five-tier method must not silently degrade to a
two-tier one and report five-tier language.

### 1.4 Open finding, carried forward

Per-tier coverage under-covers the extremes:

    Benign        0.882
    Pathogenic    0.880
    middle tiers  ~0.918 - 0.922

Marginal coverage is met; conditional coverage is not. This is exactly the
failure a Mondrian (class-conditional) construction fixes, and it is the natural
next piece of conformal work.

---

## PART TWO -- THE PACKAGE THAT DID NOT EXPORT ITSELF (`5120dd6`)

`conformal/__init__.py` imported five of its seven submodules. `calibrate` and
the newly added `ordinal` were unreachable through the package.

Rewritten with explicit imports, `__all__`, and a full docstring. The repair
that matters is not the corrected list but
`tests/unit/test_conformal_package_exports.py`: 20 tests that **walk the package
directory** and turn the suite red if any module is not exported.

**Verified by sabotage:** removing `risk_control.py` from the export list
produced 5 failures. A list that is checked against the filesystem cannot go
stale in silence.

---

## PART THREE -- THE PARTITION SET BECOMES DATA (`5b1c82b`)

### 3.1 The defect this exists to fix

Specification Finding 2, and Priority 2 which calls it "essential": probability
calibration must be fitted on data untouched by model, method and alpha
selection.

**Measured in the code on 2026-07-21:** `scripts/train.py` lines 131 and 550-551
fit the isotonic calibrator on the `tune` partition, which this module defined
as "the model/method/alpha selection set." The calibrator was therefore fitted
on the selection set. That affects every calibrated number the project reports.

### 3.2 Why a schema rather than a fifth fraction field

`PARTITIONS` was a hard-coded constant referenced in five places, beside four
separately-named fraction fields and a hand-written sum check. A fifth partition
meant six coordinated edits; a sixth would mean six more.

The partition set is now **data**: `PartitionSchema` holds an ordered tuple of
`Partition(name, fraction, role)`, and every function derives behaviour from it.
`FIVE_WAY` adds a dedicated `calib` partition with role `CALIBRATE_PROBABILITY`.

`train.py` will ask `rows_for_role(CALIBRATE_PROBABILITY)` rather than naming a
partition. Under `FOUR_WAY` that returns `None` -- the honest answer. A silent
fallback to `tune` would re-create the defect the schema exists to remove.

### 3.3 Backward compatibility was proven, not asserted

The previous revision was loaded alongside the new one and bucket assignments
compared directly:

| Check | Result |
|---|---|
| Hash assignment, seeds 42 / 7 / 123 / 2026 | identical |
| Hash assignment, non-default fractions | identical |
| `group_shuffle` on uniform gene sizes | identical |
| `tests/test_split_protocol_v2.py` | 10 passed, unchanged |
| `real_data_prep.py`'s three imported names | all resolve |

Had one gene moved bucket, stability invariant I8 would have broken across the
upgrade and no result computed before today would be comparable to one computed
after.

### 3.4 Two defects repaired in the same pass

**`group_shuffle` carve() rescaled by ROW counts** while scikit-learn's
`GroupShuffleSplit` interprets `test_size` as a proportion of **groups**.
Verified empirically: ten genes, one holding 100 of 109 rows, `test_size=0.30`
carved exactly 3 of 10 genes -- 30 per cent of groups, 3 per cent of rows.

The row ratio is an unbiased estimator of the gene ratio, so it is usually
close. Its variance grows with gene-size skew, and it **overflows** when a
high-row-count gene is carved early. Mean worst-partition deviation, 12 seeds:

| Gene-size skew | By rows | By genes |
|---|---|---|
| Uniform | 0.0000 | 0.0000 |
| Heavy-tailed (Pareto) | 0.0329 | 0.0000 |
| One gene = 90 per cent of rows | 0.0434, **3/12 crash** | 0.0065, 0/12 |

ClinVar per-gene variant counts *are* heavy-tailed, so the middle row is the
realistic regime and the bottom row is reachable.

**`genes_are_stable_under_growth` had an if/else whose branches were identical**,
under a comment implying they differed. Dead code removed.

### 3.5 Recorded honestly

The carve defect was hypothesised from reading, then **tested and refuted**
(worst deviation 0.013, better than hash mode at 0.038), and only found on a
wider sweep over twelve seeds -- the first adversarial test used a seed that did
not trigger it. Reading produced the hypothesis; only running it across enough
cases produced the evidence.

### 3.6 A forensics script repaired rather than appeased

`scripts/forensics/verify_dtype.py` line 30 asserted the literal source text
`"for p in PARTITIONS:"`, which the schema-driven loop no longer contains.

That check verified code *looked* a certain way, not that the dtype promotion
*worked*: a clean refactor failed it, and a silent breakage with the loop text
intact would have passed it. **Both directions wrong.**

Replaced with a behavioural check -- int32 fixture, `FutureWarning` escalated to
an error, real split, real remap, assert int64 out -- plus checks that every
partition **in the schema** is remapped and that genes unseen in train score
zero (incident 2026-06-13).

**Proven by sabotage:** removing the promotions fails 5 checks; reintroducing the
leak is caught with **2,217 violations**. The old text check passed both.

---

## PART FOUR -- PANEL Q (`5dcb932`)

### 4.1 The gap, verified

The metric specification has sixteen panels, A through P. Verified against
`project_metrics.txt` (SHA-256 `db987039...`): Davies-Bouldin, silhouette,
Calinski-Harabasz and Dunn appear **zero** times; adjusted Rand index, adjusted
mutual information and normalized mutual information appear nowhere as agreement
metrics. The only "cluster" occurrences are gene-cluster bootstrap, a resampling
unit.

Panel Q asks whether a **representation** has coherent structure. It says nothing
about whether predictions are correct, and nothing it produces may be reported as
clinical superiority.

### 4.2 Five failure modes closed by construction

**(1) The silhouette guard refuses before allocating**, and the limit is
expressed in **memory** rather than a sample count, because memory is the
constraint and a count silently assumes a dtype:

    maximum_distance_matrix_gib=3.0 -> floor(sqrt(3*2^30/8)) = 20,066

Measured on 1280-dimensional Evolutionary Scale Modeling version 2 embeddings:

| n | time | memory |
|---|---|---|
| 2,000 | 0.32 s | 0.03 GiB |
| 10,000 | 7.53 s | 0.75 GiB |
| 30,000 | 42.44 s | 1.00 GiB |
| 60,000 | -- | ~27 GiB predicted |
| 1,500,000 | -- | 16.4 TiB predicted |

**(2) No silent not-a-number.** Enforced in `__post_init__`, not only in the
factory -- an earlier draft guarded one of two construction paths.

**(3) Three exclusion routes, counted separately, in a fixed order.** A row
qualifying for two counts once, in the earliest. Reconciliation alone cannot
detect a reordering because the total is unchanged either way.

**(4) The two Davies-Bouldin geometries are different estimands.** Measured on a
directional fixture: 2.9188 Euclidean-on-normalized against 1.3538 spherical,
with cluster mean norms near 0.47.

**(5) The confounder gate is pure and fail-closed.** `decide_confounder_gate()`
takes estimates and returns a verdict with no randomness; comparison is on
intervals, strictly: covariate upper 0.40 against target lower 0.40 refuses,
0.3999 passes.

### 4.3 The permutation null respects dependence

Variants within a gene share a covariate, so permuting rows barely disturbs the
association. Measured:

| Null unit | 95th percentile |
|---|---|
| Gene-level (correct) | 0.0516 |
| Row-level (naive) | 0.0011 |

A **47-fold** difference. This follows the logic of the `n_pathogenic_in_gene`
permutation ablation, which established that feature's contribution against a
permuted null (observed area under the receiver operating characteristic curve
0.9666 against a permuted 95th percentile of 0.8016).

### 4.4 Sabotage results

| Sabotage | Result |
|---|---|
| Drop the reason enforcement | 1 failed |
| Reverse the exclusion order | 1 failed |
| Spherical Davies-Bouldin becomes Euclidean | 1 failed |
| Permuter shuffles rows, not gene blocks | 2 failed |
| Gate compares point estimates | 3 failed |
| Remove the memory ceiling | 1 failed |
| Restored | 78 passed |

### 4.5 A check that could not fail

The first version of the silhouette-guard test used a 500-row fixture, where
`effective = min(100000, 500) = 500` is legitimately below the ceiling. Nothing
could trigger the refusal. Rewritten against a 50,000-row cohort.

---

## PART FIVE -- THE INSTRUMENT THAT MEASURED THE DISK (`0021a72`)

### 5.1 Three wrong answers, no tests, and no home

| Version | Date | Reported for `data/` |
|---|---|---|
| v1 | 2026-07-20 | 161.38 GiB / 15,260 files |
| v2 | 2026-07-20 | every subdirectory 0.0 MiB -- budget expired, zeros printed as measurements |
| v3 | 2026-07-21 | **3.21 GiB** -- against a true 98.75 GiB measured the same hour |

Each fix was correct as far as it went. None shipped with a test. And the script
existed only at `C:\Users\monzi\Downloads`, so the capability that resolved the
2026-07-20 drive emergency was one folder-clear from being lost.

### 5.2 Defect 5, characterised by prediction

`Walker` keeps visited sets keyed on `(st_dev, st_ino)` so a **census** never
counts an overlapping root twice -- without it the legacy junction
`C:\Documents and Settings` made the census report 5.94x the volume size.

`data_breakdown()` reused the **same** Walker that had already walked the whole
volume. Every subdirectory of `data/` was in `_seen_dirs`, and `size_of()` skips
any child found there while never checking the root it is handed. So each
subtotal counted **only files sitting loose in that directory**:

| Directory | True | In subdirs | Loose only | v3 reported |
|---|---|---|---|---|
| `data/external` | 75.18 GiB | 75.18 GiB | 0.00 GiB | 0.003 GiB |
| `data/processed` | 3.50 GiB | 0.54 GiB | 2.96 GiB | 2.950 GiB |
| `data/raw` | 19.80 GiB | 19.80 GiB | 0.00 GiB | 0.000 GiB |
| `data/_drift_check` | 0.26 GiB | 0.00 GiB | 0.26 GiB | 0.264 GiB |

Four for four. The mechanism was characterised by **predicting each value before
reading the code**, not inferred afterwards.

The de-duplication was never the bug. Applying **census** semantics to a
**standalone measurement** was.

### 5.3 The repair

`size_of()` now takes an explicit `independent` flag and the two semantics are
**named** rather than implied. Cycle safety holds in both modes; only the scope
of the visited sets differs, and `hardlink_savings` is not mutated in independent
mode because that statistic describes the census.

### 5.4 Worse than wrong

The section printed *"Compare the figure above against 161.38 GiB"* directly
beneath its own bad number. A reader concludes 158 GiB was reclaimed. Nothing
was. That line is replaced by the section's own error history, and a test asserts
the old line is gone.

### 5.5 Confirmed against an independent implementation

After the fix, `audit_disk_census.py` and the separately-written
`diagnose_storage_2026-07-21.py` agree to the megabyte and to the exact file
count in every directory:

| Subdirectory | diagnose_storage | audit_disk_census |
|---|---|---|
| `external` | 75.18 GiB / 3,311 files | 75.18 GiB / 3,311 files |
| `raw` | 19.80 GiB / 43 files | 19.80 GiB / 43 files |
| `processed` | 3.50 GiB / 47 files | 3.50 GiB / 47 files |
| `_drift_check` | 0.2640 GiB / 2 | 0.2641 GiB / 2 |
| `_pandas3` | 0.0041 GiB / 130 | 0.0041 GiB / 130 |
| **subtotal** | **98.75 GiB** | **98.75 GiB** |

### 5.6 Sabotage results

| Sabotage | Result |
|---|---|
| Call site drops `independent=True` | 2 failed |
| Independent shares the directory set | 5 failed |
| Independent leaks into the shared set | 5 failed |
| **Shared de-duplication removed entirely** | **2 failed** |
| Restored | 13 passed |

The last line matters as much as the first: the fix cannot be "achieved" by
discarding the de-duplication it depends on.

### 5.7 Renamed

From `audit_disk_reclaim_v3_2026-07-20.py`. A committed tool must not carry a
version number and a date in its filename; the project's own data-layout standard
forbids version suffixes on directory names for the same reason. "Reclaim" was
also inaccurate -- it deletes nothing.

---

## PART SIX -- A FINDING I REPORTED, AND THEN RETRACTED

### 6.1 What I claimed

After the `5b1c82b` suite run I reported that the suite had grown from roughly
ten and a half minutes to nearly nineteen across the day, that the 44 new tests
ran in 11.78 seconds standalone, and that **"roughly 228 seconds is unaccounted
for."** I called it a finding and asked for a `--durations` measurement.

### 6.2 Why it was wrong

The same commit `5b1c82b` subsequently ran in **894.88 seconds** against the
**1131.67 seconds** I had reported -- a 236.79-second spread, 26.5 per cent, on
byte-identical code. The "unaccounted" 228 seconds was smaller than the noise.

**I reported a finding from a single measurement.** The correct response to one
unusual timing is a second measurement, not a hypothesis.

### 6.3 The full record, nine measurements

| Ratchet | Seconds | Minutes | Milliseconds per test |
|---|---|---|---|
| 2071 | 632.92 | 10.55 | 305.6 |
| 2071 | 648.62 | 10.81 | 313.2 |
| 2071 | 605.08 | 10.08 | 292.2 |
| 2128 | 894.68 | 14.91 | 420.4 |
| 2148 | 891.69 | 14.86 | 415.1 |
| 2192 | 1131.67 | 18.86 | 516.3 |
| 2192 | 894.88 | 14.91 | 408.2 |
| 2270 | 768.34 | 12.81 | 338.5 |
| **2283** | **728.76** | **12.15** | **319.2** |

Range 605.08 to 1131.67 seconds: a **1.87x** ratio. The highest test count ever
recorded (2283) produced the second-fastest run ever recorded. Per-test time
spans 292 to 516 milliseconds and moves with the **run**, not the **count**.

**Conclusion: there is no growth trend. There is host variance of roughly 1.9x,
which swamps any signal from adding tests.**

### 6.4 What the durations did show

Concentration, measured on the 894.88-second run:

| | Seconds | Share |
|---|---|---|
| Top 25 tests | 608.97 | 68.1 per cent |
| Remaining ~2,167 tests | 285.91 | 31.9 per cent (0.132 s each) |
| **`test_drift_reference_profile.py` alone** | **271.08** | **30.3 per cent** |

Two tests -- `test_ks_and_wasserstein_are_flagged_approximate_only_in_profile_mode`
at 147.24 seconds and `test_per_feature_action_is_identical` at 106.38 seconds --
are 28 per cent of the entire suite. Not addressed this session; recorded.

---

## PART SEVEN -- THE STORAGE AUDIT (MEASURED, NOT ACTED ON)

### 7.1 The volume

| | |
|---|---|
| Total | 935.59 GiB |
| Used | 852.09 GiB |
| Free | 83.50 GiB (8.925 per cent) |
| **Fixed volumes** | **C: only -- there is nowhere else to put anything** |

Below ten per cent free is where Windows behaviour degrades, and this is a
credible mechanism for the 1.87x suite variance recorded in Part Six. The two
problems may be one problem.

### 7.2 The project data footprint

`data/` is **98.75 GiB**, which is *larger than the free space*. There is no
headroom to stage, duplicate, or rebuild any large artifact in place.

| Directory | GiB | Files | Share of `data/` |
|---|---|---|---|
| `external/finngen` | **57.64** | **3** | **58.4 per cent** |
| `raw/cache` | 19.62 | 41 | 19.9 per cent |
| `external/eve` | 9.92 | 3,211 | 10.0 per cent |
| `external/grch38` | 3.76 | 3 | 3.8 per cent |
| everything else | 7.81 | 279 | 7.9 per cent |

### 7.3 What the audit established about test dependencies

A dependency scan produced **false positives in both directions** and must not be
treated as an answer:

- Every FinnGen hit was synthetic. `test_audit_run17_assets.py` takes `tmp_path`
  and passes it as the audit root, so `data/external/finngen/...` are relative
  strings resolved against a temporary directory. **FinnGen has zero test
  dependencies** -- established by reading the code.
- `1000g`, `reactome` and `gtex` hits are string assertions on a generated
  command, plus a class whose HTTP calls are all mocked. None requires the
  directory to exist.
- `data/raw/cache` is **never read**. It is fingerprinted by `conftest.py` to
  detect tests writing into the repository, after the 2026-07-11 incident where
  run 1 gave 1805 passed / 17 skipped and run 2 gave 1812 / 10 on the same
  checkout.

### 7.4 Governance already exists

`docs/standards/DATA_LAYOUT_STANDARD.md` and `configs/data_manifest.yaml` (347
lines) already define per-source `tier`, `class`, `sync`, `acquire` and
`regenerate`, with `setup_data_tree.py`, `audit_data_tree.py`,
`sync_data_to_gdrive.py` and `preflight_data_guard.py` reading the manifest.
Building a parallel policy would have been patchwork.

The manifest classifies the decisive item:

```yaml
finngen:
  tier: public
  class: public_redownloadable
  sync: false
  acquire: "https://www.finngen.fi/en/access_results"
```

Public, re-downloadable, deliberately not mirrored to Google Drive. FinnGen is
nonetheless **actively wired** -- 45 files reference it, including
`src/genomic_variant_classifier/data/finngen.py`, `train.py`,
`variant_ensemble.py` and the Run 17 launchers.

### 7.5 The one real gap in existing tooling

`preflight_data_guard.py` is 52 lines. It verifies `data/` is a real directory,
not a dangling junction, with `external`, `raw` and `processed` present. **It
does not check free space at all.** The guard that exists to catch storage
problems before a run cannot catch the storage problem the machine has.

### 7.6 Non-project consumers, measured

| Item | GiB |
|---|---|
| `C:\Projects\genomic-variant-classifier` | 112.93 |
| `docker_data.vhdx` | 73.49 |
| `C:\Users\monzi\OneDrive` | 31.00 |
| `C:\pagefile.sys` | 25.65 |
| `AppData\Local\Packages` (incl. Ubuntu `ext4.vhdx` 16.71) | 21.89 |
| `AppData\Local\Programs` | 14.00 |
| `Windows\WinSxS` | 8.35 |
| `AppData\Local\pip\Cache` | 5.53 |
| `anaconda3` | 4.21 |
| `System Volume Information` | 4.00 |
| `.cache\huggingface` | 2.82 |
| `C:\cabal` | 1.00 |
| `AppData\Local\Temp` | 0.58 |

**Genuinely disposable temporary files total roughly 6.5 GiB**, against 57.33
GiB needed to reach 15 per cent free. Temporary-file cleanup delivers about 29
per cent of the target when pushed to include component-store cleanup, the
Hugging Face cache and restore points. It helps; it does not solve.

`hiberfil.sys` is absent -- reclaimed 2026-07-20. `pagefile.sys` is present at
25.65 GiB; an earlier diagnostic of mine reported it as "absent or inaccessible",
an ambiguous message that conflated two very different states. That was a defect
in my script, now known to be the second case.

### 7.7 Two documentation drifts found

`tests/conftest.py` states that `data/raw/cache` **"holds 36,202 files (36,074 of
them AlphaFold structures -- the 8.77 GB cache"**. It measures **41 files, 19.62
GiB**. The comment's cost arithmetic -- 132 million `stat()` calls, a 58x runtime
penalty -- reasons about a directory that no longer exists in that form.

`docs/STORAGE_ACTION_LEDGER_2026-07-03.md` is cited as the record of that cache
and has not been updated through either the 2026-07-20 reclamation or today.

### 7.8 No storage decision was taken

Deliberately. Every such decision depends on numbers the census produces, and the
census was wrong until `0021a72`. The instrument was repaired first.

---

## PART EIGHT -- OPEN ITEMS

### 8.1 Immediate, from this session

1. **Free-space check in `preflight_data_guard.py`**, thresholds from the
   manifest, fail-loud, with tests. The one real gap in existing tooling.
2. **The storage decisions**: Docker 73.49 GiB, FinnGen 57.64 GiB, OneDrive Files
   On-Demand 31.00 GiB (reversible, deletes nothing), pagefile cap. The first two
   of those alone reach roughly 46 GiB with nothing deleted.
3. **`PartitionRole.STRUCTURE`** -- Panel Q's partition policy requires a
   gene-disjoint structure partition so cluster discovery never touches the
   locked test set. Bump `PARTITION_SCHEMA_VERSION`; rename `FIVE_WAY`, since a
   constant named for a count it no longer has is the same stale-name problem
   repaired four times today.
4. **The Panel Q orchestrator** (`evaluate_panel_q`, `StructureEvaluationMode`),
   third, once it has a legitimate partition to run on.
5. **Wire `train.py` to `rows_for_role(CALIBRATE_PROBABILITY)`** and flip
   `--split-protocol` off `legacy` (train.py line 126). Behavioural; own commit,
   own test.
6. **Class-conditional (Mondrian) ordinal conformal**, to fix the extreme-tier
   under-coverage measured in Part One.
7. **`test_drift_reference_profile.py`** at 271.08 seconds, 30.3 per cent of the
   suite, in three tests.
8. **Correct the `conftest.py` cache comment** (36,202 files against a measured
   41) and update `STORAGE_ACTION_LEDGER`.

### 8.2 Carried from earlier

- Full metric stack (Panels A-P plus Q) re-arms five skipped tests.
- Own rclone Google Drive client identifier -- dated risk, the shared one is
  being retired during 2026.
- `data/external/gtex` and `data/rnaseq` empty: hard blocker on ribonucleic acid
  infrastructure. `phylop_score`, `esm2_llr` and `eve_score` remain stub features
  returning constants pending the HGVSp parser.

### 8.3 Suite state at close

2283 collected, 2276 passed, 7 skipped. The seven skips are unchanged in identity
all session: five in `test_mc_dropout_calibration.py` awaiting metric-stack
infrastructure, one POSIX-symlink platform skip, one degenerate-fixture skip in
`test_tabular_nn_mc_dropout.py`.

Skip *positions* were verified on every commit, not merely their count. The
pattern differed by commit according to alphabetical insertion point, and matched
prediction each time: `5b1c82b` shifted them +0 and +44 (the new file sorts
between them); `5dcb932` and `0021a72` shifted both by the full +78 and +13 (the
new files sort before both).
