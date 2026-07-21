# SESSION 2026-07-20 -- Part Four: The Run-16 Gate, and Eighty-Five Gigabytes

**Date:** 2026-07-20 (continuing into the early hours of 2026-07-21 Coordinated Universal Time)
**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `0208998f7fb991b0a4c776c351797af46f662280`, authored 2026-07-20 11:09:17 -0400,
"docs(session): 2026-07-20 part three -- the calibration surface"
**Suite at start:** 2060 collected
**Suite at end:** 2071 collected; 2063 passed, 8 skipped
**Free space at start:** 6.70 gibibytes (0.716 per cent of volume)
**Free space at end:** 91.82 gibibytes (9.81 per cent of volume)

---

## PART ZERO -- WHAT WAS ASKED, AND WHAT ACTUALLY HAPPENED

The session opened with an instruction to implement five deliverables: the Joint
Embedding Predictive Architecture (JEPA), conformal prediction, the ribonucleic
acid (RNA) infrastructure, the full metric stack, and conformal quantile
regression.

**None of the five was implemented.** What happened instead, in order:

1. A ground-truth audit of the live repository found that the Run-16 input
   preflight gate had been returning a failure exit code on **every invocation,
   on a clean tree**, for an extended period. This was repaired and tested.
2. Establishing whether JEPA could proceed required measuring free disk space.
   That measurement found the working volume at **0.716 per cent free**, which is
   an operational hazard to the entire project rather than a JEPA-specific
   blocker.
3. Settling the drive consumed the remainder of the session. It required four
   successive versions of an audit tool, because the first three each carried a
   defect of my own making. All four defects are documented in Part Three.
4. **85.12 gibibytes were reclaimed and verified.** The drive is settled.

This ordering was not a drift of attention. Monzia directed it explicitly:
"We *must* settle the local/C drive memory issue before anything else." It is
recorded here so that the next reader does not mistake the absence of the five
deliverables for neglect of them.

---

## PART ONE -- THE RUN-16 PREFLIGHT GATE WAS FAILING UNCONDITIONALLY

### 1.1 The defect

`scripts/preflight_run16_inputs.py` is the input gate for Run-16 cohort
regeneration. It carries five checks and uses the exit convention 0 for
all-pass, 2 for any-fail, 3 for environment problems.

Its fourth check read:

```python
EXPECTED_COUNT = 81
...
ok = C == EXPECTED_COUNT
```

where `C` is `EXPECTED_TABULAR_FEATURE_COUNT` imported from
`src/genomic_variant_classifier/models/variant_ensemble.py`.

Measured on 2026-07-20 against commit `0208998`, by abstract syntax tree parse
(the module imports XGBoost, LightGBM, CatBoost and PyTorch at module scope, so
it cannot simply be imported for inspection):

```
EXPECTED_TABULAR_FEATURE_COUNT = 95
len(TABULAR_FEATURES)          = 95
duplicates                     = 0
```

So the check evaluated `95 == 81`, returned False, and the aggregate returned
exit code 2. **The entire Run-16 input preflight could not pass.** Not on a
broken tree -- on a clean one, every time.

The tabular feature contract advanced 81 -> 88 -> 91 -> 95 through
`scripts/patch_ve_maxentscan_delta.py`, `scripts/patch_tabular_omim_molecular.py`
and `scripts/patch_variant_ensemble_finngen_r13.py`. The gate never followed.

### 1.2 Why it survived review

`tests/unit/test_preflight_run16_inputs.py` covers `check_cohort_ref_alt` (five
tests), `aggregate`, `check_exists`, `check_gnomad_constraint` (four tests) and
`check_cohort_reviewstatus` (three tests). It **never calls
`check_feature_count`**.

The single untested check in the file was the single check that drifted. This
is the pattern recorded in the 2026-07-20 handoff section 6.5, now with another
instance: outcome-asserting checks catch what careful reading does not.

The practical damage is larger than one broken check. A fail-loud guard that
fails for a false reason is worse than no guard at all, because it trains the
operator to ignore the four checks beside it that are correct.

### 1.3 The repair -- drift-proof by construction, not by being re-pinned

Setting the literal to 95 would have reproduced the defect at the next roster
growth. `check_feature_contract()` instead asserts two properties, neither of
which is a magic number:

1. **The fail-loud contract invariant.** `EXPECTED_TABULAR_FEATURE_COUNT ==
   len(TABULAR_FEATURES)`, with no duplicate names. This is the project's own
   stated guard, restated at the gate, and it never needs editing when the
   roster grows.
2. **Membership by NAME** of the three features the gate's other checks exist to
   protect: `esm2_llr` (guarded by the ESM-2 UniProt index check),
   `maxentscan_delta` (guarded by the `fasta_seq_ref` / `fasta_seq_alt` cohort
   check) and `gene_constraint_oe` (guarded by the Genome Aggregation Database
   constraint check).

A count never told us whether the right features were present. Names do.

`check_feature_count()` is retained as a delegating alias. Silently deleting a
public name is a loss this project has had to reverse before -- see the
restoration of `metrics.py` from `87e32ad^` recorded in the part-three handoff.

### 1.4 Verification

All five branches were outcome-asserted against the real 95-name contract:

| Case | Contract supplied | Result | Exit |
|---|---|---|---|
| A | real live contract, count 95 | PASS | 0 |
| B | count 81, list 95 | FAIL -- reports both numbers | 2 |
| C | duplicate name, count still "matches" | FAIL -- names the duplicate | 2 |
| D | `esm2_llr` unregistered | FAIL -- names the feature | 2 |
| E | import fails | ENV -- not conflated with data failure | 3 |

**Case C is the one that justifies the shape of the repair.** A duplicated
feature name lets the count agree while a feature is silently lost. The
pre-repair, count-only gate **passed** that case.

Eleven tests were added at `tests/unit/test_preflight_run16_feature_contract.py`.
They inject synthetic contracts into `sys.modules` before import, which is the
only way to exercise a module that imports four heavyweight machine learning
libraries at module scope, and the only way to construct contracts that must not
exist in production.

**Recorded honestly:** the first draft of the module-restore fixture in that
file wrote `del sys.modules[m]` using a comprehension variable, which does not
leak in Python 3. All eleven tests reported PASSED while teardown raised
`NameError`. It was found by running the file, not by reading it.

### 1.5 Delivered artefacts

| File | SHA-256 |
|---|---|
| `install_preflight_run16_feature_contract_2026-07-20.py` | `deb8e95081d834afcd7453762c68854400e2076d84e84310489aef272ed40966` |
| `tests/unit/test_preflight_run16_feature_contract.py` | `1bfd0476bb8f2785ba64a6130776176c72f69f8ad1454470fb40399e657897da` |

Installed on Monzia's machine 2026-07-20 11:43:49. All six post-checks OK.
Backup written to `preflight_run16_inputs.py.bak_20260720_114349`.
Verification: **25 passed** (11 new plus 14 pre-existing) in 5.75 seconds on
Python 3.12.10, pytest 9.0.3.

---

## PART TWO -- THE DISK EMERGENCY

Determining whether JEPA could begin required knowing free disk space. The
measurement, taken 2026-07-20:

```
total  1,004,584,038,400 bytes   935.59 GiB
used     997,387,636,736 bytes   928.89 GiB
free       7,196,401,664 bytes     6.70 GiB   0.716 %
```

Below roughly five per cent free, Windows cannot reliably grow the pagefile,
`git gc` and `git checkout` can fail mid-operation, and pytest fixtures that
write temporary Parquet files -- this suite has several -- fail for reasons that
look like code defects and are not. At 0.716 per cent this was not a JEPA
blocker. It was a hazard to everything.

Free space was also observed to fall by roughly four gibibytes during the day,
and by a further 31.2 mebibytes and then 4.3 mebibytes between successive
readings taken minutes apart.

---

## PART THREE -- FOUR DEFECTS IN MY OWN AUDIT TOOLS

Four successive versions were required. Each defect is recorded with the
evidence that exposed it, because the pattern across all four is the same and is
worth more than any individual fix: **a measurement and a non-measurement must
never be formatted identically.**

### 3.1 Version 1 -- `83e5d048e36c33295b4ae5bf3ec77a710c558b733ea8d3ab06809882512c27f1`

**Defect 1: the verdict ignored operational headroom.**

Version 1 printed `JEPA embedding cache needs ~14.7 GiB : SUFFICIENT`. Its
arithmetic was correct and its conclusion was wrong. Clearing everything it
found would give 19.01 gibibytes free; allocating a 14.7 gibibyte cache from
that leaves 4.31 gibibytes, or 0.46 per cent of the volume -- while the same
script's header prints a warning whenever free space falls below five per cent.

The check compared against the artefact size alone. A verdict that contradicts
the tool's own warning threshold is not a verdict.

Corrected requirement: artefact size plus a floor of the greater of five per
cent of the volume or 20 gibibytes. On a 935.59 gibibyte volume that floor is
46.78 gibibytes, so the requirement is **61.48 gibibytes** and the true deficit
at that moment was **54.78 gibibytes**, not the "sufficient" reported.

**Defect 2: the census covered one fifth of the problem.**

Version 1 scanned only the repository and the Downloads folder:

```
repository root      175.45 GiB
Downloads              0.80 GiB
user caches           11.60 GiB
---------------------------------
accounted for        187.85 GiB
volume reports used  928.89 GiB
UNACCOUNTED          741.04 GiB   (79.8 per cent of used space)
```

Nothing could be decided while four fifths of the volume was unexamined.

### 3.2 Version 2 -- `fb6e53b0dedc55dd0df2b317eea3821ce0471078a998ae69a08126e8f45134e3`

**Defect 3: Windows directory junctions were followed. (Critical.)**

Version 2's walker guarded against cycles with `os.DirEntry.is_symlink()`, and
its docstring claimed it "never follows links or reparse points", naming Google
Drive File Stream junctions specifically. **That claim was false.**

On Windows a symbolic link carries reparse tag `IO_REPARSE_TAG_SYMLINK` and
`is_symlink()` detects it. A **directory junction** carries
`IO_REPARSE_TAG_MOUNT_POINT` and `is_symlink()` returns `False`. Windows ships
legacy compatibility junctions that are therefore invisible to that check, two of
which point at their own parent:

```
C:\Documents and Settings                       -> C:\Users
C:\Users\monzi\Local Settings                   -> C:\Users\monzi\AppData\Local
C:\Users\monzi\AppData\Local\Application Data   -> itself     (SELF-REFERENTIAL)
C:\ProgramData\Application Data                 -> itself     (SELF-REFERENTIAL)
```

Measured consequences on a 935.59 gibibyte volume:

| Reported | Value | Reality |
|---|---|---|
| `C:\Documents and Settings` | 5557.03 GiB | 5.94x the entire volume |
| `C:\ProgramData` | 2039.15 GiB | 2.18x the entire volume |
| census total | 5593.90 GiB | 5.98x the entire volume |
| reconciliation difference | **-502.2 per cent** | negative; physically impossible |

The same 73.49 gibibyte file `docker_data.vhdx` appeared **thirty times** in the
largest-files list, each occurrence one `Application Data` level deeper than the
last -- 2204.70 gibibytes of double-counting from a single file. The walk
processed 31,083,584 files and ran to exactly 1800.0 seconds, the configured time
budget. **It never terminated on its own.**

**Defect 4: after the time budget expired, zeros were reported as measurements.**

Version 2's deadline check caused `size_of` to break immediately and return
`(0, 0)`. Every section running after the budget was exhausted printed confident
zeros. The repository data directory printed fifteen subdirectories at
"0.0 MiB, 0 files" -- a directory version 1 had measured at 161.38 gibibytes
across 15,260 files. Duplicate detection reported "none found". These were not
measurements; they were the absence of measurement, formatted identically to one.

### 3.3 Version 3 -- `e23353ce2eaa92a45091c49dd1fa7bac09e24ece5b77f6286151da4b36e2d785`

Three independent guards were added, because one is insufficient:
`is_junction()` (available in Python 3.12, which this project runs), the
`FILE_ATTRIBUTE_REPARSE_POINT` attribute bit, and `is_symlink()`; plus a visited
set keyed on device and inode number; plus a hard depth ceiling.

Verified by emulating Windows junction semantics exactly -- forcing
`is_symlink()` to return `False` on a self-referential directory, so that only
the cycle guard could prevent non-termination:

```
elapsed        : 0.00 s      (version 2 ran 1800 s and never finished)
complete       : True
size           : 305.2 MiB   exactly the constructed size
hardlink saved : 19.1 MiB    hard link not double-counted
```

The version 3 run on Monzia's machine was clean: **656,659 files, 121,909 unique
directories, 3,656 reparse points skipped, 0 unreadable paths, no timeout, 60.5
seconds.** Inspection of the skipped reparse points confirmed they were
legitimate: three Docker sockets, and Hugging Face snapshot symbolic links --
which is why that cache reports 2.82 gibibytes across only 65 real blob files.

**Defect 5: cross-measurement contamination of the visited set.**

Version 3 nonetheless reported the repository data directory as 3.22 gibibytes
across 50 files, against version 1's 161.38 gibibytes across 15,260 files.

My first diagnosis -- that the breakdown iterated only directories and missed
loose files -- was **wrong**. The true cause is worse, and the evidence is exact:

| Directory | Version 3 reported | Inventory: loose files at that level | Inventory: true total |
|---|---|---|---|
| `external` | 2.8 MiB / 5 files | **2.8 MiB / 5 files** | 137.81 GiB |
| `processed` | 2.95 GiB / 38 files | **2.95 GiB / 38 files** | 3.50 GiB |
| `_drift_check` | 270.4 MiB / 2 files | **270.4 MiB / 2 files** | 270.4 MiB |
| `raw` | 0.0 MiB / 1 file | **0.1 KiB / 1 file** | 19.80 GiB |

Every version 3 figure equals *exactly* that directory's loose-file count.

The `Walker` keeps **one visited set for its whole lifetime**. The targeted
checks measured the entire repository first, registering every directory beneath
it. When the data breakdown ran later on the same walker, every subdirectory was
already marked visited and was skipped, leaving only the loose files at each
level. **A cycle guard that is correct within one walk silently suppressed a
legitimate second measurement of the same subtree.** The visited set must be
per-measurement, not per-walker.

### 3.4 The inventory tool -- `947ec5762f86b3f263310bcca0eae1f342b84024ffb3dd120eb945d69f3f6841`

A dedicated tool was written that reports loose bytes and subdirectory bytes
**separately at every level**, so the omission that produced defect 5 cannot
recur silently.

**A sixth defect, caught before delivery: the accounting check was a tautology.**

The first draft asserted `total == loose + children` and printed `HOLDS`. But
`total` is *defined* as that sum. The assertion could never fail, yet it printed
a verdict implying verification had occurred. **A check that cannot fail is
worse than no check, because it manufactures confidence.**

It was replaced with an independent second walk -- a flat traversal sharing no
code path with the tree walk -- and then proved genuine by deliberately
reintroducing version 3's defect into the tree walk. The two methods disagreed
(51,000,000 against 651,000,000 bytes) and the disagreement was reported.

On Monzia's machine both methods returned **161.38 gibibytes across 15,260
files**, in agreement, in 1.0 second, with zero reparse points and zero
unreadable paths.

---

## PART FOUR -- WHAT THE 161.38 GIBIBYTES ACTUALLY WERE

The loose files directly in `data\` totalled **13.1 kibibytes in one file**. The
volume was in `data\external\` at 137.81 gibibytes and `data\raw\` at 19.80
gibibytes.

### 4.1 Three assumptions checked against the code, one of which was destructive

Before proposing any removal, each candidate was checked against
`src/genomic_variant_classifier/monitoring/registry.py` and
`src/genomic_variant_classifier/models/variant_ensemble.py`.

| Item | Size | Registry verdict | Finding |
|---|---|---|---|
| `external\finngen\` | 57.64 GiB | **ACTIVE** | R12 **and** R13 both feed features |
| `external\eve\` | 63.36 GiB | **CACHE** | runtime path is `data/raw/cache/eve_eve_lookup.parquet` |
| `external\phylop\` | 9.19 GiB | **STUB** | path `None`; connector not wired |
| `raw\cache\esm2_cache.sqlite` | 18.11 GiB | live default in `esm2.py:80` | must keep |

**The FinnGen assumption was wrong and acting on it would have broken the
model.** I had reasoned that Release 12 was superseded by Release 13.
`variant_ensemble.py` lines 451-461 register **six** FinnGen features -- three
from R12 (`finngen_af_fin`, `finngen_af_nfsee`, `finngen_enrichment`) and three
from R13 (`finngen_r13_af_fin`, `finngen_r13_af_nfsee`, `finngen_r13_enrichment`).
`registry.py:114` marks the R12 file ACTIVE and notes it is "74% of corpus".
Deleting those 29.92 gibibytes would have destroyed three of the ninety-five
features. The filename typo `finnge_R12_annotated_variants_v1.gz` is already
documented at `registry.py:116` and is therefore tracked, not new.

### 4.2 Two findings unrelated to disk

**`data\external\gtex\` and `data\rnaseq\` are both completely empty.** The
part-three handoff listed Genotype-Tissue Expression (GTEx) connector coverage as
"unmeasured". The reason is now established: **there is no GTEx data on this
machine at all.** This is a hard blocker on the RNA infrastructure deliverable
and was invisible until this inventory.

**`phylop_score` is a live feature backed by a stub.** It appears in
`TABULAR_FEATURES` at line 355 with a default of 0.0 at line 746, while the
registry records the connector as STUB with a null path. It therefore joins
`esm2_llr` and `eve_score` as features in the 95-column contract whose connectors
return constants. **Three of ninety-five features contribute nothing.** This
matters directly to the metric stack deliverable, which would otherwise measure
them as though they were real.

### 4.3 Accumulated debris recorded but not yet addressed

- Eleven near-identical ClinVar cohort Parquet files in `data\processed\`
  (`_clean`, `_clean_v2_verified`, `_clean_v3_verified`, `_pathfix`, `_fresh`,
  `cohort_stale`, `cohort_fresh`, plus a 523.6 mebibyte `.bak_2026-07-18`),
  totalling roughly 1.9 gibibytes.
- `spliceai_index.parquet` present twice at exactly 336.8 mebibytes, in both
  `external\spliceai\` and `processed\`.
- Two `eve_eve_lookup` Parquet files with different content hashes at 423.8
  mebibytes each; only one can be current.
- Twelve empty directories, including the two named in section 4.2.
- `data\_pandas3\` holding eight near-identical fixture sets from a pandas 3
  migration study (4.2 mebibytes, 130 files).
- 5,550 Portable Network Graphics plot files (591.8 mebibytes) inside the EVE
  distribution, now offloaded.

---

## PART FIVE -- THE RECLAMATION, WITH EVIDENCE

### 5.1 Offload, verified by checksum

The canonical store for this project is Google Drive at
`genvarcla:genomic-variant-classifier/data/`. The correct action for build
inputs was therefore offload, not deletion.

An earlier verification script generated with `--size-only` was **withdrawn
before use**. Equal size is not equal content, and that is not an adequate basis
for removing 63 gibibytes of irreplaceable scientific data. The flag was dropped
so that rclone compares MD5 checksums; Google Drive supplies them and rclone
computes the local side by reading the files.

| Directory | Transferred | Result | Exit | Duration |
|---|---|---|---|---|
| `data/external/eve` | 0 bytes | **14,932 files matching, 0 differences** | 0 | 1 min 45 s |
| `data/external/phylop` | 0 bytes | **1 file matching, 0 differences** | 0 | ~20 s |

Both were **already mirrored** to Google Drive; nothing needed uploading. The
EVE check reading 63.36 gibibytes in 105 seconds is consistent with a genuine
local hash pass rather than a size-only shortcut.

### 5.2 Actions taken and their measured effect

| Action | Expected | Outcome |
|---|---|---|
| `powercfg /hibernate off` | 6.29 GiB | succeeded |
| `Remove-Item C:\$GetCurrent` | 4.22 GiB | succeeded |
| Clear `AppData\Local\Temp` | 2.48 GiB | succeeded |
| `Remove-Item C:\Config.Msi` | 0.52 GiB | **FAILED** -- see below |
| Remove `data\external\eve` | 63.36 GiB | succeeded, after checksum verification |
| Remove `data\external\phylop` | 9.19 GiB | succeeded, after checksum verification |

`C:\Config.Msi` failed with `Access to the path 'C:\Config.Msi\7b290ad.rbf' is
denied`. Those are Windows Installer rollback files. Forcing their removal while
an installation is pending can break a rollback. **This was correctly left
alone** and should be handled by `cleanmgr` if wanted.

**A defect in my own instruction is recorded here:** the Temp deletion was given
with `-ErrorAction SilentlyContinue`, which suppressed any error it encountered.
Locked temporary files are routine, so the command could have partially failed
without saying so. That is precisely the silent failure this project forbids, and
I wrote it into the instruction. The subsequent measurement happened to exceed
expectation, which is indirect evidence it worked -- but it is indirect.

### 5.3 The result

| | Free (GiB) | Per cent of volume |
|---|---|---|
| Start of session | 6.70 | 0.716 |
| After the four cleanups | 21.23 | 2.27 |
| **End of session** | **91.82** | **9.81** |
| **Total reclaimed** | **85.12** | |

Against the headroom-aware requirement of 61.48 gibibytes, the margin is **+30.34
gibibytes**. After a 14.7 gibibyte JEPA embedding cache is allocated, 77.12
gibibytes would remain, or 8.24 per cent -- still above the five per cent floor.

**The drive is settled.**

**One unexplained discrepancy, recorded rather than dismissed.** Projected free
space was 93.78 gibibytes (21.23 + 63.36 + 9.19); actual is 91.82, a shortfall of
**1.96 gibibytes**. Three full suite runs of roughly ten minutes each regenerate
`__pycache__` (142.2 mebibytes when last measured), write `.pytest_cache`, and
create temporary Parquet fixtures; the ratchet installer also wrote two backup
files. That accounts for it plausibly. It has **not** been measured, and is
logged as small-and-unexplained.

---

## PART SIX -- THE SUITE, THE RATCHET, AND A REGRESSION WE CAUSED

### 6.1 The ratchet fired correctly and blocked verification

The first attempt to verify the EVE removal produced:

```
no tests ran in 26.16s
SUITE-SIZE RATCHET FAILED (roadmap 6.14)
  expected (tests/EXPECTED_SUITE_SIZE): 2060
  actually collected:                   2071
  11 MORE test(s) than expected.
```

This is the ratchet working exactly as designed. The eleven tests were the ones
added in Part One, and the ratchet had not been bumped in the same commit -- an
omission of mine, flagged at the time and then displaced by the disk emergency.

The consequence that matters is the first line: **collection aborted, so nothing
was verified.**

### 6.2 The bump

The count was **measured on the staged tree**, not assumed:
`--collect-only -q` reported **2071 tests collected in 14.39 seconds**.

Installer `089c88d20fae4922d4ab45dc1bc84f835a351597376f02b30384afd9c141975d`
moved three things in one operation: the final line of
`tests/EXPECTED_SUITE_SIZE` from 2060 to 2071; a new ledger entry above it in the
file's existing format; and the README test badge on line 8. The ratchet and the
badge must never land separately -- a README advertising a different number from
the gate is exactly the class of stale snapshot this project keeps finding.

The edited file was validated against **the repository's own parser**, extracted
verbatim from `tests/conftest.py`, rather than a reimplementation of it. It
returned 2071.

### 6.3 Two green suite runs

| Run | Result | Duration |
|---|---|---|
| After removing `data/external/eve` (63.36 GiB) | **2063 passed, 8 skipped** | 632.92 s (10:32) |
| After removing `data/external/phylop` (9.19 GiB) | **2063 passed, 8 skipped** | 648.62 s (10:48) |
| Skip-reporting run (`-rs`) | **2063 passed, 8 skipped** | 605.08 s (10:05) |

2063 + 8 = 2071, matching the ratchet, so the gate **passed** rather than merely
not firing. The skip positions in the progress output are byte-identical between
runs, indicating stable skips unrelated to the deletions.

### 6.4 The skip census -- and the regression

Eight skips, enumerated with `-rs`:

| Count | Location | Reason | Class |
|---|---|---|---|
| 1 | `test_mc_dropout_calibration.py` | needs Run 15 cohort + gene-family-disjoint split infrastructure | infrastructure |
| 2 | `test_mc_dropout_calibration.py` | needs Spearman correlation infrastructure + real holdout labels | infrastructure |
| 1 | `test_mc_dropout_calibration.py` | needs expected calibration error infrastructure (10-15 reliability bins) + Run 15 holdout | infrastructure |
| 1 | `test_mc_dropout_calibration.py` | requires multiple K-value runs against real cohort | infrastructure |
| 1 | `test_eve_entry_name_resolution.py:217` | EVE variant_files and/or UniProt index not present | **REGRESSION, this session** |
| 1 | `test_preflight_data_paths.py:45` | POSIX symlink stands in for a Windows dangling junction | platform, permanent on Windows |
| 1 | `test_tabular_nn_mc_dropout.py:232` | test corpus does not span both boundary and extreme prediction regions | degenerate fixture |

**Five of the eight are waiting on the metric stack deliverable.** The
`test_mc_dropout_calibration.py` skips name expected calibration error
infrastructure with 10 to 15 reliability bins, and Spearman rank correlation
infrastructure -- both of which fall inside Panel B of the metric specification.
Building the metric stack will re-arm five currently dormant tests.

**One is a regression introduced by this session's work.**
`test_real_corpus_resolution_fraction` is gated on:

```python
@pytest.mark.skipif(
    not (_UNIPROT_INDEX.exists() and _EVE_VARIANT_DIR.exists()),
    reason="EVE variant_files and/or UniProt index not present",
)
```

with `_EVE_VARIANT_DIR = Path("data/external/eve/EVE_all_data/variant_files")` --
precisely the tree removed in section 5.2. It globs the 3,211 EVE score files in
comma-separated-value format and asserts that at least 99 per cent resolve from
entry name to Human Genome Organisation Gene Nomenclature Committee (HGNC)
symbol. `_UNIPROT_INDEX` at `data/external/uniprot/uniprot_human_reviewed.parquet`
still exists (11.1 mebibytes, intact), so the deletion is the sole cause.

**The ratchet did not and could not catch this.** The suite still reports 2063
passed and 8 skipped, because a skipped test is still a **collected** test. A
pass-to-skip conversion is structurally invisible to a collection-count gate.
This is a real gap in the project's guard rail, distinct from anything in the
part-three handoff, and it is the most important thing in this document after the
Run-16 gate itself: **the suite can lose verification coverage without any gate
noticing.**

The remedy is a decision for Monzia, with the numbers stated plainly:

- **Restore only `EVE_all_data/variant_files`** from Google Drive. That is the
  9.94 gibibytes of comma-separated-value files, leaving 81.88 gibibytes free
  (8.75 per cent) -- still far above the 61.48 gibibyte requirement. This re-arms
  a genuine scientific verification for 11.7 per cent of the reclaimed space.
- **Leave it skipped**, accepting that EVE entry-name resolution is no longer
  verified locally, and rely on the offloaded copy when the HGVS protein-level
  (HGVSp) parser lands and EVE becomes live.

The first is recommended. The check exists because EVE filename-to-symbol
resolution is not obvious, and it will matter the moment the connector stops
returning zero.

**RESOLVED 2026-07-21 -- see Part Nine.** The first option was taken. This
section is preserved as written because it records what was known at the time;
Part Nine records what was then done and measured.

---

## PART SEVEN -- OPEN ITEMS

### 7.1 Uncommitted at time of writing

Four paths staged, verified by `git diff --cached --stat`:

```
README.md                                          |   2 +-
scripts/preflight_run16_inputs.py                  |  60 ++++++-
tests/EXPECTED_SUITE_SIZE                          |  42 ++++-
tests/unit/test_preflight_run16_feature_contract.py| 185 +++++++++++++++++++++
4 files changed, 281 insertions(+), 8 deletions(-)
```

This session document belongs in the same commit.

### 7.2 The handoff was never filed

The 2026-07-20 part-three handoff
(`HANDOFF_2026-07-20_calibration-and-deliverable-status.md`, SHA-256
`4d481e5131426aa22c9658142f281c4ecb3020c99265cae72cd84b2c3ae8c78a`, 23,978 bytes,
456 lines) is still only in the Downloads folder. The copy failed because
`docs\handoffs\` did not exist; the directory was then created, but the copy was
never retried.

**`docs\handoffs\` is also the wrong location.** The repository convention is
`docs/status/`, where `HANDOFF_2026-07-15_cowork-to-project.md` lives. The empty
`docs\handoffs\` directory should be removed and the handoff filed under
`docs/status/`. The wrong path was mine.

### 7.3 A dated dependency risk

Every rclone operation emitted:

```
NOTICE: genvarcla: This remote uses rclone's shared Google Drive client_id,
which is being retired and will stop working during 2026.
```

It is July 2026. The canonical data store now holds the **only** copies of 72.55
gibibytes of EVE and phyloP data, and it depends on a credential scheduled to
stop working this year. If it lapses, every operation against `genvarcla:`
fails, **including retrieval**. Creating a private client identifier takes
roughly ten minutes and should not be deferred past this week. This belongs on
the roadmap as a dated dependency risk.

### 7.4 Carried forward, unchanged

- **`split_protocol_v2` exists, is wired, and is tested** -- contrary to the
  part-three handoff, which stated it "is not built". It is a **four-way** split
  (train / tune / conformal / test). The metric specification's strongest design
  requires **five** internal partitions plus an external one, separating
  probability calibration from hyperparameter selection. At present
  `train.py:131` and `:550-551` fit isotonic probability calibration on the
  `tune` partition, which `split_protocol_v2.py:52` defines as "the model/method/
  alpha selection set". Specification Finding 2 forbids this sharing and
  Priority 2 calls the separation essential.
- **`--split-protocol` defaults to `legacy`** at `train.py:126`. The four-way
  conformal split is wired but not active by default.
- **`ordinal.py` is gated by specification Finding 1.** The system collapses
  Pathogenic and Likely Pathogenic to 1, Benign and Likely Benign to 0, excludes
  Variants of Uncertain Significance, and derives five tiers by thresholding.
  Panel C says ordinal evaluation is to be used "only after establishing
  legitimate five-class targets".
- **Conformal quantile regression** targets continuous outcomes (delta percent
  spliced in, change in Gibbs free energy of folding, expression effect, assay
  activity, penetrance) and explicitly **not** intervals around a pathogenicity
  score, per Panel J.
- **`conformal/__init__.py` omits `calibrate`** from its imports, verified by
  outcome: `hasattr(conformal, 'calibrate')` returns `False`.

### 7.5 Not applicable this session

No model was trained, no ensemble run, no cohort regenerated. The per-model
algorithm comparison and the living metrics glossary required by the project's
documentation convention after every run therefore have **no new content to
record**. This is stated explicitly so that their absence is not read as an
omission.

---

## PART EIGHT -- THE DURABLE LESSON

Six defects were found in tools written during this session, four of them in
tools whose entire purpose was to measure accurately. Every one shares a shape:

1. The headroom-blind verdict compared against the wrong quantity and printed a
   confident answer.
2. The junction traversal claimed a safety property in its own docstring that it
   did not have.
3. The post-deadline zeros formatted an absence of measurement exactly like a
   measurement.
4. The visited-set contamination let a guard that is correct in one context
   silently suppress data in another.
5. The tautological identity check printed `HOLDS` for something that could not
   fail.
6. The `-ErrorAction SilentlyContinue` instruction suppressed errors in a
   deletion whose success was then assumed.

None was found by reading. Every one was found by **running the thing and
checking the output against an independently known answer** -- a synthetic tree
of exact sizes, a deliberately reintroduced defect, an emulation of Windows
junction semantics, the repository's own parser instead of a reimplementation.

This is the twenty-ninth through thirty-fourth instance of the same lesson
recorded in this project's ledger. It is now very well evidenced:

> **A check that cannot fail is worse than no check, because it manufactures
> confidence. Outcome-asserting checks catch what careful reading does not.**

---

---

## PART NINE -- ADDENDUM, 2026-07-21

Written after commit `a5caeba`. Everything in Parts Zero through Eight stands as
it was recorded; this part states what happened next and supersedes the pending
decision in section 6.4.

### 9.1 Pushed

```
0208998..a5caeba  main -> main
a5caeba (HEAD -> main, origin/main, origin/HEAD)
```

The commit landed with **1,432 insertions across 6 files**, which reconciles
exactly against independently known figures: 281 insertions measured earlier for
the four code and test files, plus 695 lines of session document, plus 456 lines
of part-three handoff. The agreement confirms both documents transferred whole.

### 9.2 The EVE regression is resolved

The remedy recommended in section 6.4 was carried out on 2026-07-21 at 00:59:09.

```
rclone copy genvarcla:.../data/external/eve/EVE_all_data/variant_files -> local
Transferred:  9.922 GiB / 9.922 GiB, 100%, 7.881 MiB/s
Transferred:  3211 / 3211 files
Elapsed:      20m55.6s
```

**The file count is itself a verification.** The test's own docstring names "the
real 3,211 EVE filenames"; rclone transferred exactly 3,211. Had the restore been
partial, the test could have passed at a degraded resolution fraction while
silently covering less of the corpus.

Only `EVE_all_data/variant_files` was restored. The 24.15 gibibytes of multiple
sequence alignments in a2m format, the 28.70 gibibytes of variant call format
files, and the 5,550 Portable Network Graphics plots remain offloaded to Google
Drive.

Module result: **17 passed, 0 skipped, 6.03 seconds**, including
`test_real_corpus_resolution_fraction`, which asserts that at least 99 per cent
of EVE filenames resolve from entry name to Human Genome Organisation Gene
Nomenclature Committee symbol.

### 9.3 New suite baseline

```
2064 passed, 7 skipped in 912.14s (0:15:12)
```

Collection remained 2071, so the suite-size ratchet passed rather than merely
not firing.

**Skip reconciliation, position by position.** The progress output places skip
markers at fixed points in the run. Before the restore: five early, then one each
at approximately 50, 76 and 95 per cent, totalling eight. After: five early, then
one each at approximately 76 and 95 per cent, totalling seven. The marker that
vanished sits at approximately 50 per cent, and unit tests execute in
alphabetical order, where `test_eve_entry_name_resolution.py` falls. **Exactly one
skip cleared, it was the EVE one, and no other skip changed position.** This is
stronger evidence than the count alone, which could have concealed one skip
clearing while another appeared.

The seven remaining skips:

| Count | Location | Reason | Class |
|---|---|---|---|
| 1 | `test_mc_dropout_calibration.py` | needs Run 15 cohort + gene-family-disjoint split infrastructure | infrastructure |
| 2 | `test_mc_dropout_calibration.py` | needs Spearman rank correlation infrastructure + real holdout labels | infrastructure |
| 1 | `test_mc_dropout_calibration.py` | needs expected calibration error infrastructure (10-15 reliability bins) + Run 15 holdout | infrastructure |
| 1 | `test_mc_dropout_calibration.py` | requires multiple K-value runs against real cohort | infrastructure |
| 1 | `test_preflight_data_paths.py:45` | POSIX symlink stands in for a Windows dangling junction | platform, permanent on Windows |
| 1 | `test_tabular_nn_mc_dropout.py:232` | test corpus does not span both boundary and extreme prediction regions | degenerate fixture |

**Five of the seven are waiting on the metric stack deliverable**, naming expected
calibration error infrastructure with 10 to 15 reliability bins and Spearman rank
correlation infrastructure. Both fall inside Panel B of the metric specification.
Building the metric stack will re-arm five currently dormant tests, which is a
concrete argument for its priority that does not depend on the specification
document at all.

### 9.4 Suite runtime rose 45 per cent

| Run | Duration |
|---|---|
| Before restore (three runs) | 632.92 s, 648.62 s, 605.08 s -- mean 628.87 s |
| After restore | **912.14 s (15 min 12 s)** |
| Change | **+283.27 s, +45.0 per cent** |

The restored module itself completes in 6.03 seconds, so it does not account for
283 seconds. The most likely cause is real-time antivirus inspection of 3,211
newly written files, possibly combined with directory globbing of that tree from
more than one test. **This has not been measured and is recorded as a hypothesis.**
It matters because a fifteen-minute suite changes what is tolerable in a
pre-launch gate, and because a 45 per cent regression with no identified cause is
the kind of thing that is easy to normalise and then never explain.

### 9.5 Free space -- the drive is settled

| | Free (GiB) | Per cent of volume |
|---|---|---|
| Start of session 2026-07-20 | 6.70 | 0.716 |
| Peak after all reclamation | 91.82 | 9.81 |
| After restoring 9.922 GiB of EVE data | **82.85** | **8.86** |

Against the headroom-aware requirement of 61.48 gibibytes the margin is **+21.37
gibibytes**. After a 14.7 gibibyte Joint Embedding Predictive Architecture
embedding cache is allocated, 68.15 gibibytes would remain -- 7.28 per cent, still
above the five per cent floor.

**Net reclamation for the session: 76.15 gibibytes**, after returning 9.92
gibibytes to re-arm a scientific check.

### 9.6 Two small unexplained figures, logged not dismissed

1. **-1.96 gibibytes** on 2026-07-20: projected 93.78 after the offloads, measured
   91.82. Three ten-minute suite runs regenerate `__pycache__`, write
   `.pytest_cache`, and create temporary Parquet fixtures; the ratchet installer
   also wrote two backups. Plausible, not measured.
2. **+0.95 gibibytes** on 2026-07-21: projected 81.90 after the restore, measured
   82.85 -- more free space than expected. A plausible cause is pytest's temporary
   directory factory, which retains only the three most recent temporary roots and
   deletes older ones; further suite runs would therefore have reclaimed earlier
   fixture output. Plausible, not measured.

Neither is large enough to affect any decision. Both are recorded because a
project that explains away small discrepancies loses the ability to notice large
ones.

### 9.7 Still open

- **The rclone shared Google Drive client identifier is being retired during
  2026.** It is July 2026. The canonical store holds the only copies of 62.6
  gibibytes of offloaded EVE and phyloP data. If the credential lapses, every
  operation against `genvarcla:` fails, including retrieval. Roughly ten minutes
  of work; it should not be deferred.
- `data/external/gtex` and `data/rnaseq` remain empty -- a hard blocker on the RNA
  infrastructure deliverable.
- `phylop_score`, `esm2_llr` and `eve_score` remain features in the 95-column
  contract whose connectors return constants.
- `split_protocol_v2` remains four-way with `legacy` as the default, and
  probability calibration is still fitted on the `tune` partition.

---

*End of session document. Parts Zero through Eight written 2026-07-20; Part Nine
added 2026-07-21.*
