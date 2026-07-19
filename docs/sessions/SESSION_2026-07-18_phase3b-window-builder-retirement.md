# SESSION 2026-07-18 — Phase 3b: retiring the superseded window builder, and moving four blind detectors onto provenance

Project: GenAssoc whole-genome variant pathogenicity classifier, `C:\Projects\genomic-variant-classifier`.
Environment: Windows, PowerShell 5.1, Python 3.12.10, venv `.venv312`, CPU-only local machine.
Remote: `github.com/monzia-moodie-repo-projects/genomic-variant-classifier`. Drive remote `genvarcla:` (rclone only; never bulk-write through `G:`).

This session closed a defect class rather than a defect. Twenty-one thousand eight hundred and fourteen
cohort rows had been carrying fabricated sequence into the one-dimensional convolutional neural network
and into the Nucleotide Transformer, and four separate detectors written to catch exactly that had all
gone blind at the same moment for the same reason. The work was to retire the module that produced the
fabrication, rebuild its replacement with coverage first, and move every consumer from inferring
provenance out of window CONTENT to reading the provenance column the builder had been writing all along.

Both commits are on `origin/main`. The suite is green at 1,968 collected, 1,961 passed, 7 skipped, zero
failures, and the suite-size ratchet gate has been run and passes.

---

## 1. THE CENTRAL DEFECT

Two window builders existed in the repository, and both were live.

The surviving, correct one is `src/genomic_variant_classifier/data/delta_window_builder.py`. When it
cannot build a window it writes `POLY = "N" * 101` and records `ok=False` together with a machine-readable
reason.

The superseded one was `src/genomic_variant_classifier/data/seq_windows.py`, driven by
`populate_fasta_seq.py` (which existed twice, once under `src/` and once under `scripts/`). It used
`PAD_CHAR = "A"` and wrote **no provenance column at all**.

Those two facts compound into the defect. `"A"` is a member of `encode_sequence`'s `BASES`, so every
placeholder position one-hot-encoded to a **confident adenine** — a positive assertion about the genome
that no one had made. `"N"` is absent from `BASES`, so it encodes to an honest all-zero vector. And
because the superseded builder wrote no `ok` column, `attach_delta_windows` took its `has_ok=False`
branch and declared every row usable behind a logger warning that nothing was reading.

Four consumers tried to catch this by comparing window content against `"A" * 101`. When the placeholder
base moved to `"N"` on 2026-07-15, all four comparisons began matching nothing. **They did not start
failing; they started passing unconditionally.** A working detector became a rubber stamp, and the change
that disarmed it was itself a correct change.

The lesson recorded here, because it generalises beyond this file: **content can never establish
provenance.** A window reading `"A" * 101` may be real — poly-adenine tracts are genuine biology. Nothing
about the string distinguishes "the reference genuinely says adenine" from "we gave up and typed
adenine". Only the builder knows, and only the builder can say. Widening the checks to also match `"N"`
would have been patchwork on an error of principle, and would have gone blind again at the next change of
placeholder base.

## 2. THE COHORT, MEASURED FOUR WAYS

`data/processed/clinvar_grch38_clean_seq.parquet`:

| Quantity | Value |
|---|---:|
| Total rows | 4,399,089 |
| Usable (`ok=True`) | 4,398,366 |
| Placeholder (`ok=False`) | 723 |
| — `non_acgt_allele` | 668 |
| — `ref_mismatch` | 53 |
| — `fetch_failed` | 2 |

That 723/668/53/2 split was re-derived **four independent times through four different code paths** over
the course of the session: by the replacement producer's dry run, by the rewritten density probe, by the
rewritten launch-gate preflight, and by direct inspection. Agreement across four routes is the reason it
is stated as fact rather than as a reading.

Two further measurements, both confirming older findings rather than discovering new ones:

- `fasta_seq` is non-null on **0 of 4,399,089 rows**. The column is entirely empty, which confirms
  `INCIDENT_2026-05-23`. Any detector reading that column was grading nothing.
- `fasta_seq_ref` and `fasta_seq_alt` both have `len_min == len_max == 101`, so window width is uniform
  and no length-based heuristic was ever going to separate real from placeholder.

## 3. WHAT WAS RETIRED, AND WHY THE ORDER MATTERED

Deleted in commit `e57835e`:

```
src/genomic_variant_classifier/data/seq_windows.py          (197 lines, 7 poly-ban offenders)
src/genomic_variant_classifier/data/populate_fasta_seq.py   (221 lines)
scripts/populate_fasta_seq.py                               ( 85 lines)
tests/unit/test_seq_windows.py                              (155 lines, 16 tests)
tests/unit/test_populate_fasta_seq.py                       (154 lines,  5 tests)
```

**A retirement audit refused this deletion earlier the same day, and was right to.**
`scripts/populate_fasta_seq.py` was at that point the sole repository-resident producer of a 534 MB
artifact that eighteen files consume. Deleting it then would have left that artifact unreproducible from
the repository — a far worse condition than the defect being fixed. The audit's refusal is the reason the
replacement was built first.

Replacement, built and tested **before** the deletion:

```
scripts/build_clean_seq_from_windows.py       252 lines, join-based producer
tests/test_build_clean_seq_from_windows.py      9 tests, mutation-tested
tests/test_build_seq_windows.py                10 tests, ported coverage for the surviving builder
```

The retirement script itself carried four preconditions, each re-verified at run time rather than trusted
from the audit: the replacement exists, the replacement is tested, every target is git-tracked and
unmodified against HEAD, no module imports the targets (checked by abstract syntax tree, not by string
search), and no launcher invokes them.

## 4. THE FOUR BLIND DETECTORS, MOVED ONTO PROVENANCE

| File | What it was doing | What it does now |
|---|---|---|
| `scripts/preflight_run16_inputs.py` | **A LAUNCH GATE.** Reported 100% real sequence on a cohort with 723 placeholders. Also sampled only the first 4,000 rows and applied a 50% threshold. | Reads the whole `ok` column. Where provenance is absent, reports that it cannot tell. |
| `scripts/probe_cohort_seq_density.py` | Reported `dummy=0` when it had no way to know | Reads `ok`; distinguishes "zero" from "cannot tell" |
| `correctness_harness.py` stage 3a | **A CORRECTNESS GATE, triply dead** — wrong column (100% null), obsolete `"A"*101` constant, and a synthetic random-adenine-cytosine-guanine-thymine fixture. Structurally incapable of firing. | Three outcomes: fail above 50% unusable, warn when `ok` is absent, quiet when all usable |
| `scripts/run9_ablations.py` | Fed fabricated adenine to `cnn_1d` | Uses the centralised placeholder; `cnn_1d` is now excluded from ablations outright, since constant input cannot be learned from |

Placeholder construction was centralised into `delta_window_builder.placeholder_window(window=101)` so
that the literal exists in exactly one place. The left-edge padding at lines 91, 128 and 132 — which is
`"N" * n` with `ok=True`, and is legitimate — was deliberately left untouched.

Poly-ban offenders went from **12 to 0**. Eight of the twelve lived in the retired modules; the other four
were the live detectors above.

## 5. A TEST THAT WAS PASSING FOR THE WRONG REASON

`test_cohort_all_dummy_fails` wrote windows of all `"A" * 101` and asserted the gate refused them. The
gate did refuse — **but because the `ok` column was absent, not because the windows were placeholders.**
The test's name described a behaviour the test never exercised, and pytest reported it green throughout.

A test that passes for a reason unrelated to its name is not coverage; it is a decorative assertion. It
was replaced by `test_cohort_mostly_placeholder_fails`, which supplies provenance so that the gate is
tested on the axis the name claims.

Two test files were realigned to the new contract: `tests/unit/test_correctness_harness.py` (6 tests to 8)
and `tests/unit/test_preflight_run16_inputs.py` (13 tests to 14).

## 6. DEFECTS FOUND IN THIS SESSION'S OWN DELIVERABLES

Recorded because the pattern matters more than any individual instance: **every one of these was caught by
running or re-deriving, and not one was caught by reasoning about the code.**

1. **The retirement script counted its own backups as importers.** Its post-check scanned for modules
   importing the retired files and found the copies it had just written. Fixed by excluding the
   `retired_2026-07-18/` and `.wiring_backup/` directories.
2. **The same script silently overwrote a backup.** It keyed backups on basename, and there were two files
   named `populate_fasta_seq.py`. Fixed to a flattened full path plus an explicit count verification.
3. **A collision check flagged its own success** in the retirement-safety audit.
4. **The ablation fix would have raised `ImportError` at run time** because it called the centralised
   placeholder helper without verifying the helper had been installed. A precondition guard was added.
5. **A patch script used the wrong anchor bytes** — an em-dash (U+2014) where a hyphen was assumed. Fixed
   by constructing the character via `chr(0x2014)` so the patch source stays pure ASCII.
6. **A content-based substring check fired on a comment**, twice, in two different scripts. Both were
   converted to abstract-syntax-tree checks. This is the same failure the poly ban was written to prevent,
   committed inside the tooling written to fix it.
7. **The `.wiring_backup` disposal tool compared only against HEAD** and reported four files as unique.
   Version 2 queried the full git object database and found all four already committed in `eba5c40`,
   `80eb9c8` and `6166fa6` — swept in by the `git add -A` hazard that has fired once before.
8. **`docs/measurements/retired_2026-07-18/` repeated that same mistake.** The retirement script's own
   precondition had verified every target was tracked and *unmodified* against HEAD, which means those
   five archived files were byte-identical to blobs git already held at `87b670e`. Committing them would
   have added redundant objects to the permanent record. The directory was deleted before staging; the
   content remains recoverable with `git show 87b670e:<path>`.
9. **An unrunnable command was issued to the owner.** The instruction to update the ratchet was written as
   `--collected <the number pytest printed>`. Angle brackets are a parse error in PowerShell — `The '<'
   operator is reserved for future use` — so both invocations died before Python started, the commit
   message was never generated, and `git commit -F` then failed on a file that did not exist. Nothing was
   committed and nothing was damaged, but the instruction was wrong and it cost a cycle.

## 7. THE SUITE-SIZE RATCHET, CORRECTED WITH THE TREE NAMED

`tests/EXPECTED_SUITE_SIZE` read **1962** at the start of this session, and its final history entry
explains why that was lower than any recent working-tree measurement: the count of 1966 written on
2026-07-15 had included four tests from `tests/unit/test_no_content_based_poly_detection.py`, a file that
**had never been committed**. The measurement was real; the tree it measured was not the tree being
guarded.

That entry set a condition for moving the number, and this session met it: the file is now committed. The
new value is **1968**, and per the entry's standing instruction every future record states the tree it was
taken on — here, the staged tree at `87b670e` with 44 paths staged, not a bare working tree.

The number was measured with `pytest tests/ --collect-only -q` and copied from that output. It was not
computed. The ratchet's history records four consecutive occasions on which its author computed the number
by hand and was wrong — 1882, 1891, 1932, 1944 — each caught by the ratchet itself.

**The gate was then run**, which is the step that distinguishes a number that is enforced from a number
that is merely written down:

```
python -m pytest tests/ --collect-only -q --assert-suite-size   ->  1968 tests collected, no failure
```

## 8. LINE-ENDING GOVERNANCE FOR THE RATCHET FILE

Staging produced a warning in the opposite direction from the other thirteen: `LF will be replaced by CRLF`.

The cause is precise. `.gitattributes:2` is `* text=auto`, which normalises to line-feed in the repository
and converts to the platform's native ending in the working tree — carriage-return line-feed on Windows.
Every other governed text file carries an explicit `eol=lf` override keyed to its extension, and
`EXPECTED_SUITE_SIZE` **has no extension**, so nothing matched it except the `*` default.

The single source of truth for suite size was the one governed text file in the repository without an
explicit line-ending rule. Not a functional defect — `conftest.py` strips each line — but it would have
re-warned on every future bump. Closed in commit `9362f2c`, and verified with `git check-attr` rather than
by reading the file: `text: set`, `eol: lf`, 0 carriage-return bytes in the working copy.

## 9. COMMITS AND VERIFICATION

```
9362f2c  chore: govern EXPECTED_SUITE_SIZE line endings explicitly     ( 1 file,     4 insertions)
e57835e  retire the superseded window builder; move 4 blind detectors  (46 files, 5,673 insertions, 953 deletions)
87b670e  (previous origin/main)
```

Pushed 2026-07-18: `87b670e..9362f2c  main -> main`.

Final state, every item measured rather than asserted:

| Check | Result |
|---|---|
| Full suite | 1,961 passed, 7 skipped, **0 failed**, 630.21s |
| Collected | 1,968 |
| Ratchet gate `--assert-suite-size` | **PASSED** |
| Poly-ban offenders | 12 → **0** |
| Invalid escape sequences | **0** across 810 files |
| `git status` | clean |
| `EXPECTED_SUITE_SIZE` carriage returns | 0 |

Diff arithmetic reconciles exactly: 44 files before the ratchet commit became 46 (adding
`EXPECTED_SUITE_SIZE` and the derived commit message); insertions 5,526 to 5,673 is +147, being 65 ratchet
insertions plus 82 message lines; deletions 952 to 953 is +1, the retired `1962` value line.

## 10. ALSO CLOSED THIS SESSION

- **Two invalid escape sequences removed**, each a `SyntaxError` in a future Python version:
  `download_finngen_R10_DEPRECATED.py:6` (`\M`, `\g`, from a Windows path in a docstring) and
  `patch_clinvar_alleles.py:12` (`\S`, `\p`). Two different strategies were needed — a raw-string prefix
  where no valid escape was present, and doubled backslashes where one was, because `patch_clinvar_alleles`
  had a trailing-backslash line continuation that a raw prefix would have broken.
- **`.wiring_backup/` disposed of**, all four files proven redundant against the full git object database.
- **The commit message was derived**, not written, from `git diff --cached --stat`, so it describes what
  was actually staged rather than what anyone remembered staging.

## 11. OPEN ITEMS

**Documentation gap.** `docs/sessions/` runs to `SESSION_2026-07-06.md` and then stops, yet the ratchet's
own history documents substantial work on 07-13, 07-14 and 07-15 — roughly four working sessions with no
session document. This file covers 07-18 only. The gap is recorded rather than silently backfilled,
because reconstructing those sessions from the ratchet history alone would produce a plausible narrative
rather than a measured one.

**Standing documentation debts.** The living metrics glossary (area under the receiver operating
characteristic curve, area under the precision-recall curve, F1, Matthews correlation coefficient, Brier
score, out-of-fold, calibration, graph-neural-network area under curve, odds ratio, Cramer's V, bootstrap
confidence interval, feature importance, and the gates) and the per-model algorithm comparison remain
unwritten.

**Unblocked by this session.** Part 3 of the hybrid — making `X_seq` optional in `VariantEnsemble.fit`,
`evaluate` and `predict_proba`, failing loudly when `cnn_1d` is active without sequence. This removes the
CLASS of defect rather than the instance. It was gated on committing the 2026-07-15 work, which landed in
`e57835e`.

**Experimental design, recorded in `run9_ablations.py`'s module docstring.** Six external annotation
families have no ablation mask: ribonucleic-acid sequencing (5 features), COSMIC (2), KEGG (2),
GenomicLM/Nucleotide Transformer (2), Reactome (1), and the heterogeneous graph neural network (1 — the
existing `no_gnn` mask covers `gnn_score`, not `hetero_gnn_score`). Thirty-two of the 95 contract features
match no prefix, most of them core descriptors and legitimately so. This bears directly on the project's
primary goal, since ablation is the instrument by which feature-class contribution is quantified.

**Minor.** `test_no_content_based_poly_detection.py` lines 238 and 251 still cite
`populate_fasta_seq.py:59` in prose describing a file that no longer exists.
`scripts/download_finngen_R10_DEPRECATED.py` declares itself deprecated for a FinnGen release superseded
in commit `77c66f5`. The `*.bak_*` rule is duplicated at `.gitignore` lines 155 and 158.
`build_alphafold_parquet.py:707` shows interface drift (`unrecognized arguments: --workers 1`). Roughly
36.5 MB of `af*` debris and a 202.7 MB `.venv` drift sit at the repository root.

## 12. ARTIFACT FINGERPRINTS (SHA-256, this session)

Every delivered script was byte-order-mark-free, pure ASCII, line-feed-only, fixture-tested before
delivery, and hash-verified on the owner's machine before execution.

```
039ddec3  apply_placeholder_helper_2026-07-18.py
54768825  apply_detector_rewrites_2026-07-18.py
ad9a4bd7  apply_run9_ablation_fix_2026-07-18.py
a2248fab  apply_run9_help_correction_2026-07-18.py
c1498a36  audit_retirement_safety_2026-07-18.py
aeb83cb1  retire_superseded_builder_2026-07-18.py
0f855468  apply_harness_stage3_rewrite_2026-07-18.py
aa1bfbdf  fix_invalid_escape_sequences_2026-07-18.py
bc6fe5a0  apply_test_contract_updates_2026-07-18.py
89493253  dispose_wiring_backups_2026-07-18_v2.py
36bd3919  tests/test_build_clean_seq_from_windows.py
4163fe22  audit_precommit_2026-07-18.py
aa9d4d9e  update_suite_ratchet_2026-07-18.py
```

## 13. METHOD

Stated because it is the reason the numbers above can be trusted.

Every delivered script carried a SHA-256 fingerprint, content-hash preconditions that are insensitive to
line-ending differences, all-or-nothing application, timestamped backups, an idempotency sentinel unique
to that patch, and a `--dry-run` mode. Expectations were derived rather than hardcoded wherever
derivation was possible. Every verification was accompanied by a negative control, on the principle that
**a verification which cannot fail is not a verification** — and that principle earned its keep twice this
session, once in the retirement audit's collision check and once in the harness stage-3 rewrite, where all
six branches were exercised deliberately.

Uploaded outputs were read from disk by byte count and hash before being interpreted, because attachments
have rendered empty intermittently throughout this session. On two occasions a file that appeared empty
was in fact 166,702 and 159,839 bytes. Concluding from the absence would have been wrong both times.

## 14. CONTINUOUS INTEGRATION CYCLE (2026-07-18, after the first push)

The first push turned Continuous Integration RED, and the record would be dishonest without it.

### What failed

Runs #522 (`9362f2c`) and #523 (`d1c2c4e`) both failed, identically, on Python 3.11 and 3.12:

```
_______ test_readme_test_count_equals_the_suite_size_ratchet_exactly _______
tests/unit/test_readme_claims.py:275
E   AssertionError: README.md states the wrong test count in 1 place(s):
E         shields.io badge              says 1962
E     tests/EXPECTED_SUITE_SIZE says 1968. These must be EQUAL -- no tolerance.
```

Collection was CORRECT on both interpreters, which is what makes the diagnosis unambiguous:
1,952 passed + 14 skipped + 1 xfailed + 1 failed = **1,968**. The ratchet was right. One line of
README.md was not.

### The tenth defect in this session's own deliverables

Section 6 above lists nine. This is the tenth, and unlike the others it reached `origin/main`.

The suite-size ratchet was bumped from 1962 to 1968 without updating the README claim BOUND to
that number, and the post-bump suite run was placed in the delivered script's trailing "Next"
block rather than in the command sequence actually executed. The last green local run -- 1,961
passed, 7 skipped, zero failures -- was taken BEFORE the bump, at a moment when the README badge
and the ratchet both said 1962 and agreed for entirely the correct reason. Moving one of the two
numbers invalidated that agreement, and nothing re-ran to notice.

**The gate did exactly what it was built to do.** `test_readme_test_count_equals_the_suite_size
_ratchet_exactly` was written on 2026-07-14 as roadmap 6.25, specifically because its first
version had carried a 50-wide tolerance that let a real 17-test drift pass silently. Its
docstring says so. It fired at the first opportunity it was given, named the file, named both
numbers, and refused to approximate. The failure was in the process feeding it, not in the gate.

**The standing correction, recorded so it is not re-learned: after moving the ratchet, the full
suite runs BEFORE the push, not after.** `tests/EXPECTED_SUITE_SIZE`'s own header already
required the number to be bumped in the same commit as the tests; what it does not say, and what
this session establishes, is that a second number elsewhere in the repository is bound to it and
that only a full run surfaces the binding.

### The fix

`8975b1a` -- `fix(readme): re-derive the test badge from the ratchet (1968)`. One line changed.

The number was **derived, never typed**: the repair reads `tests/EXPECTED_SUITE_SIZE` and writes
that value using the SAME regular expression the test uses to read it back,
`tests-([\d,]+)-success`. Sharing the detector between gate and fixer is deliberate -- roadmap
6.24(iii) records a repair tool that could not fix what its own gate reported, which converts a
red test into a manual chore, and manual chores are what rot. The repair also refuses if the
number of matching claim sites is anything other than exactly one, so a restructured README
causes a stop rather than a silent rewrite of the wrong line.

Verified before the push this time:

```
python -m pytest tests/unit/test_readme_claims.py -q   ->  10 passed in 10.29s
python -m pytest tests/ -q --assert-suite-size         ->  1961 passed, 7 skipped in 629.92s
```

The second command is the one that had been skipped. It runs the ratchet gate against the full
suite rather than against a collection-only pass.

### The environment split, which is the ratchet's whole argument

| environment | passed | skipped | xfailed | failed | total |
|---|---:|---:|---:|---:|---:|
| Windows, full cohort | 1,961 | 7 | 0 | 0 | **1,968** |
| Linux runner, Python 3.11 | 1,952 | 14 | 1 | 1 | **1,968** |
| Linux runner, Python 3.12 | 1,952 | 14 | 1 | 1 | **1,968** |

Same collection, three different pass/skip splits. This is precisely the divergence
`EXPECTED_SUITE_SIZE`'s header cites as its reason for asserting COLLECTED and never PASSED, and
it is why the README is forbidden from quoting a passing count at all
(`test_readme_does_not_quote_an_environment_dependent_passing_count`). The Continuous Integration
skips are legitimate: Windows-targeted post-flight tests, the ESM-2 tests (the HuggingFace Hub is
offline by design), and tests needing gitignored cohort data.

### Final state, 2026-07-18

```
433d2e8  docs(roadmap): 6C snapshot -- Phase 3b, suite 1968, 6.29a unknown answered   CI #525 GREEN
8975b1a  fix(readme): re-derive the test badge from the ratchet (1968)                CI #524 GREEN
d1c2c4e  docs: session record for 2026-07-18 (Phase 3b)                               CI #523 red
9362f2c  chore: govern EXPECTED_SUITE_SIZE line endings explicitly                    CI #522 red
e57835e  retire the superseded window builder; move 4 blind detectors onto provenance
87b670e  (previous origin/main)
```

All five are on `origin/main`. Continuous Integration is GREEN for the first time since
`87b670e`. Runs #522 and #523 are red for the README badge alone and for nothing in Phase 3b;
they are left in the history rather than re-run, because a red run that was genuinely red is
evidence.

### One measurement worth preserving from the roadmap 6C run

`clinvar_grch38_clean_seq.parquet` reported **21 columns**, against section 6.29a's recorded
*"19 columns, NO `ok`"*. Two columns (`ok`, `reason`) were added when the artifact was
regenerated during this session. That is what closed 6.29a's blocker, and it was confirmed by
reading the parquet metadata at write time rather than assumed -- the 6C appender carries both
branches and would have written "6.29a's premise CONFIRMED, still open" had the column been
absent.
