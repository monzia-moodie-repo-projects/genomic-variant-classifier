# SESSION 2026-07-30 — carried item CI-i becomes verifiable, and two numbers are corrected

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `a2e176a`, ratchet 3851, badge `tests-3851-success`
**Ending state:** ratchet 3856, badge `tests-3856-success`, 3850 passed / 6 skipped
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. What changed

Three files edited for one subject, and a fourth derived from one of them.

**CI-i moved from *Unverifiable from the repository alone* to *Open*.** The register
recorded its reason for being unverifiable as "needs the cohort; a skip count alone
cannot distinguish these from other skips". That is **correct about counts and wrong
about identities**. A count cannot tell one skip from another. Five node identifiers
can, and they were measured on 2026-07-30 by `pytest -v`:

    tests/integration/test_mc_dropout_calibration.py::TestOODEpistemicElevation::test_held_out_gene_families_have_higher_epistemic
    tests/integration/test_mc_dropout_calibration.py::TestUncertaintyErrorCorrelation::test_spearman_correlation_between_epistemic_and_error_positive
    tests/integration/test_mc_dropout_calibration.py::TestUncertaintyErrorCorrelation::test_accuracy_decreases_monotonically_across_epistemic_quartiles
    tests/integration/test_mc_dropout_calibration.py::TestCalibrationImprovement::test_ece_lower_with_mc_dropout_vs_single_pass
    tests/integration/test_mc_dropout_calibration.py::TestMonteCarloConvergence::test_epistemic_estimate_converges_with_k

Each is a method on a class carrying a class-level `pytest.mark.skip`. The Run 15
cohort' arrival is still not observable from the working tree — and does not need
to be. When it lands the skips are removed, `_condition_i` returns `False`, and
`test_every_open_item_still_has_its_condition` fails until the register moves CI-i to
Discharged. The register' own rule, *discharge is proved, not asserted*, does the work.

**Two checks, deliberately separate.** `_condition_i` flips when the skips are
REMOVED — the register asking to be updated. The parametrised
`test_the_ci_i_nodes_still_exist` fails when the tests THEMSELVES go away, naming the
node. Without that separation CI-i could be discharged by its subject quietly
disappearing, which is exactly how CI-l read as open for eleven commits.

**Parsed, not grepped.** The predicate walks the abstract syntax tree and matches the
decorator' dotted name, so `pytest.mark.skipif` — which is conditional and belongs to
CI-j — can never satisfy it. This follows `_discharged_o`, which was strengthened after
a sabotage replacing `import ast` with `import os` survived a `"ast" in text` check.

**A known limitation is stated in the code rather than papered over.** A marker bound
to a name first — `_GATE = pytest.mark.skip(...)` then `@_GATE` — is invisible to this
and to every static scan. It was proven invisible on 2026-07-30 against a purpose-built
fixture. The five markers here are written directly, so the check is sound for them; it
is not a general skip detector.

**`tests/EXPECTED_SUITE_SIZE` line 4274 corrected.** It read:

    JEPA IS NO LONGER DISK-BLOCKED: 56.01 GB free against ~14.7 GB needed.

Wrong twice. `56.01` came from an idiom dividing by PowerShell' `1GB` literal, which is
1073741824 bytes, so the figure was GIBIBYTES under a label reading GB. And ~14.7 GiB is
the cache-only estimate withdrawn on 2026-07-20 when the operating floor was added —
line 1944 of the same file already recorded 61.48 GiB and noted that it "reproduces
EXACTLY what audit_disk_census.py printed". One line contradicted another 2,330 lines
earlier.

**The README test badge is now DERIVED, not typed.** Measured 2026-07-30: nothing in
`.github`, `scripts`, `tools` or `.git/hooks` updates it. There is an alert workflow that
DETECTS drift, a local preflight that READS the ratchet, four one-shot historical patch
scripts whose literals are all stale, no `tools/` directory, and no non-sample git hooks.
The badge was maintained by memory, and the ratchet file' own header records what
happens to numbers maintained by memory: the pre-flight floor rotted five times in two
days, each time beneath a comment ordering the next person to raise it.

---

## 2. Verification

| Check | Result |
|---|---|
| installer `--dry-run` | wrote nothing; files byte-identical |
| installer `--apply` pre-checks | four anchors each occurring exactly once |
| installer `--apply` post-checks | ten structural checks, five nodes `exists=True skip=True` |
| ratchet, COMPUTED by collection | 3851 -> 3856, delta +5, in 25.12 s |
| full suite with `--assert-suite-size` | 3850 passed, 6 skipped, 0 failed |
| 3850 + 6 | = 3856 = the ratchet |
| **skip set** | **byte-for-byte unchanged, five plus one** |
| badge derivation, byte invariants | non-ASCII 110->110, CRLF 0->0, LF 502->502, delta +0 |
| badge idempotence | second run reports IN STEP and writes nothing |

**Sabotage matrix, run against a faithful fixture before delivery.**

| Mutation | Detected by |
|---|---|
| one class-level skip removed | `_condition_i` flips; existence tests hold |
| a class renamed | existence test names the node; predicate also flips |
| `skipif` substituted for `skip` | predicate refuses the conditional gate |
| target file deleted | all five existence tests plus the predicate |
| two badge occurrences | derivation refuses; no backup created |
| ratchet with two bare integers | derivation refuses |
| README made invalid UTF-8 | derivation refuses, naming byte `0xf9` at offset 881 |
| pre-existing backup | both tools refuse, exit 1 |
| dirty working tree | installer refuses |
| second apply | installer refuses |

Nine of nine mutations detected, zero undetected.

---

## 3. Defects found in the instruments used to do this work

Recorded because the project' own convention is that a tool wrong three times gets
tests, and because two of these are failure shapes the repository had already named.

1. **A probe crashed and its handler made it worse.** Reading a right-arrow (U+2192)
   out of a repository document and writing it to a cp1252 console raised
   `UnicodeEncodeError`; the handler then formatted `repr(exc)`, which EMBEDS the
   offending character, so the handler raised the error it was handling. Two whole
   sections never ran. An error handler that is not safe against the error it handles
   is not a handler. Fixed by routing every line through an ASCII-safe writer and never
   formatting the exception object.

2. **A probe printed a verdict that contradicted the measurement directly above it.**
   It stated "identical byte counts make Docker.backup a confirmed duplicate" beneath a
   table showing three of four shared files differing in size and a fifth present only
   in the live tree. The sentence was a hard-coded string, not a computed result. The
   ratchet file records the same shape at line 1871 for the census tool: a verdict
   printed "directly beneath its own bad measurement". Corrected reading: `Docker.backup`
   is the PREVIOUS VERSION retained by the 2026-07-27 update, not a copy of the current one.

3. **A probe missed five of the six actual skips.** It walked only function definitions,
   so it never inspected class decorators or a module-level `pytestmark`, and reported
   eleven decorated functions as the whole surface while the file producing five of the
   six runtime skips did not appear at all. An earlier text scan had found four
   decorators there; the two methods disagreed and the probe did not compare them.

4. **Directories counted as files.** `rglob("*INCIDENT*")` is case-insensitive on
   Windows, so the `docs/incidents` DIRECTORY matched and its directory-entry size
   printed as though it were a document.

5. **Compiled bytecode read as text, three times.** `__pycache__` was never excluded, so
   `.pyc` files were decoded through a text lens and one was mislabelled
   `latin1-NOT-UTF8`. Noted after the first occurrence and not fixed until the third.

6. **A test fixture that tested nothing.** `/bin/sh`' `printf` does not interpret `\xNN`,
   so a fixture intended to contain the exact byte that caused defect 1 contained the
   literal text instead. The "test" passed while exercising nothing. Caught only by
   reading the output rather than the summary line.

7. **Instructions with the same defect as the code they replaced.** A command block was
   split across two fenced sections so a variable defined in the first was consumed by
   the second; run alone, the second bound to `null`. And the README badge update was
   ordered AFTER the full suite run, so sixteen and a half minutes of compute were spent
   discovering a consequence of the step numbering.

---

## 4. Hypotheses raised and disconfirmed

Recorded so the reasoning is not repeated. Each was stated as a hypothesis, tested, and
withdrawn.

| Hypothesis | Disconfirmed by |
|---|---|
| `maximum_calibration_error` is a silent absence in the catalogue | `test_the_named_clinical_metrics_are_all_accounted_for` already covers it; the probe measured `status=IMPLEMENTED` |
| the ESM-2 cache lets a 650M-configured run silently consume 8M vectors | `esm2.py:163–165` constrains `model` in the WHERE clause; and `esm2_t6_8M_UR50D` is the DECLARED DEFAULT at `esm2.py:78` |
| the cache is orphaned because keys need the HGVSp parser | keys are `seq_hash` and gene name; the parser-dependent cache is a SECOND, score-level one at `esm2.py:634` |
| README.md carries a non-UTF-8 byte | valid UTF-8, 110 non-ASCII bytes, all em-dash and en-dash sequences; the `\xf9` was a console rendering artefact |
| a badge auto-update hook would fight a manual edit | no non-sample git hooks exist at all |
| the disk census walker is defective | `test_disk_census_walker.py:103` SPECIFIES the shared-mode behaviour; the fault is in the report' presentation, not the walker |
| a CI-i predicate would close the 2026-07-20 offload gap | it would not; that regression is in a THIRD skip category with no register entry |

---

## 5. Findings recorded and deliberately not acted on

Each is measured, none is fixed here, and none should be lost.

**The third skip category has no register entry.** CI-i covers five unconditional
skips; CI-j covers four platform skips. Eight further node identifiers plus one whole
module are gated on DATA PRESENCE or an optional dependency:
`test_eve_entry_name_resolution.py` (2), `test_drift_reference_profile.py` (1),
`test_build_cohort_v2.py` (4), `test_launch_run17.py` (1), and
`test_lovd_annotation_reaches_training_matrix.py` as a module-level gate. **The
2026-07-20 regression, where `test_real_corpus_resolution_fraction` went from passing to
skipped when the Expression of Variant Effects corpus was offloaded, is in this
category.** It needs a scope decision on the item identifier and wording.

**CI-j may be fully verifiable.** Its recorded reason is "needs the continuous-integration
matrix, not the working tree", but the matrix is declared in `.github/workflows/`, which is
in the tree. If no leg names a Windows runner, that reason is stale too.

**`esm2._compute_delta` returns `0.0` on four distinct conditions** — index out of range
(`:513`), embedding `None` (`:531`), embedding one-dimensional (`:534`, commented
"flat storage fallback"), and legitimate non-missense — with only a `logger.debug` at
`:516`, invisible at default verbosity. The module' own likelihood-ratio path COUNTS its
zeros and warns with counts at `:991` and `:993`. Same file, two standards. Total failure
is caught by the constant-column argument the author records at `:976–981`; PARTIAL
failure produces a mixed column that no constant-feature audit can see, and `:534` reports
a STORAGE-SHAPE condition as a biological value of zero.

**`scripts/audit_run17_assets.py:40` names a file that does not exist.** On disk the
archive is `finnge_R12_annotated_variants_v1.gz` (one `n`, 32,126,590,987 bytes); the
auditor names `finngen_R12_annotated_variants_v1.gz` with the comment
"# corrected (non-typo) name". Its own docstring at `:16` reads "We do NOT want to
re-download 30 GB of FinnGen if the data is sitting" — so it will do the exact thing it
exists to prevent. Every other reference uses the on-disk spelling.

**A unit error introduced by a correction.** `scripts/patch_run17_plan_doc.py:31–32`
replaced "29.77 GB" with "27.72 GB" for the R13 archive. Measured: 29,768,495,399 bytes
= 29.77 GB = 27.72 GiB. The original was right; the replacement carries the right value
under the wrong unit.

**The census report' reconciliation is inflated.** The top-level walk shares a visited
set with the targeted checks that run before it, so `C:\Windows.old`, `C:\cabal` and
`C:\Projects` each printed twice with different values in one run. `C:\Projects` measured
147.55 GiB independently against 35.14 GiB printed. The stated "256.24 GiB unaccounted"
is nearer 105 GiB. The walker is correct and tested; the caller passes `independent=True`
to the repository-data section only.

**Storage.** Roughly 488 GiB is recoverable without irreversibly touching a scientific
artifact: 195.897 GiB of gnomAD Variant Call Format files at `C:\Users\monzi\data`, which
no code path reads and which Drive already holds; 73.49 GiB in a WSL2 virtual disk that
never self-shrinks; 38.03 GiB of `C:\Windows.old`; 32.366 GiB of a nineteen-day-old repo
backup; and 14.484 GiB of fingerprint-confirmed duplicates. **The ESM-2 embedding cache is
the largest unbacked artifact on the machine**: 19,447,939,072 bytes locally against
88,834,048 bytes on Drive dated 2026-06-12, a ratio of 218.9 and an unbacked delta of
18.030 GiB.

**Two latent defects in the register' own test module.** `_ids_under` at line 350 skips
`len(heading)` characters where the heading occupies `len(heading) + 3`, so `remainder`
always begins with the last three characters of the heading word. Harmless today because
no heading remnant can contain a `**CI-x**` pattern; fragile. And the filter at line 480
contains a dead clause: `type(lambda: 0)` is `<class "function">`, the type of every
`def`, so `not isinstance(...)` is always `False` and the expression reduces to the name
check.

**Suite runtime spread is 2.05x** across four runs of materially identical work on an
unchanged head: 738.52 s, 1333.44 s, 988.33 s, 1512.18 s. Not monotonic, so not a
regression, but any future timing-sensitive gate has to live inside that spread.

**Free disk space baseline is ~83 GiB, and a full suite run transiently consumes ~28 GiB.**
A 55.36 GiB reading at 02:24 was taken immediately after a suite run and was an artefact
of timing; the ratchet file' own lines 1898 and 1931 corroborate the ~83 GiB baseline. A
capacity decision must clear the MINIMUM observed across a working session, and a suite run
overlapping a cache build would collide. The 14.7 GiB cache estimate itself remains
UNDERIVED: a bare literal at `scripts/forensics/audit_disk_census.py:137` with no embedding
dimension, row count, dtype width or replica count behind it.

---

## 6. Figures

    ratchet                3851 -> 3856  (+5, computed by collection in 25.12 s)
    README badge           3851 -> 3856  (derived from the ratchet, not typed)
    full suite             3850 passed, 6 skipped, 0 failed, 25 warnings, 1512.18 s
    skip set               unchanged: 5 unconditional + 1 POSIX-only
    files changed          4 (3 edited, 1 derived)
    sabotage               9 mutations, 9 detected, 0 undetected
    README.md              26,317 bytes, 110 non-ASCII, 0 CRLF, 502 LF, valid UTF-8

---

*Written 2026-07-30. Amend by editing this file; the carried-item register decides status,
and `tests/EXPECTED_SUITE_SIZE` decides the count.*
