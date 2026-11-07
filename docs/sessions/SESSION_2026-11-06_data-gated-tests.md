# Session 2026-11-06 -- DATA-GATED-TESTS-1: no test's outcome depends on git-ignored local data, except three declared ones

Author: Monzia Moodie

## Why

Set "ai" (TESTS-REACH-THE-NETWORK-1) was validated on the owner's Windows machine and merged as pull request #66 (merge commit
0e5d73c4, tree 2fd84792). Its post-merge block then STOPPED: the targeted tests gave 897 passed and 13 skipped in the owner's checkout
against 896 and 14 in the validation's clean clone. The one difference was test_esm2_activation.py::test_esm2_not_in_stub_mode,
skipped in the clone and passed in the checkout, which holds the git-ignored data/external/uniprot/uniprot_human_reviewed.parquet.
The block compared counts between a clone and a checkout that holds local data -- a case its sandbox harness never built. Its
revision B compares each test by node identity with the validation's own record and allows that one case by name; it passed.

A skip reason names only the FIRST condition a test does not meet, so reading the validation's skip reasons cannot say which tests
depend on local data. That was measured instead.

## The instrument (outside the repository)

A sitecustomize.py on PYTHONPATH installs, in every Python process of a run:

- an audit hook (Python Enhancement Proposal 578, PEP 578) for every event that carries a path -- open, os.listdir, os.scandir,
  glob.glob, sqlite3.connect, shutil's copy, move and remove operations, os.remove and similar;
- wrappers of os.stat, os.lstat and os.access (which raise no audit event), each added to the os.supports_* sets, and of
  os.path.exists, lexists, isfile, isdir, islink and isjunction -- on Windows, Python 3.12 binds these to C functions that never call
  os.stat (ntpath.py lines 840 to 845).

Each record carries the test and its phase (a pytest plugin keeps them in a context file every measured process reads), the
repository frames, and, for a program start, whether the child inherits the measurement. Windows command lines are split by the
Microsoft C runtime's rules; events go to one file per process under a lock (the Windows C runtime's append is not atomic). Revision 4
is the one used; its calibration (24 checks, on every platform a synthetic Windows-form program start and concurrent writers) ran on
each machine before its run. Every path inside the checkout is then answered by git: tracked, or git-ignored (`git check-ignore
--no-index`, which answers for absent paths too), present or absent.

In the sandbox (Linux, a clean clone of 0e5d73c4): the instrument and PYTHONPATH change no outcome -- 0 of 8,724 tests differ from
the clean-clone validation without them, also with the owner's block's options.

## What the owner's machine measured

- **Block SQ** (2026-11-06 07:05 Coordinated Universal Time (UTC), read-only; record SHA-256 f68be0c4...): the three sequences
  test_esm2_not_in_stub_mode uses, from the owner's index (SHA-256 3d8380bd..., 20,190 rows) and from UniProt's current FASTA
  entries, are identical: BRCA1 P38398 (1,863 residues, sequence SHA-256 80d7199c...), TP53 P04637 (393, e3ca60f1...), PTEN P60484
  (403, 2675a6cc...), each with the residue the test changes (R1699, R175, G129).
- **Block DC, the data census** (08:07 to 09:28 UTC; record c677c092..., evidence zip 35e165b9...): the whole suite in the
  owner's CHECKOUT, on 0e5d73c4, with the validation's locked environment (re-checked: SATISFIED against its Windows lock) and the
  validation's options apart from two measured in the sandbox to change nothing (pytest's cache off; coverage's report in the run
  folder). Calibration passed on that machine.
  * 8,639 passed, 80 skipped, 5 xfailed, 5 subtests passed in 4,630.70 s; the network guard 8,724 checked, 0 tried; the working-tree
    guard unchanged (49,272 entries); after the run no tracked, untracked or ignored entry of the checkout had changed.
  * 125 events files, 233,803 records, 0 instrument errors; 18 child processes started with an explicit environment or -I were not
    measured, and are listed. 13,483 paths answered by git.
  * Compared test by test with the validation's clean clone (full.xml, SHA-256 899e8bf3..., 8,635 passed, 84 skipped, 5 xfailed):
    EXACTLY 4 tests differ, each skipped in the clone and passed in the checkout, and each was found by its paths:

| Test | The git-ignored data |
|---|---|
| test_esm2_activation.py::test_esm2_not_in_stub_mode | data/external/uniprot/uniprot_human_reviewed.parquet |
| test_eve_entry_name_resolution.py::test_uniprot_index_has_entry_name_column | the same index |
| test_eve_entry_name_resolution.py::test_real_corpus_resolution_fraction | the same index; data/external/eve/EVE_all_data/variant_files |
| test_drift_reference_profile.py::test_psi_exact_on_the_real_run15_matrix | outputs/run15_rerun_report/full/splits/X_train.parquet |

  * 46 tests looked at git-ignored data PRESENT in the checkout. The other 42 had the same outcome either way:

| Tests | What they read | Disposition |
|---|---|---|
| 5 that run the Application Programming Interface's (API's) startup: test_runtime_attribution.py (4), test_containment_serving.py (1) | data/processed/dbsnp_index.parquet -- api/main.py's defaults are relative to the current folder, and the loaded tables stay in the module for later tests | fixed: api_without_local_data |
| test_preflight_run17.py (3) | 11 and 12 real data files (the command's paths, checked relative to the current folder); the real STRING graph in data/raw/cache | fixed: a temporary folder; temporary cache and links folders |
| test_data_tree_gate.py and test_storage_guard.py, the two run_all tests | the STRING graph; the storage test also the real data tree through the other gate (3,374 paths) | fixed: the other gate stubbed, temporary folders, and an exception from run_all no longer turned into a skip |
| test_orchestrator_lazy_registry.py::test_ci_data_freshness_pipeline_runs_without_sklearn (in a child) | 81 paths under data/ (data/raw 65, data/external 13, data/processed 3) -- check_local looks under the agent's root, PROJECT_ROOT | fixed: every local check under an empty folder |
| test_data_tree_gate.py::test_the_REAL_tree_is_auditable | the real data tree | by design; unchanged |
| test_ingest_clinvar_snapshot.py (2) | data/processed/clinvar_grch38.parquet, a path comparison | a look, no dependence; unchanged |
| test_dependency_census.py (1) | src/genomic_variant_classifier.egg-info (PKG-INFO, top_level.txt) | EGG-INFO-SHADOW-1, open |
| 24 source scans in 17 files; 3 whole-checkout hygiene walks | ignored folders under src (the egg-info, agent_layer/data, agent_layer/logs) and at the root | SOURCE-SCANS-ENTER-IGNORED-DIRS-1, open |

- **The drift suite** (tests_drift/, 22 tests) was not part of the owner's census. Measured in the sandbox under the same instrument
  (the drift environment; 22 passed, 0 instrument errors): no drift test looked at a git-ignored path in setup, call or teardown.

## The change

- **The ESM-2 sequences, committed** (ESM2-REAL-MODE-FIXTURE-1). tests/fixtures/uniprot/esm2_real_mode_BRCA1_TP53_PTEN.fasta holds
  UniProt's three FASTA entries as served on 2026-11-06 (3,070 bytes, SHA-256 0530d882..., line feeds only, ASCII).
  tests/unit/_uniprot_fixture.py pins each entry -- accession, entry name, gene, sequence version, length, the sequence's SHA-256,
  the residue the test changes -- and builds, in the test's own temporary folder, the index the connector reads, with the columns
  scripts/build_uniprot_index.py writes. test_esm2_not_in_stub_mode uses it and no longer depends on local data.
- **The fixture's bytes, pinned** (FASTA-FIXTURE-EOL-1). Measured before the rule existed: the fixture resolved to text=auto, and a
  clone under core.autocrlf=true -- the owner's Windows configuration -- wrote it with 49 carriage returns (3,119 bytes), which its
  canonical-bytes test refuses. `.gitattributes` now carries `tests/fixtures/**/*.fasta text eol=lf`, after the .cif rule; with it the
  clone holds the committed 3,070 bytes.
- **The three checks of local data, declared** (tests/local_data.py). They exist to check the data itself, so a fixture would defeat
  them. `@local_data(path, ...)` takes each path relative to the repository root -- two of the three read `Path("data/...")`, relative
  to the CURRENT folder, so a run started elsewhere skipped them silently -- and skips the test, by pytest's own skipif, with one reason
  for every such test naming the absent paths ("local data absent (git-ignored; held by a checkout, never by a clone): ..."), unless
  all exist. It also applies the `local_data` mark, registered in pyproject.toml, so `pytest -m local_data` runs exactly them.
- **The API startup, without local data.** tests/unit/conftest.py's api_without_local_data points the three optional tables at
  absent paths (gnomAD's off, as when its variable is unset) and puts the module's tables back afterwards; the five tests that run the
  startup request it.
- **The other readers.** Two preflight tests run in a temporary folder and the third passes temporary STRING cache and links
  folders; the two run_all tests stub the other gate and pass temporary folders, and no longer turn an exception from run_all into a
  skip. The freshness child checks every local source under an empty folder: the orchestrator builds the agent with root
  PROJECT_ROOT (the repository), so the child replaces the detector's check_local -- as it already replaced the network probe --
  records the root the agent passed, and must find exactly the one asset the test plants there (bypassing the redirection fails it).
- **Bytecode** (PYCACHE-FROM-ISOLATED-CHILDREN-1). The dependency-profile test's isolated child (`-I` ignores
  PYTHONDONTWRITEBYTECODE) runs with -B. Measured: without it the one test leaves three bytecode files under src; with it, none.
- **Contracts.** tests/unit/test_local_data_contract.py (21 tests): the local-data tests are exactly the measured three (parsed:
  module-level tests and methods of module-level classes; a class may not carry it); every declared path is repository-relative POSIX,
  git-ignored and untracked (asked of git), so the decorator cannot hide an ordinary missing file; `pytest.mark.local_data` is never
  written by hand (it would carry no skip) and is registered; the decorator refuses no path or a bad one and applies the mark and
  pytest's skip; in a real child pytest session it skips with the one reason where the data is absent, runs the test where it is
  present, and `-m local_data` selects exactly those; every test or fixture that runs the API startup requests api_without_local_data,
  which points every table at an absent path. tests/unit/test_uniprot_sequence_fixture.py (6): the canonical bytes, each pinned entry,
  the built index's columns, and the connector serving the sequences with no network. tests/unit/test_gitattributes_contract.py (+2):
  FASTA fixtures are pinned to LF, today's and the next.

## Errors of mine, recorded

- The first version of the marker was implemented by hooks in tests/conftest.py (pytest_configure, pytest_runtest_setup,
  pytest_terminal_summary). tests/unit/test_session_guard_placement.py failed it: the session guards' hooks are the root
  conftest.py's alone, because a hook implemented twice runs twice. Caught by running the contract tests before any full run. The
  decorator, which needs no hook, replaced it.
- The first fix of the freshness test changed the child's CURRENT folder: I had reasoned from check_local's default root (".")
  without reading its caller, which passes PROJECT_ROOT. The sandbox measurement of the candidate showed the child still looking at
  24 git-ignored paths of the clone, exactly as before. Corrected (above) and measured again: 0.
- A code comment written before the measurement was re-read said the data-tree run_all test had audited the real data tree (3,374
  paths); the census shows that test read one git-ignored path, the STRING graph -- the 3,374 belong to its storage-guard twin, and
  the freshness child's 81 paths are under data/raw, data/external and data/processed, not data/external alone. Both corrected
  from the census before commit.
- In the development clone, while probing the decorator's predecessor, I removed a whole folder a probe had created a file in
  (`rm -rf data`), which deleted 11 tracked files; restored at once with `git checkout -- data`. Rule: remove exactly what a probe
  created.

## Revision B: the owner's first V1 (2026-11-06 18:21 to 19:25 UTC)

-Validate stopped at the targeted tests: 797 passed, 6 failed, 7 skipped; preservation of the checkout verified unchanged; nothing
committed. Two causes, both carriage returns, both mine:

- FIXTURE-WRITTEN-BY-APPLY-1 (5 failures: the fixture's canonical bytes, its three entries, the connector). The installer ran `git apply`
  on the working tree; git wrote the new FASTA file under the .gitattributes read before the patch changed it, and the owner's
  core.autocrlf=true (recorded in the run's environment stage) converted it: 49 carriage returns. The installer's blob check passed
  because `git hash-object --path` normalises line ends. Reproduced in the sandbox with core.autocrlf=true: that path writes 3,119
  bytes with 49 carriage returns; a clone of the commit, a fast-forward merge to it (the post-merge block's path) and a fast-forward
  to a merge commit of it each write 3,070 bytes and none. I had measured the rule on a clone only, and the harness had never run
  with core.autocrlf. The installer now applies the patch to the index (`git apply --cached`), checks the staged tree, has git write
  the changed files from the index (`git checkout-index --force -u`), and compares every changed file's bytes with
  `git cat-file --filters` (the bytes a checkout writes); on the old path that comparison names exactly the fixture.
- STDIN-TEXT-MODE-1 (1 failure: test_every_declared_path_is_git_ignored_and_untracked). The test sent the paths to
  `git check-ignore --stdin` in Python's text mode, which writes "\r\n" on Windows; git answered for "<path>\r" (reproduced on Linux by
  sending those bytes). test_c2_qualification_record.py had recorded exactly this on 2026-10-02 and uses NUL-separated bytes; I did
  not read the repository's existing calls of `check-ignore` before writing a new one. The test now sends NUL-separated bytes, with a
  tracked, never-ignored control path that must not be reported; mutations each fail it: the "\r\n" form, an ignored control, a
  declared path that is not ignored.

## Verification (sandbox)

- The whole suite on a clean clone of the candidate (c3765a36, tree f9cb3a42), under the same instrument, Python 3.12.3 and the
  training-and-test lock's environment, Git 2.47.1 (2026-11-06 10:57:20 to 11:11 UTC): 8,721 passed, 26 skipped, 6 xfailed, 5
  subtests passed in 820.16 s, exit 0; 147,101 records, 0 instrument errors, every test started and ended (8,753); the network guard
  8,753 checked, 0 tried; the warning-filter guard 8,753 checked, 0 changed; the working-tree guard unchanged. Test by test against
  the same measurement of main (run_20261106T073014Z): the 29 new tests passed; the three local-data tests skipped, now with the one
  reason; nothing else differs. After the run the clone held pytest's cache and the gate's two coverage files -- and, unlike main's
  run, no bytecode file (PYCACHE-FROM-ISOLATED-CHILDREN-1 closed in the whole suite).
- Tests looking at a git-ignored path in the clone (every such path is absent there): 29 -- 30 in that run, the freshness test file
  then measured again after its correction (4 passed, 0 git-ignored paths, its child measured) -- against 38 on main. No longer: the
  eleven fixed tests. Newly: the two contract tests that probe one deliberately absent path. The other 27 look only at paths absent
  in a clone: alphamissense_protein_index.parquet relative to the current folder (TestAnnotationPipeline 10, the LOVD wiring 2),
  models/classification_thresholds.json (2), data/reference/reclassification (2), the ClinVar ingest output (2), the real-tree audit
  (1), names a test makes up to be absent (preflight inputs 3, the .gitattributes probe 1), pytest's own probes of models/ in two
  child sessions (2) -- and 2 that are not looks in the clone at all: tempfile.TemporaryDirectory's clean-up removed src.tsv inside a
  temporary folder by name relative to a directory descriptor, and instrument revision 4 resolves such removal events against the
  current folder (its docstring says only "counted, not resolved", which holds for the wrapped stat functions). All 155 removal
  records of the run were read that way; none of the owner's 46 readers was found by a removal (measured), so the census stands.
  INSTRUMENT-DIR-FD-AUDIT-EVENTS-1, recorded with DATA-CENSUS-TOOL-1. In the owner's checkout the census found no git-ignored data present that
  any test reads beyond the 46 above. (The source scans that enter ignored folders under src do not appear in a clone: there those
  folders do not exist.)
- The drift suite under the instrument: 22 passed, no git-ignored path.
- The changed and new test files, and the guard-placement contract: 142 passed, 3 skipped (the three local-data tests, each with the
  one reason), 1 xfailed (the ESM-2 model weights are not in the sandbox).
- Node identities against main (0e5d73c4): +29 / -0; 8,753 collected; no collection warning.
- Sabotage, each caught: the mark written by hand (2 tests fail); the decorator without its skip (2); the mark not registered (1); paths
  relative to the current folder (2); a fourth decorated test (1); a tracked path declared (2); the skip reason without its paths (2);
  the API fixture dropped from a startup test (1); the .fasta rule removed (2); the freshness redirection bypassed (1).
- The harness's synthetic local data (a UniProt index with the two anchors, two EVE files, a 50,000-row matrix) makes the three
  local-data tests run and pass: 3 passed under `-m local_data`, 35 deselected.

## Open, recorded

- SOURCE-SCANS-ENTER-IGNORED-DIRS-1: 24 source scans and 3 whole-checkout hygiene walks enter git-ignored folders (in the owner's
  checkout: under src the egg-info, agent_layer/data and agent_layer/logs; no Python file was seen inside them). Same outcome today;
  a scan that walks local data depends on it.
- DATA-CENSUS-TOOL-1: the instrument and its analysis live outside the repository; committing them as a maintenance tool would let
  any checkout be censused again. With it, INSTRUMENT-DIR-FD-AUDIT-EVENTS-1: removal audit events carrying a directory descriptor
  (shutil.rmtree's descriptor-based walk) must be recorded as such, not resolved against the current folder.
- EGG-INFO-SHADOW-1 (open since 2026-11-01): the census measured test_dependency_census reading the git-ignored egg-info.
