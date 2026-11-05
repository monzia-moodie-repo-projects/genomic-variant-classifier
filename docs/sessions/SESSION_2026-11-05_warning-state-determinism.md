# Session 2026-11-05 -- WARNING-FILTER-REORDER-AT-FIRST-IMPORT-1: every test file passes alone

Author: Monzia Moodie

The session began on 2026-11-04 at 22:35 UTC with set "af" (DRIFT-SUITE-UNGUARDED-1, pull request #64) verified complete on the
owner's machine, and closed the finding that change's independent review recorded: a test's warning filters depended on which
tests ran with it.

## Set "af", verified on Windows (2026-11-04)

- -Validate once, 17:34:52 to 19:36:11 UTC (121.3 minutes; package downloads 18.3-67.5 MB/s, no retries). Stage minutes:
  inventory 5.0, apply 1.0, the training-and-test install 17.4, the drift install 21.6, targeted 10.6, collection 1.8, the full
  suite 55.2, drift 1.3, the rest 7.4. The full suite: 8,559 passed, 83 skipped, 5 xfailed, 0 failed, no warnings, coverage 83%;
  the warning-filter guard 8,647 checked, 0 changed; the clone's working tree unchanged. Targeted: 635 passed, 3 skipped. Drift:
  21 passed, the guard 21 checked, 0 changed. Receipt 9730aa22, candidate 45bf3891.
- Pull request #64 was merged with a merge commit: adb2e2fd663ea300dbfce250d948b6ec260d3a0a (parents c98c5b92 and 45bf3891, tree
  679596de -- the sandbox's final tree), read from GitHub with `git ls-remote` at 22:36:03 UTC: main only.
- The post-merge block (21:57 UTC): targeted 635 passed, 3 skipped, no warnings, the guard 638 checked, 0 changed; drift 21
  passed; the owner's checkout unchanged (49,267 entries); both candidate environments still exactly their Windows locks; 81
  untracked files unchanged. The branch was deleted at 22:12 UTC.
- Every block the owner pasted equals its delivered file. Lines 3252-3813 of the uploaded console output repeat lines 2689-3251
  (one prompt line apart): not new runs.

## What was wrong, measured

- pytest 9.0.3 (read: _pytest/warnings.py) runs collection inside `warnings.catch_warnings()` and discards what collection
  registered; its configuration opens a second catch that lasts until unconfigure; each test runs in its own catch with the
  configured entries at the front.
- Some libraries register filters when first imported. A census plugin that wrapped `warnings._add_filter` from pytest's plugin
  loading to the end of a full collection (8,647 tests; 2026-11-04 22:42 Coordinated Universal Time (UTC)) counted 124 calls: 61
  by pytest itself (its configured entries per phase, and 29 pytest.importorskip calls) and 63 by libraries and Python's os
  module. The ones that persist -- made outside the library's own catch_warnings -- come from thirteen libraries: requests,
  urllib3, numpy, pandas, scipy (three modules), torch (two), sympy, torch_geometric, accelerate, mlxtend, srsly, spaCy and
  pyspark; the rest are inside a catch_warnings of their own (pandas, matplotlib, tqdm, os, importorskip) and leave nothing
  behind (each read). The independent review's census, loaded at a different point, counted 116 calls with the same libraries.
- In a full run collection imports them all, and no test runs under their entries. In a run of fewer files a library could be
  imported for the first time inside a test, and its entries went to the front of that test's filters, ahead of pyproject.toml's.
  mlxtend's `simplefilter("always", DeprecationWarning)` is identical to an entry pytest registers, so it MOVED that entry: the
  guard failed the test. A NEW entry was only listed, though its effect is the same.
- Each of the 425 test files run ALONE on main (adb2e2fd), in the candidate clone whose editable install is its own src, with
  Continuous Integration's options apart from coverage, one at a time (2026-11-04 22:52 to 23:52 UTC, resumed after two container
  restarts; 3,216 s of runs; tools: per_file_census.py, preserved with the logs): 21 files not clean.
  * 17 failed the warning-filter guard at one teardown each: test_1kgp, test_alphafold, test_calibration_implementations_agree,
    test_catboost, test_containment_scripts, test_core, test_correctness_harness, test_data_tree_gate,
    test_genomiclm_llr_is_computed, test_invariant_ownership, test_lovd_annotation_reaches_training_matrix, test_model_admission,
    test_model_input_width, test_runtime_attribution, test_storage_guard, test_train_cnn_activation,
    test_variant_ensemble_save_load.
  * 2 leaked tests/unit onto sys.path: test_api (two tests insert it inside the test) and test_variant_ensemble_save_load. The
    full run hides it: other modules insert the same folder at module level during collection.
  * tests/unit/test_dbsnp_pop_preference.py: "no tests ran" (exit 5). It was a script: its checks ran as module-level asserts
    during collection, it defined no test, and pytest counted and reported nothing for it.
  * test_interpretation_contract and test_release_approval failed only because that run's PATH had Git 2.43, which lacks
    `--no-lazy-fetch`; with Git 2.47.1 they passed (32 and 85). The suite needs Git 2.45 or later and says so nowhere a failing
    test would show: GIT-VERSION-REQUIREMENT-UNSTATED-1, open.
  * 378 of the 425 passed with library entries listed as "added during a test".
- A first census, run from a worktree whose environment's editable install pointed at a different clone, also showed sys.path
  leaks in test_alphafold.py: scripts/build_alphafold_parquet.py inserts src/ unless that exact path is already on sys.path. Not a
  defect of the suite under its own installation; recorded because a checkout run against another checkout's environment leaks.

## The design

- PREVENTION. The root conftest.py imports every module of `_FILTER_REGISTERING_MODULES` at session start -- after the
  working-tree guard's first snapshot, before collection -- inside `warnings.catch_warnings(record=True)`. What a module registers
  is discarded, as collection discards it; the module stays imported, so no later import registers anything again; what its import
  warns goes to pytest's report through `pytest_warning_recorded`, as a warning during collection would. A module the environment
  lacks is named as absent (no test can import it either); one that is installed and fails to import stops the session with a
  usage error that also carries what the import warned and what it changed in the working tree, because pytest runs neither the
  terminal summary nor sessionfinish after session start raises. A trade-off, chosen: a test guarded by pytest.importorskip used
  to skip when its library was installed but broken; now the session stops. Rejected: a list in pyproject.toml read as an ini
  option (a configuration-only child pytest would warn about an unknown option); importing under `simplefilter("ignore")` (it
  would hide an `error::` entry's failure during import).
- COMPLETENESS. A `pytest_collection` wrapper, trylast so it runs innermost (measured by the review: inside pytest's collection
  catch_warnings), compares the whole warning state -- the filters, warnings.showwarning and pytest's recording hook -- before and
  after collection, and stops the session with a usage error naming every difference: a library missing from the list, or a test
  module that registers a filter at module level (collection discards it, so it never governed a test). Under `-p no:warnings`
  nothing is discarded, and the check stops the session all the same (measured by the review).
- DETECTION. The per-test guard fails an added entry as it fails a moved one. One comparison, `_filter_differences`, serves both
  places. It says where each entry now stands -- "new: ... (position i of n; ahead of k of the N entries this test started with)"
  and "moved: ... (now ahead of k entries it used to follow)" -- and fails closed on any other difference (an entry duplicated by a
  direct list operation, which simplefilter() and filterwarnings() never produce).
- `child_conftest` (a fixture of the root conftest): the one source of a child project's conftest.py -- this file with the list
  replaced by given names, found by parsing; a test proves nothing else differs. Child sessions that test another guard import no
  heavy library; the scikit-learn contract's children list scipy.sparse and scipy.special, which their collection imports
  (measured: with an empty list the completeness check stops them, naming those entries).
- pyproject.toml: torch_geometric 2.7.0 warns three times when imported, about its own code (it imports its deprecated
  torch_geometric.distributed, and calls torch.jit.script on two of its classes at import, which torch 2.11.0 deprecates). Until now
  the suite never showed them: its first import of torch_geometric was inside pytest.importorskip, under `simplefilter("ignore")`. Two
  entries, each scoped by message and issuing module; a test imports torch_geometric in a fresh interpreter and requires exactly those
  three warnings without the entries and none with them, and the project calls neither interface (parsed).
- The tests/unit helpers are imported as package modules (`tests.unit._admission_support`, `tests.unit._review_map_detector`):
  seven files no longer edit sys.path for them (this relies on the repository root being on sys.path: pytest's default import
  mode provides it, and `python -m pytest`, which Continuous Integration and the installers run, always does).
  test_dbsnp_pop_preference.py's checks are 8 tests.

## Verification

- Full collection with the check: 8,647 tests at the base, 8,667 after (+21 / -1 by node identity), the check passed;
  session-start imports 3.0 to 4.2 s in the training-and-test environment (this session's runs and the review's).
  Mutation: spaCy removed from the list -> the full collection stops (exit 4) naming spaCy's and srsly's entries.
- Eleven mutations of the final mechanism, each caught by the tests (run_mutations.py, mutations_v2.json; 1 to 5 failing each):
  no session-start import; added entries accepted; no collection check; import warnings dropped; imports outside
  catch_warnings; absent modules imported; no duplicate fallback; moves not reported; the collection check comparing filters only;
  the failure message without the working tree; without the warnings. (A first run was stopped by a kill that bypassed its
  restore and left one mutation in conftest.py: found by a search for every mutation's text, undone exactly, and the runner now
  keeps a backup on disk and restores on termination.)
- The drift suite in the drift candidate environment: 22 passed; the guard 22 checked, 0 changed; "absent in this environment:
  torch, torch_geometric, accelerate, mlxtend.frequent_patterns, spacy, pyspark".
- EVERY TEST FILE ALONE, ON THE CHANGE (2026-11-05 00:44:57 to 01:46:21 UTC; clone c with the final patch 43185772 applied --
  the committed tree apart from this record's and the CHANGELOG's text; one file at a time; 3,680 s of runs): 426 of 426 files
  exit 0; the warning-filter guard "0 changed" in all 426; the working tree unchanged in all 426; 8,635 passed, 26 skipped, 6
  xfailed in total -- the whole suite, 8,667. Before the review's fixes, the work-in-progress patch ddcf7123 had run 328 files,
  all clean (stopped to re-run on the final code).
- THE FULL SUITE IN ONE INVOCATION, as Continuous Integration runs it (coverage on, --assert-suite-size, the six publication
  variables unset; 2026-11-05 01:46:59 UTC; the same clone and patch): 8,635 passed, 26 skipped, 6 xfailed, 5 subtests passed, 0
  failed, no warnings, coverage 84%; the warning-filter guard 8,667 checked, 0 changed; session-start imports 6.44 s; the working
  tree unchanged; 1,056.95 s. Then the drift suite in the drift candidate environment: 22 passed; the guard 22 checked, 0 changed;
  imports 11.28 s; the working tree unchanged.
- After those runs only the CHANGELOG, the ROADMAP, this record and one assertion of test_filter_registering_imports.py (split
  into two lines, the same checks) changed. Re-run on the final tree: that file and the eight document and repository tests, 84
  passed; the working tree unchanged. The installer's -Validate runs the whole suite on the committed tree again, in fresh
  environments built from the locks.

## The independent review

A separate review agent read the change against the base and measured in its own copies (2026-11-04 23:57 to 2026-11-05
00:35 UTC, resumed once after a container restart). It was given the code and the questions, not the author's conclusions. Its
findings, and what was done with each:
1. DEFECT: the first comparison inferred a "front" part, and named entries "now at the front" that were not: an entry appended at
   the end, one inserted in the middle, and a swap of two entries (measured with four-entry cases). REPLACED: each new entry is
   reported with its position and how many of the starting entries it now precedes, each moved entry with how many it now
   precedes that it used to follow; a test places every shape exactly.
2. DEFECT: "23 subpackages" -- 24 (22 packages, and monitoring/ and training/ without __init__.py). CORRECTED.
3. RISK: an entry duplicated by a direct list operation was not a difference. The comparison now fails closed on any difference
   left over; a direct re-insertion of an existing entry is reported as a move.
4. RISK: a session-start usage error dropped the import's warnings and its writes into the checkout (pytest skips the summary
   and sessionfinish), and the summary's "not imported" line could never appear. The message now carries both, measured by two
   new child tests; the line now describes the one case it can appear in (the conftest loaded after the session started).
5. RISK: the collection check compared the filters only. It now compares the whole state the per-test guard compares (a test
   module that replaces warnings.showwarning stops the session).
6. RISK: an installed but broken library now stops the session where importorskip used to skip. Kept, and stated in the code,
   this record and the CHANGELOG; an exception that is a warning is named as an `error::` filter's doing.
7. WORDING, all corrected: "the way Continuous Integration runs the suite" (it runs one invocation: each file alone used its
   options apart from coverage); the census count (124 calls, 61 of them pytest's own; the review's census counted 116); one list of
   thirteen libraries; measurement dates; "emptied" where the scikit-learn contract's children list two modules; the collection
   message under `-p no:warnings`; the dbSNP mutation claim (every adjacent swap measured: 2, 1, 1, 1, 2 failed); acronyms in new
   prose; line widths and the README alignment; torch_geometric "calls" torch.jit.script; rows, not variants.
8. RISK (low), recorded: not run on Python 3.11 here (Continuous Integration runs it on the pull request); the helpers' package
   imports need the repository root on sys.path (pytest's default import mode and `python -m pytest` provide it).
Found sound by the reviewer, measured: the collection wrapper runs innermost, inside pytest's catch; the session-start order
(snapshot, imports, collection) under the configuration's filters; `--collect-only`, `-p no:warnings`, a collection error, an
interrupt during collection; both full collections and the +21 / -1 transition; the spaCy mutation; the drift environment's five
and six; the torch_geometric premises and the entries' scope; no test asserting a module absent from sys.modules, and every
sys.modules stub passing alone; one module object for each helper; the dbSNP rewrite's inputs and values; the child conftest's one
difference. Its own checks: all 27 simplefilter/filterwarnings/resetwarnings calls in src, scripts, tests and tests_drift sit inside
a catch_warnings or pytest.warns; after the listed imports, importing all 1,741 module names the project imports registered nothing
further; 22 files run alone in the change, each 0 changed.

## Open, recorded

- GIT-VERSION-REQUIREMENT-UNSTATED-1 (above).
- TESTS-REACH-THE-NETWORK-1: during the per-file runs the sandbox's proxy refused connections to storage.googleapis.com,
  databases.lovd.nl, bioconductor.org, huggingface.co (seven times in one window) and rest.uniprot.org; tests/unit/test_core.py took
  261 to 270 s alone in three censuses, with the processor nearly idle for minutes (load average 0.05), which is the shape of waiting
  on refused connections. The tests that opened them are not yet attributed. A suite run should not reach the network: the next change
  measures which tests do (a socket audit hook recording each connection with its test) before deciding a guard.
- PYTHON-FLOOR-DECLARED-3.10-1: pyproject.toml declares requires-python ">=3.10"; the tests use tomllib (3.11+), and Continuous
  Integration and both locks are Python 3.11 and 3.12.
- PYPROJECT-HOMEPAGE-STALE-1: pyproject.toml's Homepage names github.com/monzia-moodie/genomic-variant-classifier; the repository
  is github.com/monzia-moodie-repo-projects/genomic-variant-classifier.
- docs/ROADMAP.md section 5 has not been re-derived since 2026-09-04; its priorities are the owner's ruling.
- Carried: WINDOWS-RSCRIPT-NOT-ON-PATH-1; the check logs of pull requests #63 and #64; the 101 tracked root files (15
  Install_*.ps1, 3 *_survey.txt); scikit-learn #35089 for production threads; SILENT-FALLBACKS-1; EGG-INFO-SHADOW-1;
  REPORT-PROVENANCE-ABSENT-1; the 14 install paths not lock-qualified.
