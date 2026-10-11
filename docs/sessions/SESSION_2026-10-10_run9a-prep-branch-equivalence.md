# Session record, 2026-10-10: run9a-prep holds nothing main lacks -- the branch-equivalence record

Author: Monzia Moodie. Design authority: owner instruction 2026-10-10 ("Move forward with all necessary commits" on the GitHub branch
run9a-prep, which GitHub showed as "180 commits not in main"); ADR-0004 (records, placement by role); PATH-BUDGET-1. Base: main at
`ffc549f4` (pull request #54, the lockfile migration; tree `4dfc615f`).

## 1. The question, and why a commit count does not answer it

"Commits not in main" counts commit IDENTIFIERS. A history rewrite gives every rewritten commit a new identifier while its content is
unchanged, so the count can be large while the branch holds nothing new. Retiring the branch is irreversible on GitHub; the decision
needs the CONTENT question answered and kept: does the branch hold any file content, or any change of history, that main lacks?

## 2. Measurements taken before designing

| Question | Measurement | Consequence |
| --- | --- | --- |
| What is run9a-prep? | Tip `c1c01920` (2026-04-30), 183 commits, one root `08e881b`. | The subject. |
| How did it relate to main? | Pull request #1 merged it at 2026-05-01 01:44 UTC: `61c1b2b`, the commit tagged run9a-baseline, whose second parent `c568971` has the tip's tree, identity and dates. | It WAS merged; it is a stale copy. |
| Why does it differ now? | The rewrite of 2026-07-11 (scripts/forensics/verify_postrewrite.py: git filter-repo excised data/alphafold_cif_cache_2026-07-03.tar.gz and notebooks./gitkeep) renamed every later commit on main and the identifiers quoted in messages. | Commits must be paired by content, and quoted identifiers mapped, not compared as text. |
| Is the owner's preserved bundle the same branch? | `history/run9a-prep_c1c01920aa0b_20261010T221514Z.bundle`, sha256 `1e83929f...e4f4`, 3,796,302 bytes: `git bundle verify` passes (complete history); one head, the tip; the same 183 commits; 0 of its 1,237 objects absent from the GitHub mirror. | It can stand in for the GitHub branch once that is deleted. |
| Do tags hold anything only the branch reaches? | Every remote tag (run9a-baseline, v2.0.0, v4.0.0) points into main. | Nothing else pins the old history. |
| Does an existing pairing corroborate? | The pre-rewrite commit map compiled on 2026-09-21 (58 rows) covers July-era commits only; no overlap with run9a-prep. | Neither corroborates nor contradicts; not used. |

## 3. The design in one line per component

- **Owner** (`repository_records/branch_equivalence.py`, VERIFICATION_RESULT, family `repository-history/branches/`): `derive` pairs
  each branch-only commit with a reference commit by tree, author, committer and both dates, then requires the parents' counterparts
  and the message to agree (quoted identifiers mapped exactly, same length, to the counterparts'); a commit with no counterpart is
  explained only if it changes declared excised paths alone AND its tree without them equals its parent's counterpart's tree; every
  object the branch reaches and the reference does not is typed and listed. Equivalent only if every commit is explained and no blob
  is missing. `render` is deterministic; `load` is strict and re-derives every figure; a command line derives and verifies.
- **Record** (`REC-9ccf9fa0863b447292230fe1130850b0.json`, 61,210 bytes, sha256 `9141d73e...19ff`): bound to the preserved bundle
  (store, path, digest, size, head), pinned against line-ending conversion (BRANCH-EQUIVALENCE-RECORDS-PINNED-1).
- **Installer**: verifies the bundle (digest, size, `git bundle verify`, the one head), fetches it into its disposable clone under a
  private ref, re-derives the record with the module being committed and requires byte equality before anything is committed.

## 4. The result

180 branch-only commits: 169 identical, 10 identifier_rewrite (13 identifier pairs, e.g. `cb9a4cb` -> `a9f5ff0`), 1 excised_paths_only
(`9bdd270`, which only added notebooks./gitkeep), 0 differs, 0 ambiguous, 0 no_counterpart. Objects the branch reaches and main does
not: 180 commits, 2 trees (`3a113fe2`, `793bc8da`), 0 blobs. Verdict: equivalent. Derived byte-identically three times: from the GitHub
mirror, from the bundle fetched into a fresh clone, and again after the last edit to the module (its docstring).

## 5. Verification

- 43 tests on histories built with git plumbing (every outcome; the rewrite's pattern reproduced; trees written with `git mktree`,
  never through an index -- see section 6), the strict loader (17 tamper cases, malformed bytes, duplicate keys, a non-canonical
  rendering, key order) and the committed record; 4 attribute-contract tests.
- The first run had 3 failures, all the tests' own: the synthetic preserved copy named no head (the loader rightly refuses that), and
  one expectation counted 2 new trees where git stores 1 (the excised `notebooks./` subtree holds the same single blob as main's
  `notebooks/`, so it IS main's subtree object).
- Seeded defects: 15 against the module; the first run left 3 survivors. Two were test gaps, now closed: the remaining-tree
  comparison (a declared excision the reference still holds must explain nothing) and untyped blob identifiers in `load`. The third
  -- removing the subset-of-excised-paths test -- is EQUIVALENT: with every parent counterpart's tree equal to the parent's tree
  without the excised paths, the tree comparison implies the subset test; it is kept, documented, as a short-circuit. Second run:
  14 of 15 detected, the survivor being that equivalent mutant.
- Full suite (development sandbox, by node identity, against main `ffc549f4`): 0 outcome changes on 8,345 shared tests and 47 added,
  all passing -- with the sandbox's git 2.43.0, with a Linux build of Git for Windows 2.55.0.windows.5 whose index applies its Windows
  path check, and with that build under the Windows-newline simulation (its 14 simulation-specific failures identical on both sides).
  The failures common to both sides are the sandbox's own (see the correction named at the end of section 6). Measured 2026-10-11 on
  the committed tree; the 2026-10-10 runs (44 added) measured the unit before the history helper changed.
- The installer's re-derivation script, run in a simulated clone: byte-identical (exit 0); three negative controls refused (the
  bundle ref moved one commit back: a different record; one byte added to the record: refused by the strict loader; no bundle ref:
  refused by git), each exit 1.

## 6. 2026-10-11: the owner's runs

| Question | Measurement | Consequence |
| --- | --- | --- |
| Does `wsb stop` close a sandbox from an ordinary window? | The orphaned first sandbox (started from an elevated window): exit 0, closed at 0.7 s. | Revision c's closing works, even across integrity levels. |
| Does revision c pass? | 26 of 26 checks (21 before the closing, 5 after); closed by `wsb stop`; archive `d142b790`, terminal record `50d36b09`. | The isolation mechanism is qualified; the replay itself is not yet built. |
| Does the record re-derive on the owner's machine? | `rederive` stage: byte-identical, `9141d73e`, verdict equivalent. | The evidence chain is closed on his machine. |
| Why did the first installer issue fail? | 372 of 372 targeted tests errored at setup: PermissionError on `TEMP\pytest-of-monzi` (os.scandir). | Expected cause, measured next on his machine: Python 3.12.4+ makes that folder with an owner-only access list, and one made from an administrator window is unreadable from an ordinary one. |
| Did the failure change the checkout? | Inventories before and after: 8,661 files, identical except the two timing fields. | Preservation verified. |
| Does a private temporary root change any test outcome? | Full suite with its own root: 0 outcome changes on 8,389 tests; deepest temporary path 180 characters below the basetemp. | Revision b of the installer uses one per run. |
| Who owns TEMP\pytest-of-monzi? (Blocks T1, T2; 01:53 UTC, ordinary window) | Created 2026-10-09 06:51:28 UTC, 0.8 s before the first of 973 tmp* folders five test sessions left in TEMP. Its listing and its access list cannot be read, and renaming it is refused (IOException, access denied). | The owner is not this window's account; it is read next from an administrator window, which can also rename it. My T2 comment that an ordinary window could rename it was wrong. |
| Does revision b pass on the owner's machine? | Every stage through `temp_root` passed (the private root's probe: 2 passed) and the record re-derived byte-identically again; then 14 of 372 targeted tests failed with one cause: "error: Invalid path 'notebooks./gitkeep'" from `git update-index` in the tests' history helper. Preservation verified unchanged. | Git for Windows 2.55.0.windows.5 checks every index path against Windows naming while core.protectNTFS is on (read-cache.c verify_path, compat/mingw.c is_valid_win32_path). |
| Is that the whole cause? | A Linux build of that Git for Windows source with is_valid_win32_path copied verbatim into its index check: the owner's run reproduced exactly (14 failed, 356 passed, 2 skipped; the same 14 node identities). With the helper writing trees by `git mktree`: 373 passed, 2 skipped (the Windows-only junction tests). | The helper no longer uses an index (+3 tests); installer revision c. |
| What were the owner's 2 skips? | Positions 101 and 102 of 372; PowerShell orders test_artifact_inventory_committed_records.py (5 tests) before test_artifact_inventory.py, which places there the two tests that need a symbolic link. | A privilege an ordinary window lacks; both junction tests ran and passed. Revision c prints every skip reason. |

The full-suite failures of the development sandbox were attributed, here and on 2026-10-09, to missing packages; measured on
2026-10-11, 8 of them come from its git 2.43.0, 5 from a lightgbm and scikit-learn mismatch and 2 from its relative PYTHONPATH:
docs/sessions/CORRECTION_2026-10-11_the-sandbox-failures-are-not-all-missing-packages.md.

## 7. Not done here

- The GitHub branch is NOT deleted by this change. Deleting it is a separate, irreversible owner step, after this record is on main,
  and only while GitHub's run9a-prep still points exactly at `c1c01920`.
- TEMP on the owner's machine is repaired by owner blocks, not by this change: the folder's owner read from an administrator window
  and the folder renamed aside there (reversible), a real pytest run proving a fresh folder from an ordinary window, and the three
  folders the two failed validations left removed after their identities are checked against their own logs.
- The 72 tempfile.mkdtemp() call sites in nine test files that leave directories in TEMP move to tmp_path in a separate change.
