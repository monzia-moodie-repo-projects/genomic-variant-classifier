# CORRECTION 2026-10-11 -- the development sandbox's failures are not all missing packages

**Author: Monzia Moodie**
**Measured at commit:** ffc549f4 (main; pull request #54)

**Applies to two tracked documents, each pinned at all sixty-four characters as they stand on main at ffc549f4:**

```
docs/CHANGELOG.md
    4a1eac3fc7a2e86f93337576fc8ca3573579d9d44d30dbcbcb54ad7c4d54c475
docs/sessions/SESSION_2026-10-09_run-intent-and-evaluator-boundary.md
    60ed493607c2180b62ae6f5c357e8b328b40697dceb0211cbddf6cba28d2f62a
```

Corrections belong BESIDE records, never inside them. No file named here is edited by this document.

## 1. What was written

Three statements about full-suite runs in the development sandbox (the cloud container in which changes are built before the owner's
Windows validation) said its "60 failures and 30 errors come from packages absent there":

- `docs/CHANGELOG.md`, entry "2026-10-09 (environment qualification 9, completed)", its FULL SUITE paragraph;
- `docs/CHANGELOG.md`, entry "2026-10-09 (science stage)", its FULL SUITE paragraph;
- `docs/sessions/SESSION_2026-10-09_run-intent-and-evaluator-boundary.md`, section 5.

## 2. What was measured (2026-10-11, the full suite at main ffc549f4, by the first line of every failure and error)

| Count | Cause |
| --- | --- |
| 33 | the starlette test client needs httpx (28 errors at setup, 5 failures) |
| 25 | pyfaidx absent |
| 13 | seaborn absent (11 directly, 2 lazy-import probes) |
| 5 | the sandbox's lightgbm 4.5.0 (the owner's is 4.6.0) calls scikit-learn's check_X_y with force_all_finite, which scikit-learn 1.8.0 no longer accepts |
| 2 | collection errors: networkx absent |
| 2 | a child Python process started from another working directory cannot import the package: the sandbox sets PYTHONPATH to the relative path `src` instead of installing it |
| 1 | the import-resolution gate: 9 names that need seaborn or torch (absent) |
| 1 | the roster-fitting test: torch absent, and the lightgbm mismatch above |
| 8 | the sandbox's git is 2.43.0, which refuses `--no-lazy-fetch` ("unknown option: --no-lazy-fetch"); all 8 are release-approval and interpretation-contract tests that read git objects |

Total: 60 failures and 30 errors.

The 8 git failures are present in every sandbox full-suite run of 2026-10-09 to 2026-10-11 that was checked (five runs). They pass
when the same tests run with git 2.55.0.windows.5 (a Linux build of that Git for Windows source, the version on the owner's machine):
the base then has 52 failures and 30 errors.

## 3. What this changes

Nothing about any outcome comparison: every one of those comparisons was by node identity, before against after, in the same
sandbox, and the failing set was identical on both sides. What was wrong is the stated CAUSE: two version differences (git, lightgbm) and a sandbox setup
choice were described as missing packages. The run9a-prep unit's documents (2026-10-10 and 2026-10-11) state the measured causes.
