# Session record, 2026-10-09 / 2026-10-10: the lockfile migration admitted (ruling 2026-10-08f, step 7)

Author: Monzia Moodie. Design authority: owner rulings 2026-10-08c, 2026-10-08d, 2026-10-08e and 2026-10-08f; ADR-0004 (records,
preservation, AUTHORITY-SUCCESSION-1). Base: the artifact-inventory record unit (tree `474b9746`), itself on the run-intent unit.

## 1. What was admitted, and what was not

Approved (ruling 2026-10-08e): proposal `f21ac4bca99dabd28f2201fec4e27115c3ad4cc8791050f7d7ddbaf0dee645b9` -- 90 field transitions in
13 existing packages (all provenance corrections of the baseline lock to what the qualified library contains), 15 additions (DANDELION,
its dependencies and the locally built qvalue), R 4.6.0 -> 4.6.1. Admitted now: that the baseline lockfile, the candidate lockfile, the
selected artifacts and the proposal AGREE. Not established here: that the environment is qualified -- the isolated replay re-verifies
every byte it installs, and the method tests run on that library.

## 2. Measurements taken before designing

| Question | Measurement | Consequence |
| --- | --- | --- |
| Does the existing transition owner admit the approved proposal? | Yes: 90 transitions, 15 additions, 102 packages after. | Reuse it; add only what was missing. |
| What did it not check? | The restoration effect (ruling c), the plans (ruling d), the run's own report, the regeneration, the run evidence. | admission policy in lockfile_admission.py. |
| How is the candidate encoded? | 228,247 bytes, 4,002 CRLF line ends, no lone CR; canonical text 224,245 bytes, `ce6aa8b4`. | Two digest domains; `-text` for the preserved copy. |
| Would git keep the CRLF candidate? | `*.lock text eol=lf` matched it (git check-attr). | MIGRATION-ARTIFACTS-PRESERVED-1. |
| Which tests read the live renv.lock as the baseline? | test_environment_qualification (runtime-only admission) and test_environment_admission (ported cases). | They now read the preserved baseline. |
| Is the evidence free of local paths? | Path-like matches were `https://` and JSON escapes; "downloads" occurs in curl's own description. | Public verbatim preservation. |
| Which role? | ADR-0004: a migration manifest is part of the migration's evidentiary record. | MIGRATION_RECORD under records/migrations/. |

## 3. The design in one line per component

- **Admission policy** (`environment_qualification/lockfile_admission.py`): the acceptance contract is code (the rulings' digests),
  never the evidence; seven ordered checks, one reason code each; corroboration is reported, never promoted to authority.
- **Record owner** (`repository_records/lockfile_migration.py`): eight fixed parts, placement by role, strict parsing,
  deterministic rendering, exact inventory and bytes; it does not decide admission.
- **Succession**: renv.lock now holds the admitted successor's canonical text; the predecessor survives verbatim with the new
  provenance relation SUPERSEDED_AUTHORITY; a test re-derives the admission from the preserved bytes (the behavioural gate).

## 4. Verification

- The policy admits the real evidence; its result equals the committed manifest's admission.
- Every semantic check refuses a consistent forgery (all digests re-bound), after the forger was shown to reproduce the preserved
  bytes exactly; every binding refuses a one-byte change with the contract unchanged.
- Seeded defects: 45 of 45 detected in a disposable worktree. The first battery run found one survivor (a plan whose label no longer
  describes its body); two controls were added. An interrupted run left one mutant applied in the DISPOSABLE worktree only; it was
  restored with git checkout and the battery re-run to completion in the background.
- The committed blobs equal the preserved bytes, including the CRLF candidate (`e391298a`).
- Full suite of the first issue (development sandbox, by node identity, against the record unit): 0 outcome changes on 8,230 shared
  tests and 91 added, all passing, natively and under the Windows-newline simulation.

## 5. 2026-10-10: the first issue failed to apply on Windows; the record is flat and the repository has a path budget

| Question | Measurement | Consequence |
| --- | --- | --- |
| Why did `-Validate` stop at `apply`, twice? | Clone root `C:\Users\monzi\AppData\Local\Temp\gvc_lockfile_migration_candidate_<stamp>`: 90 characters. Longest patched path (per-part layout): 169. Together with the separator: 260. | Over the 259 characters a Windows path may hold; Git for Windows refuses it unless `core.longpaths` is set. |
| Why did no simulation see it? | Every installer simulation ran on Linux, which has no such limit. | The repository itself must hold the budget. |
| What did the repository bound before? | Nothing. Longest tracked path on main: 141; longest directory: 93. | PATH-BUDGET-1. |
| Is a directory per part needed? | The eight basenames are distinct. | Flat `artifacts/<original basename>`: longest 148. |

- **Budget** (`repository_records/path_budget.py`): file path at most 150 characters and directory at most 138 (UTF-16 code units),
  derived from the Windows limits (259 for a file, 247 for a directory) and a declared root allowance of 108 characters. Enforced by
  `identity.ArtifactInstance` for every record owner and tested over every tracked path (`git ls-files`).
- **Record**: rebuilt by the same builder from the same three bound sources; new record id `REC-722981980c854434ac9e3e7ec86956fe`,
  manifest `6cc2f488`; every preserved artifact has the digest it had.
- **Verification**: 18 seeded defects, all 18 detected (a first run left one survivor -- `windows_length`'s own refusal of a
  non-text value was never exercised because its caller refuses first -- and a direct test was added). Full suite (development
  sandbox, by node identity, against the record unit): 0 outcome changes on 8,230 shared tests, 115 added, all passing; the same
  under the Windows-newline simulation (its 14 simulation-specific failures exactly the record unit's).
- **Installer**: before cloning it measures the full Windows length of every path the clone will hold, from its actual root, and
  refuses before any work; a failing stage now prints its evidence.
