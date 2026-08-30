# SESSION 2026-08-30 -- a tool that was already printing it

**Author: Monzia Moodie**
**Commits:** `95f6c44`, `c8c3240`, `6545b64`, `6f9ae43`, `fd6cd4e`
**Ratchet:** 5705 -> 5719
**Preceding head:** `b3619f2`
**Ending head:** `fd6cd4e`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `95f6c44` | CORRECTION-PART-3 | NEUTRAL | 5705 | 5690p/15s |
| `c8c3240` | INCIDENT-GENCODE-UNDECLARED | NEUTRAL | 5705 | 5690p/15s, 1084.4s |
| `6545b64` | CORRECTION-TOOL-ALREADY-KNEW | NEUTRAL | 5705 | 5690p/15s, 1114.5s |
| `6f9ae43` | CORRECTION-PART-2 | NEUTRAL | 5705 | 5690p/15s, 1106.2s |
| `fd6cd4e` | DATA-TREE-GATE | ADDITION +14 | 5705 -> 5719 | 5704p/15s, 1087.1s |

One work commit and four corrections, three of which correct the one before.

---

## 1. The arc: a question about a type became a question about callers

It began with `ARTIFACT-KEY-INSUFFICIENT-1`, recorded at `482c0c9` as blocking
Phase 1C. Testing the manifest's own pattern against all four measured
collision classes left exactly ONE genuine product case:

| collision | files | already modelled by |
|---|---|---|
| `ClinVar/vcf` | 2 | `CoordinateContext` -- they differ by ASSEMBLY |
| `ClinVar/primary_release` | 18 | `acquire`/`regenerate` -- project-derived |
| `EVE/csv` | 3,212 | nothing; PARTITIONS, a different axis |
| `GENCODE/sequence_fasta` | 3 | **nothing. The genuine case.** |

And GENCODE is declared by no registry. Chasing that produced `c8c3240`:
636,522,106 bytes under `data/external/gencode/`, acquired with the publisher's
own manifest and File Transfer Protocol listing, validated by a dedicated
script, consumed by nothing.

Then `configs/data_sources.json` -- a second NAMING authority, fifteen absolute
Google Drive paths rooted at the superseded data root, disagreeing with the
manifest on six of fifteen names.

And a validator whose `GENOMIC_DATA_ROOT` is SET and points at a directory that
does not exist, so it exits 2 on intact data.

---

## 2. `fd6cd4e` -- the auditor acquires a caller

`AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1`. MEASURED: ten tracked files name
`audit_data_tree`, and every one is documentation, a `.gitignore` comment, or
the script itself. The runbook's "Run it at session start and before every run"
is an instruction to a human.

**It had already reported the finding nobody had seen.** Run on 2026-08-30 it
named `gencode`, `grch38` and `eve_smoke` as orphans -- 4,685,941,722 bytes --
and classified `processed/` and `raw/` correctly as untracked BY DESIGN, which
my own probe did not distinguish.

### The refactor

The computation was locked inside a 159-line `main()` with 25 `print()` calls.
Split into ONE COMPUTATION AND TWO RENDERINGS:

```
audit_tree(data_dir, manifest) -> AuditReport      computes, prints nothing
audit_rows(data_dir, manifest) -> [(sev, msg)]     renders gate rows
main(argv) -> int                                  renders the table + JSON
```

`AuditReport.return_code` derives the exit code, so the gate and the command
line cannot disagree. `audit_rows` invents no severities: FAIL for a blocked
tree or a controlled-tier sync violation, WARN for orphans, aliases, naming and
review-tier, OK otherwise.

**BEHAVIOUR-PRESERVING, PROVEN BY BYTE COMPARISON.** An earlier version grouped
findings by category; the original interleaves them per directory. The diff
caught it, and every finding is now recorded in ENCOUNTER ORDER as
`(kind, name, extra)` with both renderers walking one sequence.

### The gate

`data_tree_gate()` loads the auditor by path -- the same
`importlib.util.spec_from_file_location` mechanism `storage_gate` uses,
registering in `sys.modules` BEFORE exec because the auditor now declares
dataclasses -- and returns a FAIL row rather than raising. `run_all` composes
seven gates.

### The number that mattered

Four of the fourteen tests SKIP without an importable `preflight_run17`, and
`test_storage_guard.py` says such a skip "MUST NOT skip on the development
machine or in CI". They skipped in the authoring sandbox.

**MEASURED at the gate: `skipped` stayed at 15, `passed` rose 5690 -> 5704,
exactly +14.** The four gate tests ran. Had they skipped, the gate would have
been green with the wiring untested.

---

## 3. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | Wrote a next action into TWO session records without measuring whether the kernel had a caller | zero production construction sites across six types |
| 2 | Quoted `preflight_data_guard`'s 2026-07-21 finding as present tense | line 27 of the same paragraph states the repair |
| 3 | Built a "systemic, three instances" claim on that quotation | it is two |
| 4 | A probe counted 31 "invocations" that were all Markdown prose | every hit was a `.md` file |
| 5 | The same probe reported ZERO imports while the wiring was real | it uses `importlib`, which no import scan sees |
| 6 | Wrote an orphan probe duplicating an existing auditor | the auditor named the same three, better classified |
| 7 | Claimed `mim2gene` / `omim` shows the form for one publisher's products | `data/external/mim2gene` does not exist; both files are in `omim/` |
| 8 | `PlannedTarget` called with six arguments | it takes four; two correct examples were adjacent |
| 9 | A `SyntaxWarning` from `G:\My Drive` in a commit message | invalid escape `\M` |
| 10 | The refactor grouped findings, changing output order | byte diff against the original |
| 11 | An alias fixture whose canonical name was a SUBSTRING of the alias | sabotage changed no test |
| 12 | A sabotage case that was a no-op | the mutated branch came second |
| 13 | The installer's pin stem transform did not match its payload names | simulation before delivery |

Errors 2, 3 and 7 are one shape: **a claim published before reading the thing
it depends on.** Errors 4 and 5 are the shape
`test_storage_guard.py` already named on 2026-07-21 -- *a source check passes on
dead code and fails on a clean refactor, both directions wrong.*

---

## 4. Findings

### Closed
`AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1` -- repaired at `fd6cd4e`.

### Registered and OPEN
- `GENCODE-ACQUIRED-VALIDATED-UNDECLARED-1` -- 636,522,106 bytes
- `CONFIG-DECLARES-A-SECOND-PATH-VOCABULARY-1`
- `VALIDATOR-CHECKS-A-LOCATION-THE-DATA-LEFT-1`
- `MANIFEST-DECLARES-TWO-SOURCES-IN-ONE-DIRECTORY-1`
- `AUDITOR-TREATS-AN-EMPTY-DIRECTORY-AS-PRESENT-1` -- `tcga` reports `ok` and
  `topmed` `MISS`, both zero bytes, both controlled and irreplaceable
- `DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1`
- `QUOTED-A-FINDING-PAST-ITS-OWN-REPAIR-1`
- `PROBE-COUNTS-PROSE-AS-INVOCATION-1`
- `PROBE-CANNOT-SEE-A-DYNAMIC-IMPORT-1`

### Measured and not a defect
Nothing is currently cloud-backed: five sources carry `sync: true`, four are
`MISS`, and the fifth is `public_redownloadable` so it does not meet the
must-back-up bar. 341.1 MB of controlled data is offline-only by policy.

---

## 5. Ending state

```
HEAD     fd6cd4e
ratchet  5719
gate     5704 passed, 15 skipped, 0 failed, 0 errors
suite    60c7535c9a4ffeea -> 02535ddfc579feab
```

## 6. Next intended action

The drift source kernel is the last subsystem measured as correct and without a
caller. Phase 1C's reference profile is what would construct a
`SourceEvidenceManifest` from real acquisition data.

It is blocked on `ARTIFACT-KEY-INSUFFICIENT-1`, which now has a narrower
subject than when it was recorded: three of four collision classes dissolve
into existing axes, and the fourth belongs to a source the manifest does not
declare. Declaring GENCODE does NOT answer it -- the manifest declares
directories, not artifacts, which is what `95f6c44` and this record establish.
