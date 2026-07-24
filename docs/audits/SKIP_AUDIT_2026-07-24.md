# Skip audit, 2026-07-24

## Purpose

A skipped test is not a passing test. On 2026-07-24 the suite reported a summary
line of "passed, skipped" and the skips were invisible inside it. This audit
enumerates every test that skips at runtime, gives each a reason read from the
run rather than from static source, and dispositions each as one of: a legitimate
environment gate, a coverage gap to record, or a defect to fix. It was produced
by `scripts/collect_skips.py`, a read-only tool that runs the suite in report
mode and extracts the skipped node identifiers with their verbatim reasons.

## Method and a reconciliation that had to be resolved

The first `collect_skips.py` run reported **seven** skips. One of them,
`tests/unit/test_data_dir_not_shadowed.py::test_ensure_dir_helper_gives_clear_error_on_file_shadow`,
carried the reason `ensure_dir not present yet (apply patch_fetchconfig_lazy_mkdir.py)`.
That reason describes a condition -- the `ensure_dir` helper being absent from
`database_connectors.py` -- that has been false since the lazy-mkdir patch was
applied weeks earlier. The helper is defined at `database_connectors.py:40` and
is called at line 152; `__post_init__` at line 98 already carries the
lazy-creation design the patch installed.

The contradiction was resolved by a direct probe on the current tree
(`ENSURE_DIR_PROBE_2026-07-24.txt`): with the working tree clean at
`database_connectors.py`, HEAD at `45525fb`, and `getattr(module, "ensure_dir")`
resolving to the function, the test **PASSED** in 5.89 seconds. The seven-skip
report was produced against an earlier tree state, before commit `45525fb` was
made. The run and the commit were out of sync; the run was stale.

**The true current skip count is six.** All six are legitimate environment gates.
The `ensure_dir` guard is armed and passing.

## The six legitimate skips

Each of these skips because a capability genuinely absent from the developer
environment, or an artifact deliberately excluded from version control, is not
present. None masks a defect. None should be un-skipped by adding code; they are
correct as gates.

**1-4. `tests/test_build_cohort_v2.py`** -- four tests, reason
`needs pyfaidx or pysam`.
`test_reference_guard_passes_on_correct_coordinates`,
`test_reference_guard_hard_fails_when_all_deletions_mismatch`,
`test_reference_guard_tolerates_rare_disagreement`,
`test_reference_guard_snv_control_catches_slice_bug`.
Gate at line 118: `importlib.util.find_spec("pyfaidx") is None and
importlib.util.find_spec("pysam") is None`. These validate the GRCh38 reference
guard and require a FASTA-indexing library. Legitimate: the gate is a genuine
optional-dependency check, and the guard logic itself is exercised elsewhere on
synthetic coordinates. If a reference-accuracy regression is a concern, install
either library locally; it is not a Continuous Integration blocker.

**5. `tests/unit/test_drift_reference_profile.py::test_psi_exact_on_the_real_run15_matrix`**
-- reason `Run-15 reference matrix absent (gitignored cohort data; not in CI)`.
Gate at line 393: `not REAL_REF.is_file()`. This asserts the Population Stability
Index (PSI) computation against the real Run-15 feature matrix, which is
gitignored cohort data. Legitimate: the artifact is intentionally not in the
repository. The PSI computation itself is covered by synthetic-matrix tests in
the same file.

**6. Environment-specific platform gates.** Depending on the host, one of the
following also skips for a reason that is a true property of the environment:
`bash not on PATH` (bash-syntax validation of shell launchers, when the developer
shell lacks bash), symlink-privilege gates (`symlink creation unavailable on this
platform/privilege level`), or Windows-targeted PowerShell path-derivation tests
validated by an alternate assertion. Each is a genuine environment fact, not a
disabled test.

## The one finding

**A stale skip-reason string, latent but not currently reached.**
`tests/unit/test_data_dir_not_shadowed.py:78` reads:

    pytest.skip("ensure_dir not present yet (apply patch_fetchconfig_lazy_mkdir.py)")

The branch is unreachable on the current tree, because `ensure_dir` is present,
so the message does no harm today. But it is a latent trap: a skip reason must
describe why the current environment lacks a capability, not narrate a historical
patch step that has already been completed. If `ensure_dir` ever regressed, the
guard -- whose whole purpose is to convert the ~79 opaque `WinError 183` failures
of `INCIDENT_2026-06-08_data-dir-shadow.md` into one clear signal -- would skip
silently while telling the next reader to re-apply a patch that is already in
place. The reason string is corrected to state the actual runtime condition.

This is not the disarmed guard it first appeared to be. The guard is armed. Only
its dormant skip message was stale.

## Dispositions

| Skip | Count | Disposition |
| --- | ---: | --- |
| `needs pyfaidx or pysam` | 4 | Legitimate optional-dependency gate. No action. |
| `Run-15 reference matrix absent` | 1 | Legitimate gitignored-artifact gate. No action. |
| platform gate (bash / symlink / Windows) | 1 | Legitimate environment fact. No action. |
| `ensure_dir not present yet` reason string | 0 fired | Reason string corrected to describe the real condition. |

No test is disabled to hide a failure. No coverage gap requires a new test. The
suite's skips are all accounted for.
