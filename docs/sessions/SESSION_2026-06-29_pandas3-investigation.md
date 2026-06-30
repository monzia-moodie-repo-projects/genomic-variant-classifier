# SESSION 2026-06-29 -- pandas 3.0.4 upgrade investigation (attempted, rolled back, 3 fixes kept)

## Outcome
pandas-3 upgrade ATTEMPTED and ROLLED BACK to 2.3.3. Blocked by a `pd.date_range` segfault in the pandas
3.0.4 Windows cp312 wheel (not a numpy ABI issue -- disproven across 7 numpy versions). Three genuine
improvements kept, each proven feature-hash-identical (49e98393...) on 2.3.3. Pins unchanged at 2.3.3.

## Environment
- Windows / Python 3.12.10 / venv .venv312. pandas 2.3.3 (restored), numpy 2.4.4, torch 2.11.0+cpu,
  torch_geometric 2.7.0, sklearn 1.8.0, xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10, pyarrow 23.0.1.

## What was proven
- DATA-PREP EQUIVALENCE on pandas 3.0.4: feature_hash 49e983935291605e1c7179b77c857ccf805a40dfeb6d92496f434f66616433c2
  (709 rows x 88 cols), all 7 merges result_rows=709, dtypes identical, warnings empty. The 2026-04-29
  string-dtype break the pin guarded against does NOT occur. Verified via the seeded equivalence harness on
  a fixed 2,000-variant cohort (cohort sha256 b404ebc30eab8cc5b0df6edf90cf9667bb171709af46e57384558a272d51bf1a;
  regenerable by seed=42 from scripts/pandas3_equivalence_harness.py -- not committed, working artifact).
- AGENT LAYER on pandas 3.0.4: 22/22 agents registered + scheduled + active, 0 dormant (check_agents_active.py).

## The blocker (date_range)
- `pd.date_range(...)` segfaults (0xC0000005) at pandas/core/indexes/datetimes.py:1442 on pandas 3.0.4.
  Reproduces with/without freq=, with a 2-line faulthandler repro. `DatetimeIndex`, `Timestamp`, and numpy
  `datetime64` all work -- isolates the fault to the date_range range-generator in the 3.0.4 wheel.
- numpy-ABI hypothesis DISPROVEN: segfault persists across numpy 2.0.2 / 2.1.3 / 2.2.0 / 2.2.6 / 2.3.0 /
  2.3.2 / 2.3.3. Clean --force-reinstall of pandas 3.0.4 also did not fix it. -> wheel-level defect.
- `date_range` is used NOWHERE in runtime code (src/ + scripts/ greps empty); only in one test fixture.

## Kept fixes (all valid + equivalent on 2.3.3)
1. allele_freq numeric cast in _join_gnomad (eliminates the lone object-downcast FutureWarning).
2. _suppress_fillna_downcast version-aware (no-op + no Pandas4Warning on pandas >= 3).
3. river test fixture builds dates via Timestamp+Timedelta, not date_range.

## Retry trigger
Re-attempt pandas-3 when pandas > 3.0.4 ships a fixed Windows cp312 wheel. The equivalence harness +
the three fixes are already in place; re-run Install_pandas3_upgrade flow + Run_pandas3_full_validation.

## Tech-debt noted (not addressed this session)
- river 0.23.0 / nannyml 0.13.0 / evidently 0.7.6 conflict with pandas-3 (river requires pandas<3.0). Not in
  requirements.txt; contained to standalone drift scripts. nannyml/evidently/plotly conflicts pre-date this
  arc (plotly 6.6.0 vs their <6). Revisit the drift toolchain's pandas-3 compatibility as its own task.

## Provenance
Patchers committed for traceability: patch_allele_freq_numeric.py, patch_suppress_downcast_version_aware.py,
patch_river_test_fixture.py. Equivalence harness: pandas3_equivalence_harness.py. The .ps1 installers and
data/_pandas3/ bundles were session scratch (not committed).
