# Session 2026-11-02 -- DRIFT-PROFILE-LOCK: the drift-monitoring profile locked per target, one installer for every locked profile

Author: Monzia Moodie

Closes DRIFT-ENVIRONMENT-NOT-LOCKED (recorded 2026-11-02 in docs/sessions/SESSION_2026-11-01_warnings-and-resources.md). The owner
rulings of 2026-10-31 for DEPENDENCY-PROFILES-1 apply unchanged: one lock per target (Linux 3.11, Linux 3.12, Windows 3.12), pip and
pip-tools retained with the pinned generator toolchain, the owner's environments preserved (his .venv-drift is observed read-only,
never modified), the six required check names kept.

## What was wrong, measured

1. requirements-drift.txt pinned seven distributions (nannyml, evidently, numpy, pandas, scikit-learn, lightgbm, plotly); everything
   else resolved on the day. The drift jobs of pull request #59 (2026-11-02) resolved 137 distributions: pyarrow 26.0.0 while
   training locks 23.0.1 -- although the file itself says the drift monitor reads the parquet files training writes, and pins
   numpy, pandas and scikit-learn to training for exactly that reason; matplotlib 3.11.2 (training 3.10.8); pytest 9.2.0 (9.0.3);
   scipy 1.17.1 on Python 3.11 but 1.18.1 on 3.12 -- two different drift environments under one job name.
2. The drift job upgraded pip to the newest release, installed pytest and packaging unpinned, and built the project with an
   isolated, unpinned setuptools.
3. An editable install WITH build isolation runs `egg_info` in the source tree and leaves src/genomic_variant_classifier.egg-info
   (measured with setuptools 84.2.1, and with 70.2.0 forced by --build-constraint: both leave it); without build isolation the
   metadata goes to a temporary folder. The old drift recipe (`pip install -e . --no-deps`, in the job and in the file's own
   instructions) is the isolated kind: a mechanism that produces exactly the directory of EGG-INFO-SHADOW-1 (not a proof of how the
   owner's arose). The new environment contract refused such a directory on its first run here (duplicate_metadata).
4. drift_monitor.yml: job verify-model installed requirements-api.txt -- which does not contain the project -- and then imported
   genomic_variant_classifier from the checkout root, where it is not (it is under src/): ModuleNotFoundError by construction,
   reproduced. Job label-drift installed pandas and pyarrow unpinned and imports neither (its steps use urllib and echo).
5. Four hand-kept copies of the old `pip install -r requirements-drift.txt` recipe (performance_estimator, run_drift_monitor,
   verify_drift_libs, the drift tests' docstring), and scripts/verify_drift_libs.py imported nannyml directly -- bypassing the
   telemetry switch and the .env suppression of revision b.

## The resolution, measured (evidence: the session's measurement folder, SHA256SUMS)

- Sources: requirements-drift.txt (+ pyarrow==23.0.1) and the new requirements-drift-test.txt (pytest>=8.0,<10.0 as
  requirements-dev.txt; packaging==26.0 as requirements.txt; setuptools>=61.0 as pyproject.toml's build-system).
- Constraints: constraints-drift.txt is DERIVED from the training-and-test locks -- every pin active on any of their targets, less
  the drift profile's declared departures -- by `python scripts/check_dependency_profiles.py --derive-constraints drift` (171
  constraints); the lockfile job refuses any other content.
- Resolution with the pinned toolchain (pip 26.1.2, pip-tools 7.5.3), `--generate-hashes --allow-unsafe --strip-extras`: 137 pins.
  Python 3.11.17 and 3.12.3 gave byte-identical bodies (only the header's "with Python" line differs; the committed lock is the
  3.12 one, as for training). 74 distributions are shared with training; all equal except exactly three DEPARTURES, each measured
  necessary -- constrained to the training version of that one distribution, the resolution is impossible:
    lightgbm 4.5.0 (training 4.6.0): nannyml 0.13.1 requires >=3.3,<4.6;
    plotly 5.24.1 (6.6.0): nannyml requires >=5.6,<6.0, evidently 0.7.6 <6;
    xgboost 2.1.4 (3.2.0): nannyml requires FLAML[automl], whose "automl" extra requires xgboost>=0.90,<3.0.0 in every release from
    2.3.3 to 2.7.0.
- Every one of the 137 has a wheel for CPython 3.11 and 3.12 on Linux (`--only-binary=:all:` dry runs: 137 of 137 each), so the
  drift profile installs wheels only and needs no build constraints.
- Installed on 3.11 and 3.12: pip check clean; scripts/verify_drift_libs.py OK; tests_drift 19 passed each. evidently 0.7.6's import
  and a Report run make no python-dotenv call, open no network connection, load no telemetry module and leave no unclosed resource.

## The change

- src/genomic_variant_classifier/deps/profiles.py (standard library only): THE declaration of both profiles -- sources, constraint
  file, locks per target, install flags, build constraints, the aligned profile, departures with their measured reasons, forbidden
  distributions (pyspark, torch, torch-geometric) -- plus the targets, the Python policy and `install_command()`, the one remediation
  text every message cites.
- scripts/check_dependency_profiles.py: `--offline` checks EVERY profile (sources on every target, constraints, forbidden
  distributions; for an aligned profile, that its constraint file is exactly the derivation and that each departure still departs);
  `--environment` requires `--profile`; `--derive-constraints PROFILE` writes an aligned profile's constraint file.
- scripts/install_locked_profile.py (standard library only, run by the base interpreter): refuses an existing environment, an
  existing observation or a missing input before creating anything; makes a NEW virtual environment; installs pip from the
  generator toolchain's hashed record, the target's lock with `--require-hashes --no-deps --no-cache-dir` and the profile's flags
  (`--only-binary=:all:` for drift, `--build-constraint` for train-test), the project editable with `--no-deps
  --no-build-isolation`; observes the environment and proves it IS the lock; every child interpreter isolated (-I).
- requirements-drift-linux.lock (137 pins) and requirements-drift-windows.lock (generated on the owner's machine by block D1).
- .github/workflows/ci.yml: the lockfile job installs packaging from the toolchain's hashed record (unpinned until now) and checks
  every profile offline; the pytest and drift jobs install through the installer and then prove the job's `python` is that
  environment. .github/workflows/drift_monitor.yml: feature-drift installs through the installer; label-drift and verify-model
  install nothing (verify-model runs with PYTHONPATH=src: model_registry's import closure is the standard library and this package,
  measured in a virtual environment without pip, and the step now computes its verdict -- exit 3, NOT CHECKED, with the committed
  registry -- on 3.11 and 3.12); PIP_RETRIES and PIP_DEFAULT_TIMEOUT as in ci.yml; no `cache: pip` anywhere a cache is not read.
- deps/governance.py: four new artifacts declared (requirements-drift-test.txt, constraints-drift.txt with the new purpose
  ALIGNMENT_CONSTRAINTS, the two drift locks); requirements-drift.txt is now a declaration, not an installation file; ci.yml and
  drift_monitor.yml are lock-qualified (4 of 18 install paths).
- monitoring/performance_estimator.py: `import_nannyml()` (public; the only import of nannyml in src/ and scripts/, enforced by
  tests/unit/test_nannyml_import_boundary.py); scripts/verify_drift_libs.py imports nannyml through it; the remediation texts cite
  `install_command("drift")`.

## Verification

All in the development sandbox on 2026-11-02 unless stated; evidence in the session's measurement, emulation and mutation folders.

- Every changed job, emulated step by step on candidate commit 2e29aafb (each `run` step read from the workflow and run with
  GitHub's bash options; the checkout listed in full before and after every installation step):
  - lockfile drift check (Python 3.11, packaging 26.0 from the toolchain record): all five steps pass; every profile's locks and
    sources SATISFIED.
  - pytest (3.12): the installer built the training-and-test environment, contract SATISFIED, the checkout unchanged; release
    approvals append-only; suite: 8,575 passed, 26 skipped (each environmental), 6 xfailed and no failure; coverage 84%; 8,607 collected.
  - drift monitor (3.11) and (3.12): installed and proven, the checkout unchanged, isolation and pip check pass, 19 passed each.
  - Monthly Drift Monitor: feature-drift installed and proven, then its designed outcome (readiness UNKNOWN; the schema gate exits 3
    with no matrix supplied); verify-model, with nothing installed, computes its verdict: exit 3, NOT CHECKED (the committed registry
    declares no production model); label-drift, with nothing installed, reaches its network request (this sandbox's proxy refuses
    the ClinVar server).
- The owner's machine (block D1, 10:46-10:50 UTC): the Windows lock generated with the pinned toolchain -- 136 pins, every one with a
  Windows wheel (a hash-checked dry run, 136 of 136); the 135 pins it shares with the Linux lock have equal versions and identical
  hash sets (colorama is Windows-only; nvidia-nccl-cu12 and uvloop are Linux-only); .venv312 and .venv-drift unchanged, the
  checkout's tracked files unchanged. The owner's existing .venv-drift, judged against this lock: 82 differences (72 versions, among
  them pyarrow 25.0.0 and scipy 1.18.0; 5 missing; 1 extra) and the duplicate project metadata of EGG-INFO-SHADOW-1.
- Block D1's harness: 14 of 14 cases (the Linux resolution reproduced through the block byte for byte apart from the output name;
  every refusal; configuration cleared, recorded by name and restored); 13 mutants of the block, each removing one check, all
  killed. The checks, the installer and the import boundary: 11 mutants killed; 5 planted defects in the workflows and locks each
  failed its test.
- The Windows path budget of the drift environment, from its 136 Windows wheels (61,037 installed files, the measurer checked
  against five real installations): longest file 130 and longest directory 105 characters below the environment's root.
- Not run here: the pytest job on Python 3.11 (GitHub's pytest (3.11) check) and the Windows validation (the installer's -Validate
  on the owner's machine).

## Open, recorded

- EGG-INFO-SHADOW-1 (the owner's environment): the mechanism above is now known; resolution still waits for the candidate-adoption
  decision, by quarantining the stale directory -- never by uninstalling an ambiguous identity.
- The owner's .venv-drift was built by the old recipe; block D1 observed it read-only. Replacing it is a separate decision: the
  installer builds a NEW drift environment and never modifies an existing one.
- Install paths not yet lock-qualified: 14 of 18 (deps/governance.py INSTALL_CONSUMERS).
