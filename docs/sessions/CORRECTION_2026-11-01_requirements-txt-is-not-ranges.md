# CORRECTION, 2026-11-01: requirements.txt is not "ranges"

Author: Monzia Moodie

Three records written on 2026-10-31 say that Continuous Integration's pytest job installs "requirements.txt and requirements-dev.txt,
which are ranges": docs/CHANGELOG.md (the 2026-10-12 entry), docs/ROADMAP.md (the same entry), and
docs/sessions/SESSION_2026-10-12_test-workspace-hygiene.md (CI-INSTALLS-DEV-RANGES-1). They also imply that requirements-dev.lock's pins
were overridden.

MEASURED 2026-10-31 with the repository's own parser at 9feaf1a1:

- requirements.txt holds 65 requirements: 57 exact pins, 7 ranges (jinja2, optuna, polars, duckdb, pykan, pybigtools, pyBigWig) and one
  with no bound (gudhi). requirements-dev.txt holds 7 requirements, all ranges. Neither pins a single transitive dependency.
- The job never installed requirements-dev.lock. pytest is not in requirements-api.lock; pytest 9.2.0 came from requirements-dev.txt's
  range "pytest>=8.0,<10.0".

The records stand as written; this note is their correction. The repair is DEPENDENCY-PROFILES-1
(docs/sessions/SESSION_2026-11-01_dependency-profiles.md).
