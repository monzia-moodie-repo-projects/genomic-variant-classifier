# Session 2026-11-01 -- DEPENDENCY-PROFILES-1: the training-and-test profile, locked per target

Author: Monzia Moodie

## Rulings (owner, 2026-10-31)

- OWNER-ENVIRONMENT-VERSUS-LOCK-1: preserve .venv312; do not synchronise it to requirements-dev.lock; qualify a separate candidate.
- CI-INSTALLS-DEV-RANGES-1, renamed DEPENDENCY-PROFILES-1: measured versions enter as CONSTRAINTS for the packages a profile needs (never
  the whole inventory); one lock per target (Linux 3.11, Linux 3.12, Windows 3.12); pip and pip-tools retained; this first change covers
  the training-and-test profile only; the six required check names are preserved; a Windows job, when added, is a separate job.

## Evidence, in order

1. The owner's environment, observed read-only (2026-10-31 23:19 UTC; observer SHA-256 abce5326..., observation a4fc5829...): Python
   3.12.10, 251 distribution records under 250 names, pip check clean; its file inventory and attributes unchanged by the observation
   (path, size and last-write time of 55,794 files compared; contents not hashed). It satisfies every declaration except pillow
   (12.2.0 installed, 12.1.1 declared) and holds two metadata records for the project itself (EGG-INFO-SHADOW-1: the editable install's
   dist-info and a leftover src/genomic_variant_classifier.egg-info; a fresh editable install with setuptools 70.2.0 does not recreate it).
2. Locks: see docs/CHANGELOG.md. Linux 174 pins (3.11 and 3.12 identical), Windows 152; every Windows pin equals the owner's version.
3. Of the owner's 97 packages outside the profile, an import census (1,234 files) found three with importers: ntqr (guarded; its admission
   waits on the SR #31 check, so the profile omits it and test_d1_d2's stub test runs instead of skipping -- NTQR-ADMISSION-PENDING-1),
   python-docx (guarded, two scripts) and python-barcode (a false positive: it ships a top-level `tests` namespace fragment, harmless here
   because the repository's tests/ is a regular package). The two pytest plugins the profile drops (Faker, pytest-reportlog) are used by
   no test or configuration.
4. The locked Linux environment ran the full suite of 9feaf1a1 with Continuous Integration's command: 8,505 passed, 26 skipped, 6
   xfailed, 0 failed; every skip is environmental (data, R, Windows-only code, offline models), none is a missing Python package.
5. Python 3.11.17, a dry run of the pytest job's exact install (2026-11-01, about 02:59-03:19 UTC; nothing installed): pip planned
   exactly the lock's 174 pins active on linux-3.11 (none missing, none extra, no version different), every download hash-checked;
   three source archives had their metadata prepared, every other distribution has a cp311 or pure-Python wheel. The 3.11 installation
   and suite are proven by the 'pytest (3.11)' job.
6. The offline check runs on a bare Python 3.11 holding only packaging (26.3, the newest, as the lockfile job installs it) and the
   standard library: SATISFIED.
7. The pytest job's `cache: pip` was removed: every install passes --no-cache-dir, and the pinned setup-python's pip cache key hashes
   only requirements.txt, restores an older cache by prefix and saves it again (src/cache-distributions/pip-cache.ts, src/cache-save.ts
   at a309ff8b, read 2026-11-01).
8. The pytest job emulated (2026-11-01, 03:32-04:03 UTC): every `run` step of the job, read from the change's ci.yml and run in
   order with GitHub's bash options on Python 3.12.3, against a shallow detached checkout of tree 2a3ea0a2 (this change without the
   observer's tests), event pull_request with base 9feaf1a1. Install 423 s (the three source-only wheels record "Generator:
   setuptools (70.2.0)"); contract SATISFIED (176 distributions); coverage-critical imports all present; pip check clean; release
   approvals append-only against 9feaf1a1; suite 8,545 passed, 26 skipped (all environmental), 6 xfailed, 0 failed, 1,409 s,
   coverage 83%; working-tree guard unchanged; a full listing of the checkout before and after the install step: identical.
9. That run measured deps/environment_observation.py at 0 of 91 statements covered: the observer had no test. Six were added
   (tests/unit/test_environment_observation.py). Suite 8,577 -> 8,583.

## Open, recorded

- HF-XET-WINDOWS-MARKER-1: a Windows candidate built from its lock lacks hf-xet (present in .venv312) -- expected.
- EGG-INFO-SHADOW-1: resolve in the owner's environment only after a clean candidate exists, by quarantining the stale directory; never
  by uninstalling an ambiguous identity.
- Install paths not yet lock-qualified (16 of 18): deps/governance.py INSTALL_CONSUMERS, each with its measured status.
- TEMP-EMPTIED-EXTERNALLY-1: TEMP was emptied at or after a restart onto a newer Windows Insider build (2026-10-31 17:29 UTC); the
  deleting component is not identified. Long-lived candidate workspaces live under the project's own folder, never TEMP.
- Withdrawn for confirmatory use (owner ruling): the uploaded paired bootstrap reference (low coverage with few clusters; equivalence
  claimed from degenerate intervals). The statistical follow-up builds inside evaluation/metrics.py.
