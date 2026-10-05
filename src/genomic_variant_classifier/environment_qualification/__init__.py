"""Environment qualification for the scientific stage (owner ruling 2026-10-05b).

r_runtime        -- the file-based R runner and probe, and the runtime-only lockfile admission (R 4.6.0 -> 4.6.1, nothing else).
required_tests   -- the required-test OUTCOME gate over a dedicated JUnit report (skips, absences, substitutions refused).
receipt          -- the record binding qualification evidence to the exact candidate being admitted.
source_repair    -- admission of a source-only lockfile repair (provenance fields of selected packages; nothing else).
install_plan     -- the sealed installation plan: one inspected artifact per package, bound to lockfile, runtime, platform and
                    the R series a binary was built for.
artifact_inspector -- the pre-install gate: an archive's INTERNAL identity, build metadata and digest, never its filename.

Expectations always come from the reviewed plan, never from the evidence under test -- the principle of
operations/admission_verifier.py, which binds a suite's COLLECTION identity; this package judges OUTCOMES.
Pure Python standard library; no logging configuration (library convention).
"""
from __future__ import annotations
