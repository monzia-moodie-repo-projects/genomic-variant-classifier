"""Environment qualification for the scientific stage (owner ruling 2026-10-05b).

r_runtime        -- the file-based R runner and probe, the runtime-only lockfile admission (R 4.6.0 -> 4.6.1, nothing else), and the
                    runtime COMPONENT manifest (files under R_HOME/bin and R_HOME/etc -- the runtime, not only its launcher).
required_tests   -- the required-test OUTCOME gate over a dedicated JUnit report (skips, absences, substitutions refused), exact
                    membership of replay qualification rows, and what a fixture process actually loaded (versions, locations).
receipt          -- the record binding qualification evidence to the exact candidate being admitted.
install_plan     -- the sealed installation plan: one inspected artifact per package, bound to lockfile, runtime, platform,
                    the R series a binary was built for, and its native libraries counted from its contents.
artifact_inspector -- the pre-install gate: Python checks an archive's bytes and STRUCTURE; the qualified R reads its DESCRIPTION
                    and judges the dependency closure; identity is the exact recorded strings, never the filename.
r_semantics      -- the R program (text) that interprets R's own package metadata: read.dcf records, package_version.
build_plan       -- route selection, rebuild closure, installation order, and admission of build receipts and installed dependencies.
isolation        -- the network claim a replay may make, from an enforcement MECHANISM's evidence (Windows Sandbox configuration,
                    detached virtual machine); a host disconnection is a procedure, never "offline"; probes are diagnostic.
admission        -- ONE definition of admission reused by every script: exact artifact-set coverage (re-hashed), the plan digest, the
                    library content digest, exact lockfile transitions, the pure qualification decision, and the run record
                    (execution separate from admission).

Expectations always come from the reviewed plan, never from the evidence under test -- the principle of
operations/admission_verifier.py, which binds a suite's COLLECTION identity; this package judges OUTCOMES.
Pure Python standard library; no logging configuration (library convention).
"""
from __future__ import annotations
