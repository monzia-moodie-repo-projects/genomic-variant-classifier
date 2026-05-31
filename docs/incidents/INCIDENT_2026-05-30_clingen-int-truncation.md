---
incident_id: INCIDENT_2026-05-30_clingen-int-truncation
date: 2026-05-30
status: RESOLVED (2026-05-30; .astype(int) -> .astype(float); regression test added; full suite 596 passed)
severity: latent (no current production impact; activates at R10-G)
component: src/genomic_variant_classifier/models/variant_ensemble.py (engineer_features)
related: INCIDENT_2026-04-29 (GCS deletion), 2026-04-30 connector silent-zero audit, R10-G (ClinGen integration)
discovered_by: correctness-harness build (Task 2), HEAD 25b5eaf
---

# INCIDENT 2026-05-30 -- clingen_validity_score integer truncation

## Summary

`engineer_features` casts `clingen_validity_score` to integer via `.astype(int)`
(variant_ensemble.py, approx. L166-169). ClinGen gene-disease validity confidence
is a fractional value on a 0-1 scale. Casting it to `int` truncates every value in
`[0, 1)` to `0`, silently destroying the signal. This is a latent data-quality bug:
it has **no impact today** because the ClinGen connector is not yet wired with real
fractional scores (it is one of the dead connectors in the 2026-04-30 audit), but it
will silently zero the ClinGen feature the moment R10-G feeds genuine fractional
validity scores.

## Evidence (empirical, 2026-05-30)

Running `engineer_features` on `build_reference_slice()`:

- `clingen_validity_score` fed `rng.integers(1, 5, n)` (integer) -> **survives**
  (nonzero fraction > 0): values 1-4 are preserved by `.astype(int)`.
- `clingen_validity_score` fed `rng.uniform(0.1, 1.0, n)` (fractional, the real
  ClinGen scale) -> **nonzero fraction 0.0**: every value truncates to 0.

Contrast: `pli_score` (variant_ensemble.py approx. L293-298) is handled as
`.astype(float).clip(0, 1)` and survives fractional input (nonzero fraction 1.0).
The difference is the `int` vs `float` cast, not the clip.

## Why it is not in KNOWN_ZERO_DEFAULT

The correctness harness (agent_layer/harness/correctness_harness.py) defines
`KNOWN_ZERO_DEFAULT` -- 21 columns that are legitimately ~all-zero because their
connector default is zero/sub-threshold and no input currently populates them.
`clingen_validity_score` is **deliberately excluded** from that allowlist. It is not
a dead connector; it is a column that *would* carry signal but is corrupted by a
cast. Keeping it outside the allowlist means Stage 5 of the harness will **hard-fail**
(not warn) if `clingen_validity_score` ever silently zeroes on real data -- which is
exactly the tripwire we want for R10-G.

In the synthetic reference slice the column is fed integers specifically so the
fixture exercises the surviving path and the 21-column allowlist stays exact; the
truncation bug is recorded here rather than masked by the fixture.

## Proposed fix (defer to R10-G; do not patch pre-emptively)

When R10-G wires real ClinGen scores, change the cast in `engineer_features` from
`.astype(int)` to `.astype(float).clip(0, 1)` (mirroring `pli_score`), and add a
post-condition unit test asserting fractional input survives (nonzero fraction > 0).
Until R10-G supplies fractional scores, no code change is required -- but the bug must
be fixed *before* ClinGen validity is treated as a label-quality signal, or the
"higher-quality VCEP labels" R10-G is built to exploit will be silently nulled.

## Cross-references

- 2026-04-30 connector silent-zero audit (30+ all-zero features; ClinGen among the
  dead connectors at that time).
- SESSION_2026-05-30.md (correctness-harness build; this finding surfaced during the
  Stage-5 zero-audit allowlist derivation).
- R10-G roadmap item (ClinGen gene-disease validity + VCEP curated variants as
  higher-quality labels).