# AlphaFold Phase-D Cohort Build — COMPLETION RECORD (2026-07-03)

Addendum to `ALPHAFOLD_SESSION_2026-07-02.md`. This closes the AlphaFold
structural-feature arc: the cohort was built end-to-end and verified against the
on-disk parquet (not merely reported by the run log).

## Build invocation

- Command: `python scripts/build_alphafold_parquet.py --workers 8`
- Builder commit: `7ff8e89` (SHA-256 of builder file:
  `2BD52996BE3848D8D7C3CEA35874D842BE020D7DAC4F018DABC53000DAB4B4F5`),
  the parallel + deterministic + canonical-selection version that passed the
  live serial-vs-parallel identity gate.
- Start: 2026-07-03 01:09:46 ; finish: 2026-07-03 10:10:57 (fetch phase ran
  hard; per-gene progress logging was buffered by the stdout pipe and flushed at
  completion — a known observability limitation of the `| Tee-Object` launch, to
  be run with `python -u` next time; it did NOT affect correctness).

## Final cohort artifact

- Path: `data/external/alphafold/alphafold_cohort.parquet`
- File size: 110.2 MB
- Rows (residues): 9,960,360
- Distinct UniProt accessions (structures): 18,034
- Columns: `uniprot_accession`, `residue_pos`, `plddt`, `rsa`, `ss`, `dist_active`
- Nulls: 0 in every column

## Per-feature verification (read from the on-disk parquet)

| feature | min | max | mean | notes |
|---|---|---|---|---|
| residue_pos | 1 | 2699 | 430.3 | max = AlphaFold single-fragment ceiling (~2700 aa); giants exceed it -> unusable set |
| plddt (predicted Local Distance Difference Test) | 1.99 | 99 | 72.78 | real confidence distribution; mean in the "confident" band |
| rsa (relative solvent accessibility) | 0 | 1 | 0.4168 | correctly clamped [0,1]; ~42% mean exposure is biologically sensible |
| ss (secondary-structure class) | 0 | 2 | 0.6367 | all three classes present |
| dist_active (Angstrom to nearest active/binding site) | 0 | 376.5 | 78.97 | zeros = residues AT active sites; real geometry, not sentinel |

## Sentinel (stub-fallback) fractions — the "are the features real?" check

- plddt == 50.0 : 0.0002  (a few genuinely-disordered residues near pLDDT 50, not stubs)
- rsa == 0.5 : 0.0000
- Interpretation: effectively ZERO sentinel rows. All four structural features
  are genuine for essentially every residue in the cohort.

## Coverage accounting

- Coverage report: `data/external/alphafold/alphafold_coverage.json`
- Total cohort genes: 18,302
- Usable (canonical AlphaFold structure matched by exact UniProt sequence): 18,034 (98.5%)
- Unusable (documented gap -> structural sentinel): 268
- Reconciliation: 18,034 + 268 = 18,302  (exact)
- Consistency across three independent measurements: 400-gene canonical audit
  98.2%; 50-gene probe 98.0%; full build 98.5% — mutually consistent.
- Nature of the 268 unusable: AlphaFold-DB length-ceiling giants (e.g. TTN,
  NEB, OBSCN, PLEC) and isoform-only entries with no record whose sequence
  matches canonical. These correctly receive structural sentinels and retain all
  non-structural features plus the ESM-2 (Evolutionary Scale Modeling 2) sequence
  branch, which has no length limit. Enumerated in the coverage JSON.

## Separate, earlier drop-out (not part of the 268)

- 1,081 cohort gene symbols did not resolve to a reviewed UniProt accession in
  the local `uniprot_human_reviewed.parquet` index (name mismatch / non-reviewed
  / obsolete symbol). These never reached AlphaFold. This is distinct from the
  268 canonical-structure misses and occurs at the gene->accession resolution
  step (19,383 cohort missense genes -> 18,302 resolved).

## Correctness lineage (defects fixed to reach this artifact)

1. Stale AlphaFold-DB URL (v4 -> v6): resolve current cifUrl via the prediction API.
2. Non-CIF payload guard: reject any non-`data_`/`_atom_site` body (never parse an error page).
3. O(n^2) Shrake-Rupley RSA -> O(n) via cKDTree + numpy; proven numerically
   identical (max RSA diff 0.0); ~10x on large proteins (A2M 1474 res: 23s -> 2s).
4. Silent isoform mis-selection -> canonical-sequence-exact match; no match -> None
   (documented sentinel), never a mis-numbered isoform structure.
5. Coverage report + dormant hard gate (<0.90 usable fraction aborts).
6. Parallel fetch (--workers) + atomic CIF cache write + deterministic row order
   (sort by all columns); output proven byte-identical serial vs parallel.

Each delivered as a hash-verified, guarded (count==1) installer with byte-compile,
full-unit-suite (25 passed, 0 skipped), and live end-to-end post-checks; every
patcher additionally re-validated by extraction from the finished installer.

## Status

AlphaFold structural features: COMPLETE and VERIFIED. Cohort parquet on disk,
feature-validated, coverage-reconciled. This closes the "wired, stubbed pending
build" status that opened the 2026-07-02 session.
