# INCIDENT 2026-06-10 -- ESM-2 coverage capped at 3,451 by a stale protein-coord index

**Status:** Resolved (code) / operational follow-up pending for Run 16
**Severity:** Medium -- silent feature degradation; one model's input (`esm2_delta_norm`)
was near-dead across runs, but the ensemble remained functional and no incorrect
result was shipped externally.
**Components:** `data/protein_coords.py`, `data/esm2.py`, `data/real_data_prep.py`
(step 10b), Vast.ai training-box data sync.

## Summary

In Run 15 (sealed, commit `032a2ab`), ESM-2 scored only **3,451 of ~2,488,889
missense** variants (`esm2_delta_norm` > 0). The same fixed 3,451 appeared in both
the 3k-row smoke and the 1.49 M-row full run, which is impossible for a
cohort-scaled quantity and was the first clue.

Root cause: ESM-2's candidate set is exactly the missense rows carrying
`protein_pos`/`wt_aa`/`mut_aa`, which are populated only by `ProteinCoordConnector`
(step 10b) from the AlphaMissense protein-coordinate index. The Vast training box
merged against a **stale, small `alphamissense_protein_index.parquet`** (a
smoke-era build of ~3,461 loci) that was never refreshed from the healthy local
index. The local index covers **96.6%** of missense (`2,405,448`), proven by the
connector's own coverage report. The defect was therefore operational (a data-sync
gap), compounded by the fact that step 10b logged the low count at INFO and the
pipeline trained for 11.5 hours regardless -- a silent degradation.

## Investigation (measure-first; theories tested and discarded)

1. **Empty `protein_change` / "build an HGVSp parser."** Discarded: `hgvsp_parser.py`
   and `protein_coords.py` already exist; coordinates come from AlphaMissense, not
   from parsing ClinVar `Name`. The empty `protein_change` column is irrelevant to
   ESM-2.
2. **Stale cache *file* locally.** Discarded: `probe_protein_coord_cache.py` showed
   the local index at 2,411,089 rows (96.9% by row count).
3. **Stale training splits.** Discarded: the Run 15 log shows a fresh
   `DataPrepPipeline` run (`=== DataPrepPipeline: starting ===`) and
   `ProteinCoord: loading index cache`, not pre-built splits.
4. **Key drift / merge-dtype miss.** Discarded: `probe_coord_merge_repro.py`
   reproduced the real connector merge locally at 164,400/169,677 missense (96.7%)
   on a 300k sample; `[A]` normalization-only overlap, `[B]` real connector, and
   `[C]` replicated merge all agreed.
5. **Stale cache on the Vast box (confirmed).** Local merge lands ~96.7% but Run 15
   logged 3,461 with `protein_pos` and 3,451 ESM-2 candidates -> the Vast box ran an
   older/smaller index than the local one.

Key Run 15 log lines:
```
11:04:16  ProteinCoord: loading index cache: .../alphamissense_protein_index.parquet
11:04:20  Score annotation 10b (protein coords): 3461 variants with protein_pos.
11:04:41  Computing ESM-2 delta for 3451 missense variants ...
11:04:41  ESM-2: gene(s) absent from the UniProt index ... (first missing: MYH11;NDE1).
11:04:42  ESM-2: 3435/3451 variants scored (>0).
```

## Resolution

- **Code (landed):** a fail-loud coverage gate in `real_data_prep._annotate_scores`
  step 10b. Two pure helpers -- `_protein_coord_source_present(cache_path, am_path)`
  and `_assert_protein_coord_coverage(df, min_cov)` -- plus
  `AnnotationConfig.min_protein_coord_coverage = 0.50`. The gate is enforced ONLY
  when a coord source (cache file or AlphaMissense TSV) is present; in stub mode
  (no source) it is skipped, matching the connector's documented degradation
  contract. A source-present run whose missense coverage falls below the threshold
  now raises before any model trains.
  - Regression note: the first version of the gate raised unconditionally and broke
    12 stub-mode tests; the conditional version restores them and is covered by
    `tests/unit/test_protein_coord_coverage_gate.py` (13 cases).
- **Operational (Run 16):** rebuild the AlphaMissense protein-coord index on the
  Vast box from the TSV already present there, and confirm the coverage report
  prints ~96-97% before training. The gate now enforces this at train time even if
  the step is skipped.

## Follow-ups (open)

- **UniProt gene-symbol alias gap** -- `MYH11`/`NDE1` absent from the 20,190-gene
  reviewed index (HGNC symbol/alias drift). At full coverage this silently zeroes a
  slice of `esm2_delta_norm`; needs an alias map plus a logged unmatched-gene count
  and an ESM-2-side coverage gate.
- **`wt_aa`-vs-UniProt mismatch counter** in `esm2._compute_delta` -- currently
  computes a delta even on a residue mismatch without aggregate signal.
- **Cache cohort-fingerprinting** in `protein_coords` -- defense-in-depth so a cache
  built on cohort A is not silently reused for cohort B.
- **Two divergent AlphaMissense score indices** --
  `data/raw/cache/alphamissense_scores_hg38.parquet` (71,034,269 rows, `lookup_key`)
  vs `data/processed/alphamissense_index.parquet` (71,697,556 rows,
  `chrom/pos/ref/alt`). Confirm which `ac.alphamissense_path` the launch uses and
  retire the other.
- **Disk:** delete the 570 MB byte-identical duplicate
  `data/processed/clinvar_grch38_clean_seq (1).parquet`.
- **Tech-debt:** pandas-3.0 `FutureWarning` on `.fillna` downcasting in
  `variant_ensemble.py` (~90 hits) before any pandas bump.

## Lessons

- A connector logging a low count at INFO is not a safeguard; coverage-critical
  steps need a fail-loud gate that aborts before expensive downstream work.
- Cache artifacts must be keyed on, or validated against, the cohort they serve, and
  must be treated as build outputs that travel with the data to every box.
- Every patch made from a theory the data had not yet confirmed would have been
  wrong here; each fix waited on a measurement (probe or log line).
