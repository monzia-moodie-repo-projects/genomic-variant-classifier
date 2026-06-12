# INCIDENT 2026-06-12 -- protein-coord index corrupted by sample rebuild

## Status: RESOLVED

## Summary
During Run-16 AlphaMissense verification, the protein-coord index
`data/external/alphamissense/alphamissense_protein_index.parquet` (17.8 MB, full
cohort) was overwritten with a 50k-sample-only index (0.29 MB). A coverage probe was
re-run after the cache file was deleted; `ProteinCoordConnector._build_index` filters
to the cohort passed to `annotate_dataframe` and writes the result to the canonical
cache path, so passing only a 50k sample produced a sample-sized cache in place of the
full one.

## Detection
Cache file size: 0.29 MB vs the expected ~18 MB full index. (Coverage on the same
`random_state=0` sample still read 0.9672, a false pass -- the tiny index covered
exactly the sample it was built from.)

## Impact (averted before launch)
A Run-16 regen would have loaded the tiny cache as-is (the connector never rebuilds
when a cache file exists), yielding ~1% `protein_pos` coverage on the full 2.49M-
missense cohort -> the protein-coord coverage gate would have aborted the regen. This
is the same silent-ESM-2-zero class that capped Run 15 at 3,451 of ~2.49M.

## Resolution
- Hardened the probe (`scripts/probe_protein_coord_coverage.py`, v2): default mode is
  READ-ONLY and refuses to build from a sample; size-checks the cache (full ~18 MB vs
  sample <1 MB) so a corrupt cache FAILS even when the reused sample would match; full
  rebuild is an explicit `--rebuild-full` that reads the entire cohort.
- Rebuilt the FULL index from the full cohort + TSV: 4,399,089 cohort rows ->
  18.64 MB cache, full-cohort coverage 0.9665 (2,405,448 / 2,488,889 missense).
- Read-only verify: 18.64 MB, coverage 0.9672, exit 0.

## Standing lessons
1. The protein-coord index MUST be built from the FULL cohort, never a sample.
2. Any diagnostic that calls `annotate_dataframe` on a cache-miss WILL rebuild and
   overwrite the canonical cache -- diagnostics must be read-only or use a temp dir.
3. Validate the cache by SIZE (full ~18 MB), not existence or coverage-on-the-same
   sample.
4. Run-16 `--alphamissense` = the TSV `data/external/alphamissense/AlphaMissense_hg38.tsv.gz`,
   NOT the scores parquet. `train.py` help points at `alphamissense_scores_hg38.parquet`,
   but that parquet's directory lacks the protein-index, so `ProteinCoordConnector`
   would deadzone ESM-2. (The connector reads the TSV for scores via `_parse_tsv`.)
5. Ship the rebuilt 18.64 MB index to the Vast.ai box, co-located with the
   `--alphamissense` source dir, so the regen loads it instead of re-scanning the
   613 MB TSV.
