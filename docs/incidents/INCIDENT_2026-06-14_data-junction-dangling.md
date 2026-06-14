# INCIDENT 2026-06-14 -- data/ Junction -> Google Drive (G:) went dangling; 20 tests failed loud

**Status:** RESOLVED -- data/ restored from git; full suite 1100 passed / 6 skipped / 41 warnings.
**Class:** Environmental (developer machine). NOT a code regression.

## Summary
After the CI-repair commit 5a6b0d0, the full local suite showed 20 failures, ALL raising the codebase's own
fail-loud guard:

    NotADirectoryError: Cannot create data\splits: a path component exists as a non-directory (stray file or
    dangling symlink/junction shadowing data/). Remove or rename it and restore data/ from git, then retry.

(real_data_prep.py:222 DataPrepConfig.__post_init__; protein_pipeline.py:376 ProteinStructurePipeline.__init__;
esm2 _open_cache also surfaced FileExistsError [WinError 183] on 'data'.)

## Root cause
The repo's data/ was a Windows **Junction**:

    data  ->  G:\My Drive\genomic-variant-classifier\data    (Google Drive for Desktop)

When G: was not mounted/synced, the junction DANGLED. `Get-Item .\data -Force` still showed the reparse point
(Mode d----l, LinkType Junction, Target G:\My Drive\...\data), but any mkdir/write through it failed, and
`git status` reported "could not open directory 'data/'" with all 6 tracked data/ files shown as deleted (D).
No src/ code writes a bare `data` file -- verified by grep over connect/open/to_parquet/to_csv/write_* with a
bare-'data' filter (empty). The corruption was purely the dangling junction; the fail-loud guard (added
proactively in an earlier session) caught it exactly as designed -- no silent stubbing.

## Affected tests (20)
test_core.py::TestAnnotationPipeline (12; default DataPrepConfig output_dir=data/splits),
test_npathogenic_train_only.py (4; same default), test_esm2_activation.py (2; data/raw/cache),
test_lovd_annotation_reaches_training_matrix.py (2; data/raw/cache/alphafold). All write to the REAL data/
(not tmp_path) -- a pre-existing test-isolation weakness (see Prevention).

## Remediation
1. Inspected: Get-Item .\data -Force | Format-List ... -> LinkType=Junction, Target=G:\My Drive\...\data.
2. Cleared the dangling junction (cmd /c rmdir / Remove-Item reported it already unresolvable -> gone).
3. git checkout -- data/  -> recreated a PLAIN LOCAL data/ with the 6 tracked files:
   external/.gitignore, processed/.gitignore, processed/gene_pathogenic_counts.parquet,
   processed/gene_summary.parquet, raw/.gitignore, reference/schema/schema_baseline.json.
4. Re-ran: 1100 passed / 6 skipped / 41 warnings.

CI was never affected -- a fresh checkout has an intact, real data/.

## Consequence (ACTION REQUIRED before any real-data run)
data/ is now a PLAIN LOCAL directory; the junction to Google Drive is GONE. The large UNTRACKED assets that
lived under the junction target on G:\My Drive\...\data\ are NOT in the new local data/:
data/external/spliceai/spliceai_index.parquet (336.8 MB), dbNSFP, gnomAD-constraint TSV, LOVD, AlphaMissense,
data/raw/cache/*, data/agent_state.json, ... The next real run (Run 15 prep,
build_reclassification_reference.py, ESM-2, _annotate_scores) will silent-stub these (connector
"file not found -> default scores" warnings). Re-hydrate first: re-point the junction to a MOUNTED G:, OR
robocopy the assets from G:\My Drive\...\data into local data/. Also check whether `outputs/` is a similar
(possibly dangling) junction -- the Run-15 splits live under outputs/run15_rerun_report/full/splits/.

## Prevention
- Recommended infra (matches the documented architecture): keep data/ and outputs/ as PLAIN LOCAL directories
  and use rclone `genvarcla:` for Drive durability (agent-layer only), NOT a live G: junction that dangles
  whenever Drive-for-Desktop is unmounted.
- If a junction is kept: ensure Drive-for-Desktop auto-starts, AND add a preflight check that FAILS FAST if
  data/ or outputs/ is a dangling reparse point before a run.
- Test hardening (separate): point the data/-writing tests (TestAnnotationPipeline, test_npathogenic_train_only)
  at tmp_path so the suite is hermetic and a clobbered data/ cannot take out 20 tests.
