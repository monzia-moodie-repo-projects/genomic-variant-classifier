# Session 2026-06-11 (late PM) -- CNN real-sequence + RNA MaxEntScan-delta activation

## Summary
Two pre-Run-16 feature activations landed on origin/main, full suite green throughout
(893 passed / 6 skipped):
- fb12c0f -- 1D-CNN activated on real [fasta_seq_ref, fasta_seq_alt] delta windows.
- e3bcd79 -- RNA MaxEntScan delta (maxentscan_delta); EXPECTED_TABULAR_FEATURE_COUNT 80 -> 81.

Both were view-first / probe-before-change: every consumer and call site was read and the core
logic validated in a sandbox before any file was written, in line with the project's
"wired != populated != non-zero" failure mode.

## CNN real-sequence activation (fb12c0f)
The cohort's single `fasta_seq` column is deprecated and empty (notna=0 across 4.4M rows), while
[fasta_seq_ref, fasta_seq_alt] are ~100% populated. train.py gated the CNN on the empty column,
so the CNN always ran on "A"*101 placeholders and the real-sequence branch raised
NotImplementedError (claiming meta_train was unavailable). In fact DataPrepPipeline._save_splits
already persists meta_train.parquet (gene-split-aligned to X_train: both are
df.iloc[train_idx].reset_index, scaling preserves order, parquet preserves order). The fix:
gate on the ref/alt columns; build test-side windows from meta_test and train-side from
meta_train.parquet via seq_window_join.attach_delta_windows; remove the raise. No run() signature
change. The ensemble routes X_seq to the CNN via .iloc[idx].reset_index, which is DataFrame-safe,
and CNN1DClassifier._pair_arrays hits its real delta path on a 2-column frame. The CNN is a base
estimator (its OOF feeds the stacker), so NO feature-matrix schema change.

## RNA MaxEntScan delta (e3bcd79)
rna_pipeline.annotate_dataframe (wired at _annotate_scores step 13, rna_pipeline default True) read
the same empty single `fasta_seq`, so maxentscan_score defaulted to 0.0 everywhere. Repointed to a
_window_score helper applied to BOTH ref and alt windows: maxentscan_score stays the ref-window
score; maxentscan_delta = score(alt) - score(ref) is the new variant-specific signal. Graceful
fallback to legacy single fasta_seq (ref==alt -> delta 0) then defaults.

Registration touched the feature-count contract: maxentscan_delta added to TABULAR_FEATURES, both
_engineer_features blocks (variant_ensemble.py and real_data_prep.py), and the RNA-off default
tuple; EXPECTED_TABULAR_FEATURE_COUNT 80 -> 81. INFERENCE_FEATURE_COLUMNS is list(TABULAR_FEATURES),
so it auto-tracks; the contract test stays green at 81/81 with no edit to it or api/pipeline.py.

## Activation precondition (load-bearing)
real_data_prep does NOT add fasta_seq* columns anywhere; _load_and_label is pd.read_parquet with no
column subsetting, so ref/alt ride through ONLY if the input cohort carries them. Run 16 MUST pass
--clinvar data\processed\clinvar_grch38_clean_seq.parquet or BOTH the CNN and maxentscan_delta
degrade silently to inert. Added to the Run-16 launch contract.

## Verification
- Full suite 893 passed / 6 skipped (was 892 + 1 failed before the harness slice fix).
- CNN: torch-gated end-to-end test trains on a 2-column ref/alt delta frame, returns finite proba.
- RNA: maxentscan_delta nonzero for a real ref!=alt splice variant; 0 for ref==alt / non-splice /
  legacy single fasta_seq; graceful (no crash) when neither ref/alt nor fasta_seq is present.
- The correctness harness stage-5 tripwire caught maxentscan_delta added without its reference-slice
  entry -- the only gate that did. Fixed by populating the slice (NOT allowlisting it).

## Carry-forward
- Schema baseline refresh: regenerate from post-Run-16 X_train; target EXPECTED_TABULAR_FEATURE_COUNT
  (now 81). Reconciles the pre-existing 78/79/80 spread at the regen.
- Run-16 launch contract: --clinvar ref/alt cohort + --esm2-model 650M + --esm2-uniprot-index +
  --alphamissense, all Test-Path-gated in the ONE preflight script.
- Always-donor MaxEntScan selection bug: biology-correct donor/acceptor selection from
  dist_to_donor/dist_to_acceptor is the next RNA item (does not block Run 16).
- New standing run-gate: every new feature must appear populated in the correctness-harness slice.

## Key learnings
- meta_train.parquet was already persisted; the "needs run() plumbing" raise was unnecessary.
  Reading the actual _save_splits body (not memory) found the signature-free path.
- A feature can be wired (in the contract, in engineer_features) and still be silently zero if its
  upstream column is absent from the input cohort -- the launch-contract --clinvar requirement is
  the real activation switch.
- The correctness harness earns its keep: it caught drift that py_compile, the feature-count
  contract, and the targeted unit tests all passed.
