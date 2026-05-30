# INCIDENT 2026-05-30 - train.py paired CNN sequences with wrong labels after gene-aware split (PM11c/PM11d)

## Status
RESOLVED. Latent in production (never fired); live on synthetic / real-sequence runs. Fixed + regression-tested at HEAD 2a63210.

## Symptom
scripts/train.py built CNN input sequences with raw_df["fasta_seq"].iloc[: len(y_test)]
(and the train equivalent) -- a positional slice of the PRE-split DataFrame paired
with labels produced by the gene-aware GroupShuffleSplit. Because the split permutes
row order, sequence i was paired with the label of a different variant.

## Severity: latent in production, live on synthetic
- Production data/processed/clinvar_grch38.parquet has fasta_seq present but notna=0.
  has_sequences (notna > 100) is therefore always False in production, cnn_1d is always
  popped, and the dummy-sequence path runs -- so the misalignment has NEVER fired.
- It WOULD fire on any synthetic run (fixture emits real fasta_seq) or once real
  sequences are introduced. Regression test confirms agreement < 0.85 under old logic.

## Root cause
run() returns X_train/y_train already gene-split and index-reset, but train.py realigned
sequences against the unsplit raw_df by position. No shared key was used. X_train carries
no identity column (1197216x73, zero id columns); run() does not return meta_train.

## Resolution
- Test side: X_seq_test = meta_test["fasta_seq"].reset_index(drop=True). meta_test is
  df.iloc[test_idx], split-aligned by construction (verified 349067 == y_test).
- Train side: raise NotImplementedError when real training sequences are present, since
  no signature-free realignment exists. Loud failure instead of silent corruption.
- has_sequences keyed on meta_test.

## Prevention
- NEW tests/unit/test_train_sequence_alignment.py reproduces the bug and locks the fix.
- Full train-side fix (if CNN-on-real-sequences is ever wanted): plumb meta_train out of
  DataPrepPipeline.run() (Option-B-wide), then key X_seq_train on meta_train["fasta_seq"].
  Deferred until needed.

## Related
- docs/incidents/INCIDENT_2026-05-23_cnn1d-0.5-auroc.md (the CNN_1D AUROC 0.5 symptom this
  class of sequence-handling defect produces).

## Commit
fix(train): realign test sequences to gene-aware split; guard train side (PM11c/PM11d)