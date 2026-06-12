# Session 2026-06-11 (PM) -- ESM-2 650M LLR Fix + train.py Wiring

<!-- docs-close: ecd0474 esm2-llr+train-wiring -->

## Summary
Activated and validated ESM-2 650M for Run 16. Caught a regen-blocking OOM in the
LLR path on free CPU before any GPU spend, fixed it by windowing, then closed the
train.py wiring gap that would otherwise have silently run the 8M model with live
per-gene UniProt REST.

## Commits (chain: fd92e2d -> 1db43f1 -> ecd0474)
- `1db43f1` fix(esm2): window LLR forward pass so long proteins do not OOM
- `ecd0474` feat(train): add --esm2-model and offline UniProt index flags

## What happened

### ESM-2 650M activation probe (scripts/probe_esm2_650m_activation.py)
CPU, read-only probe exercising the real ProteinCoordConnector + ESM2Connector
interfaces. Caught a STEP-4 OOM: `_mlm_logit_matrix` ran a full-length forward pass
per protein; for TTN (~34k aa) the O(L^2) attention attempted a ~94 GB allocation,
which would OOM the RTX 4090 as well as CPU. Latent until protein_pos was populated
(prior silent-zero left the candidate set empty, so the forward pass never ran).

### LLR windowing fix (src/genomic_variant_classifier/data/esm2.py, +47/-5)
Added `_MLM_MAX_RESIDUES = 1022` and `_windowed_logit_row`. Proteins longer than the
cap window the WT- and masked-marginal reads to a context centered on the variant
residue; short proteins keep the existing one-pass-per-protein fast path. Index math
validated against a mock: residue identity preserved at N-term/mid/C-term, and short
proteins reproduce the full-sequence index exactly (no regression). Patcher anchored
on the unique `_mlm_logit_matrix` return line (dash-count-independent).

Probe re-run GREEN at 650M: esm2_delta_norm nonzero_frac=0.967 (max 9.394),
esm2_llr nonzero_frac=0.960 (range -14.600..+2.653), protein-coord coverage 0.970,
3 wt-vs-sequence mismatches counted. 9 esm2 unit tests pass.

### DECISION: ESM-2 650M for Run 16; ESM C 600M deferred
650M is validated end-to-end and connector-compatible (HuggingFace
`facebook/<name>` EsmForMaskedLM), is the ROADMAP target, and its weights are local.
ESM C (Cambrian) is net-new connector code (different SDK/loader/tokenizer/positional
range) and would conflate a model-family change with the activation under test.
Single-variable discipline: activate with the validated model now, A/B ESM-2 vs ESM C
later with the model as the controlled variable against the 650M baseline.

### train.py ESM-2 wiring (scripts/train.py, +41)
Confirmed `AnnotationConfig.esm2_model_name` defaults to `esm2_t6_8M_UR50D` and
`esm2_uniprot_index_path` defaults to None (-> live REST), and train.py overrode
neither. `ESM2Connector` is constructed once (line 893), 4-arg
(model_name/cache_path/uniprot_index_path/device); the earlier 2-arg reading was a
Select-String -Context truncation artifact, not a duplicate block. Added
`--esm2-model` / `--esm2-uniprot-index` / `--esm2-cache` / `--esm2-device`, threaded
into AnnotationConfig; extended metrics `annotation_sources` to record
esm2_model / esm2_uniprot_index / finngen / dbnsfp. 2 wiring tests pass.

## Process note
First commit attempt used a bash heredoc (`git commit -F - <<'MSG'`); PowerShell 5.1
has no heredoc, so the commit failed and HEAD stayed at fd92e2d. Recovered via
file-based `git commit -F <file>`. Reinforced: no heredocs in PS5.1; multi-line
commit messages use `-F <file>` or multiple `-m`.

## Run 16 launch contract (mandatory -- miss one and a feature deadzones)
- `--esm2-model esm2_t33_650M_UR50D`            (else 8M)
- `--esm2-uniprot-index data\external\uniprot\uniprot_human_reviewed.parquet`  (else live per-gene REST)
- `--alphamissense <AlphaMissense scores path>` (else protein_pos empty -> esm2 all-zero)
Belongs in the ONE preflight script (Test-Path each before launch).

## Open / next
- CNN blocker: train.py raises NotImplementedError when meta_test fasta_seq notna > 100
  (INCIDENT_2026-05-30; meta_train not plumbed through run()). Verify Run 16 cohort
  fasta_seq density; decide plumb-meta_train vs placeholder sequences.
- Schema baseline refresh 78 -> 79 AFTER the regen (esm2_llr becomes a live column).
- EVE still dead (needs EVE_scores_ASM acquisition) -- separate data track.
- Doc drift: AnnotationConfig docstring lists 17 steps; code runs 18 (ReactomeConnector
  already exists/runs -- stale memory note). Per-step log labels inconsistent
  (15/16, 16/17, 17/17, 18/18). Reconcile.
- Hygiene: non-ASCII em-dash in the real_data_prep.py esm2_delta_norm comment --
  confirm real byte vs Get-Content display artifact, clean if real.
- Standing run gates before launch: all-models smoke, full suite green, zero known bugs.
