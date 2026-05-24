---
incident_id: 2026-05-24_run10b-premature-destroy
date: 2026-05-24
severity: high
status: salvaged
data_loss: deep_ensemble model + cloud-computed metrics + GNN
salvaged_via: local CPU inference (Phase 2 v2)
final_test_auroc: 0.9970
---

# Incident: Run 10b cloud instance destroyed mid-pipeline

## Summary

The Vast.ai instance hosting Run 10b training (instance 37429606) was destroyed at
approximately 06:00 UTC on 2026-05-24, while the `deep_ensemble` base estimator was
fitting member 5/5. The destroy fired because `vastai destroy instance $INSTANCE_ID`
was placed in the same paste block as the preceding SCP and verification commands.
PowerShell executed all commands sequentially without pausing for the manual
verification step suggested in the inline comment.

## Sequence of events (UTC)

| Time | Event |
|---|---|
| 05:52:48 | Last successful SSH probe; deep_ensemble member 5/5 fitting |
| 05:53 -> 05:58 | User pasted "completion sequence" block (SCP + verify + destroy + git) |
| 05:58 | SCP pulled all complete artifacts (9 base models + splits + log) - successful |
| ~06:00 | `vastai destroy instance 37429606` fires; instance termination begins |
| ~06:02 | Subsequent SSH probe fails: `Connection refused` |
| 06:48 | Local diagnostic confirms: `vastai show instances` empty, instance gone |

## Root cause

UX failure in the completion-sequence script. The destroy command was visually
separated only by a `#` comment:

```powershell
# After visual confirmation of all 10 models + metrics.json present locally:
echo y | vastai destroy instance $INSTANCE_ID
```

PowerShell ignores `#` comments and runs the next executable line. The intended pause
for manual verification didn't happen because comment lines don't halt execution.

## Impact

### Lost (cannot recover without re-training)
- `deep_ensemble.joblib` - was in member 5/5 fit, never saved
- CV-stacked meta-learner - pipeline never reached this stage
- GNN model (STRING DB) - pipeline never reached this stage
- Cloud-computed metrics.json + per_model_metrics.csv
- Locked test AUROC as computed by the production pipeline

### Preserved (Phase 1.7.1 patch saved these)
- 9 base model triplets on disk (~291 MB total)
- 9 OOF arrays (uniform length 1,017,633)
- 9 meta JSON files with OOF AUROCs
- All split parquets (X_train/val/test, y_train/val/test, meta_*)
- Final training log

## Salvage path

Phase 2 v2 local CPU inference (`scripts/run10b_partial_phase2_eval_v2.py`):
- Loaded 8 of 9 base models successfully (cnn_1d failed cross-platform unpickle; see
  separate incident INCIDENT_2026-05-24_cnn1d-cross-platform-unpickle.md)
- Predicted on X_val (154,404 rows) and X_test (349,067 rows)
- Computed simple-average ensemble: **TEST AUROC = 0.9970**
- Attempted OOF-stacked meta-learner: alignment sanity check failed (OOF rows not in
  X_train order), correctly fell back to simple-average

Wall time: 2.3 min on CPU.

## What worked exactly as designed

- **Phase 1.7.1 incremental checkpoint patch** (commit f147112): without per-model
  joblib+OOF+meta dumps right after each AUROC log, the destroy would have erased
  everything in cloud RAM. With them, 9 of 10 models were already persisted.
- **Phase 2 v2 alignment sanity check**: correctly detected that the OOF arrays don't
  align with `y_train[:1017633]` row order, refused to report inflated meta-learner
  numbers, and fell back to the reliable simple-average ensemble.
- **Phase 2 v2 path auto-discovery**: found splits at `full/splits/` despite my Phase 1
  inventory's wrong assumption of `full/`.

## Mitigation (committed to memory as STANDING RULE #30)

> Irreversible cloud commands (`vastai destroy`, `rm -rf`, force-push, force-add
> deletions, anything irrecoverable) NEVER share a paste block with preceding
> setup/copy commands. Always isolate in a separate code block requiring explicit
> re-paste after manual verification of expected state.

Operational implementation:
1. SCP + verification blocks are ONE paste
2. Manual eyeball verification of artifacts on local disk
3. Destroy command is a SEPARATE paste, explicitly typed (not just pasted from a script)

## Cost

- Wasted cloud time after destroy fired: ~10 minutes of deep_ensemble that wouldn't
  have saved anyway = ~$0.13
- Salvage compute: $0 (local CPU)
- Net cost of incident: <$1 + ~1 hour of session time

## Permanent record

Test AUROC 0.9970 is the official Run 10b headline number, recorded in
`outputs/run10b_final/full/metrics_partial.json` with `status: "partial - Run 10b
instance destroyed mid-pipeline at ~06:00 UTC"`. Run 11 should retrain only the
missing pieces (deep_ensemble + meta + GNN) rather than redoing the full ~2.5h
GPU pipeline.
