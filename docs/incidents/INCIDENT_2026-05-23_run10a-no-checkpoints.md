---
incident_id: INCIDENT_2026-05-23_run10a-no-checkpoints
severity: high
status: open
opened: 2026-05-23
affected_runs: [Run 9, Run 10a]
related_commits: [66593d6, ac64665]
---

# Run 10a -- sixteen hours of training, zero recoverable model artifacts

## One-line summary

A 16-hour, $13 Vast.ai run produced seven valid OOF AUROCs in the log, then was killed mid-KAN -- and *no model checkpoints existed on disk anywhere in /workspace*. Same root cause as the Run 9 PicklingError: the ensemble persists only at pipeline end, never mid-training.

## What happened (timeline, UTC)

| Time | Event |
|------|-------|
| 07:46 May 23 | Run 10a launched on Vast.ai inst 37429606 (RTX 4090, $0.76/hr) |
| 07:52 | DataPrep complete (~6 min); 17 annotations done; LOVD fix validated (369 variants); DbNSFP fix validated (204K SIFT scores) |
| 07:57 → 09:16 | 7 base models trained successfully, OOF AUROCs all healthy except cnn_1d=0.5000 (separate incident) |
| 09:16 | KAN training begins |
| 13:41 | KAN cycle 1 of 6 complete (4h 25m) |
| 19:41 | KAN cycle 2 complete |
| 21:08 | KAN cycle 3 begins |
| 23:38 | Monitoring round: `find /workspace -name "*.pkl" -o -name "*.joblib" -o -name "*.cbm"` returns **empty**. Models dir present but contains only `.` and `..`. |

## Root cause

`variant_ensemble.py` writes nothing to disk during training. The training loop:

```python
for name, estimator in self.base_estimators.items():
    # ... CV fit ...
    oof_preds[:, i] = ...
    logger.info(f"  {name} OOF AUROC: {auroc:.4f}")
    # ← NO PERSISTENCE HERE
```

`EnsembleConfig.__post_init__` creates `model_dir` but never writes to it:

```python
def __post_init__(self) -> None:
    self.model_dir = Path(self.model_dir)
    self.model_dir.mkdir(parents=True, exist_ok=True)
```

Persistence happens only at end-of-pipeline `ensemble.save()`. In Run 9 that call raised PicklingError on the nested `_CNN1D._build_model.<locals>._CNN1D` class and lost all training. In Run 10a the run was killed before reaching the save call -- same outcome.

The Phase 1.7 patch (commit `66593d6`) was documented as "per-model checkpoint" but empirically does not write per-model. It appears to have only added the `model_dir.mkdir` line plus the pickle fix for the nested CNN class.

## Why this is unacceptable

| Run | Cost | Hours | Salvageable artifacts |
|-----|------|-------|----------------------|
| Run 9  | ~$9.70 | 11.4 | Log file only |
| Run 10a | ~$13.00 | 16+ | Log file only |
| **Total wasted** | **~$22.70** | **27.4 hr** | -- |

Two consecutive runs lost all training work to the same architectural omission. The standing pre-flight rule did not catch it because the rule did not require *runtime verification* that checkpoints actually fire.

## The fix -- incremental persistence patch

Append to the training loop in `variant_ensemble.py`, immediately after the OOF AUROC log line:

```python
import joblib
import json
import numpy as np
from datetime import datetime

# Persist this base model immediately
model_path = self.config.model_dir / f"{name}.joblib"
oof_path   = self.config.model_dir / f"{name}_oof.npy"
meta_path  = self.config.model_dir / f"{name}_meta.json"

try:
    joblib.dump(estimator, model_path, compress=3)
    np.save(oof_path, oof_preds[:, i])
    with open(meta_path, "w") as f:
        json.dump({
            "name": name,
            "oof_auroc": float(auroc),
            "saved_at_utc": datetime.utcnow().isoformat(),
            "n_samples": int(len(y_fit)),
        }, f, indent=2)
    size_mb = model_path.stat().st_size / 1e6
    logger.info(f"    {name} checkpoint saved: {model_path.name} ({size_mb:.1f} MB)")
except Exception as exc:
    logger.error(f"    {name} checkpoint FAILED to save: {exc}", exc_info=True)
    # do not abort -- training continues, but flag the failure
```

After the loop finishes, the stacker and GNN train against the full `oof_preds` array. If any pipeline step crashes after that, all base models and their OOF arrays are already on disk and the stacker can be retrained offline in seconds.

## Verification post-patch

Within 30 minutes of relaunching, the following must be true:

```bash
ssh -i $KEY -p $VAST_PORT -T $REMOTE 'ls -la /workspace/outputs/run10a/full/models/'
# Expected to contain: random_forest.joblib + random_forest_oof.npy + random_forest_meta.json
# at minimum (random_forest finishes first, ~5 min)
```

If after first base model completes (visible by "Training xgboost ..." appearing in the log) the models dir is still empty, **the patch did not take effect**. ABORT immediately, do not let the run continue. The cost of an early abort is $0.10; the cost of letting it run is ~$13.

## Prevention going forward

1. **Memory rule added (memory edit #29):** every long cloud training >30 min MUST use incremental checkpointing; pre-flight must verify a checkpoint file appears within the first 30 min.
2. **Pre-flight script extension:** add a `verify_checkpoint_emission` step to `scripts/preflight_review.py` that polls the models dir after first base model log line and fails if no files appear.
3. **Unit test:** add `tests/integration/test_ensemble_persistence.py` that fits the ensemble on a 1K-row fixture and asserts every base estimator produces a `.joblib + _oof.npy + _meta.json` triple before pipeline end.
4. **Dockerfile:** add post-fit checkpoint verification to the `trainer` image's CMD so the image refuses to call `ensemble.save()` if no per-model checkpoints exist.


## Precursor incidents

This issue has a documented history. The same root cause hit Run 9 and was filed two weeks ago:

- `docs/incidents/INCIDENT_2026-05-12_no-per-model-checkpoint.md` -- first occurrence, Run 9 (May 12). PicklingError raised by `ensemble.save()` on the nested `_CNN1D._build_model.<locals>._CNN1D` class, losing all training.
- `docs/incidents/INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md` -- related root cause for the pickle failure.
- `docs/incidents/INCIDENT_2026-05-13_phase17-apply-failure.md` -- attempt to apply the Phase 1.7 patch failed to actually add per-model writes; only `model_dir.mkdir()` made it in.

Today's run is the third occurrence of "long expensive training, no salvageable checkpoints." The standing rule added in memory (#29) closes the gap going forward.
## Status

- [x] Root cause identified
- [x] Standing rule added to memory
- [ ] Patch applied locally → push to main
- [ ] Run 10c launched with patched code
- [ ] Verification: checkpoints appear within 30 min
- [ ] Unit test added under tests/integration/
- [ ] Pre-flight extension committed

## Status update (2026-05-31): RESOLVED

Per-model incremental checkpointing is present in the base-model loop of
`variant_ensemble.py`: immediately after each model's OOF AUROC is logged it writes
`{name}.joblib`, `{name}_oof.npy`, `{name}_oof_indices.npy`, and `{name}_meta.json`
(with `saved_at_utc`) to `config.model_dir`, inside a log-but-do-not-abort try/except.
Locked by `tests/unit/test_ensemble_persistence.py::test_per_model_checkpoints_written`,
which fits a fast-tabular ensemble on a 300-row fixture and asserts the four-file quartet
plus OOF/index length parity for every base model. A regression that drops the emission
fails CI. Closed.
