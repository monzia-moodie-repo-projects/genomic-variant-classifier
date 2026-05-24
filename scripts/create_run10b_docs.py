"""
Run 10b-partial salvage - docs creation helper.

Writes 4 doc files (session + 2 incident reports + CHANGELOG entry merge)
inline, sidestepping the .md download issue. The 4 docs are embedded as
raw triple-quoted strings; Python handles backticks/quotes/em-dashes
natively, no escaping needed.

Usage (from C:\\Projects\\genomic-variant-classifier\\ in .venv312):
    python scripts\\create_run10b_docs.py
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(r"C:\Projects\genomic-variant-classifier")
SESSIONS_DIR = REPO / "docs" / "sessions"
INCIDENTS_DIR = REPO / "docs" / "incidents"
CHANGELOG = REPO / "docs" / "CHANGELOG.md"


# ---------------------------------------------------------------------------
# SESSION_2026-05-24.md
# ---------------------------------------------------------------------------
SESSION_DOC = r"""---
date: 2026-05-24
phase: 1.7.1
run: 10b
status: partial-salvage
test_auroc: 0.9970
test_size: 349067
val_auroc: 0.9970
val_size: 154404
working_models: 8
broken_models: 1
incidents:
  - INCIDENT_2026-05-24_run10b-premature-destroy.md
  - INCIDENT_2026-05-24_cnn1d-cross-platform-unpickle.md
---

# Session 2026-05-24 - Run 10b launch, instance destroyed mid-pipeline, local salvage

## Outcome

**HEADLINE: Simple-average ensemble TEST AUROC = 0.9970** on 349,067 locked-test variants
(prevalence 0.1999), reconstructed locally from 8 working base models after the cloud
instance was destroyed prematurely.

## Timeline (UTC)

| Time | Event |
|---|---|
| 03:25:44 | Run 10b launched with `launch_run10b_skip_kan_v2.sh` (commit f147112) |
| 03:25:48 | 5-stage v2 preflight passed (4 seconds total) |
| 03:26:03 | Score annotation starts |
| 03:28:17 | Annotation complete (matched Run 10a annotation counts exactly) |
| 03:28:39 | Ensemble training starts (10 estimators expected) |
| 03:33:52 | random_forest done, OOF AUROC 0.9961, **first checkpoint landed @ 03:34:04** |
| 03:34:52 -> 04:17:19 | xgb, lgb, lr, gbm, cb checkpoints landed in sequence |
| 04:38:01 | tabular_nn done, OOF AUROC 0.9971 |
| 04:59:40 | cnn_1d done, OOF AUROC 0.5000 (known regression) |
| 05:36:58 | mc_dropout done, OOF AUROC 0.9971 (NEW - was hidden behind KAN in Run 10a) |
| 05:36:59 -> ~05:55 | deep_ensemble fitting 5/5 members |
| ~06:00 | **vastai destroy instance fired mid-deep_ensemble** (destroy in same paste block as SCP) |
| ~06:00 | SCP had already pulled 9 base model triplets + splits + final log |
| ~06:00 -> 06:48 | Local diagnostics confirm instance gone, artifacts on disk |
| 06:48 -> 06:51 | Phase 2 v2 ran locally, 2.3 minutes wall clock, produced metrics_partial.json |

## What was lost vs. what was preserved

### Lost forever (instance destroyed in RAM/cloud disk)
- **deep_ensemble.joblib** - was fitting member 5/5 when killed
- **CV-stacked meta-learner** - pipeline never reached this stage
- **GNN model** - never reached this stage
- **Cloud-computed locked-test AUROC** - never computed
- **metrics.json / per_model_metrics.csv** - written at pipeline end

### Preserved locally (~291 MB total)
- 9 base model triplets: `.joblib` + `_oof.npy` + `_meta.json` for rf, xgb, lgb, lr, gbm, cb, tnn, cnn_1d, mc_dropout
- All 6 splits: `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test` parquet files at `full/splits/`
- 3 meta-feature parquets: `meta_train`, `meta_val`, `meta_test`
- Final training log (23,997 bytes)

## Per-model TEST AUROC (locally reconstructed)

```
catboost              0.9970   (cloud OOF 0.9977, delta -0.0007)
gradient_boosting     0.9965   (cloud OOF 0.9970, delta -0.0005)
lightgbm              0.9970   (cloud OOF 0.9978, delta -0.0008)
logistic_regression   0.9937   (cloud OOF 0.9952, delta -0.0015)
mc_dropout            0.9969   (cloud OOF 0.9971, delta -0.0002)
random_forest         0.9942   (cloud OOF 0.9961, delta -0.0019)
tabular_nn            0.9969   (cloud OOF 0.9971, delta -0.0002)
xgboost               0.9968   (cloud OOF 0.9979, delta -0.0011)
cnn_1d                FAILED   (cross-platform pickle bug; AUROC was 0.5 anyway)
```

Average OOF->TEST degradation: -0.0009. Healthy generalization.

## Simple-average ensemble (8 useful models)

| Split | AUROC | Size |
|---|---|---|
| Val  | 0.9970 | 154,404 |
| Test | 0.9970 | 349,067 |

Headline matches best-single model (catboost, lightgbm), reflecting the high cross-model
correlation we saw in Run 9 (OOF blend delta=+0.0005 over best-single).

## Lessons learned

### Attempted, Failed, Fixed

**Attempted:**
- Full Run 10b training with --skip-kan to bypass KAN runaway
- Phase 1.7.1 incremental per-model checkpoint patch
- End-to-end SCP + destroy + git commit sequence in single paste block
- Approximate meta-learner stacking via OOF arrays + y_train[:1017633]

**Failed:**
1. **Premature `vastai destroy`** - destroy command shared paste block with SCP commands;
   PowerShell ran all sequentially, killing instance 37429606 ~30 min before
   deep_ensemble + meta + GNN + test eval would have completed
2. **OOF alignment for meta-learner** - OOF arrays are stored in CV-prediction order,
   not X_train row order; sanity check detected and refused bad meta-learner numbers
3. **cnn_1d local unpickle** - nested-class closure does not resolve across cloud Linux
   -> local Windows; `NoneType.__new__(X)` error

**Fixed / Worked as designed:**
1. **Phase 1.7.1 patch validated in disaster recovery** - per-model checkpoints saved
   9 of 10 models when the instance died unexpectedly
2. **Phase 2 v2 auto-discovery** - found splits at `full/splits/` despite Phase 1
   inventory's wrong assumption of `full/`
3. **Sanity check caught misalignment** - refused to publish inflated meta-learner numbers,
   fell back to reliable simple-average ensemble

### Learned (durable insights for Run 11+)

1. **STANDING RULE #30** - Irreversible cloud commands (`vastai destroy`, `rm -rf`,
   force-push, force-add deletions) NEVER share a paste block with preceding setup/copy
   commands. Always isolate in a separate code block requiring explicit re-paste after
   manual verification of expected state.
2. **OOF arrays need row-index sidecar** - to reconstruct meta-learner stacking from
   saved OOF arrays, the per-fold prediction-to-row mapping must be saved alongside
   (e.g., `{name}_oof_indices.npy`).
3. **cnn_1d serialization is fragile** - nested closure classes don't survive
   cross-platform pickle. Run 11 should refactor `_CNN1D._build_model` to module-level
   class definition OR switch CNN serialization to `torch.save(state_dict)`.
4. **Split parquets live at `<run_dir>/splits/`**, not `<run_dir>/` directly.
5. **Local CPU inference for 503K rows is FAST** - Phase 2 v2 took 2.3 min total.
   RF alone was 33 sec for predict on val + test combined.
6. **mc_dropout + deep_ensemble were hidden behind KAN in Run 10a** - they're real
   estimators (10 total when --skip-kan); Run 10a memory of "8 base models" was an
   undercount because KAN cycled forever and we never saw past it.

## Cost ledger

| Phase | Wall time | Cost |
|---|---|---|
| Run 10a (KAN runaway from prior session) | 19h 22m | $14.72 |
| Reset + v2 prep | 17 min | $0.22 |
| Run 10b through 9 of 10 base models | 2h 27m to 05:52 UTC | $1.86 |
| Run 10b deep_ensemble in flight at destroy | ~10 min wasted | $0.13 |
| Local salvage (CPU) | 2.3 min | $0.00 |
| **Total** | **~22h 25m** | **~$16.93** |

## Run 11 priority backlog

1. **cnn_1d module-level refactor** - move `_CNN1D` class out of `_build_model` closure;
   add unit test for round-trip pickle on Windows
2. **OOF row-index sidecar** - save `{name}_oof_indices.npy` alongside `{name}_oof.npy`
3. **HGVSp parser** (`src/data/hgvsp_parser.py`) to fix ESM-2 + EVE silent-zero
4. **deep_ensemble retrain** on small CPU Vast.ai instance - only the deep_ensemble
   model needs to be retrained; could complete in ~30 min for ~$0.20

## Files written this session

- `outputs/run10b_final/run10b_master_final.log` (23,997 B)
- `outputs/run10b_final/full/phase1_inventory.json`
- `outputs/run10b_final/full/metrics_partial.json` (HEADLINE NUMBERS)
- `outputs/run10b_final/full/per_model_metrics_partial.csv`
- `outputs/run10b_final/full/models/` (9 base model triplets, ~291 MB)
- `outputs/run10b_final/full/splits/` (6 split parquets + 3 meta-feature parquets)
- `outputs/run10b_final/phase2_eval.log`
- `scripts/launch_run10b_vm.sh` (committed in 927e8d6)
- `scripts/run10b_partial_phase1_inventory.py`
- `scripts/run10b_partial_phase2_eval_v2.py`

## Commits

- `f147112` - Phase 1.7.1 incremental checkpoint patch + incident docs (pre-launch)
- `927e8d6` - Run 10b launch script committed (post-destroy, before salvage)
- `9b1400e` - Phase 1.7.1 Run 10b-partial salvage: TEST AUROC 0.9970
- `8e1b21f` - (misnamed; only contained a CHANGELOG blank-line modification)
- (next) - Docs proper: session + incident reports + CHANGELOG entry
"""


# ---------------------------------------------------------------------------
# INCIDENT_2026-05-24_run10b-premature-destroy.md
# ---------------------------------------------------------------------------
INCIDENT_DESTROY = r"""---
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
"""


# ---------------------------------------------------------------------------
# INCIDENT_2026-05-24_cnn1d-cross-platform-unpickle.md
# ---------------------------------------------------------------------------
INCIDENT_CNN1D = r"""---
incident_id: 2026-05-24_cnn1d-cross-platform-unpickle
date: 2026-05-24
severity: medium
status: deferred-to-run-11
affects: cnn_1d base estimator local loading
workaround: exclude cnn_1d from ensemble (AUROC was 0.5000 anyway, no loss)
---

# Incident: cnn_1d.joblib fails to load on local Windows after successful cloud Linux save

## Summary

The `cnn_1d.joblib` file (284,436 bytes) was successfully saved on the Vast.ai cloud
instance (Linux, Python 3.10+) by the Phase 1.7.1 incremental checkpoint patch. When
loaded on the local development machine (Windows, Python 3.12.10, `.venv312`), joblib
raises:

```
TypeError: NoneType.__new__(X): X is not a type object (NoneType)
  File "joblib/numpy_pickle.py", line 749, in load
    obj = _unpickle(...)
```

## Root cause

The CNN1D model class is defined as a closure inside a method:

```python
class _CNN1DWrapper:
    def _build_model(self):
        class _CNN1D(nn.Module):  # nested closure class
            def __init__(self):
                ...
        return _CNN1D()
```

When `joblib.dump` pickles the trained instance, the nested class `_CNN1D` is stored
with a qualified-name path like `..._CNN1DWrapper._build_model.<locals>._CNN1D`. On
unpickling:
1. Pickle attempts to look up the class by its qualified name path
2. The closure-local class isn't reachable through normal module attribute lookup
3. Returns `None` as the class object
4. `None.__new__(X)` then fails with the TypeError above

## Why the bug manifests cross-platform but not in-process

- **Cloud Linux save -> cloud Linux load**: would also have failed if attempted
  (this is the same root cause as the Run 9 PicklingError during `ensemble.save()`)
- **Cloud Linux save -> local Windows load**: the bug we hit
- **In-process** (no save/load): works fine because the closure class is reachable
  via the active function's local scope

The cnn_1d.joblib individual save succeeded on the cloud because joblib's per-instance
dump uses a more permissive pickling path than `ensemble.save()`'s full-graph dump. But
the LOAD path is the same restrictive lookup either way.

## Impact

- cnn_1d cannot be loaded for inference on any machine other than the original
  training process
- cnn_1d's OOF predictions (stored in `cnn_1d_oof.npy`) ARE usable - they're just a
  numpy array, no pickling required
- BUT cnn_1d's OOF AUROC was 0.5000 (deferred regression from Run 9) so excluding
  cnn_1d from the simple-average ensemble loses no signal
- **Net impact on Run 10b salvage: zero**

## Fix (deferred to Run 11)

### Option A (preferred): module-level class definition

```python
# src/genomic_variant_classifier/models/cnn1d.py

class _CNN1D(nn.Module):  # module-level, picklable
    def __init__(self, n_features: int):
        ...

class _CNN1DWrapper(BaseEstimator):
    def _build_model(self):
        return _CNN1D(self.n_features_)  # construct, don't define
```

Pros: pickle-safe across processes, machines, and Python versions.
Cons: small refactor; requires care if `_CNN1D` needs access to wrapper attrs.

### Option B: switch to torch.save(state_dict)

```python
def __getstate__(self):
    state = self.__dict__.copy()
    buf = io.BytesIO()
    torch.save(self._model.state_dict(), buf)
    state['_model_state_dict'] = buf.getvalue()
    del state['_model']
    return state

def __setstate__(self, state):
    self.__dict__.update(state)
    self._model = self._build_model()
    buf = io.BytesIO(state['_model_state_dict'])
    self._model.load_state_dict(torch.load(buf))
```

Pros: PyTorch-recommended pattern; portable across Python+torch versions.
Cons: more code; need to ensure `_build_model` is deterministic.

**Recommendation: Option A.** It's a smaller change and Option B doesn't solve the
underlying anti-pattern.

## Unit test to add in Run 11

```python
# tests/unit/models/test_cnn1d_persistence.py
import joblib
import numpy as np
import tempfile
from pathlib import Path

def test_cnn1d_roundtrip_pickle():
    \"\"\"Regression test: cnn_1d must survive cross-process joblib roundtrip.

    See INCIDENT_2026-05-24_cnn1d-cross-platform-unpickle.md
    \"\"\"
    from genomic_variant_classifier.models.cnn1d import CNN1DWrapper

    X = np.random.randn(100, 78).astype(np.float32)
    y = np.random.randint(0, 2, size=100)

    model = CNN1DWrapper(n_features=78, epochs=2)
    model.fit(X, y)
    p1 = model.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cnn1d.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)  # must NOT raise TypeError

    p2 = loaded.predict_proba(X)
    np.testing.assert_allclose(p1, p2, rtol=1e-5)
```

## Related incidents

- `INCIDENT_2026-04-17_esm2-hgvsp-parser.md` - separate ESM-2 issue
- Run 9 `PicklingError` on `ensemble.save()` - same root cause, manifested at save
  time instead of load time
- `INCIDENT_2026-05-23_cnn1d-0.5-auroc.md` - cnn_1d functional regression
  (different bug, also deferred to Run 11)
"""


# ---------------------------------------------------------------------------
# CHANGELOG entry (no metadata header - ready to merge directly)
# ---------------------------------------------------------------------------
CHANGELOG_ENTRY = r"""## 2026-05-24 - Run 10b launch, premature destroy, local salvage to TEST AUROC 0.9970

### Attempted
- Full Run 10b training with `launch_run10b_skip_kan_v2.sh` (KAN disabled, 10 base estimators)
- Phase 1.7.1 incremental per-model checkpoint patch (commit f147112) tested in production
- End-to-end SCP + destroy + commit sequence in single PowerShell paste block
- Approximate meta-learner stacking from saved OOF arrays + y_train

### Failed
- **Premature `vastai destroy`**: destroy command shared paste block with SCP; PowerShell ran all sequentially, killing instance 37429606 at ~06:00 UTC while deep_ensemble member 5/5 was fitting. Lost deep_ensemble + meta-learner + GNN + cloud test eval. See INCIDENT_2026-05-24_run10b-premature-destroy.md
- **OOF meta-learner alignment**: OOF arrays stored in CV-prediction order, not X_train row order. Pairing OOF with `y_train[:1017633]` gave reconstructed AUROC ~0.50 across all 8 models. Sanity check caught this; fell back to simple-average.
- **cnn_1d cross-platform unpickle**: `joblib.load` of cloud Linux-saved cnn_1d.joblib fails on local Windows with `TypeError: NoneType.__new__(X)` due to nested-class closure. See INCIDENT_2026-05-24_cnn1d-cross-platform-unpickle.md

### Fixed / Worked as designed
- **Phase 1.7.1 patch fully validated** in disaster recovery scenario. Per-model joblib + OOF + meta JSON saved right after each AUROC log preserved 9 of 10 base models when the instance died unexpectedly. Without the patch, Run 10b would have been a total loss.
- **Phase 2 v2 auto-discovery** located splits at `full/splits/` despite Phase 1 inventory's wrong assumption of `full/` root
- **Alignment sanity check** in Phase 2 v2 correctly detected misaligned OOF rows and prevented false meta-learner results from being published

### Learned
- **STANDING RULE #30**: Irreversible cloud commands NEVER share a paste block with preceding setup/copy commands. Always isolate in a separate code block requiring explicit re-paste after manual verification.
- **OOF row indices need sidecar**: To enable post-hoc meta-learner reconstruction, the per-fold prediction-to-row mapping must be saved alongside OOF arrays (`{name}_oof_indices.npy`).
- **Closure-defined classes are pickle-fragile**: `_CNN1D._build_model.<locals>._CNN1D` doesn't survive cross-process pickle. Run 11 must move `_CNN1D` to module-level.
- **Split parquets live at `<run_dir>/splits/`**, not `<run_dir>/` directly.
- **Local CPU inference is fast enough**: 503K rows x 8 models in 2.3 min wall-clock; the no-local-training rule applies to training only, inference is fine.
- **mc_dropout + deep_ensemble are real estimators**: They were hidden behind the KAN dam in Run 10a. With `--skip-kan` we see 10 base estimators, not 8.

### Outcome
Locked **TEST AUROC = 0.9970** on 349,067 variants via simple-average ensemble of 8 working base models. Matches best-single performance (catboost, lightgbm both at 0.9970). Mean OOF->TEST degradation -0.0009 across 8 working models indicates healthy generalization.

### Commits
- `f147112` Phase 1.7.1 incremental checkpoint patch (pre-launch)
- `927e8d6` Run 10b launch script committed (post-destroy)
- `9b1400e` Run 10b-partial salvage results
- `8e1b21f` (CHANGELOG blank-line modification only; superseded by this commit)

"""


def main() -> int:
    print("=" * 70)
    print("Run 10b-partial docs creator")
    print("=" * 70)

    # Pre-flight: repo exists?
    if not REPO.exists():
        print(f"FAIL: repo not found: {REPO}")
        return 2
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  REPO        = {REPO}")
    print(f"  SESSIONS    = {SESSIONS_DIR}  (exists)")
    print(f"  INCIDENTS   = {INCIDENTS_DIR}  (exists)")
    print(f"  CHANGELOG   = {CHANGELOG}  (exists: {CHANGELOG.exists()})")

    # Write session doc
    session_path = SESSIONS_DIR / "SESSION_2026-05-24.md"
    session_path.write_text(SESSION_DOC, encoding="utf-8")
    print(f"\n  Wrote: {session_path}  ({len(SESSION_DOC):,} chars)")

    # Write 2 incident reports
    inc1_path = INCIDENTS_DIR / "INCIDENT_2026-05-24_run10b-premature-destroy.md"
    inc1_path.write_text(INCIDENT_DESTROY, encoding="utf-8")
    print(f"  Wrote: {inc1_path}  ({len(INCIDENT_DESTROY):,} chars)")

    inc2_path = INCIDENTS_DIR / "INCIDENT_2026-05-24_cnn1d-cross-platform-unpickle.md"
    inc2_path.write_text(INCIDENT_CNN1D, encoding="utf-8")
    print(f"  Wrote: {inc2_path}  ({len(INCIDENT_CNN1D):,} chars)")

    # Repair + merge CHANGELOG
    if not CHANGELOG.exists():
        # Create from scratch with header
        new_content = "# CHANGELOG\n\n" + CHANGELOG_ENTRY
        print(f"\n  CHANGELOG.md missing -> creating new with entry")
    else:
        current = CHANGELOG.read_text(encoding="utf-8")
        # Strip leading whitespace (cleans up the blank-line creep from earlier botched prepends)
        original_len = len(current)
        current = current.lstrip()
        trimmed = original_len - len(current)
        if trimmed > 0:
            print(f"\n  Trimmed {trimmed} leading whitespace bytes from CHANGELOG")

        # Find "# CHANGELOG" header followed by blank line(s)
        match = re.match(r'^(# CHANGELOG\s*?\n\n+)', current)
        if match:
            header = match.group(1)
            body = current[len(header):]
            # Normalize header to exactly "# CHANGELOG\n\n"
            new_content = "# CHANGELOG\n\n" + CHANGELOG_ENTRY + body
            print(f"  Inserted new entry after '# CHANGELOG' header")
        else:
            # Fallback: prepend without header (shouldn't happen)
            new_content = CHANGELOG_ENTRY + current
            print(f"  WARN: no '# CHANGELOG' header found; prepended entry")

    CHANGELOG.write_text(new_content, encoding="utf-8")
    print(f"  Wrote: {CHANGELOG}  ({len(new_content):,} chars)")

    # Post-flight: verify all 4 files on disk
    print(f"\n{'=' * 70}")
    print("Verification:")
    for path in [session_path, inc1_path, inc2_path, CHANGELOG]:
        if path.exists():
            size = path.stat().st_size
            print(f"  OK   {path.relative_to(REPO)}  ({size:,} bytes)")
        else:
            print(f"  FAIL {path.relative_to(REPO)}  MISSING")
            return 3

    print(f"\n{'=' * 70}")
    print("DONE. Next steps (run in PowerShell):")
    print()
    print("  cd C:\\Projects\\genomic-variant-classifier")
    print("  git add docs\\CHANGELOG.md")
    print("  git add docs\\sessions\\SESSION_2026-05-24.md")
    print("  git add docs\\incidents\\INCIDENT_2026-05-24_run10b-premature-destroy.md")
    print("  git add docs\\incidents\\INCIDENT_2026-05-24_cnn1d-cross-platform-unpickle.md")
    print("  git status")
    print('  git commit -m "Phase 1.7.1: docs (session + 2 incidents + CHANGELOG entry); supersedes 8e1b21f"')
    print("  git push origin main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
