---
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
