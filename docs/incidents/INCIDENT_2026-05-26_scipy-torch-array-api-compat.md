# INCIDENT 2026-05-26 — scipy / array_api_compat / torch import-time crash on local Windows venv

## Status
OPEN. Pre-existing latent issue surfaced (not caused) by the phylop test
relocation on 2026-05-26. Local-only; does not affect Vast.ai training images.

## Symptom
Importing `scipy.stats` on the local `.venv312` (Windows, Python 3.12.10) raises:

    TypeError: issubclass() arg 2 must be a class, a tuple of classes, or a union

at `scipy/_lib/array_api_compat/common/_helpers.py:69` (`_issubclass_fast`).
The transformers package additionally raises:

    ValueError: torch.__spec__ is not set

These cascade through every test file that imports sklearn (because sklearn
imports scipy.stats), producing 12 collection errors on full-suite pytest runs.

## Visibility timeline
Pre-2026-05-26: hidden behind `test_phylop_block.py:32` NameError, which was
the first collection failure pytest hit under `-x`.

2026-05-26 (this session): unmasked when the phylop NameError was cleared
by relocating the file under `tests/unit/` with proper module imports
(commit follow-up to 3a166f6). Twelve collection errors now visible.

## Root Cause Hypothesis
1. **Most likely**: torch is partially or incorrectly installed. The
   `torch.__spec__ is not set` error suggests the package metadata is
   corrupt, which propagates into array_api_compat's torch-detection path
   (`_issubclass_fast` ends up calling `issubclass(cls, "Tensor")` with
   a string instead of a class).
2. **Possible**: scipy upgraded to a version that eagerly evaluates
   distribution docstrings at import time, AND array_api_compat shipped
   a version where `_issubclass_fast` fails ungracefully when its target
   class is unimportable.
3. **Possible**: a numpy major-version change broke torch's metadata.

## Reproduction
    python -c "import scipy.stats"

on the local `.venv312` venv. Reproduces every time.

## Impact
- Local full-suite pytest: 12 collection errors. Blocks Charter v1.1 G1 gate.
- Vast.ai training: UNAFFECTED. Run 14 trained successfully on Linux GPU image.
- Run 15 training: not blocked by this incident, but the G1 gate must be
  cleared before launch.

## Why Not Fixed in This Commit
Environmental fix (likely `pip install --force-reinstall torch` on Windows,
~750 MB download). Holding the phylop relocation commit hostage to a
multi-hour torch reinstall would delay forward progress on the anomaly
sweep without benefit.

## Planned Resolution
1. Capture version inventory: `pip show torch scipy array_api_compat numpy`
   and `python -c "import importlib.util; print(importlib.util.find_spec('torch'))"`.
2. Try `pip install --force-reinstall --no-deps torch` first (smallest
   blast radius).
3. If that does not resolve, pin scipy to `<1.15` to disable the eager
   docstring generation, or pin array_api_compat to its last known-good
   version.
4. Re-run full pytest suite. Must reach 0 collection errors.
5. Update the Charter v1.1 G1 gate text to require this resolution before
   any `vastai create` for Run 15.

## Affected Test Files (12 collection errors via the same chain)
- tests/unit/test_alphamissense.py
- tests/unit/test_clingen.py
- tests/unit/test_dbsnp.py
- tests/unit/test_esm2_activation.py (alternate failure path: torch.__spec__)
- tests/unit/test_eve.py
- tests/unit/test_hgmd.py
- tests/unit/test_lovd_annotation_reaches_training_matrix.py
- tests/unit/test_mc_dropout_uncertainty.py
- tests/unit/test_omim.py
- tests/unit/test_prediction_artifacts.py
- tests/unit/test_splice_ai_promotion.py
- tests/unit/test_vep.py

## Related
- 3a166f6 — A1 false-anomaly closure (which exercised the full-suite gate
  and indirectly surfaced this incident).
- Phylop relocation commit (follow-up to 3a166f6) — directly surfaced this
  issue by clearing the test_phylop_block.py NameError.

---

## Corrections (2026-05-27)

**Status**: OPEN -> RESOLVED.
**Verified by**: Phase B.1-B.9 of session 2026-05-27.
**Fix commit**: `9eec8eb` - fix(pytest): restrict discovery to tests/ to stop sys.modules["torch"] pollution.
**Session record**: `docs/sessions/SESSION_2026-05-27.md`.

### Root Cause Hypotheses (L28-39 above) - all DISPROVED

| # | Original hypothesis | Status | Disproved by |
|---|---|---|---|
| 1 | torch is partially or incorrectly installed | DISPROVED | B.1.3 of session 2026-05-27: `python -c "import torch; print(torch.__file__, torch.__version__, torch.__spec__, torch.Tensor)"` all work cleanly in plain Python. torch is fully and correctly installed. |
| 2 | scipy upgraded with eager docstring evaluation | NOT investigated, IRRELEVANT given actual root cause | n/a |
| 3 | numpy major-version change broke torch metadata | NOT investigated, IRRELEVANT given actual root cause | n/a |

### Verified Root Cause

`src/genomic_variant_classifier/agent_layer/test_message_bus.py` at L87-89 contains module-level code:

    for _mod in ("ewc_utils", "shap", "torch", "feedparser", "requests"):
        if _mod not in sys.modules:
            sys.modules[_mod] = MagicMock()

This file matches pytest's default `test_*.py` auto-discovery pattern. During full-suite collection, pytest imports it. The import runs the L87-89 loop, replacing `sys.modules["torch"]` with a MagicMock. Subsequent collection of 12 downstream test files triggers scipy.stats's array_api_compat `_issubclass_fast`, which calls `getattr(sys.modules["torch"], "Tensor")` and gets back a MagicMock - hashable so it passes scipy's lru_cache key check, but NOT a class, so the `issubclass(cls, parent_cls)` call raises TypeError.

The `test_esm2_activation.py` `ValueError: torch.__spec__ is not set` (the alternate failure noted in the original Symptom section above) is the same pollution viewed via a different lookup path.

### Proof (B.6.4 of session 2026-05-27)

| Run | Tests collected | Errors |
|---|---|---|
| Baseline (full rootdir) | 416 | 12 |
| Minus `test_message_bus.py` | 552 | 0 |

The +136 tests = exactly the 12 victim files' counts (17+10+10+3+18+10+2+7+10+11+22+16 from per-file collection in B.4.1). Test-count arithmetic decisive.

### Fix Applied (commit 9eec8eb)

`pyproject.toml` gained `[tool.pytest.ini_options]` with `testpaths = ["tests"]`. This restricts pytest's auto-discovery to the canonical `tests/` tree. `test_message_bus.py` is unmodified and remains runnable by explicit path:

    python -m pytest src/genomic_variant_classifier/agent_layer/test_message_bus.py
    python src/genomic_variant_classifier/agent_layer/test_message_bus.py

The "Planned Resolution" at L57-67 above is NOT what was done. No `pip install --force-reinstall torch` was needed - torch was never broken. No scipy or array_api_compat pinning was needed.

### Side Effect (B.8.1)

Root-level `test_catboost.py` (17718 B, untracked per `.gitignore:95`, in "Scratch and generated files" section per `.gitignore:92`) is no longer auto-discovered. It contains 26 scratch tests, still runnable by explicit path. This is a correctness improvement - cloud/CI runs (Vast.ai) never saw this file because it is untracked. Local pytest behavior now aligns with cloud pytest behavior. The canonical tracked file at `tests/unit/test_catboost.py` (20551 B) remains in default discovery.

### Verified Post-Fix

- `python -m pytest --collect-only -q`: 526 tests, 0 errors (was 416 collected + 12 errors).
- `tests/unit/test_mc_dropout_uncertainty.py`: 7 passed in 5.35s (A1 regression - shipped in 3a166f6 on 2026-05-26 but never actually ran under pytest until 2026-05-27 because it was among the 12 erroring files).
- `tests/unit/test_alphamissense.py`: 17 passed in 20.28s.
- `tests/unit/test_eve.py`: 18 passed in 20.73s.
- `tests/unit/test_prediction_artifacts.py`: 11 passed in 5.13s.

### Deferred Follow-ups

- **D12**: Refactor `test_message_bus.py` L87-89 sys.modules pollution into pytest `monkeypatch.setitem` fixtures with proper teardown. The current module-level design pollutes sys.modules even when the file is run standalone (`python test_message_bus.py`). Post-Run-15 cleanup.
- **D14**: Codify the 6 process lessons from `docs/sessions/SESSION_2026-05-27.md` section 9 into standing rules (memory).

### Lesson on Incident Documentation

This file preserves the original hypotheses (L28-39 above) and "Planned Resolution" (L57-67) in their original form. Future readers can see the diagnostic trail of "what we believed on 2026-05-26 vs what turned out to be true on 2026-05-27". Incident docs should APPEND corrections, not rewrite earlier content. Archaeological context is valuable for future debuggers facing similar symptoms.
