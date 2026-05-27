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