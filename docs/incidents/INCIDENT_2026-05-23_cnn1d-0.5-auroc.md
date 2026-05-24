---
incident_id: INCIDENT_2026-05-23_cnn1d-0.5-auroc
severity: medium
status: open
opened: 2026-05-23
affected_runs: [Run 10a]
related_commits: [post-C5 namespace refactor, ac64665]
---

# cnn_1d returns OOF AUROC = 0.5000 in Run 10a

## One-line summary

The `cnn_1d` base estimator produced exactly 0.5000 OOF AUROC across all CV folds in Run 10a — mathematically equivalent to constant predictions. Every other base model in the same run achieved 0.995+ on identical input. This is a code regression, not a data issue.

## Clarification of the model's purpose

**`cnn_1d` is a 1-D convolutional network operating on the 78-dim tabular feature vector.** It is NOT an image classifier. Input shape is `(batch, 78)`, reshaped to `(batch, 1, 78)` for 1D convolution sliding across the feature axis. It exists in the ensemble alongside `tabular_nn` (MLP) to capture local feature-feature interactions that a fully-connected layer would not specialize on.

This means: no image data is required, was ever required, or will fix the regression. The bug is in the wrapper code.

## Diagnostic findings

| Probe | Result |
|-------|--------|
| OOF AUROC | 0.5000 (exact) |
| Training duration | 9 min 49 sec (09:07:04 → 09:16:53 UTC) |
| Log lines for cnn_1d | Only the "Training cnn_1d ..." entry and the AUROC line — no fit progress, no warnings, no errors |
| `grep -iE "fallback|silent|nan|inf|warning|skip"` in cnn window | No hits |
| Run 9 cnn_1d behavior | Trained successfully (non-0.5 AUROC) — regression introduced between Run 9 and Run 10a |
| Likely cause (from project memory) | "Pickle-refactor likely broke inner `_CNN1D._build_model.<locals>._CNN1D` closure" |

A 0.5000 AUROC exactly means every prediction received the same probability score. The most common causes:

1. **Closure / scope breakage:** `_build_model` defines a nested `_CNN1D` class whose forward pass references variables from the outer closure. A refactor that moved imports or changed the class hierarchy can leave the inner forward returning a stale-bound constant tensor.
2. **Silent device mismatch:** model on CPU, optimizer on GPU (or vice versa) → no error, just no parameter updates → constant init weights → constant outputs.
3. **Loss not connected to model parameters:** if the wrapper accidentally detaches gradients (e.g. `model.eval()` set during training, or `with torch.no_grad():` covering the wrong block).
4. **Wrong output head:** if the post-refactor model returns logits from an uninitialized layer, all outputs collapse.

## Why this slipped through

- Run 9 worked, so the test wasn't exercised on the new code path.
- No unit test gates AUROC > 0.55 before allowing a wrapper to enter the ensemble.
- The wrapper does not log fit progress (no epoch/loss prints), so the 9-min window was opaque.

## Reproduction & fix plan (Run 10b)

```bash
# 1. Local diff of the cnn_1d wrapper between Run 9 and current HEAD
git log --oneline -- src/genomic_variant_classifier/models/cnn_1d.py
git diff <run9-commit>..HEAD -- src/genomic_variant_classifier/models/cnn_1d.py

# 2. Smoke test on tiny fixture (must run before any cloud relaunch)
python -m pytest tests/unit/test_cnn_1d_wrapper.py -v
# Add assertion: AUROC > 0.55 on a 1K-row synthetic binary classification
```

The unit test must:
1. Generate 1000 synthetic samples with a known signal (one feature carries the label with noise).
2. Fit `cnn_1d` for 20 epochs.
3. Assert OOF AUROC ≥ 0.65.
4. Fail loudly if predictions are constant (`np.std(preds) < 1e-6`).

## What this is NOT

- NOT a data shortage. Other models trained on identical input got 0.995+.
- NOT an indication that image data should be added. The architecture is tabular.
- NOT a reason to remove cnn_1d from the ensemble — fix the wrapper.

## Roadmap note on image data

Per the project memory (edit #15 in standing rules): the project is multi-modal-from-start with ResNet + image + DNA/RNA/protein pipelines in the GenAssoc v1 roadmap. However:

- The current Phase 0 work is establishing the tabular ACMG-5 baseline.
- The locked test AUROC for Phase 0 has not yet been produced (Run 10a was the third attempt).
- Image data integration is NOT scoped to a numbered phase yet.
- The ablation matrix (lookup-only, feature-permutation, unseen-gene-holdout, true-generalization) must complete before adding modalities — otherwise AUROC changes cannot be attributed to images vs upstream fixes.

**Recommendation: do NOT begin image data acquisition until Phase 0 baseline is locked and ablations complete.** Image data scoping deserves its own design doc covering: which images (cryo-EM? IHC? AlphaFold-predicted structures? histology?), licensing, storage cost on Vast.ai / Drive, and which architectures consume them. That doc is out of scope for the current run cycle.

## Status

- [x] Root cause hypothesized (closure breakage from post-C5 refactor)
- [ ] Diff cnn_1d wrapper Run 9 vs HEAD
- [ ] Add tests/unit/test_cnn_1d_wrapper.py with AUROC > 0.55 gate
- [ ] Fix the wrapper, verify on smoke test locally
- [ ] Include in Run 10b on Vast.ai (alongside KAN re-introduction on CPU-only instance)
