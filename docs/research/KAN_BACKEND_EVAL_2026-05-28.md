# KAN Backend Evaluation (2026-05-28)

**Status**: decision of record (no code change). Supersedes the earlier working
assumption that imodelsx lacks batching.
**Scope**: which KAN backend the ensemble uses for Run 15 and beyond, and
whether a pre-Run-15 backend swap is warranted.
**Provenance**: source audit recorded in docs/sessions/SESSION_2026-05-28.md;
commit chain 3cf287a..4989a70 (2026-05-28).

---

## 1. Why this evaluation exists
KAN (Kolmogorov-Arnold Network) is one base estimator in the stacked ensemble;
its value is representational diversity against the GBDTs (CatBoost/LightGBM/
XGBoost) and the other neural members. Run 10a saw a pykan runaway (~17.9 GB
peak, ~19h22m, ~$14.72) at 1.2M samples. Open question entering Run 15: is the
current KAN path memory-safe, and should the backend change before launch?

## 2. Authoritative source findings (code audit, 2026-05-28)
These come from reading the installed/repo source directly and supersede the
earlier research-report risk ranking.

- Backend ladder (src/genomic_variant_classifier/models/kan.py): imodelsx
  (primary, _fit_imodelsx) -> pykan (_fit_pykan) -> efficient-kan
  (_fit_efficient_kan) -> MLP (_fit_mlp).
- imodelsx BATCHES. imodelsx/kan/kan_sklearn.py: def fit(self, X, y,
  batch_size=512, ...) builds TensorDataset + DataLoader(batch_size=512). X and
  y are CPU tensors; only the model is moved to device; each batch moves to GPU
  inside the loop. Peak GPU memory is bounded by batch size, NOT dataset size.
  The Run-10a 17.9 GB event was pykan-specific (full-batch spline-grid
  intermediates) and cannot recur on the imodelsx path.
- Subsample cap. kan.py max_fit_samples default = 100_000 (no override anywhere
  in src/ or scripts/, so Run 15 uses the default). _fit_imodelsx subsamples
  1.2M -> 100k via sklearn.utils.resample(n_samples=100_000, stratify=y,
  random_state=...), preserving pathogenic/benign balance. Minor: resample's
  default replace=True makes this a stratified bootstrap (~4% duplicate rows at
  100k/1.2M); the pykan path's _subsample_if_needed uses replace=False.
  Cosmetic, not a correctness issue; not changed.
- imodelsx v1.0.13 bug (bare-name test_size/random_state/shuffle inside fit())
  is handled twice: kan.py sets the missing instance attrs after construction
  (L181-183), and scripts/launch_run11_vm.sh sed-patches the source on fresh VM
  installs (L191-197). The local .venv312 copy is already patched.

Implication: the imodelsx primary path is memory-safe at any N. No pre-Run-15
backend swap is warranted on memory grounds.

## 3. Backend comparison (future optimization, not a Run-15 action)
- imodelsx (wraps Blealtan/efficient-kan): current primary. Batches;
  memory-safe; KANClassifier-compatible. Correct choice now.
- efficient-kan (Blealtan): the engine inside imodelsx; direct fallback
  (_fit_efficient_kan). Not on PyPI (vendored).
- FastKAN (ZiyaoLi/fast-kan, RBF basis): ~3.3x faster than efficient-kan in
  published microbenchmarks. Candidate FUTURE second backend (ahead of pykan)
  IF KAN becomes the wall-clock long-pole. Not on PyPI (would need vendoring).
- pykan (official): source of the Run-10a runaway at full batch; retained as a
  fallback only, not primary.
- PolyKAN (arXiv 2511.14852, fused CUDA kernels, PPoPP 2026): promising on
  paper but UNRELEASED; watch, do not depend on.

## 4. Decision (2026-05-28)
- Keep imodelsx as the primary KAN backend for Run 15. No swap.
- KAN trains on a 100k stratified subsample at batch_size=512 -> memory-safe;
  not expected to be the memory or time bottleneck at 100k.
- FastKAN is a FUTURE speed optimization, off the Run-15 critical path. Revisit
  only if Run 15 logs show KAN as the wall-clock long-pole.
- Do not modify the backend ladder or drop pykan without an explicit decision
  plus measurements.

## 5. Profiling protocol (only if/when KAN time becomes a concern)
Run ON the GPU VM (a CPU-only local box cannot measure GPU memory):
- Sizes: 10k, 100k, 1M rows.
- Capture torch.cuda.max_memory_allocated() and wall-clock per epoch per size.
- Compare imodelsx vs FastKAN at matched width/depth before any swap.

## 6. Files inspected
- src/genomic_variant_classifier/models/kan.py
- .venv312/Lib/site-packages/imodelsx/kan/kan_sklearn.py + kan_modules.py
- scripts/launch_run11_vm.sh
- warpKAN was raised and dismissed: warp-lang on PyPI = NVIDIA Warp (GPU
  simulation framework), not a KAN; no such package exists.