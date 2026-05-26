# Reference Card — genomic-variant-classifier Runs 9 → 14

> A reusable, self-contained reference for this project and similar future projects.
> Cross-references session docs, INCIDENT docs, and commits, but stands on its own.

---

## Section A — Runs at a glance

| Run | Commit | Wall-clock | Cost  | Test AUROC | Models | KAN backend | LightGBM | Headline |
|-----|--------|-----------:|-----:|----------:|------:|-----|----|---|
| 9   | `b15a625` | 11.4h     | $9.70 | — (pickle crash) | 11/11 trained, save failed | pykan (19h OOM) | gpu | First Vast.ai run; ensemble.save PicklingError on nested class |
| 10a | various   | aborted   | —     | —         | —     | pykan timeout | — | Process aborted; multiple failed launches |
| 10b | various   | aborted   | —     | —         | partial | — | — | Lost deep_ensemble + meta + GNN to bad destroy paste |
| 11  | `7d91386→61a8d99` | 7.9h | $5.60 | 0.9974    | 9/11  | MLP fallback (fastkan missing) | SKIPPED (gpu/OpenCL fail) | First locked test AUROC ever produced |
| 12  | `a968e28→a6fa7c5` | 6.5h | $4.80 | 0.9974    | 8/11  | NameError torch | failed (cuda fail) | KAN added via imodelsx; first attempt |
| 13  | `f4dbeed`         | 6.3h | $4.90 | 0.9974    | 9/11  | NameError test_size | **OK (cpu)** | LightGBM restored; best AUPRC + Brier ever |
| 14  | `bf2f665`         | TBD   | TBD   | TBD       | TBD   | TBD | TBD | TBD |

---

## Section B — Algorithms in the ensemble (project reference)

### Decision-tree family (the workhorses)

**CatBoost** — Gradient boosting with two distinctive choices: (1) ordered boosting, which trains each iteration on a permutation where the current sample hasn't been used yet, reducing target leakage; (2) oblivious trees, which use the same split feature/value at every node in a given depth. Excels at categorical features without one-hot encoding. Top OOF AUROC across Runs 11-13. Worst calibration among top-3 GBDTs (Brier 0.0166 vs XGBoost 0.0120) — ranks well, predicts extreme probabilities.

**XGBoost** — Reference implementation of regularized gradient boosting. Second-order Taylor expansion of the loss; L1+L2 regularization on leaf weights; depth-wise tree growth. Best calibration among top-3 GBDTs in Run 13 (F1 0.9769, MCC 0.9539, Brier 0.0120). Stable across all runs.

**LightGBM** — Histogram-based gradient boosting with leaf-wise tree growth. Bins features into 255 discrete values, making cache-line access patterns extremely efficient. Two novel components: GOSS (gradient-based one-sided sampling — keep large-gradient samples, downsample small-gradient ones); EFB (exclusive feature bundling — merge sparse features that rarely co-occur). **Operational note:** PyPI binary is CPU-only; `device_type: gpu` requires OpenCL not in the wheel, `device_type: cuda` requires a CUDA-compiled build. On this dataset (1.2M rows), CPU mode matches XGBoost-on-GPU wall-clock — the GPU overhead is not amortized below ~5M rows.

**Gradient Boosting (sklearn)** — Reference GBM. No histogram binning, depth-wise growth, no GPU. ~10-100× slower than modern GBDTs but useful as a control: shows that on this dataset, modern algorithmic tricks (histograms, leaf-wise growth) buy you speed, not accuracy.

**Random Forest** — Bagging of decision trees with feature subsampling at each split. Trains trees in parallel (no sequential residual fitting). Tends to underfit signal vs GBDTs but produces well-calibrated probabilities natively. Consistent 0.9964 OOF AUROC across runs.

### Linear baseline

**Logistic Regression** — L2-regularized linear model. Establishes a critical scientific anchor: on this dataset's 78 features, LR achieves OOF AUROC 0.9942 — meaning 99.4% of discriminative signal is linearly separable in this feature space. Any nonlinear model gets at most 0.3% additional signal, and a large fraction of that may be memorization of gene-level prevalence (open question, planned for ablation matrix).

### Neural family

**TabularNN** — Feedforward neural network on tabular features. Fixed ReLU activations, learned weights at edges. On tabular data with structured heterogeneous features, generally 0.3 percentage points below the best GBDT — consistent with Grinsztajn et al. 2022.

**MC Dropout** — Same architecture as TabularNN but keeps dropout active at inference; averages T stochastic forward passes to estimate epistemic uncertainty. Computational cost of one TabularNN + T forward passes.

**Deep Ensemble** — 25 independently-initialized TabularNNs averaged. Genuine epistemic uncertainty via model diversity rather than dropout noise. Highest F1 / MCC of any single model in Run 13 (0.9771 / 0.9542) despite 25× the compute cost.

**KAN (Kolmogorov-Arnold Network)** — Learns the activation functions themselves (parameterized as B-splines) on the edges of the network, rather than placing fixed activations at the nodes. Theoretical motivation: any continuous multivariate function can be written as a sum of compositions of univariate functions (Kolmogorov-Arnold representation theorem). The hope is that learnable activations capture nonlinear thresholds without the piecewise approximation that ReLU networks need.
- **Backends evaluated:** `pykan` (v0.2.8, reference impl, ~10× slower, GPU issues); `imodelsx` (v1.0.13, wraps `efficient-kan` with sklearn fit/predict_proba); `ikan` (v1.3.0, backup).
- **Trains on a 100K subsample**, not the full 1.2M rows. KAN scales poorly on tabular data because every edge has spline parameters (memory and compute go up quadratically with width).

**CNN_1D** — 1-dimensional convolutional network on FASTA sequence one-hot encodings. Currently SKIPPED — the pipeline does not populate `fasta_seq`; if not skipped, fed dummy sequences and predicts 0.5000 for everything.

---

## Section C — The KAN remediation chain (4 bugs across 3 runs)

Pattern that explains every "fix one bug, the next one surfaces" run since Run 10:

| # | Bug | Symptom | Why nothing earlier caught it | Fix |
|---|---|---|---|---|
| 1 | `fastkan` not on PyPI | `pip install` fails on Vast.ai despite passing locally because someone had it from a wheel that no longer resolves | requirements.txt drift; SR #31 didn't exist | Remove `fastkan`, add `pykan` |
| 2 | Missing `import torch` in `_fit_imodelsx` | `NameError: name 'torch' is not defined` raised only when imodelsx is the active KAN backend on the VM | Local tests use the pykan backend; the imodelsx codepath was untested | Add `import torch` to `_fit_imodelsx` |
| 3 | LightGBM CUDA not available | `device_type: gpu` → OpenCL missing in PyPI binary; `device_type: cuda` → CUDA Tree Learner not enabled either | PyPI ships CPU-only binary; both gpu and cuda paths fail equivalently | Drop device_type flag entirely; force CPU |
| 4 | imodelsx v1.0.13 bare-name refs + missing `__init__` attrs | `fit()` references `test_size`/`random_state`/`shuffle` as bare names instead of `self.X`; `__init__` never defined them either | Upstream package bug | Two-part: (a) sed-patch installed package on VM; (b) explicitly set the three attrs on the KANClassifier instance before `.fit()` |

**Standing rule that came out of this:** *Verify pip installability and successful import of every requirement BEFORE adding it to requirements.txt.* This is Standing Rule #31.

---

## Section D — Vast.ai operational lessons

1. **GPU filter:** `dlperf >= 80, pcie_bw >= 12, gpu_name = RTX_4090`. Norway DLP 16 / PCIE 0.7 (Run 11) cost +1.5h vs Hungary DLP 97 / PCIE 12.7 (Runs 12-13). Neural training is GPU-bound and benefits from PCIe bandwidth.

2. **CLI version:** vastai >= 1.0.12 is interactive on `destroy`. Always pipe `echo y |` for non-interactive destroy in scripts; never embed destroy with setup commands.

3. **Path convention:** repo-relative inside `/workspace/genomic-variant-classifier/`. The launch script anchors via `cd "$REPO_ROOT"`. Symlinks `/workspace/data` and `/workspace/outputs` must be bootstrapped with `rm -rf` first — `ln -s` does NOT replace existing real directories.

4. **2FA:** Vast.ai API keys 2FA toggles must all be OFF on User Read or the CLI returns 401 on every command.

5. **PowerShell → bash heredocs:** Use `@'...'@ | ssh ... bash -s` with `-replace "\`r\`n", "\`n"` to strip CRLF. Never embed bash code as a single quoted string in PowerShell; the inner quotes get stripped.

6. **Long runs (>30 min):** Each base estimator and OOF column must be checkpointed RIGHT AFTER the AUROC log line, not at the end of the pipeline. Run 9's PicklingError lost the entire run; Run 10b's bad destroy paste lost 9/10 models. Per-model checkpoints turn a multi-hour disaster into a survivable rerun.

7. **Irreversible commands isolated:** `vastai destroy`, `rm -rf`, `git push --force`, force-delete — NEVER share a paste block with setup or copy commands. Always a separate code block, after manual verification.

---

## Section E — Open scientific questions

The Run 13 results raise three questions that Run 14 alone does not answer but lays the groundwork for:

1. **Is `n_pathogenic_in_gene` driving the 0.9974 AUROC?** Importance score 3.3× the next feature. If yes, the ensemble is partially a gene-prevalence lookup, not a variant-level pathogenicity model. Resolution: feature permutation ablation, planned for Run 15+.

2. **Why are 30+ of 78 features silently zero?** Connector wiring bugs (ESM-2, EVE missing HGVSp parser; LOVD missing `parquet_path`; etc.). Run 14 measures the dead-feature count as a baseline; Run 15+ fixes them one by one and measures AUROC delta per fix.

3. **Does KAN contribute orthogonal signal?** Run 14 will tell us if KAN's OOF AUROC lands in the [0.996, 0.998] cluster (just one more model in the noise) or beats CatBoost's 0.9975 (genuinely novel nonlinear signal worth investigating).

---

## Section F — Documentation cadence (standing rule)

After every run/session/milestone:

1. **CHANGELOG entry** — `docs/CHANGELOG.md`, append-only, Attempted/Failed/Fixed/Learned format.
2. **Session doc** — `docs/sessions/SESSION_YYYY-MM-DD.md`. Run-by-run narrative with timestamps.
3. **Results doc** — `docs/runs/RUN_NN_RESULTS.md`. Numbers, per-model analysis, hypothesis test results.
4. **Observability JSON + MD** — `outputs/run_NN_report/`. Structured data for programmatic comparison across runs.
5. **Incident doc (if applicable)** — `docs/incidents/INCIDENT_YYYY-MM-DD_slug.md`. Root-cause analysis for any unexpected failure.
6. **Roadmap update** — `docs/ROADMAP.md` (and `.docx` in Google Drive). Captures direction changes and lessons.

All committed and pushed at session close. **No exceptions.**
