# RUN 14 — Plan and Decision Points

**Target commit:** `bf2f665`
**Planned date:** 2026-05-26
**Planner:** Monzia Moodie (with Claude)
**Predecessor:** Run 13 (commit `f4dbeed`), completed 2026-05-26

---

## 1. Why this run exists

Run 14 is the first run where the full **10-base-model ensemble** has any realistic chance of training end-to-end. KAN has failed in every prior run since its introduction:

| Run | Outcome | Root cause |
|----|---|---|
| 10a | 19h pykan runaway | OOM at 17.9 GB on prior hardware |
| 11  | MLP fallback     | `fastkan` not on PyPI |
| 12  | NameError: torch | Missing `import torch` in `_fit_imodelsx` |
| 13  | NameError: test_size | imodelsx v1.0.13 references bare `test_size` |

The fix shipped at `bf2f665` is two-part: (a) a `sed`-based patch in `scripts/launch_run11_vm.sh` rewrites the three bare-name references in the installed imodelsx package to `self.test_size` / `self.random_state` / `self.shuffle`; (b) `kan.py` explicitly sets those three attributes on the `KANClassifier` instance before calling `.fit()`. Both are necessary because imodelsx v1.0.13's `KANClassifier.__init__` never defines the attributes at all — patching the bare references alone leaves an `AttributeError` waiting downstream.

Run 14 validates that this combined fix holds on a real 100K-row KAN training (the smoke test was only 200 rows) and that the meta-learner can integrate a KAN OOF column alongside nine other base models without numerical issues.

---

## 2. Hypotheses (what we're testing)

### H1 — Primary: KAN trains successfully and produces a stable OOF AUROC.
**Pass criterion:** `==> kan OOF AUROC: 0.NNNN` appears in the master log with NNNN ≥ 0.95.
**Why:** Confirms the entire KAN remediation chain (4 bugs across 3 runs) actually works.

### H2 — KAN's OOF AUROC falls in [0.996, 0.998].
**Pass criterion:** The interval matches the cluster of every other base model except logistic regression and CNN_1D.
**Why:** Both literature (Grinsztajn et al. 2022) and our prior LR baseline (0.9942) suggest that on this dataset, 99.4% of discriminative signal is linearly separable; any neural model — KAN included — that exploits feature interactions should land roughly where TabularNN/MC Dropout/Deep Ensemble landed in Runs 11-13.
**Falsification:** If KAN materially beats CatBoost (which holds top spot at 0.9975), we have evidence of a genuine non-linear signal that GBDTs are not capturing. That would be a real scientific finding.

### H3 — Adding KAN does not move the blend AUROC.
**Pass criterion:** Test-set blend AUROC stays at 0.9974 ± 0.0002.
**Why:** Across Runs 9 → 13, every model addition has been within blend noise. We expect this to continue. The value of KAN in this run is benchmarking, not blend lift.
**Falsification:** A blend AUROC of 0.9978+ would indicate KAN contributes orthogonal signal worth keeping. Anything below 0.9974 suggests KAN harms the stacker (over-confidence injection) and meta-learner regularization should be tuned.

### H4 — LightGBM stays in CPU mode at 0.9974 ± 0.0002.
**Pass criterion:** Log line confirms `device_type: cpu` (no CUDA attempt). Test AUROC matches Run 13.
**Why:** Verifies the Run 13 fix has held and the PyPI-binary-CPU-only lesson is stable.

### H5 — Dead-feature count from `run14_observability.py` matches prior estimate (30+ of 78).
**Pass criterion:** Markdown report's "Dead features" section lists between 25 and 35 features with non-zero rate < 0.001.
**Why:** Sets a baseline measurement for the post-Run-14 HGVSp parser + connector audit work. Without this baseline, we cannot quantify improvement from fixing ESM-2, EVE, LOVD, etc.

---

## 3. What we are NOT testing in Run 14

These are explicitly deferred and not part of this run's success criteria:

- HGVSp parser (Run 15+ — fixes ESM-2 and EVE silent-zero)
- LOVD connector parquet_path wiring (Run 15+, partial work in Phase 1.5b)
- HGMD Professional integration (procurement-gated, Run 16+)
- Ablation matrix (Run 15+, after baseline 10-model ensemble is locked)
- Phase 2 REST API and Docker deployment work
- Feature permutation testing for `n_pathogenic_in_gene` leakage

If any of these creeps into Run 14, stop and re-scope.

---

## 4. Instance constraints (standing rules)

- **GPU filter:** `dlperf >= 80, pcie_bw >= 12` (lesson from Run 11: Norway DLP 16 cost +1.5h vs Hungary DLP 97).
- **GPU:** RTX 4090 (sufficient; KAN trains on 100K subsample, not the full 1.2M).
- **Hourly rate ceiling:** $0.80/hr.
- **SSH key:** `C:\Users\monzi\.ssh\id_lambda_run8`.
- **Data path convention:** repo-relative inside `/workspace/genomic-variant-classifier/`. Bootstrap symlinks `rm -rf` first, never trust `ln -s` to overwrite.
- **Destroy:** `echo y | vastai destroy instance ID` (CLI ≥1.0.12 is interactive). Manual paste, separate code block, AFTER local artifact verification.

---

## 5. Expected wall-clock + cost

Runs 12 and 13 averaged 6.4h on Hungary instances. Run 14 adds one more trained model (KAN at ~5-10 min on 100K subsample) so expected:

| Metric | Estimate | Source |
|---|---|---|
| Wall-clock | 6.4 – 6.7 h | Runs 12-13 + KAN delta |
| Cost @ $0.74/hr | $4.75 – $4.95 | Hungary tier pricing |
| Cost @ $0.80/hr | $5.10 – $5.35 | Worst-case price |
| Idle post-completion | < 5 min | If destroy fires immediately |

**Budget approval needed before launch.** Stop here if not comfortable with ~$5 outlay.

---

## 6. Sequence of operations

1. Run `Run14_Preflight.ps1`. Expect exit code 0.
2. Search Vast.ai with the standing filter.
3. Create instance, wait for SSH ready.
4. Bootstrap symlinks on VM (`rm -rf` before `ln -s`).
5. SCP 7 data files into repo-relative paths.
6. Push `run14_observability.py` to VM scripts dir.
7. Launch `scripts/launch_run11_vm.sh` under nohup with master log.
8. Detach. Monitor with `Run14_Monitor.ps1 -Mode Quick` every ~30 min.
9. Spot-check KAN status with `Run14_Monitor.ps1 -Mode KAN` once the run is ≥2h in.
10. When training completes, run `Run14_Postflight.ps1`.
11. Verify local artifacts. Inspect `run14_observability.md`.
12. **In a separate paste block**, run `echo y | vastai destroy instance ID`.
13. Commit: session doc, RUN_14_RESULTS.md, observability report, CHANGELOG.
14. Push to origin/main.

---

## 7. What success looks like

A green run means:
- All 10 base models log an `OOF AUROC` line in `[0.99, 1.00)` (LR will be ~0.994).
- KAN backend confirmed as `imodelsx` (not pykan, not MLP fallback).
- LightGBM device confirmed as `cpu`.
- Test-set blend AUROC ≥ 0.9974.
- No tracebacks in the master log other than the known imodelsx_patch sed line.
- Observability report shows 25-35 dead features (matches prior estimate).
- Instance destroyed within 10 minutes of training completion.
- Session doc + results doc + observability report committed and pushed within 24h.

A useful failure means:
- KAN fails for a NEW reason we haven't seen before (any error other than NameError test_size / NameError torch / fastkan missing).
- We document the new failure mode in detail before relaunching.

---

## 8. Decision tree for in-run anomalies

| Symptom | Decision |
|---|---|
| KAN fails before 1h elapsed | Stop, document, fix locally, push, relaunch on a fresh instance |
| KAN fails after 1h elapsed | Let the run finish without KAN; investigate post-run |
| LightGBM fails | Let the run finish; we have 9-model baseline from Run 12 |
| Test AUROC drops below 0.995 | This is unprecedented; do NOT destroy instance until log is fully extracted |
| Master log grows beyond 100 MB | Possible loop bug; check with Monitor `-Mode Errors` |
| Per-model wall-clock > 2× Run-13 baseline | Possible instance throttling; check `nvidia-smi`, consider re-launching on a different host |

---

## 9. References

- `docs/sessions/SESSION_2026-05-25.md` — Runs 11/12/13 + KAN bug chain
- `scripts/launch_run11_vm.sh` — actual launch entrypoint (still named `_run11_` historically)
- `scripts/run14_observability.py` — VM-side post-run observability collector
- `docs/incidents/INCIDENT_2026-04-17_esm2-hgvsp-parser.md` — ESM-2 silent-zero
- `PHASE_1_ASSESSMENT.md` — bug catalogue from Phase 1 codebase audit
