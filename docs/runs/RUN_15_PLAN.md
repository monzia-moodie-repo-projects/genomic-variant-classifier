# Run 15 Plan

**Status**: DRAFT (created at Run 14 close-out, 2026-05-26)
**HEAD at creation**: dc95dab
**Author**: Monzia Moodie

This plan must be fully populated and Charter v1.1 gates G1 + G2 must PASS before any Vast.ai instance create for Run 15.

---

## A. Hypothesis

**H_Run15**: <DECISION>
(Examples: "Closing 5+ silent-zero feature gaps will lift OOF blend AUROC above 0.9990, without inflating gene-prevalence memorization." OR "KAN at full 814K training tractor adds genuine diversity beyond catboost.")

## B. Decisions to lock before launch

### B.O — Open Items From Run 14 Backlog
- **B.O1** KAN status: <DECISION: scale subsample 100K→250K-500K | drop from base learners | keep at 100K>
  - Justification: Run 14 OOF→test gap was 0.0025 (~3.5× catboost's gap), indicating 100K subsample overfits.

- **B.O2** A1 (np.log(0) at mc_dropout.py:87): **CLOSED — verified false anomaly (2026-05-26)**. Line 86 already does `clipped = np.clip(probs_stack, 1e-8, 1.0 - 1e-8)`, so line 87 never sees log(0). At the worst-case boundary, log(1e-8) ≈ -18.42 and 1e-8 * log(1e-8) ≈ -1.84e-7 are finite. Behaviour locked by tests/unit/test_mc_dropout_uncertainty.py (7 cases, all green). Probe in scripts/probe_a1_boundary.py confirms with the real function. No source code change required.

- **B.O3** A2 fix (mc_dropout uncertainty degenerate, missing _predict_proba_single_pass): <DECISION>

- **B.O4** A7 fix (observability per_model dict empty): **CLOSED — rewrite to read structured files (2026-05-27, da41f27)**. Run 14 master log emits per-model metrics via Python's logging module without the "==>" prefix the regex required. Fix: new `read_per_model_metrics_files()` reads `per_model_metrics.csv` + `per_model_metrics_val.csv` + `models/*_meta.json`; log-grep retained as fallback. Verified locally: `per_model_source: structured`, catboost OOF=0.9982 TEST=0.9975 VAL=0.9975 matches CHANGELOG Run 14. Regression test: `tests/unit/test_run14_observability.py` (7 cases).

### B.D — Data Source Decisions
- **B.D1** 1KGP AF parquet build: <DECISION: build before Run 15 | defer to Run 16>
  - Unlocks 5 dead features (af_1kg_{afr,eur,eas,sas,amr}).

- **B.D2** FinnGen TSV transfer: <DECISION: transfer + integrate | defer>
  - Unlocks 3 dead features (finngen_af_fin, finngen_af_nfsee, finngen_enrichment).

- **B.D3** STRING parquet index build: <DECISION: build | defer>
  - Unlocks gnn_score (1 dead feature) and full GNN training path.

- **B.D4** HGVSp parser: <DECISION: build | defer>
  - Unlocks ESM-2 + EVE (2 dead features).

- **B.D5** PrimateAI-3D: <DECISION: license-review | defer | drop>

- **B.D6** CNN-fasta input: <DECISION: provide | use --skip-cnn>

## C. Code Changes Required (with file paths)

- **C.1** mc_dropout.py:87 np.log(0) clip: <DECISION: yes | no>
- **C.2** TabularNNClassifier._predict_proba_single_pass(): <DECISION: implement | drop MC-dropout from base learners>
- **C.3** scripts/run14_observability.py per_model parser: **rewrite — closed in da41f27** (read structured files preferentially; log-grep fallback with relaxed regex)
- **C.4** scripts/launch_run11_vm.sh imodelsx_patch dedupe (A3): <DECISION>
- **C.5** Run15_Postflight.ps1 uses Test-ArtifactPresent for ALL gate checks: <DECISION: required | skip>
- **C.6** Run15_Postflight.ps1 must `exit 1` on any FAIL (SR #39): <DECISION>
- **C.7** Separate Vastai_Destroy_Confirmed.ps1 requiring gate exit 0, refusing `echo y |` (SR #38): <DECISION>

## D. Anomalies Carried Forward (must be addressed or explicitly accepted)

- A1 np.log(0) in mc_dropout.py:87  → see B.O2/C.1
- A2 mc_dropout uncertainty degenerate → see B.O3/C.2
- A3 imodelsx_patch echo dup in launch script → see C.4
- A4 KAN trained on 12% subsample → see B.O1
- A5 score annotation step numbering inconsistency → cosmetic, defer
- A6 5 silent-zero data sources → see B.D1-B.D6
- A7 observability per_model dict empty → CLOSED at da41f27 (rewrite to read structured files; see B.O4/C.3)
- A8 postflight gate path assumption → CLOSED at dc95dab (Test-ArtifactPresent helper); follow-up wiring in C.5

## E. Wall-clock + cost budget

- Estimated GPU hours: <DECISION>
- Estimated cost USD: <DECISION>
- Hard ceiling: <DECISION>

## F. Pre-launch gates (Charter v1.1)

- [ ] G1 (local): run `scripts/Run_Preflight_Local.ps1` — must PASS all checks
- [ ] G2 (VM): run `scripts/Run_Preflight_VM.ps1` on instance — must PASS all checks
- [ ] Working tree clean, HEAD pushed to origin/main
- [ ] All B.O* and C.* decisions filled (no `<DECISION>` placeholders)
- [ ] All A1-A7 either fixed in code commits OR explicitly accepted in this plan
- [ ] Test-ArtifactPresent.ps1 dot-sourced from postflight script (verified by `grep`)
- [ ] Run15_Postflight.ps1 verified to `exit 1` on FAIL
- [ ] Vastai_Destroy_Confirmed.ps1 in place, requires gate exit 0

## G. Run abort criteria

- Cost exceeds budget by >25% → abort
- Wall-clock exceeds budget by >50% → abort
- Any base learner fails to fit → abort
- Disk usage on VM > 90% → abort
- gate G2 FAIL → abort before any training launch

---

## Decision log

(Append decisions here as they are made, with date + rationale.)

- **2026-05-27 — A7 (B.O4 / C.3) decision: rewrite** (commit `da41f27`). Rationale: the training script already writes structured per-model metrics to `per_model_metrics.csv`, `per_model_metrics_val.csv`, and `models/*_meta.json`; log-grep was over-fitted to the shell launch script's "==>" echo style and never matched the Python-logger output. Rewrite reads those structured sources first; log-grep regex relaxed (drop "==>" prefix requirement) as fallback. Closes A7 with no regression to other observability outputs (feature_nonzero, KAN status, LightGBM status, artifact inventory all unchanged). Regression test: `tests/unit/test_run14_observability.py` (7 cases, all passing).