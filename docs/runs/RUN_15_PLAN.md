# Run 15 Plan

**Status**: DRAFT (created at Run 14 close-out, 2026-05-26)
**HEAD at creation**: dc95dab
**Author**: Monzia Moodie

This plan must be fully populated and Charter v1.1 gates G1 + G2 must PASS before any Vast.ai instance create for Run 15.

---

## A. Hypothesis

**H_Run15**: **DECIDED 2026-05-27 — Option C3 hybrid**. Run 15's enhancements (KAN-250K + GNN-enabled + real MC-dropout) BOTH (a) narrow OOF→test gap to ≤ 0.0010 AND (b) maintain unseen_gene_holdout AUROC ≥ 0.95 — improvements reflect genuine generalization, not memorization.
(Falsified by either: (a) OOF→test gap > 0.0010 OR (b) unseen_gene_holdout AUROC < 0.95. Selected from 4 candidates 2026-05-27 PM9; see Decision log for rationale and trade-offs vs C1/C2/C4.)

## B. Decisions to lock before launch

### B.O — Open Items From Run 14 Backlog
- **B.O1** KAN status: **DECIDED 2026-05-27 — 250K subsample for Run 15 (Option A1); 500K reserved for Run 16 (Option A2) if Run 15 OOF→test gap remains >0.001**
  - Justification: Run 14 OOF→test gap was 0.0025 (~3.5× catboost's gap), indicating 100K subsample overfits.

- **B.O2** A1 (np.log(0) at mc_dropout.py:87): **CLOSED — verified false anomaly (2026-05-26)**. Line 86 already does `clipped = np.clip(probs_stack, 1e-8, 1.0 - 1e-8)`, so line 87 never sees log(0). At the worst-case boundary, log(1e-8) ≈ -18.42 and 1e-8 * log(1e-8) ≈ -1.84e-7 are finite. Behaviour locked by tests/unit/test_mc_dropout_uncertainty.py (7 cases, all green). Probe in scripts/probe_a1_boundary.py confirms with the real function. No source code change required.

- **B.O3** A2 fix (mc_dropout uncertainty degenerate, missing _predict_proba_single_pass): **CLOSED — implement _predict_proba_single_pass on TabularNNClassifier (2026-05-27, c60e842)**. Added selective dropout-activation method between predict_proba (L874) and predict (L884); `model.eval()` puts whole network in inference mode (running-stats BatchNorm), then loop `model.modules()` and selectively `.train()` only `nn.Dropout` instances. `try/finally` ensures `.eval()` restoration so subsequent predict_proba calls aren't left dropout-active. Regression suite: `tests/unit/test_tabular_nn_mc_dropout.py` (15 tests across 5 classes — Contract, Stochasticity, SideEffects, Integration, ScientificProperties; 14 pass + 1 skip on sparse-corpus calibration test). Integration stubs: `tests/integration/test_mc_dropout_calibration.py` (5 skipped tests for post-Run-15 OOD / Spearman correlation / ECE / MC convergence). MCDropoutWrapper L216 hasattr check now succeeds → real epistemic + aleatoric uncertainty in Run 15.

- **B.O4** A7 fix (observability per_model dict empty): **CLOSED — rewrite to read structured files (2026-05-27, da41f27)**. Run 14 master log emits per-model metrics via Python's logging module without the "==>" prefix the regex required. Fix: new `read_per_model_metrics_files()` reads `per_model_metrics.csv` + `per_model_metrics_val.csv` + `models/*_meta.json`; log-grep retained as fallback. Verified locally: `per_model_source: structured`, catboost OOF=0.9982 TEST=0.9975 VAL=0.9975 matches CHANGELOG Run 14. Regression test: `tests/unit/test_run14_observability.py` (7 cases).

### B.D — Data Source Decisions
- **B.D1** 1KGP AF parquet build: **DECISION: defer to Run 16 (2026-05-27)** — `thousandgenomes.py` (6.7 KB) is a single-column gnomAD AF fallback connector (outputs `allele_freq` for variants where gnomAD AF is null), NOT a per-population feature split. Stub mode (L26-28 of connector) handles parquet absence gracefully; Runs 11-14 all completed without it. Deferring to Run 16 prioritises launch over marginal AF coverage.
  - Coverage: connector (thousandgenomes.py) is a gnomAD AF fallback that outputs a single `allele_freq` column. The original "5 features" sub-bullet was based on a misreading of the connector's scope — corrected 2026-05-27.

- **B.D2** FinnGen TSV transfer: **DECISION: defer (2026-05-27)** — 30.6 GB on disk at `data/external/finngen/finnge_R12_annotated_variants_v1.gz` but two blockers: (1) filename has typo "finnge" vs connector-expected "finngen"; (2) file is FinnGen R12, but `finngen.py` was written for R10 schema (expected columns: `#chrom, pos, ref, alt, af_fin, af_nfsee, rsid`). R12 column compatibility must be verified before integration. Deferring to dedicated post-Run-15 task.
  - Unlocks 3 dead features (finngen_af_fin, finngen_af_nfsee, finngen_enrichment).

- **B.D3** STRING parquet index build: **DECISION: build (enable GNN path) (2026-05-27)** — all cache files present: `data/raw/cache/string_links.parquet` (13,715,404 edges, 92.5 MB), `data/raw/cache/string_names.parquet` (19,699 nodes, 3.1 MB), `data/raw/cache/string_graph_700.pkl` (17.2 MB). gnn.py defensively handles missing gene_symbol (L640-644). Run 9 GNN-FREE root cause was pipeline-side `gnn_df` overwrite (`X_train_raw = pd.read_parquet(...)` clobbers gnn_df with 78-col matrix lacking gene_symbol), NOT gnn.py code. Pipeline-side fix needed in build_pyg_dataset caller — separate code change scoped under Run 15 pre-flight, not B.D3 itself.
  - Unlocks gnn_score (1 dead feature) and full GNN training path.

- **B.D4** HGVSp parser: **DECISION: defer (2026-05-27)** — no parser exists on disk (`src/genomic_variant_classifier/data/hgvsp_parser.py` absent). Substantial new code: parser + connector wiring (esm2.py, eve.py both expect `protein_pos`, `wt_aa`, `mut_aa` columns) + test suite. INCIDENT_2026-04-17 open since April 17. Not a Run 15 quick-win. Defer to Run 16+.
  - Unlocks ESM-2 + EVE (2 dead features).

- **B.D5** PrimateAI-3D: **DECISION: defer (license-review subtrack continues) (2026-05-27)** — primateai3d.py exists (7.3 KB) with `PHASE_2_PLACEHOLDER` comment (L26-28) and explicit `(must match TABULAR_FEATURES when wired)` note (L41). Connector NOT yet wired into feature matrix. License unresolved per Run 11 procurement notes. 4 GB data file (`primateai3d_scores_hg38.tsv.gz`) not on disk. License-review continues as separate track.

- **B.D6** cnn_1d fasta input misconception: **DECISION: use --skip-cnn for Run 15 (2026-05-27)** — per INCIDENT_2026-05-23, `cnn_1d` is a 1-D CNN over the 78-dim tabular feature vector (input shape `(78, 1)`), NOT a sequence model. The plan's "CNN-fasta input" framing was based on a misconception. The 0.5 AUROC regression in Run 10a is from closure breakage in `_CNN1DWrapper._build_model` after post-C5 refactor (commit ac64665), per INCIDENT_2026-05-24. Closure refactor is a separate code change. Run 14 already used --skip-cnn.

## C. Code Changes Required (with file paths)

- **C.1** mc_dropout.py:87 np.log(0) clip: **DECLINED — line structurally safe via L86 (2026-05-27)**. L86 `clipped = np.clip(probs_stack, 1e-8, 1.0 - 1e-8)` assigns to a variable; L87 uses `clipped` (NOT raw `probs_stack`) in both `np.log(clipped)` and `np.log(1 - clipped)`. Mathematical guarantee: all values in `[1e-8, 1-1e-8]` produce finite logs (`log(1e-8) ≈ -18.42`; `log(1-1e-8) ≈ -1e-8`). 4 production runs (11/12/13/14 — mc_dropout OOF AUROC 0.9971/0.9971/0.9971/~0.9968) + 7-test boundary suite (`tests/unit/test_mc_dropout_uncertainty.py`) + `scripts/probe_a1_boundary.py` confirm safety empirically. Adding a 2nd clip at L87 would clip already-clipped values to the same bounds — pure no-op. Per standing rule 'redundant defensive code is anti-pattern when tests exist', declined.
- **C.2** TabularNNClassifier._predict_proba_single_pass(): **implement — closed in c60e842** (selective dropout activation; BatchNorm preserved in eval; try/finally state restoration; 15 unit tests + 5 integration stubs). Closes A2/B.O3 in same commit.
- **C.3** scripts/run14_observability.py per_model parser: **rewrite — closed in da41f27** (read structured files preferentially; log-grep fallback with relaxed regex)
- **C.4** scripts/launch_run11_vm.sh imodelsx_patch dedupe (A3): **closed in 9628463** (removed redundant outer tee on L200; inner echoes at L197/L199 preserved). Empirical evidence (verified 2026-05-27 against `outputs/run14/run14_master.log`): the success-branch message `imodelsx_patch: fixed 3 bare-name refs` appears **2 times** in the Run 14 master log despite the branch firing only once, confirming the double-tee bug structurally and empirically.
- **C.5** Run15_Postflight.ps1 uses Test-ArtifactPresent for ALL gate checks: **required — closed in f7febbb** (Section 5 wires 7 gates: master_log, observability_md, observability_json, per_model_metrics_csv, ensemble_joblib (≥1MB), ensemble_manifest, blend_weights). Run 14 dot-sourced Test-ArtifactPresent but never called it; that wiring gap is the actual content of C.5.
- **C.6** Run15_Postflight.ps1 must `exit 1` on any FAIL (SR #39): **closed in f7febbb** (5 exit-1 paths: training-incomplete abort, obs script missing, SCP obs script fail, SCP report fail, gate FAIL block). PowerShell parser self-test PASS. Empirically verified post-commit: exit 1 total = 5 (3 line-starting + 2 inline).
- **C.7** Separate Vastai_Destroy_Confirmed.ps1 requiring gate exit 0, refusing `echo y |` (SR #38): **closed in 6107e56** (4 defense layers: stdin-redirected refusal via `[Console]::IsInputRedirected` → exit 2; gate file missing → exit 3; gate content != "0" → exit 4; interactive case-sensitive `Read-Host -cne "DESTROY"` confirmation. Exit 5 if vastai CLI fails. Implements the SR documented in `docs/sessions/SESSION_2026-05-26.md` L143).

## D. Anomalies Carried Forward (must be addressed or explicitly accepted)

- A1 np.log(0) in mc_dropout.py:87  → see B.O2/C.1
- A2 mc_dropout uncertainty degenerate → see B.O3/C.2
- A3 imodelsx_patch echo dup in launch script → CLOSED at 9628463 (redundant outer tee on L200; see C.4)
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

- **2026-05-27 — A3 (C.4) decision: removed redundant outer tee** (commit `9628463`). Rationale: `scripts/launch_run11_vm.sh` L200 had `fi | tee -a "$LOG"` after an if/else where each inner echo (L197 success branch, L199 else branch) already piped to `tee -a "$LOG"`. Effect: each imodelsx_patch status line logged to `run11_master.log` twice (empirically confirmed: success-branch message appears 2x in Run 14 log; else-branch message appears 0x). Fix: replace L200 outer-tee with bare `fi` plus forensic comment `# A3 fix 2026-05-27: removed redundant outer tee`. Inner echoes preserved; outer-else WARN echo at L202 preserved. Verified by 5/5 PS sanity checks (verbatim source substrings) + 9 internal patcher post-conditions + PS-level anchor uniqueness pre-check (defense in depth proven during session 2 re-paste: idempotency check fired correctly and patcher refused to re-apply).

- **2026-05-27 — C.5+C.6+C.7 decision: postflight + destroy infrastructure** (commits `f7febbb` + `6107e56`). Rationale: closes the Run 14 procedural-failure class (A8 root cause). Run14_Postflight.ps1 had Test-ArtifactPresent dot-sourced but never called; Run 14 also did not SCP `models/` (silent oversight — the directory containing `ensemble.joblib`); destroy was inlined in postflight output for manual paste, allowing the INCIDENT_2026-05-24 failure where a destroy command shared a paste block with SCP setup. Implemented: (a) `scripts/Run15_Postflight.ps1` (194 lines) wires 7 Test-ArtifactPresent gates, writes `.gate_exit_code` to the report dir, SCPs `models/` and `per_model_metrics*.csv`, and prints only a pointer to `Vastai_Destroy_Confirmed.ps1` instead of an inline destroy command. (b) `scripts/Vastai_Destroy_Confirmed.ps1` (114 lines) refuses execution unless 4 independent defense layers pass. Both scripts verified by PowerShell parser self-test (`[System.Management.Automation.Language.Parser]::ParseFile()`) and sanity-check suites (12 + 12 = 24 verbatim-source markers, all PASS). A3-class procedural failure (PS-throw inside `& { }` doesn't halt paste) was recurrence-checked: f7febbb's paste exhibited the bug benignly (false-positive sanity checks let commit proceed); 6107e56's paste used `try { ... } catch { ... return }` at top scope to definitively halt on throw — the fix is now proven in production usage.
- 2026-05-27: **D17 CLOSED** in 486c680 — `scripts/run15_observability.py` + `tests/unit/test_run15_observability.py` (byte-level clone of run14 variants via Python patcher; 14/14 pytest PASS in 7.67s). **D16 audit finding**: already CLOSED in bd75ed5 (2026-05-11) per today's audit; prior CHANGELOG mentions on L54/L98 of d8baaa9 are stale as a result but kept per append-only convention. Run 15 launch infrastructure now complete: Run15_Postflight.ps1 (f7febbb) + Vastai_Destroy_Confirmed.ps1 (6107e56) + scripts/run15_observability.py (486c680). All Phase C code-level items closed.
- 2026-05-27 PM5 — A4/B.O1 KAN: 250K subsample for Run 15 (Option A1); 500K reserved for Run 16 (Option A2) if Run 15 OOF→test gap remains >0.001. Staged scaling test of the 100K-overfit hypothesis (Run 14 gap 0.0025).
- **2026-05-27 PM6 — A2 (B.O3 / C.2) decision: implement** (commit `c60e842`). Rationale: per the standing rule "Monzia drives scope. NEVER propose dropping models/techniques/features", the choice between `implement` and `drop MC-dropout from base learners` was always going to be implement. TabularNNClassifier had `nn.Dropout(0.3)` layers in its architecture (L817) but no `_predict_proba_single_pass()` method, so MCDropoutWrapper L216 hasattr check failed → degenerate fallback path returning zero epistemic + aleatoric. Fix: selective dropout activation — `model.eval()` then loop `modules()` to `.train()` only `nn.Dropout` instances; BatchNorm stays in eval to preserve running-stats inference. Audit-driven implementation: Step 0 verification probe caught CRLF line-ending blocker before any mutation; Step 1 paste audit caught caplog-scoping false-negative risk in `test_no_missing_method_warning`; revised paste landed cleanly on first attempt with 14/15 unit tests PASS + 5/5 integration stubs collected.
- **2026-05-27 PM7 — C.1 decision: no** (docs-only). Rationale: B.O2 closure (2026-05-26) verified the line is safe; live probe of `mc_dropout.py` L82-88 (2026-05-27, CRLF: False, 313 lines, 11694 bytes) confirms current code matches B.O2 description. L86 `clipped = np.clip(probs_stack, 1e-8, 1.0 - 1e-8)` assigns to `clipped`; L87 uses `clipped` (NOT raw `probs_stack`) in both `np.log` calls — boundary safety is structurally enforced via variable reuse, not side-clipped. Mathematical guarantee: `clipped ∈ [1e-8, 1-1e-8]` ⇒ both `log(clipped)` and `log(1-clipped)` finite. Adding a 2nd clip at L87 would clip already-clipped values to the same bounds — pure no-op. Empirical: 4 production runs (11/12/13/14, mc_dropout OOF AUROC 0.9971/0.9971/0.9971/~0.9968) + 7-test boundary suite + `scripts/probe_a1_boundary.py` (2956 bytes, present locally) confirm safety. Per standing rule 'redundant defensive code is anti-pattern when tests exist', declined. C.1 probe Phase 4 aborted on PS automatic-variable `$matches` collision; memory rule #21 updated with auto-var blocklist. First C.1 implementation paste also failed with exit 11 (marker mismatch + non-atomic writes); memory rule #28 expanded with items (14) atomicity and (15) cross-file marker presence requirements.
- **2026-05-27 PM8 — B.D batch decisions: 6 resolved + 3 plan factual corrections** (docs-only). B.D1 defer (single allele_freq column, not 5 features); B.D2 defer (filename typo + R12/R10 schema mismatch on 30.6 GB file); B.D3 build (STRING fully cached: 13.7M edges, 19.7K nodes, 17.2 MB graph pickle); B.D4 defer (no parser, new code work); B.D5 defer (license + wiring both pending); B.D6 --skip-cnn (cnn_1d is tabular CNN not sequence CNN; closure bug from post-C5 ac64665 refactor). Plan factual corrections: B.D1 sub-bullet rewritten (5 features → 1); B.D6 heading clarified ("CNN-fasta input" was misnomer); B.D2 expanded with file-mismatch detail. Probe lessons captured: (1) guessed connector paths missed `thousandgenomes.py` and `primateai3d.py` — directory listing must inform per-item probe paths; (2) PowerShell `-and` binds tighter than `-or` (memory #21.12 updated earlier today caught this); (3) section-finder loops require single-purpose iterations to avoid precedence traps; (4) NEW: docs that discuss a validation marker must not contain the marker literal — PM8 first attempt embedded the literal placeholder token in a meta-mention, breaking the delta count by 1 and forcing a re-run with the literal removed. Plan placeholder count: 11 → 5 (10 actual decisions → 4 actual decisions remaining: H_Run15 L13, E budget L68-70 ×3).
- **2026-05-27 PM9 — H_Run15 decision: Option C3 hybrid** (docs-only). Run 15's enhancements (KAN-250K per B.O1 PM5; GNN-enabled per B.D3 PM8 pending pipeline-side gene_symbol fix in build_pyg_dataset caller; real MC-dropout per B.O3 PM6 c60e842) tested by conjunctive criterion: BOTH (a) OOF→test AUROC gap ≤ 0.0010 (vs Run 14 baseline 0.0025; threshold encoded in B.O1 PM5) AND (b) unseen_gene_holdout AUROC ≥ 0.95 (tests project's central scientific concern re gene-prevalence memorization given n_pathogenic_in_gene importance 3.3× next feature). Falsified by either criterion failing. Selected over: C1 (gene-memo only, no gap criterion), C2 (gap only, no memorization test), C4 (orthogonality, supporting-goal not primary). Plan placeholder count: 5 → 4 (L13 closed; L68/L69/L70 remaining for E budget; L77 backtick-wrapped doc-pattern). E budget L68/L69/L70 derives next: requires full 13-ablation matrix incl. unseen_gene_holdout.
