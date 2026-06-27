# Run 17 Plan

**Status**: DRAFT (created 2026-06-27, FinnGen dual-release build complete + CI-green)
**HEAD at creation**: 70904a6
**Author**: Monzia Moodie

This plan must be fully populated and Charter v1.1 gates G1 + G2 must PASS before any Vast.ai instance create for Run 17.

---

## A. Hypothesis

**H_Run17 (primary)**: The Run 17 baseline reproduces the Run 14/16 ensemble performance on the expanded 91-feature contract (88 + 3 FinnGen R13 columns) without regression, while the newly-wired Run-17 annotation sources (OMIM genemap2, FinnGen R12+R13, and the other CLI-wired connectors) measurably reduce the silent-zero feature count relative to Run 16.

**H_Run17 (dual-release sub-hypothesis — the benchmarking experiment)**: FinnGen R12 and R13, run as two independent annotation passes over the *same* variants with evolved population frequencies, produce *measurably different* feature distributions and per-model feature-importance, demonstrating that the pipeline can ingest and benchmark two releases of the same source apples-to-apples. Specifically: R13 annotates more variants than R12 (higher non-null coverage), and `finngen_r13_*` feature-importance is correlated-but-not-identical to `finngen_*`.

Falsified by: (a) baseline AUROC regression > 0.002 vs Run 16 at matched config, OR (b) R12 and R13 producing statistically indistinguishable feature distributions (which would mean the dual-release adds no benchmarking signal).

This is a PROOF-OF-CONCEPT for the project's SUPPORTING goal (empirically measure/compare/document ML behavior on large complex datasets), not a model-selection exercise. All models stay.

## B. Decisions to lock before launch

### B.O — Open items from Run 16 backlog
- **B.O1** Run-17 annotation wiring (OMIM/PhyloP/dbSNP/EVE/ClinGen CLI flags): the `run_phase2_eval.py` flags + `AnnotationConfig` fields exist (verified this session). Confirm each resolves to a real local file at preflight or is consciously absent (logged, not silently zero).
- **B.O2** HGVSp parser (unlocks ESM-2 + EVE protein-coordinate features): status to confirm at preflight — if delivered, validate against step-10b protein-coordinate coverage gate (min_protein_coord_coverage=0.50); if absent, ESM-2/EVE remain consciously-absent (logged).

### B.D — Data source decisions
- **B.D1** FinnGen R12: **WIRED (resolves RUN_15 B.D2 deferral)**. `data/external/finngen/finnge_R12_annotated_variants_v1.gz` (29.9 GB, registry filename typo 'finnge' intentional, 1017 cols, GENOME_AF_fin + GENOME_AF_nfe). The RUN_15-era R10-vs-R12 schema concern is resolved: `FinnGenConnector` reads the R12 schema and emits `finngen_af_fin/af_nfsee/enrichment`. Launcher hard-fails (exit 7) if the file is missing.
- **B.D2** FinnGen R13: **WIRED (dual-release experiment)**. `data/external/finngen/finngen_R13_annotated_variants_v0.gz` (29.77 GB, correct spelling, 1025 cols, SAME schema as R12). Second independent `FinnGenConnector` pass with `column_prefix="r13_"` → `finngen_r13_af_fin/af_nfsee/enrichment`. Launcher hard-fails (exit 7) if missing.
  - Verified earlier: R12 17085/20000 nonzero (mean AF 0.1025); R13 19318/20000 nonzero (mean AF 0.0971). Same variants, same coords, evolved frequencies — apples-to-apples by construction.
- **B.D3** Feature contract: **91 columns (88 + finngen_r13_af_fin/af_nfsee/enrichment)**. EXPECTED_TABULAR_FEATURE_COUNT=91; INFERENCE_FEATURE_COLUMNS auto-tracks as list(TABULAR_FEATURES). Contract test (test_feature_count_contract.py, 4/4) gates this.
- **B.D4** Harness reference slice: feeds all 6 finngen columns (Option B); KNOWN_ZERO_DEFAULT=25 (finngen AF removed — now actively zero-audited, not allowlisted).

## C. Code changes required (with file paths) — ALL COMPLETE + CI-GREEN

The FinnGen dual-release build is a 5-stage arc, all committed to origin/main (HEAD 70904a6, CI #477 green):
- **C.1** `src/genomic_variant_classifier/data/finngen.py` — `FinnGenConnector` parameterized by `column_prefix` + `finngen_columns()` helper. **closed in ca76482** (Stage 1; test_finngen_release_prefix.py, 9 tests).
- **C.2** `src/genomic_variant_classifier/models/variant_ensemble.py` — TABULAR_FEATURES 88→91 (R13 trio + defaults af=0.0/enrichment=1.0); EXPECTED_TABULAR_FEATURE_COUNT 88→91. **closed in 752335c** (Stage 2; test_feature_count_contract.py 4/4).
- **C.3** `src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py` — build_reference_slice feeds 6 finngen columns; KNOWN_ZERO_DEFAULT 27→25 (Option B). **closed in 1bedf52** (Stage 3; test_correctness_harness + test_harness_fixture_omim_molecular green).
- **C.4** `real_data_prep.py` + `run_phase2_eval.py` + `launch_run17_baseline.sh` — R13 real-data wiring (AnnotationConfig.finngen_r13_path; --finngen-r13-path flag; independent R13 connector pass column_prefix="r13_"; launcher R13 file pick + exit-7 guard). **closed in e284a03 + 70904a6** (Stage 4; test_finngen_r13_wiring.py 4 tests).

## D. Anomalies carried forward (must be addressed or explicitly accepted)

- **A.D1** Stage 3 took a wrong turn through Option A (allowlist 27→29, committed 752335c) before the Option B forward-fix (1bedf52) restored the project's pre-existing decision. The `test_allowlist_unchanged_size` guard caught it as designed. ACCEPTED + documented in commit history; allowlist is now 25 (stronger silent-zero detection).
- **A.D2** The Stage 4 test was committed broken over a red suite (e284a03) and forward-fixed (70904a6). The R13 wiring itself was always correct; only the test was broken. ACCEPTED + documented.
- **A.D3** Two independent ~30GB FinnGen gzip passes (R12 + R13) = ~60GB SCP up + two compute-bound annotation passes on the VM. ACCEPTED as the dual-release experiment's cost (see E).
- **A.D4** `data_freshness.yml` scipy fix (28e11cc) is in the tree; confirm its own CI run is green (it predates the Option B push).

## E. Wall-clock + cost budget

- **SCP up**: both finngen files ~60GB total (R12 29.9GB + R13 29.77GB) over the id_lambda_run8 key. Non-trivial — confirm budget acceptance before launch.
- **VM annotation**: two ~30GB gzip passes, bounding-box-filtered so RAM stays bounded by the matched subset, but two full decompression passes. Flag against MIN_RAM_GB=50.
- **Instance**: Vast.ai RTX 4090, target $0.55/hr (cap $0.77/hr), filter `dlperf>=80 pcie_bw>=12`. Offer selection via scripts/select_vast_offer.py.
- **Estimate**: state explicit wall-clock estimate + dominant cost driver at preflight; confirm Monzia accepts before any instance create (standing law for runs >15min).

## F. Pre-launch gates (Charter v1.1)

- **G1 (local)**: scripts/Run_Preflight_Local.ps1 adapted Run-15→Run-17 — §3 DELETE (imodelsx patch moved to kan.py L181); §6 test floor 566→~1483; §7 rebuild data list (FinnGen NOW BOTH files local, hard-fail); §11 repoint launchPath→launch_run17_baseline.sh; §12/13 create RUN_17 postflight; ADD agent-liveness via scripts/check_agents_active.py. Reference slice (build_reference_slice, now feeding finngen) is the G1 single-source-of-truth.
- **G2 (VM)**: scripts/Run_Preflight_VM.sh Run-17-aware (MIN_VRAM_MIB=20000, MIN_DISK_GB=150, MIN_RAM_GB=50).
- **ALL-MODELS smoke** (scripts/smoke_all_models.py): tiny --max-train ~3000, NO --skip flags, --string-db auto; fail+block if any model errors/skips/degenerate-OOF/Traceback. Run BEFORE the full run.
- **Agent liveness** (scripts/check_agents_active.py): all 21 agents registered + scheduled; ProvisioningAgent both registered AND in a pipeline. Run at session start + preflight before+after launch.

## G. Run abort criteria

- Any base estimator errors / skips / produces degenerate OOF in the ALL-MODELS smoke → ABORT, fix, re-smoke.
- Checkpoint each base estimator + OOF right after its AUROC log; if any single estimator exceeds ~30min wall-clock → ABORT (compute-bound runaway).
- FinnGen R12 OR R13 file missing at launch → launcher exits 7 (hard-fail, no silent zero-annotation).
- Baseline AUROC regression > 0.002 vs Run 16 at matched config → HALT, investigate before continuing.

## H. R12-vs-R13 comparison protocol (the dual-release deliverable)

The experiment's interpretable output. Computed post-run, per-release × per-model.

### H.1 Per-model × per-release metrics
For each of the 11 models (CatBoost, XGBoost, LightGBM, RF, GBM, LR, 1D-CNN, TabularNN, MC-Dropout, Deep Ensemble, KAN) and the stacked blend, report under three feature configs — {R12-only, R13-only, both} — to isolate each release's contribution:
- **AUROC** (area under ROC; [0,1], 0.5=chance): primary discrimination metric.
- **AUPRC** (area under precision-recall; [0,1], baseline=prevalence): discrimination under class imbalance.
- **F1** (harmonic mean precision/recall; [0,1]): threshold-dependent balance.
- **MCC** (Matthews correlation; [-1,1], 0=chance): balanced single-number quality.
- **Brier** (mean squared error of probabilities; [0,1], lower=better): calibration + sharpness.
- **Calibration** (reliability curve + ECE): probability trustworthiness.
- **OOF AUROC** (out-of-fold; the leak-free generalization estimate): primary anti-overfit metric.

### H.2 FinnGen-specific deltas (R12 vs R13)
- **Coverage delta**: fraction of variants annotated non-null per release (observed: R12 17085/20000 = 85.4%; R13 19318/20000 = 96.6%). Quantify on the full cohort.
- **AF-shift**: distribution of (finngen_r13_af_fin − finngen_af_fin) over jointly-annotated variants (observed sample means: R12 0.1025, R13 0.0971). Report mean/median/IQR + a paired test (Wilcoxon signed-rank) for whether the shift is systematic.
- **Feature-importance delta**: rank of finngen_* vs finngen_r13_* in each model's importance (permutation importance + native where available). Report Spearman correlation between the two rankings (apples-to-apples expectation: high but <1.0).
- **Enrichment comparison**: finngen_enrichment vs finngen_r13_enrichment distribution (both default 1.0 when absent; compare where present).

### H.3 Living-metrics glossary entries (FinnGen-specific, per standing documentation law)
- **Coverage (FinnGen)**: non-null annotation fraction = (variants with finngen AF) / (total variants). Range [0,1]. Why: measures how much of the cohort a release informs; the direct R12-vs-R13 benchmarking signal. Varied per run: compared R12 vs R13 head-to-head.
- **AF-shift**: per-variant (R13 AF − R12 AF). Range [-1,1]. Why: captures frequency drift between releases on identical variants; the substance of "evolved frequencies." Varied: paired distribution + signed-rank p-value.
- **Feature-importance rank correlation (Spearman)**: ρ between finngen_* and finngen_r13_* importance vectors across models. Range [-1,1]. Why: tests the apples-to-apples expectation (releases should agree on which finngen signal matters, but not perfectly). Varied: per-model ρ + aggregate.

## I. Decision log

- **2026-06-27**: Dual-release (R12 + R13) chosen as the Run-17 benchmarking experiment. Build completed as 5 stages (ca76482 → 70904a6), all CI-green at #477. Feature contract 88→91. Harness Option B (feed finngen, allowlist 27→25). R13 wired as independent connector pass (column_prefix="r13_") reading a separate ~30GB file.
- **2026-06-27**: Option A (allowlist R13) tried + reverted to Option B (feed fixture) per the pre-existing test_allowlist_unchanged_size guard. Documented in A.D1.

## J. References

- RUN_15_PLAN.md B.D2 (the FinnGen deferral this run resolves)
- docs/CHANGELOG.md (the ca76482→70904a6 commit arc)
- scripts/launch_run17_baseline.sh (the launcher with both finngen guards)
- tests/unit/test_finngen_r13_wiring.py, test_feature_count_contract.py, test_correctness_harness.py (the gates)
