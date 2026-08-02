# Run 17 Plan

**Status**: DRAFT (created 2026-06-27, FinnGen dual-release build complete + CI-green)
**HEAD at creation**: 70904a6
**Author**: Monzia Moodie

This plan must be fully populated and Charter v1.1 gates G1 + G2 must PASS before any Vast.ai instance create for Run 17.

---

<!-- FEATURE_CONTRACT: 95 -->
<!--
  ^ THE MACHINE-READABLE CONTRACT ASSERTION. G1 §13c reads this marker and hard-fails if it
  disagrees with EXPECTED_TABULAR_FEATURE_COUNT in the package. It is the single authority;
  the prose below is for humans.

  It exists because scraping the prose did not work, twice:
    * The corrected hypothesis was first written `**97**-feature`. The check matched a digit
      followed by '-' or ' ', so the '*' hid the 97 entirely and the only number it could see
      was the stale one quoted in a footnote.
    * Stripping markdown to fix that then collapsed `finngen_r13_*` feature-importance into
      "finngenr13 feature-importance" -- and the guard read it as a claim of THIRTEEN features,
      failing a correct document.

  A contract must be ASSERTED, not INFERRED from narrative text. Update this number and the
  prose together; G1 will not let them disagree with the code.
-->

## A0. MODEL CHANGE SINCE RUN 15 — read before comparing any per-model number

**2026-07-12: `logistic_regression` is now SCALED. Its Run-17 numbers are NOT comparable to
Run 15's for that model.** This is a **correction of a defect**, so any change is an
improvement, not a regression — but it must not be read as an effect of the Run-17 feature
expansion.

**The defect.** `logistic_regression` was a bare `LogisticRegression` in the base-model roster,
and `VariantEnsemble.fit` dispatches it to `X_tab.values` — the **raw** tabular matrix, where
`pos` runs to 1,000,000 alongside `allele_freq` at 1e-6. It **did not converge**, and said so in
every test run and every Continuous Integration run for weeks, on Python 3.11 and 3.12 alike:

```
ConvergenceWarning: lbfgs failed to converge after 1000 iteration(s)
```

A non-converged logistic regression was being fit, and its out-of-fold predictions fed the
stacking meta-learner.

**It was the only unprotected model.** Audited 2026-07-12 across the full roster:

| model | scaled? |
|---|---|
| random_forest, xgboost, lightgbm, gradient_boosting, catboost | scale-invariant (trees) |
| svm, svm_bagged_rbf | `ScalableSVM` → `make_pipeline(StandardScaler, …)` |
| tabular_nn | own `StandardScaler` **and** `BatchNorm1d` |
| mc_dropout, deep_ensemble | wrap `TabularNNClassifier`, inherit its scaler |
| kan | own `StandardScaler` |
| cnn_1d | consumes the **one-hot DNA sequence** (values in {0,1}) — correctly bare; scaling it would destroy the encoding |
| **logistic_regression** | **nothing — the gap** |

Every other scale-sensitive model already had a scaler. This was an oversight, not a design
choice.

**Why it matters for THIS run's hypothesis.** A stated first-class goal of the project is to
*"empirically measure/compare/validate ML algorithms … even at small performance differences."*
Comparing `logistic_regression` against XGBoost while it **alone** was handicapped by unscaled
inputs measured the **defect**, not the algorithm. Any linear-versus-tree conclusion drawn from
Run 15 or earlier is **confounded**. Run 17 is the first run in which that comparison is sound.

**Also fixed (same day):** the correctness harness's own stage-3 sanity model
(`correctness_harness.py`) was a second, separate unscaled `LogisticRegression(max_iter=200)`
that likewise never converged. Stage 3 asserts *"the pipeline can learn a signal"* — asserting
that with an unconverged optimiser is asserting it on unsound evidence, since an unconverged fit
can produce near-constant probabilities for reasons unrelated to the data, which is precisely
the condition stage 3 tests for. Both are now scaled; the harness runs warning-free.

Guarded by `tests/unit/test_logistic_regression_is_scaled.py`, which **fails on a
`ConvergenceWarning`** rather than printing one — and which also asserts that `cnn_1d` must
**not** be scaled, so the next reader does not "fix" the one model that is correctly bare.

## A. Hypothesis

> **CONTRACT CORRECTED 2026-07-12: ninety-one → ninety-seven.** This hypothesis previously
> stated the contract as ninety-one columns (eighty-eight plus three FinnGen R13). That was true
> when written (2026-06-27, `fbdcf4c`). On **2026-07-06** (`80eb9c8`) KEGG (×2), COSMIC (×2) and
> the Nucleotide Transformer (×2) landed, taking `EXPECTED_TABULAR_FEATURE_COUNT` to
> ninety-seven. The *runbook* was corrected (`61c2b04`); **this hypothesis was not**, and G1
> passed the plan anyway — it checked the document for unfilled decisions, never for truth.
>
> Run 17 spends money and is gated against this document. G1 §13c now DERIVES the count from
> the package and hard-fails on any disagreement.
>
> **The historical numbers above are spelled out in words on purpose.** Written as digits they
> would be indistinguishable, to the guard, from a live claim — and the guard would keep failing
> on a note explaining why it once failed. Any `<N>-feature` digit string in this file is treated
> as a LIVE ASSERTION about the contract. Do not write one unless you mean it, and never wrap it
> in markdown emphasis: `**97**-feature` is invisible to the check, which is precisely how a
> number a human can read but a machine cannot verify goes stale.
>
> **CONTRACT CORRECTED AGAIN 2026-08-02: ninety-seven -> ninety-five.** On
> **2026-07-14** (`4528414`, roadmap 6.21a) the two HGMD columns were REMOVED --
> not renamed, not deferred. Two independent reasons, either sufficient. First,
> no licence: HGMD Professional is a paid QIAGEN product this project does not
> hold, so both columns were CONSTANT ZERO for the life of the project, occupying
> two slots and making the roster overstate the science by two. Second, and this
> one survives the licence arriving: HGMD "DM" means *disease-causing mutation*
> and the training label here is ClinVar Pathogenic -- the same quantity under two
> vendors' names. As a VARIANT-LEVEL feature it is an answer key, and the
> gene-aware split cannot help, because the leak sits inside every fold at the
> variant level. A variant of uncertain significance, which is precisely what this
> classifier exists to score, has no HGMD entry: the flag reads zero, the model
> leans benign, and you publish an excellent area under the receiver operating
> characteristic curve on catalogued variants while systematically under-calling
> the variants that matter. If access is ever obtained, wire it GENE-LEVEL and
> LEAVE-ONE-OUT (`n_hgmd_dm_in_gene`, excluding the variant being scored),
> mirroring `n_pathogenic_in_gene` -- never as a variant-level flag.
>
> **This document was frozen on 2026-07-13 and four of its assertions went stale
> afterwards**, of which G1 §13c reads exactly one. The marker, this hypothesis,
> B.D3 and B.D4 were corrected together on 2026-08-02, because correcting only the
> marker would have restored the green light over three remaining misstatements.
> §13c itself could not run at all until 2026-08-01 (`f2cff8c`): it crashed on a
> PowerShell type detail and its crash counted as neither pass nor fail.

**H_Run17 (primary)**: The Run 17 baseline reproduces the Run 14/16 ensemble performance on the expanded 95-feature contract (86 + 3 FinnGen R13 + 6 KEGG/COSMIC/Nucleotide-Transformer columns, the 86 being the Run-16 contract less the two HGMD columns excised 2026-07-14) without regression, while the newly-wired Run-17 annotation sources (OMIM genemap2, FinnGen R12+R13, KEGG, COSMIC, the Nucleotide Transformer, and the other CLI-wired connectors) measurably reduce the silent-zero feature count relative to Run 16.

**H_Run17 (dual-release sub-hypothesis — the benchmarking experiment)**: FinnGen R12 and R13, run as two independent annotation passes over the *same* variants with evolved population frequencies, produce *measurably different* feature distributions and per-model feature-importance, demonstrating that the pipeline can ingest and benchmark two releases of the same source apples-to-apples. Specifically: R13 annotates more variants than R12 (higher non-null coverage), and `finngen_r13_*` feature-importance is correlated-but-not-identical to `finngen_*`.

Falsified by: (a) baseline AUROC regression > 0.002 vs Run 16 at matched config, OR (b) R12 and R13 producing statistically indistinguishable feature distributions (which would mean the dual-release adds no benchmarking signal).

This is a PROOF-OF-CONCEPT for the project's SUPPORTING goal (empirically measure/compare/document ML behavior on large complex datasets), not a model-selection exercise. All models stay.

## B. Decisions to lock before launch

### B.O — Open items from Run 16 backlog
- **B.O1** Run-17 annotation wiring (OMIM/PhyloP/dbSNP/EVE/ClinGen CLI flags): the `run_phase2_eval.py` flags + `AnnotationConfig` fields exist (verified this session). Confirm each resolves to a real local file at preflight or is consciously absent (logged, not silently zero).
- **B.O2** HGVSp parser (unlocks ESM-2 + EVE protein-coordinate features): status to confirm at preflight — if delivered, validate against step-10b protein-coordinate coverage gate (min_protein_coord_coverage=0.50); if absent, ESM-2/EVE remain consciously-absent (logged).

### B.D — Data source decisions
- **B.D1** FinnGen R12: **WIRED (resolves RUN_15 B.D2 deferral)**. `data/external/finngen/finnge_R12_annotated_variants_v1.gz` (29.9 GB, registry filename typo 'finnge' intentional, 1017 cols, GENOME_AF_fin + GENOME_AF_nfe). The RUN_15-era R10-vs-R12 schema concern is resolved: `FinnGenConnector` reads the R12 schema and emits `finngen_af_fin/af_nfsee/enrichment`. Launcher hard-fails (exit 7) if the file is missing.
- **B.D2** FinnGen R13: **WIRED (dual-release experiment)**. `data/external/finngen/finngen_R13_annotated_variants_v0.gz` (27.72 GB / 29,768,495,399 bytes, correct spelling, 1025 cols, SAME variant set as R12 -- see the PROBE-VERIFIED note below). Second independent `FinnGenConnector` pass with `column_prefix="r13_"` → `finngen_r13_af_fin/af_nfsee/enrichment`. Launcher hard-fails (exit 7) if missing.
  - Coverage/AF (20k reference sample): R12 17085/20000 nonzero (mean AF 0.1025); R13 19318/20000 nonzero (mean AF 0.0971) -- R13 informs more of the cohort.
  - **PROBE-VERIFIED 2026-06-28** (scripts/probe_finngen_sizes.py, full streaming pass, both files integrity=CLEAN): R12 and R13 have IDENTICAL data-row counts (21,331,644 each) and a HEAD-sample variant-key Jaccard of 1.0 (chrom:pos:ref:alt) -- so 'same variants, same coords, apples-to-apples' is verified, not assumed. The R13 file is SMALLER than R12 despite more samples, +8 net columns, and higher coverage purely because of ENCODING, not missing content: R12 (_v1) carries a per-row b37_coord string plus EXOME/GENOME nfee/nfse coord+AF string columns, while R13 (_v0) drops b37_coord and adds AC/AN integer-count columns that gzip-compress far better than coordinate strings. Both are BGZF (block-gzip; R12 ~1.74M members, R13 ~1.82M). SHA-256 recorded in data/external/finngen/CHECKSUMS.sha256.
- **B.D3** Feature contract: **95 columns (86 + finngen_r13_af_fin/af_nfsee/enrichment + KEGG/COSMIC/Nucleotide-Transformer x6)**. EXPECTED_TABULAR_FEATURE_COUNT=95; INFERENCE_FEATURE_COLUMNS auto-tracks as list(TABULAR_FEATURES). Contract test (test_feature_count_contract.py, 4/4) gates this.
- **B.D4** Harness reference slice: feeds all 6 finngen columns (Option B); KNOWN_ZERO_DEFAULT=24 (finngen AF removed — now actively zero-audited, not allowlisted).

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

- **SCP up**: ~57.6 GB total when both ship (R12 29.92 GB + R13 27.72 GB; PROBE-VERIFIED 2026-06-28) over the id_lambda_run8 key. Per-config: `r12only` ships R12 only (~29.92 GB); `r13only` ships R13 only (~27.72 GB); `both`/baseline ships both (~57.6 GB). The three configs run as THREE independent runs, so aggregate training cost is ~3x a single run -- confirm budget acceptance before launch.
- **VM annotation**: two BGZF (block-gzip) passes (R12 32.1 GB / R13 29.8 GB on disk; both decompress CLEAN end-to-end, PROBE-VERIFIED 2026-06-28), bounding-box-filtered so RAM stays bounded by the matched subset, but two full decompression passes. Flag against MIN_RAM_GB=50. (r12only/r13only each decompress one file; both/baseline decompresses both.)
- **Instance**: Vast.ai RTX 4090, target $0.55/hr (cap $0.77/hr), filter `dlperf>=80 pcie_bw>=12`. Offer selection via scripts/select_vast_offer.py.
- **Estimate**: state explicit wall-clock estimate + dominant cost driver at preflight; confirm Monzia accepts before any instance create (standing law for runs >15min).

## F. Pre-launch gates (Charter v1.1)

- **G1 (local)**: scripts/Run_Preflight_Local.ps1 adapted Run-15→Run-17 — §3 DELETE (imodelsx patch moved to kan.py L181); §6 test floor 566→1496 collected / 1485 pass (reconciled 2026-06-28, CI #486 green); §7 rebuild data list (FinnGen NOW BOTH files local, hard-fail); §11 THREE launchers (launch_run17_{baseline,r12only,r13only}.sh); §12/13 RUN_17 postflight EXISTS (scripts/Run17_Postflight.ps1, parameterized -Config {both,r12only,r13only} + -DryRun, CI #486 green); ADD agent-liveness via scripts/check_agents_active.py. Reference slice (build_reference_slice, now feeding finngen) is the G1 single-source-of-truth.
  - **Postflight usage (per config)**: run `Run17_Postflight.ps1 -Config <both|r12only|r13only> -DryRun` first (prints derived paths; NO SSH/SCP/destroy), then the real invocation, then Vastai_Destroy_Confirmed.ps1 to tear down. A downloaded .ps1 carries Mark-of-the-Web under RemoteSigned -- `Unblock-File` it before running.
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

### H.4 Ablation realization & caveat (PROBE-VERIFIED 2026-06-28)
The three feature configs {R12-only, R13-only, both} are realized by CONSTANT-FILLING the excluded release's 3 finngen columns (af_fin/af_nfsee at 0.0, enrichment at 1.0), NOT by dropping them -- the 91-column contract is fixed. Interpretation caveat: for the tree learners (CatBoost/XGBoost/LightGBM/RF/GBM) a constant column carries no split information, so constant-filled approximates 'absent'; but the column still occupies a feature slot, and the linear (LR) and neural (1D-CNN/TabularNN/MC-Dropout/Deep Ensemble/KAN) models treat a constant input differently from true absence (it contributes a bias-like constant, not nothing). So each cross-config delta measures 'release present vs constant-filled', which for tree models is a close proxy for 'present vs absent' and for linear/NN models should be read as 'present vs neutralized'. The comparison is on the SAME 21,331,644-variant universe (identical row counts + Jaccard 1.0 head sample), so coverage-delta and AF-shift are computed over a common variant set -- no intersection caveat is required.

## I. Decision log

- **2026-06-27**: Dual-release (R12 + R13) chosen as the Run-17 benchmarking experiment. Build completed as 5 stages (ca76482 → 70904a6), all CI-green at #477. Feature contract 88→91. Harness Option B (feed finngen, allowlist 27→25). R13 wired as independent connector pass (column_prefix="r13_") reading a separate ~30GB file.
- **2026-06-27**: Option A (allowlist R13) tried + reverted to Option B (feed fixture) per the pre-existing test_allowlist_unchanged_size guard. Documented in A.D1.
- **2026-06-28**: FinnGen R12/R13 integrity + size-gap VERIFIED before any paid run (scripts/probe_finngen_sizes.py, full streaming single pass). Both files integrity=CLEAN; IDENTICAL data-row counts (21,331,644 each); HEAD-sample variant-key Jaccard 1.0. The smaller R13 file (27.72 GB vs R12 29.92 GB) is an ENCODING difference (R13 _v0 drops the per-row b37_coord string and uses AC/AN integer counts vs R12 _v1's coordinate/AF strings), NOT fewer variants -- so B.D2's 'same variants, apples-to-apples' is now evidence-backed. SHA-256 of both files recorded in data/external/finngen/CHECKSUMS.sha256 (machine-checkable provenance).

## J. References

- RUN_15_PLAN.md B.D2 (the FinnGen deferral this run resolves)
- docs/CHANGELOG.md (the ca76482→70904a6 commit arc)
- scripts/launch_run17_baseline.sh (the launcher with both finngen guards)
- tests/unit/test_finngen_r13_wiring.py, test_feature_count_contract.py, test_correctness_harness.py (the gates)
