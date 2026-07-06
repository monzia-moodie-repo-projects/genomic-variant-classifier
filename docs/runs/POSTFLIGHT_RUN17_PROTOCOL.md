# RUN 17 — POSTFLIGHT AUDIT PROTOCOL (drafted 2026-07-06, BEFORE the run)

**Purpose.** A standing, exhaustive scientific audit executed after EVERY greenlit run (per the
project directive: local smoke + VM smoke must pass before greenlight; every real run gets a full,
documented postflight covering everything from cloud platform and cost to every output/readout).
This protocol is written *before* the run so it is ready to execute the moment results land, and so
the run cannot be declared "done" until every section below is completed and documented.

**Scope note.** This is the 97-feature Run 17 (NT + COSMIC + KEGG active; `EXPECTED_HEAD 659610f`).
Sections A–J are MANDATORY. A run with any unresolved FAIL in B/E/F/G is NOT publishable and NOT
"green" — it is a diagnosed failure to document and fix, not a result to report.

Output: a dated `docs/runs/POSTFLIGHT_RUN17_<UTC>.md` written from the template in §K, with every
number, every warning, and every verdict recorded in structured prose. No abridgement.

---

## A. Provenance & platform (capture BEFORE teardown — irrecoverable after `vastai destroy`)
Record, verbatim, into the postflight doc:
- **Instance:** vastai instance id, GPU model + count, VRAM, vCPUs, RAM, disk, region, image,
  `$/hr`, and the exact `vastai show instances` row. (From the Vast dashboard / CLI — not memory.)
- **Timing:** provision time, SCP start/end, `Run_Preflight_VM.sh` timestamp, launch timestamp,
  `run_phase2_eval.py` start/end (from `run17_baseline_master.log`), total wall-clock, teardown time.
- **Cost:** `$/hr × billed-hours` = run cost; add SCP/idle time. Record the actual number, not an estimate.
- **Code state:** VM `git rev-parse HEAD` (must == `659610f`), working-tree-clean assertion from G2,
  the `Run_Preflight_VM.sh` full PASS block, and the launcher's `==> HEAD:` echo from the master log.
- **Data state:** the launcher's `[1/6] Data preflight` block (every `OK: <file> (<MB>)` line,
  including the new `$COSMIC_TSV` and `$KEGG_PARQUET`), the KEGG column-probe line, and the kg+rnaseq
  column-contract line. Confirms no source silent-zeroed at the file level.

## B. Run integrity & exit (HARD GATE — any FAIL blocks publication)
- `run_phase2_eval.py rc == 0` (from the master log `==> run_phase2_eval.py rc=` line).
- The 97-feature guard passed (the run printing `Features: 97` in the Phase-2 summary; a wrong count
  raises inside `variant_ensemble` and exits non-zero — so rc=0 AND Features:97 together prove the count).
- Post-run artifact check: all of `metrics.json`, `per_model_metrics.csv`, `per_model_metrics_val.csv`,
  `oof_predictions.parquet`, `feature_importance.csv`, `models/ensemble.joblib` + `.manifest.json`
  VERIFIED present (the launcher's post-run loop).
- Checkpoint sentinel (T+45 min) fired OK, and a base estimator + OOF appeared within ~30 min of
  training start (checkpoint discipline — else the run should have been aborted).
- Model count: `models/*.joblib` count matches the expected roster (13 base + stacker artifacts).

## C. Full metrics readout (record ALL, primary = AUPRC)
From `metrics.json` + `per_model_metrics{,_val}.csv`, transcribe into the doc:
- Ensemble + per-model: AUROC, **AUPRC (primary)**, F1(macro/weighted), MCC, Brier — dev (test) AND
  holdout (val), all 14 rows (13 models + ENSEMBLE_STACKER).
- Train/val/test n; class balance per split; n_features (must be 97).
- The C3 held-out-gene falsifier: AUROC vs the 0.95 threshold (PASS/FAIL) — a core generalization claim.
- Flag any model with AUROC≈0.5 or MCC=0.0 and CLASSIFY it: expected small-fold artifact vs real
  degeneracy (the smoke showed MCC=0 threshold artifacts — verify these RESOLVE at full scale with
  calibration; if they persist, that is a finding, not a footnote).

## D. Per-split feature population (no silent defaults in the trained matrix)
`python scripts/audit_smoke_feature_population.py <run>/splits --run17` (or the full-run splits dir).
- Every FAIL-severity source must be `ok` in train AND val AND test. At full scale — unlike the
  connectors-only smoke — gnomAD/dbNSFP/1000G/hetero-GNN/reactome/rnaseq are ALL active, so `--run17`
  is now the CORRECT yardstick (it was off-yardstick for the smoke).
- The three new families must be populated: `genomiclm_delta_norm` (expect high coverage),
  `cosmic_recurrence` (expect ~partial — somatic overlap), `kegg_pathway_count` + `kegg_disease_pathway_flag`.
- Any feature dead in a split → investigate BEFORE trusting the run (a train-only-alive feature is inert
  at val/test/inference).

## E. Leakage audit (HARD GATE for the new connectors — promotes them "firing" → "trusted")
`python scripts/feature_leakage_audit.py --x <run>/splits/X_train.parquet --meta <run>/<label_file>`
- Confirm NO `CLINVAR_CLNSIG`-type column leaked into the matrix (exit 3 = stop).
- Lone-feature rank AUROC for genomiclm_*/cosmic_*/kegg_* — HONEST result ≈ 0.5–0.65 (real but not a
  label proxy). PRIORITY: `genomiclm_delta_norm` ranked #1 in the smoke — a lone-feature AUROC that is
  extreme (≥ the flag threshold) means investigate before reporting it as discovered signal.
- Point-biserial correlation + coverage-by-class for each. Document the verdict per feature (CLEAN/FLAG).
- Cross-check the smoke's feature-importance ranking against the full-run ranking; a connector that
  dominated the smoke but collapses (or explodes) at scale is a finding to explain.

## F. Calibration (clinical trustworthiness — HARD GATE if the model is to be called clinical)
On OOF/holdout predictions: Brier (already in C), plus reliability curve by decile, ECE, calibration
slope + intercept. If the MCC=0 threshold artifact from the smoke appears, this is where it is
diagnosed and the operating threshold re-derived (sensitivity ≥ 0.90 point; PPV ≥ 0.80 point).
NOTE: the expanded calibration/clinical panels are the metrics `evaluation/` package (still UNBUILT —
§8 of SESSION_2026-07-06). If that package is not yet built at postflight time, compute Brier/ECE/slope
by hand here and record that the full panel is deferred — do NOT silently skip it.

## G. Log warning/error sweep (nothing fails silently)
`grep -iE "warn|error|traceback|degenerate|skip|silent|fallback|default|NaN|inf" run17_baseline_master.log`
- Classify EVERY hit: benign-expected (e.g. a documented if-present source absent) vs real. For each
  source's annotation line, record the coverage count (`N variants with <feature> > 0`) and sanity-check
  it against expectation (e.g. COSMIC ~partial, KEGG high, gnomAD high). A source reporting `0 variants`
  that was supposed to be active = silent-zero = FAIL.
- Confirm GNN + hetero-GNN non-degenerate (std > 0, the in-run hard gate + `verify_gnn_score`).
- Confirm the imodelsx KAN patch applied (KAN present in the roster, not dropped).

## H. Degeneracy & sanity cross-checks
- No all-constant feature in the trained matrix (the audit's concatenated-constant scan).
- Feature importance: top-20 sane (no single feature at ~100% dominance without explanation; the
  Head-A vs disease-prior separation concern from the fusion design applies to interpretation).
- OOF shape matches n_train; `_train_row_idx` present; no duplicate variant_ids across splits;
  gene-disjoint split integrity (no gene in both train and test) reconfirmed at full scale.
- Inference contract: saved base models consume RAW (unscaled) X — note for any downstream inference.

## I. Reproducibility & artifact preservation
- SCP back the FULL `outputs/run17_baseline/full/` (verify each artifact's presence + size post-transfer).
- rclone the results to Drive (`genvarcla:...`); record the destination path.
- Record: exact launch command (`==> ARGS:` from the master log), seeds, n_folds, EXPECTED_HEAD,
  all input file sizes + (where cheap) checksums. A researcher must be able to reconstruct the run.
- Commit the postflight doc + metrics CSVs/JSON (NOT the large binaries — .gitignore already excludes
  `outputs/**/*.joblib|*.parquet|*.npy`).

## J. Teardown & cost close-out
- Only after I–transfer is VERIFIED: `echo y | vastai destroy <id>` (own paste block, after manual check).
- Record final billed cost; confirm the instance is destroyed (`vastai show instances` empty).

---

## K. POSTFLIGHT DOCUMENT TEMPLATE  (`docs/runs/POSTFLIGHT_RUN17_<UTC>.md`)
```
# POSTFLIGHT — Run 17 (97-feature) — <UTC timestamp>
EXPECTED_HEAD: 659610f   | actual VM HEAD: ____   | match: ____
## A. Platform & cost:  instance ___, GPU ___, $/hr ___, billed hrs ___, TOTAL $ ___
   timings: provision ___ / SCP ___ / preflight ___ / launch ___ / run ___ / teardown ___
## B. Integrity:  rc=___  Features:97 ___  artifacts ___/7  checkpoint ___  roster ___/13+stacker
## C. Metrics (AUPRC primary):  <full 14-row table, dev + holdout>  | C3 held-out AUROC ___ vs 0.95
## D. Feature population:  <per-split ok/DEAD for all sources; the 3 new families explicitly>
## E. Leakage:  <lone-feature AUROC per genomiclm/cosmic/kegg + verdict; CLNSIG-leak check>
## F. Calibration:  Brier ___ ECE ___ slope ___ intercept ___ | threshold@sens.90 ___ @PPV.80 ___
## G. Log sweep:  <every warn/error classified; per-source coverage counts; GNN std>0; KAN present>
## H. Degeneracy:  <constant-scan; top-20 importance; split integrity; OOF shape>
## I. Repro:  ARGS ___, seeds ___, artifacts SCP'd + rclone'd to ___
## J. Teardown:  destroyed ___, final cost ___
## VERDICT:  <GREEN + publishable / BLOCKED + the specific FAILs to fix>
```

## L. Execution order (once results are on the laptop)
1. §A/§B from the master log (before teardown — capture platform/cost first).
2. §C/§D/§E/§F/§G/§H from the SCP'd artifacts.
3. §I preserve + §J teardown.
4. Write §K doc; commit; rclone. Only then is the run "done".
