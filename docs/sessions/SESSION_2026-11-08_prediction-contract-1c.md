# Session 2026-11-08 -- package 1 (the prediction contract), set 1c

Author: Monzia Moodie
Base: merged set 1b, a29af854 (pull request #69; tree b0bb918b). Rulings: decisions ccffb8cf and 38f15c7a of 2026-11-07.
Acceptance: docs/package1/PACKAGE1_SET1C_DELIVERY.md (gate by gate), docs/package1/PACKAGE1_SET1C_CENSUS_LEDGER.json.

## What set 1c does

- **One bound reader of a completed run's split predictions** (`prediction_artifacts.load_completed_split_predictions`), and the
  selected predictor's evaluation evidence computed from it (`selected_predictor_evaluation_evidence`).
- **The threshold workflow migrated** onto that reader and the repository's exact operating-point machinery
  (`evaluation/decision_policy.py`, `scripts/calibrate_thresholds.py`); thresholds on the served stage; neutral bands;
  certification blockers; a record digest; refusals instead of fallbacks or clamping.
- **Export metadata bound to the selected predictor** (G7.2): `PipelineMetadata.evaluation_evidence` and `fit_input_rows`;
  `val_auroc` and `n_train` removed and refusing; the exporter requires a verified completed run and the files it lists.
- **The remaining consumers**: the import-time threshold loader removed from api/schemas.py; validate_external,
  validate_clinvar_temporal, validate_lovd_enriched and build_lovd_index re-pointed with their own defects resolved; conformal
  calibration on the selected score with a recorded substrate; conformal_prediction on the finite-sample order statistic;
  continual_trainer on the 1-D score; calibration_analysis records what it skips.
- **Dispositions**: the reconstruction exporter historical-execution denied; transition_policy.py retired; two superseded
  patchers and the OOF alignment check refuse (G2.4); the other OOF probes and the reconstruction diagnostic labelled; the three
  observability scripts read the selected blend's weights from a verified manifest.

## Measured

- calibrate_thresholds' search (the extracted function executed; synthetic split, 20,000 rows, prevalence 0.15): 1.000000 with
  238 calls for both targets; the documented lowest qualifying thresholds are 0.566457 (3,141 calls, PPV 0.8001) and 0.703091
  (1,867 calls, PPV 0.9502).
- Tail predictive values overlap on realistic runs: a completed fixture run (AUROC 0.906, prevalence 0.3775, 400 validation rows)
  -- positive_moderate `selected_score >= 0.554`, negative_moderate `1 - selected_score >= 0.382` (scores <= 0.618), 16 rows in
  both; a near-separable run (AUROC 0.9997) overlaps at moderate floors of 0.95 (positive >= 0.2275, negative <= 0.677). Hence
  DECISION-POLICY-OVERLAP-1.
- build_lovd_index `_lovd_label` (extracted and executed on its own terms): "not pathogenic" and "probably not pathogenic" -> 1,
  "Class 10" -> 0; 2 of its 12 terms wrong.
- conformal_prediction's old threshold: on a 37-row fixture at alpha 0.1, strictly between the 35th and 36th order statistics (k = 35); at
  alpha 0.01 (k = 38 > n) the largest score instead of +inf.
- apply_d1_fixes.py and fix_oof_index.py on an archive of a29af854: both exit 1 and leave real_data_prep.py and
  run_phase2_eval.py byte-identical (F-02, F-05, F-07, F-13 find no target; ten F-06 fragments already applied, two absent).
- gen_run17_observability.py reproduced run17_observability.py from run15_observability.py byte for byte before and after the change.
- Mutation checks: twenty guards sabotaged one at a time in a copy of the tree; nineteen caught by their named test at first; the
  incomplete-run export refusal was masked by a downstream refusal, so its test was strengthened (the refusal's own message, and no
  ensemble opened) and the mutation is now caught.
- The owner's 1b records re-read from Downloads through the linked computer: receipt dcee4aaa, full.xml 16f4990f (8,922 test
  cases: 8,834 passed, 83 skipped, 5 xfailed), promote record 654561db, post-merge record 3240a10a (1,105 targeted: 1,100 passed,
  5 skipped, 0 differences, 0 warnings), branch deletion record ea670257 (GitHub holds main a29af854 only), Windows derived
  record digest b1330929.

## Errors of mine, caught before delivery

- A first conformal docstring said the old `np.quantile` threshold lay BELOW the order statistic; a measurement showed it lies
  between the k-th and (k+1)-th (conservative), and that the anti-conservative defect is the k > n case. Corrected before any test
  was frozen.
- Four test assertions first matched words in my own new docstrings (a forbidden name quoted while explaining its removal); they
  now check the code (syntax-tree attribute access, call text) rather than the text.
- Three session-state headings carried guessed clock times; corrected from `date -u`.
- A snapshot for the full suite first excluded `outputs/`, which holds 19 tracked files; retaken with them.

## Still open (owner decisions)

DECISION-POLICY-OVERLAP-1 (this set); GNN-FEATURE-ORDER-1 (blocks Run 17) and MAXTRAIN-SPLIT-REPERSIST-1 (set 1b).
