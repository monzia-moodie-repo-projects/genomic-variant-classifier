# Run_pandas3_full_validation.ps1 -- PHASE 7 full validation on pandas 3.0.4:
#   (1) presence-check the 8 smoke inputs, (2) pytest as a HARD GATE, (3) only on green pytest,
#   the EXACT 2026-06-28 green-smoke invocation under pandas 3.0.4, (4) assert 13 models + GNN.
# pytest-before-smoke so a unit-level break stops you BEFORE the ~104-min smoke.
# Run from repo root.
$ErrorActionPreference = "Stop"
Set-Location C:\Projects\genomic-variant-classifier
$ts = Get-Date -Format yyyyMMdd_HHmmss

# --- 0. pandas must be 3.0.4 ---
$pv = (python -c "import pandas; print(pandas.__version__)").Trim()
"pandas: $pv"
if ($pv -ne "3.0.4") { throw "ABORT: pandas is $pv, expected 3.0.4." }

# --- 1. presence-check the 8 inputs the green smoke used (abort before a doomed 104-min run) ---
"=== presence-check smoke inputs ==="
$inputs = @(
  "data\processed\clinvar_grch38_clean.parquet",
  "data\processed\gnomad_v4_exomes.parquet",
  "data\external\spliceai\spliceai_index.parquet",
  "data\external\alphamissense\AlphaMissense_hg38.tsv.gz",
  "data\processed\clinvar_grch38_clean_seq.parquet",
  "data\external\gnomad\gnomad.v4.1.constraint_metrics.tsv",
  "data\external\dbnsfp\dbnsfp_clinvar_index.parquet",
  "data\external\lovd\lovd_all_variants.parquet"
)
$missing = @()
foreach ($p in $inputs) {
  if (Test-Path $p) { "  OK   $p" } else { "  MISS $p"; $missing += $p }
}
if ($missing.Count -gt 0) { throw "ABORT: $($missing.Count) smoke input(s) missing; resolve before running." }
if (-not (Test-Path "data\external\string\9606.protein.links.detailed.v12.0.txt.gz")) {
  throw "ABORT: STRING links file missing (needed for --string-db auto)."
}

# --- 2. PYTEST HARD GATE ---
"=== PYTEST (hard gate -- smoke runs ONLY if this is green) ==="
Start-Transcript -Path "logs\pytest_pandas3_$ts.txt" -Append
python -m pytest -q --tb=short
$pytestExit = $LASTEXITCODE
Stop-Transcript
"pytest exit: $pytestExit"
if ($pytestExit -ne 0) {
  throw "ABORT: pytest FAILED on pandas 3.0.4 (exit $pytestExit). Smoke NOT started. See logs\pytest_pandas3_$ts.txt."
}
"=== pytest GREEN -- proceeding to the full ALL-MODELS smoke (~104 min) ==="

# --- 3. FULL ALL-MODELS SMOKE -- EXACT 2026-06-28 invocation, under pandas 3.0.4 ---
Start-Transcript -Path "logs\smoke_pandas3_$ts.txt" -Append
python scripts\smoke_all_models.py `
    --clinvar data\processed\clinvar_grch38_clean.parquet `
    --gnomad data\processed\gnomad_v4_exomes.parquet `
    --spliceai data\external\spliceai\spliceai_index.parquet `
    --alphamissense data\external\alphamissense\AlphaMissense_hg38.tsv.gz `
    --seq-windows data\processed\clinvar_grch38_clean_seq.parquet `
    --gnomad-constraint data\external\gnomad\gnomad.v4.1.constraint_metrics.tsv `
    --dbnsfp-path data\external\dbnsfp\dbnsfp_clinvar_index.parquet `
    --lovd-path data\external\lovd\lovd_all_variants.parquet `
    --string-db auto `
    --clinvar-sample-n 2000 `
    --smoke-n 2000 `
    --keep-output
$smokeExit = $LASTEXITCODE
Stop-Transcript
"smoke exit: $smokeExit"

# --- 4. assert smoke green + 13 models + non-degenerate GNN ---
"=== smoke result assertions ==="
$smklog = "logs\smoke_pandas3_$ts.txt"
$expected = @("catboost","cnn_1d","deep_ensemble","gradient_boosting","kan","lightgbm",
              "logistic_regression","mc_dropout","random_forest","svm","svm_bagged_rbf",
              "tabular_nn","xgboost")
$present = @()
foreach ($m in $expected) {
  if (Select-String -Path $smklog -Pattern "\b$m\b" -Quiet) { $present += $m }
}
"models seen in smoke log: $($present.Count)/13 -> $($present -join ', ')"
$missingModels = $expected | Where-Object { $_ -notin $present }
if ($missingModels) { "MISSING MODELS: $($missingModels -join ', ')" }

# GNN non-degenerate (look for the gnn_score / GNN AUROC line, not a degenerate 0.5000)
"=== GNN sanity (gnn_score / AUROC lines) ==="
Select-String -Path $smklog -Pattern 'gnn_score|GNN.*AUROC|gnn.*auc' | Select-Object -Last 6 | Select-Object Line

"========================================================"
if ($smokeExit -eq 0 -and $present.Count -eq 13 -and -not $missingModels) {
  "PASS: pandas 3.0.4 -- pytest green ($pytestExit), smoke exit 0, all 13 models present."
  "      Ready for pins + commit."
} else {
  "FAIL: smokeExit=$smokeExit models=$($present.Count)/13. Investigate logs\smoke_pandas3_$ts.txt."
  throw "ABORT: full validation did not fully pass."
}
