# Install_pandas3_upgrade.ps1 -- PHASE 5+6: install pandas 3.0.4, prove equivalence on REAL 3.0.4
# (the string-dtype gate), then update the pins. Auto-rolls-back to 2.3.3 if the gate fails.
# Run from repo root, AFTER the v2 fix is GREEN on 2.3.3.
$ErrorActionPreference = "Stop"
Set-Location C:\Projects\genomic-variant-classifier

# --- 0. Preconditions: clean fix in place, baseline2 exists, currently on 2.3.3 ---
$pv = (python -c "import pandas; print(pandas.__version__)").Trim()
"pandas now: $pv"
if ($pv -ne "2.3.3") { throw "ABORT: expected 2.3.3 as starting point, found $pv." }
if (-not (Test-Path data\_pandas3\baseline2_pandas233\feature_hash.txt)) {
    throw "ABORT: baseline2_pandas233 bundle missing -- run the v2 fix installer first."
}
# Confirm the v2 fix is actually in the source (sentinel present).
if (-not (Select-String -Path src\genomic_variant_classifier\data\real_data_prep.py -Pattern 'Cast BOTH' -Quiet)) {
    throw "ABORT: v2 allele_freq fix not present in real_data_prep.py."
}
$BASEHASH = (Get-Content data\_pandas3\baseline2_pandas233\feature_hash.txt).Trim()
"golden baseline hash: $BASEHASH"

# --- 1. Snapshot current environment for rollback ---
$ts = Get-Date -Format yyyyMMdd_HHmmss
pip freeze > "logs\pipfreeze_pre_pandas3_$ts.txt"
"env snapshot -> logs\pipfreeze_pre_pandas3_$ts.txt (rollback reference)"

# --- 2. Install pandas 3.0.4 ---
"=== installing pandas==3.0.4 ==="
python -m pip install "pandas==3.0.4"
if ($LASTEXITCODE -ne 0) { throw "ABORT: pip install pandas==3.0.4 failed." }
$pv2 = (python -c "import pandas; print(pandas.__version__)").Trim()
"pandas now: $pv2"
if ($pv2 -ne "3.0.4") { throw "ABORT: pandas is $pv2 after install, expected 3.0.4." }

# --- 3. Import-check the heavy stack (pandas 3 can ripple into these) ---
"=== import-check: torch / torch_geometric / catboost / xgboost / lightgbm / sklearn / numpy ==="
$imp = python -c "import importlib,sys; mods=['torch','torch_geometric','catboost','xgboost','lightgbm','sklearn','numpy','pyarrow']; [importlib.import_module(m) for m in mods]; print('ALL IMPORTS OK')" 2>&1
$imp
if ($imp -notmatch "ALL IMPORTS OK") {
    "*** IMPORT FAILURE on pandas 3.0.4 -- rolling back to 2.3.3 ***"
    python -m pip install "pandas==2.3.3"
    throw "ABORT: a core dependency failed to import under pandas 3.0.4. Rolled back to 2.3.3."
}

# --- 4. THE GATE: run harness on REAL 3.0.4, compare to the 2.3.3 golden baseline ---
"=== PHASE 6 GATE: data-prep on pandas 3.0.4 (string-dtype change is now LIVE) ==="
Start-Transcript -Path "logs\pandas3_real304_$ts.txt" -Append
python scripts\pandas3_equivalence_harness.py `
    --cohort data\_pandas3\cohort_2k_seed42.parquet `
    --gnomad data\processed\gnomad_v4_exomes.parquet `
    --out    data\_pandas3\real_pandas304
Stop-Transcript

"=== EQUIVALENCE: baseline2 (2.3.3) vs real_pandas304 (3.0.4) -- THE STRING-DTYPE GATE ==="
python scripts\pandas3_equivalence_harness.py --compare data\_pandas3\baseline2_pandas233 data\_pandas3\real_pandas304
$gate = $LASTEXITCODE

"=== real 3.0.4 results ==="
"version : $(Get-Content data\_pandas3\real_pandas304\pandas_version.txt)"
"hash    : $(Get-Content data\_pandas3\real_pandas304\feature_hash.txt)"
"baseline: $BASEHASH"
"--- merge_counts (every result_rows MUST be 709) ---"
Get-Content data\_pandas3\real_pandas304\merge_counts.json
"--- warnings.json ---"
Get-Content data\_pandas3\real_pandas304\warnings.json

if ($gate -ne 0) {
    "*** PHASE 6 GATE FAILED on pandas 3.0.4 -- the upgrade changes the feature matrix or drops join rows. ***"
    "*** This is the string-dtype break you pinned against. Rolling back to 2.3.3. ***"
    python -m pip install "pandas==2.3.3"
    $pvr = (python -c "import pandas; print(pandas.__version__)").Trim()
    "rolled back to pandas $pvr. Pins NOT changed. Investigate the failing merge above before retrying."
    throw "ABORT: Phase 6 equivalence gate failed; rolled back to 2.3.3."
}

"=== GATE PASS: pandas 3.0.4 produces the IDENTICAL feature matrix + join counts as 2.3.3. ==="
"=== Pins will be updated next (separate step) now that the upgrade is PROVEN. ==="
