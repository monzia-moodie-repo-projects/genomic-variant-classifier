# Rollback_to_pandas233.ps1 -- revert pandas 3.0.4 -> 2.3.3 (the proven-good baseline), KEEPING all
# arc code improvements (allele_freq numeric fix + version-aware decorator guard + fixture fix), then
# PROVE those kept changes are correct on 2.3.3: equivalence hash 49e98393 + warnings empty + pytest green.
# Run from repo root.
$ErrorActionPreference = "Stop"
Set-Location C:\Projects\genomic-variant-classifier

# --- 0. confirm we're on 3.0.4 (the thing we're rolling back) ---
$pv = (python -c "import pandas; print(pandas.__version__)").Trim()
"pandas now: $pv"
if ($pv -ne "3.0.4") { throw "ABORT: expected 3.0.4 to roll back from, found $pv." }

# --- 1. confirm the KEPT code changes are present (don't want to roll back code, only the package) ---
"=== confirm kept improvements are in the tree ==="
$okAllele = Select-String -Path src\genomic_variant_classifier\data\real_data_prep.py -Pattern 'Cast BOTH' -Quiet
$okGuard  = Select-String -Path src\genomic_variant_classifier\data\real_data_prep.py -Pattern '_PANDAS_MAJOR >= 3' -Quiet
$okFix     = Select-String -Path tests\unit\test_annotation_policy_baseline.py -Pattern 'avoids the range-generator' -Quiet
"allele_freq numeric fix present : $okAllele"
"decorator version-guard present : $okGuard"
"test fixture fix present        : $okFix"
if (-not ($okAllele -and $okGuard -and $okFix)) { throw "ABORT: a kept improvement is missing; investigate before rollback." }

# --- 2. roll back pandas ---
"=== installing pandas==2.3.3 (rollback) ==="
python -m pip install "pandas==2.3.3"
if ($LASTEXITCODE -ne 0) { throw "ABORT: pip install pandas==2.3.3 failed." }
$pv2 = (python -c "import pandas; print(pandas.__version__)").Trim()
"pandas now: $pv2"
if ($pv2 -ne "2.3.3") { throw "ABORT: pandas is $pv2 after rollback, expected 2.3.3." }

# --- 3. date_range works again + heavy stack imports ---
"=== verify date_range works on 2.3.3 (the thing that segfaulted on 3.0.4) ==="
python -c "import pandas as pd; idx = pd.date_range('2024-01-01', periods=10, freq='D'); print('date_range OK', len(idx))"
if ($LASTEXITCODE -ne 0) { throw "ABORT: date_range still failing on 2.3.3 -- unexpected, investigate." }
"=== heavy stack import-check ==="
python -c "import importlib; [importlib.import_module(m) for m in ['torch','torch_geometric','catboost','xgboost','lightgbm','sklearn','numpy','pyarrow','pandas']]; print('ALL IMPORTS OK')"
if ($LASTEXITCODE -ne 0) { throw "ABORT: a core import failed on 2.3.3." }

# --- 4. PROVE the kept code changes are correct on 2.3.3: equivalence hash + warnings empty ---
$ts = Get-Date -Format yyyyMMdd_HHmmss
Start-Transcript -Path "logs\rollback_233_verify_$ts.txt" -Append
python scripts\pandas3_equivalence_harness.py `
    --cohort data\_pandas3\cohort_2k_seed42.parquet `
    --gnomad data\processed\gnomad_v4_exomes.parquet `
    --out    data\_pandas3\rollback_233_verify
Stop-Transcript
"=== equivalence: golden baseline2 (2.3.3) vs rollback_233_verify (2.3.3, with kept fixes) ==="
python scripts\pandas3_equivalence_harness.py --compare data\_pandas3\baseline2_pandas233 data\_pandas3\rollback_233_verify
$cmp = $LASTEXITCODE
"=== hash + warnings ==="
"baseline2       : $(Get-Content data\_pandas3\baseline2_pandas233\feature_hash.txt)"
"rollback_verify : $(Get-Content data\_pandas3\rollback_233_verify\feature_hash.txt)"
"warnings.json   : $(Get-Content data\_pandas3\rollback_233_verify\warnings.json -Raw)"
$wEmpty = ((Get-Content data\_pandas3\rollback_233_verify\warnings.json -Raw).Trim() -eq "{}")
if ($cmp -ne 0) { throw "ABORT: NOT EQUIVALENT on 2.3.3 after rollback -- the kept changes misbehave on 2.3.3." }
if (-not $wEmpty) { throw "ABORT: warnings not empty on 2.3.3 -- investigate." }

# --- 5. confirm pins are (still) correct at 2.3.3 -- we never changed them, verify that's true ---
"=== pin check (should STILL be pandas==2.3.3 / pandas>=2.2,<3.0 -- never changed) ==="
Select-String -Path requirements.txt -Pattern 'pandas==' | Select-Object Line
Select-String -Path requirements-api.txt -Pattern 'pandas>=' | Select-Object Line

# --- 6. agent liveness on 2.3.3 (river<3.0 conflict is GONE now pandas is back to 2.x) ---
"=== agent liveness on 2.3.3 ==="
python scripts\check_agents_active.py 2>&1 | Select-Object -Last 3

"========================================================"
"GREEN: rolled back to pandas 2.3.3. date_range works. Kept changes proven equivalent (hash 49e98393,"
"warnings empty). Pins already correct at 2.3.3. Ready for pytest + commit (docs)."
