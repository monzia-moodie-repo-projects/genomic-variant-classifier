# Install_suppress_downcast_versionaware.ps1 -- Finding 1 fix: make _suppress_fillna_downcast a no-op
# on pandas >= 3.0 (kills the Pandas4Warning), then RE-PROVE equivalence on the installed 3.0.4.
# Run from repo root. pandas is already 3.0.4 at this point.
$ErrorActionPreference = "Stop"
Set-Location C:\Projects\genomic-variant-classifier
$dl = "$env:USERPROFILE\Downloads"

$p = Get-ChildItem $dl -Filter "patch_suppress_downcast_version_aware*.py" |
     Where-Object { Select-String -Path $_.FullName -Pattern '_PANDAS_MAJOR >= 3' -Quiet } |
     Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $p) { throw "ABORT: patcher (marker '_PANDAS_MAJOR >= 3') not in $dl." }
$pNon = ([System.IO.File]::ReadAllBytes($p.FullName) | Where-Object { $_ -gt 127 }).Count
if ($pNon -ne 0) { throw "ABORT: patcher has non-ASCII bytes." }
Copy-Item $p.FullName "scripts\patch_suppress_downcast_version_aware.py" -Force

$pv = (python -c "import pandas; print(pandas.__version__)").Trim()
"pandas: $pv (expect 3.0.4)"
if ($pv -ne "3.0.4") { throw "ABORT: pandas is $pv, expected 3.0.4." }

# Pre-check: both the v2 allele_freq fix AND (not yet) the version guard.
"=== pre-check: current decorator wrapper ==="
Select-String -Path src\genomic_variant_classifier\data\real_data_prep.py -Pattern 'option_context|_PANDAS_MAJOR|_suppress_fillna_downcast' | Select-Object LineNumber, Line

python scripts\patch_suppress_downcast_version_aware.py
if ($LASTEXITCODE -ne 0) { throw "ABORT: patcher exited $LASTEXITCODE." }

"=== post-patch: the version-aware guard ==="
Select-String -Path src\genomic_variant_classifier\data\real_data_prep.py -Pattern '_PANDAS_MAJOR|option_context|no-op on pandas' | Select-Object LineNumber, Line

# Re-run the harness on REAL 3.0.4 -> compare to the golden 2.3.3 baseline. Must STILL be EQUIVALENT,
# and now warnings.json must be EMPTY (the Pandas4Warning is gone).
$ts = Get-Date -Format yyyyMMdd_HHmmss
Start-Transcript -Path "logs\pandas3_real304_postguard_$ts.txt" -Append
python scripts\pandas3_equivalence_harness.py `
    --cohort data\_pandas3\cohort_2k_seed42.parquet `
    --gnomad data\processed\gnomad_v4_exomes.parquet `
    --out    data\_pandas3\real304_postguard
Stop-Transcript

"=== EQUIVALENCE: baseline2 (2.3.3) vs real304_postguard (3.0.4, guard applied) ==="
python scripts\pandas3_equivalence_harness.py --compare data\_pandas3\baseline2_pandas233 data\_pandas3\real304_postguard
$cmp = $LASTEXITCODE
"=== warnings.json (MUST be empty {} now -- Pandas4Warning gone) ==="
Get-Content data\_pandas3\real304_postguard\warnings.json
"=== hashes (MUST both be 49e98393...) ==="
"baseline2     : $(Get-Content data\_pandas3\baseline2_pandas233\feature_hash.txt)"
"real304_guard : $(Get-Content data\_pandas3\real304_postguard\feature_hash.txt)"

$wEmpty = ((Get-Content data\_pandas3\real304_postguard\warnings.json -Raw).Trim() -eq "{}")
if ($cmp -ne 0) { throw "ABORT: NOT EQUIVALENT after guard -- investigate." }
if (-not $wEmpty) { throw "ABORT: warnings.json not empty after guard -- a warning remains." }
"=== GREEN: pandas 3.0.4 equivalent to 2.3.3 baseline AND zero warnings. Ready for pins + tests + commit. ==="
