# Install_allele_freq_fix.ps1 -- apply the allele_freq pandas-3 fix, then PROVE equivalence on 2.3.3.
# Run from repo root. Still on pandas 2.3.3 (we verify the fix is value-preserving BEFORE upgrading).
$ErrorActionPreference = "Stop"
Set-Location C:\Projects\genomic-variant-classifier

$dl = "$env:USERPROFILE\Downloads"

# Select patcher BY CONTENT marker; abort if absent.
$src = Get-ChildItem $dl -Filter "patch_allele_freq_numeric*.py" |
       Where-Object { Select-String -Path $_.FullName -Pattern 'pandas-3 readiness: allele_freq' -Quiet } |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $src) { throw "ABORT: patch_allele_freq_numeric.py not in $dl (content marker missing)." }
"Found: $($src.FullName)"

# ASCII guard (absolute path -- .NET CWD trap).
$nonAscii = ([System.IO.File]::ReadAllBytes($src.FullName) | Where-Object { $_ -gt 127 }).Count
"patcher non-ASCII bytes (expect 0): $nonAscii"
if ($nonAscii -ne 0) { throw "ABORT: patcher has non-ASCII bytes." }
Copy-Item $src.FullName "scripts\patch_allele_freq_numeric.py" -Force

# Confirm still on 2.3.3 (we prove the fix is value-preserving on the OLD pandas first).
$pv = (python -c "import pandas; print(pandas.__version__)").Trim()
"pandas: $pv (must be 2.3.3 for the equivalence check)"
if ($pv -ne "2.3.3") { throw "ABORT: pandas is $pv, not 2.3.3." }

# Pre-check: show the exact target line BEFORE patching.
"=== target line BEFORE ==="
Select-String -Path src\genomic_variant_classifier\data\real_data_prep.py -Pattern 'allele_freq.*fillna.*gnomad_af' | Select-Object LineNumber, Line

# Apply the patch.
python scripts\patch_allele_freq_numeric.py
if ($LASTEXITCODE -ne 0) { throw "ABORT: patcher exited $LASTEXITCODE." }

"=== target region AFTER ==="
Select-String -Path src\genomic_variant_classifier\data\real_data_prep.py -Pattern 'pandas-3 readiness|pd.to_numeric|allele_freq.*fillna' | Select-Object LineNumber, Line

# Re-run the harness on 2.3.3 into a NEW bundle, then COMPARE to the golden baseline.
$ts = Get-Date -Format yyyyMMdd_HHmmss
Start-Transcript -Path "logs\pandas3_postfix_233_$ts.txt" -Append
python scripts\pandas3_equivalence_harness.py `
    --cohort data\_pandas3\cohort_2k_seed42.parquet `
    --gnomad data\processed\gnomad_v4_exomes.parquet `
    --out    data\_pandas3\postfix_pandas233
Stop-Transcript

"=== EQUIVALENCE: baseline (pre-fix 2.3.3) vs postfix (2.3.3) -- MUST be EQUIVALENT + warnings gone ==="
python scripts\pandas3_equivalence_harness.py --compare data\_pandas3\baseline_pandas233 data\_pandas3\postfix_pandas233
$cmp = $LASTEXITCODE
"=== postfix warnings.json (MUST be empty {}) ==="
Get-Content data\_pandas3\postfix_pandas233\warnings.json
"=== feature hashes (MUST match 49e98393...) ==="
"baseline: $(Get-Content data\_pandas3\baseline_pandas233\feature_hash.txt)"
"postfix : $(Get-Content data\_pandas3\postfix_pandas233\feature_hash.txt)"

if ($cmp -ne 0) { throw "ABORT: NOT EQUIVALENT after fix -- the fix changed a value. Investigate before upgrading." }
"=== GREEN: fix is value-preserving on 2.3.3 + warning eliminated. Ready for Phase 4 (pandas-3 sim). ==="
