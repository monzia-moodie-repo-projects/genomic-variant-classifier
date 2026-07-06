# Install_allele_freq_fix_v2.ps1 -- CORRECTED: replace harness (line-insensitive merge keys) + apply
# the corrected allele_freq fix (cast BEFORE fillna), then PROVE equivalence on 2.3.3.
# Run from repo root. Still on pandas 2.3.3.
$ErrorActionPreference = "Stop"
Set-Location C:\Projects\genomic-variant-classifier
$dl = "$env:USERPROFILE\Downloads"

# --- 1. Replace the harness with the corrected (line-insensitive) version ---
$h = Get-ChildItem $dl -Filter "pandas3_equivalence_harness*.py" |
     Where-Object { Select-String -Path $_.FullName -Pattern 'line-INSENSITIVE' -Quiet } |
     Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $h) { throw "ABORT: corrected harness (marker 'line-INSENSITIVE') not in $dl." }
$hNon = ([System.IO.File]::ReadAllBytes($h.FullName) | Where-Object { $_ -gt 127 }).Count
if ($hNon -ne 0) { throw "ABORT: harness has non-ASCII bytes." }
Copy-Item $h.FullName "scripts\pandas3_equivalence_harness.py" -Force
"harness replaced (line-insensitive merge keys): $($h.Name)"

# --- 2. Apply the corrected v2 patch (reverts v1 if present) ---
$p = Get-ChildItem $dl -Filter "patch_allele_freq_numeric*.py" |
     Where-Object { Select-String -Path $_.FullName -Pattern 'Cast BOTH' -Quiet } |
     Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $p) { throw "ABORT: corrected patcher (marker 'Cast BOTH') not in $dl." }
$pNon = ([System.IO.File]::ReadAllBytes($p.FullName) | Where-Object { $_ -gt 127 }).Count
if ($pNon -ne 0) { throw "ABORT: patcher has non-ASCII bytes." }
Copy-Item $p.FullName "scripts\patch_allele_freq_numeric.py" -Force

$pv = (python -c "import pandas; print(pandas.__version__)").Trim()
"pandas: $pv (must be 2.3.3)"
if ($pv -ne "2.3.3") { throw "ABORT: pandas is $pv, not 2.3.3." }

python scripts\patch_allele_freq_numeric.py
if ($LASTEXITCODE -ne 0) { throw "ABORT: patcher exited $LASTEXITCODE." }

"=== patched region ==="
Select-String -Path src\genomic_variant_classifier\data\real_data_prep.py -Pattern 'Cast BOTH|_af = pd.to_numeric|_af.fillna' | Select-Object LineNumber, Line

# --- 3. Re-run harness on 2.3.3 -> compare to golden baseline (must be EQUIVALENT + warnings empty) ---
$ts = Get-Date -Format yyyyMMdd_HHmmss
Start-Transcript -Path "logs\pandas3_postfix2_233_$ts.txt" -Append
python scripts\pandas3_equivalence_harness.py `
    --cohort data\_pandas3\cohort_2k_seed42.parquet `
    --gnomad data\processed\gnomad_v4_exomes.parquet `
    --out    data\_pandas3\postfix2_pandas233
Stop-Transcript

# NOTE: baseline_pandas233 was produced by the OLD harness (file:lineno keys). Re-capture a fresh
# baseline with the CORRECTED harness so the comparison uses the same keying on both sides.
"=== re-capturing baseline with corrected harness (revert patch -> capture -> re-apply) ==="
Copy-Item src\genomic_variant_classifier\data\real_data_prep.py "$env:TEMP\rdp_v2.bak" -Force
# temporarily revert to clean to capture a TRUE pre-fix baseline under the new harness:
if (Test-Path src\genomic_variant_classifier\data\real_data_prep.py.bak) {
    Copy-Item src\genomic_variant_classifier\data\real_data_prep.py.bak "$env:TEMP\rdp_clean.bak" -Force
    Copy-Item src\genomic_variant_classifier\data\real_data_prep.py.bak src\genomic_variant_classifier\data\real_data_prep.py -Force
    python scripts\pandas3_equivalence_harness.py `
        --cohort data\_pandas3\cohort_2k_seed42.parquet `
        --gnomad data\processed\gnomad_v4_exomes.parquet `
        --out    data\_pandas3\baseline2_pandas233 | Out-Null
    # restore the v2-patched version
    Copy-Item "$env:TEMP\rdp_v2.bak" src\genomic_variant_classifier\data\real_data_prep.py -Force
    "baseline2 (clean, corrected-harness) captured"
}

"=== EQUIVALENCE: baseline2 (clean) vs postfix2 (v2 fix), both 2.3.3, corrected harness ==="
python scripts\pandas3_equivalence_harness.py --compare data\_pandas3\baseline2_pandas233 data\_pandas3\postfix2_pandas233
$cmp = $LASTEXITCODE

"=== postfix2 warnings.json (MUST be empty {}) ==="
Get-Content data\_pandas3\postfix2_pandas233\warnings.json
"=== feature hashes (MUST both be 49e98393...) ==="
"baseline2: $(Get-Content data\_pandas3\baseline2_pandas233\feature_hash.txt)"
"postfix2 : $(Get-Content data\_pandas3\postfix2_pandas233\feature_hash.txt)"

$wEmpty = ((Get-Content data\_pandas3\postfix2_pandas233\warnings.json -Raw).Trim() -eq "{}")
if ($cmp -ne 0) { throw "ABORT: NOT EQUIVALENT after v2 fix. Investigate before upgrading." }
if (-not $wEmpty) { throw "ABORT: warning still present after v2 fix (warnings.json not empty)." }
"=== GREEN: v2 fix is value-preserving on 2.3.3 AND the downcast warning is eliminated. ==="
