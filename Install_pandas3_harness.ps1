# Install_pandas3_harness.ps1 -- place + verify the pandas-3 equivalence harness, then run Phase 1 baseline.
# Run from repo root: C:\Projects\genomic-variant-classifier
$ErrorActionPreference = "Stop"
Set-Location C:\Projects\genomic-variant-classifier

$dl = "$env:USERPROFILE\Downloads"
# Select the downloaded harness BY CONTENT marker (avoid stale copies), abort if not found.
$src = Get-ChildItem $dl -Filter "pandas3_equivalence_harness*.py" |
       Where-Object { Select-String -Path $_.FullName -Pattern 'pandas 2.x -> 3.x upgrade equivalence harness' -Quiet } |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $src) { throw "ABORT: harness not found in $dl (content marker missing)." }
"Found: $($src.FullName)"

# ASCII guard (cp1252 trap) -- ABSOLUTE path (.NET CWD trap).
$bytes = [System.IO.File]::ReadAllBytes($src.FullName)
$nonAscii = ($bytes | Where-Object { $_ -gt 127 }).Count
"non-ASCII bytes (expect 0): $nonAscii"
if ($nonAscii -ne 0) { throw "ABORT: harness has non-ASCII bytes; would break under cp1252." }

Copy-Item $src.FullName "scripts\pandas3_equivalence_harness.py" -Force
python -c "import ast; ast.parse(open('scripts/pandas3_equivalence_harness.py',encoding='utf-8').read()); print('placed + compiles')"

# Confirm pandas is STILL 2.3.3 (baseline must be captured on the OLD pandas).
$pv = (python -c "import pandas; print(pandas.__version__)").Trim()
"pandas in venv: $pv (MUST be 2.3.3 for the baseline)"
if ($pv -ne "2.3.3") { throw "ABORT: pandas is $pv, not 2.3.3 -- baseline must be on the pinned version." }

# Step A: build the FIXED seeded cohort (both runs consume this identical file).
python scripts\pandas3_equivalence_harness.py --build-cohort `
    --clinvar data\processed\clinvar_grch38_clean.parquet `
    --cohort  data\_pandas3\cohort_2k_seed42.parquet `
    --n 2000 --seed 42

# Step B: capture the 2.3.3 BASELINE bundle (data-prep + features only; ~8-12 min, NO models/GNN).
$ts = Get-Date -Format yyyyMMdd_HHmmss
Start-Transcript -Path "logs\pandas3_baseline_$ts.txt" -Append
python scripts\pandas3_equivalence_harness.py `
    --cohort data\_pandas3\cohort_2k_seed42.parquet `
    --gnomad data\processed\gnomad_v4_exomes.parquet `
    --out    data\_pandas3\baseline_pandas233
Stop-Transcript

"=== baseline bundle contents ==="
Get-ChildItem data\_pandas3\baseline_pandas233 | Select-Object Name, Length
"=== feature hash + pandas version + downcast warning count ==="
Get-Content data\_pandas3\baseline_pandas233\feature_hash.txt
Get-Content data\_pandas3\baseline_pandas233\pandas_version.txt
"--- merge_counts.json ---"; Get-Content data\_pandas3\baseline_pandas233\merge_counts.json
"--- warnings.json (the DEFINITIVE downcast offender list) ---"; Get-Content data\_pandas3\baseline_pandas233\warnings.json
