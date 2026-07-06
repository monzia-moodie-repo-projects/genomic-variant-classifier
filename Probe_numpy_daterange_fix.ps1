# Probe_numpy_daterange_fix.ps1 -- find the highest numpy version (<= current) that makes
# pandas 3.0.4 date_range stop segfaulting. Tests candidates newest-first; STOPS at the first
# that works. Does NOT touch the project beyond numpy; records the winning version for the real pin.
# Run from repo root. pandas must be 3.0.4.
$ErrorActionPreference = "Continue"   # we EXPECT some candidates to segfault; don't abort on that
Set-Location C:\Projects\genomic-variant-classifier

$pv = (python -c "import pandas; print(pandas.__version__)").Trim()
"pandas: $pv (must be 3.0.4)"
if ($pv -ne "3.0.4") { Write-Host "ABORT: pandas not 3.0.4"; exit 1 }

"=== current numpy (the one that segfaults date_range) ==="
$cur = (python -c "import numpy; print(numpy.__version__)").Trim()
"current numpy: $cur"

# Candidate numpy versions to try, NEWEST first (so we minimize the downgrade).
# pandas 3.0.4 requires numpy>=1.26.0; we try the 2.x line down from current.
$candidates = @("2.3.3", "2.3.2", "2.3.0", "2.2.6", "2.2.0", "2.1.3", "2.0.2")

$daterange_probe = 'import pandas as pd; pd.date_range("2024-01-01", periods=10, freq="D"); print("DATE_RANGE_OK")'

$winner = $null
foreach ($nv in $candidates) {
  "=================================================="
  "=== trying numpy==$nv ==="
  python -m pip install "numpy==$nv" --quiet 2>&1 | Select-Object -Last 2
  $installed = (python -c "import numpy; print(numpy.__version__)" 2>$null).Trim()
  if ($installed -ne $nv) { "  install did not land $nv (got '$installed'); skipping"; continue }

  # Run the date_range probe in a SUBPROCESS so a segfault doesn't kill THIS script.
  $out = python -c $daterange_probe 2>&1
  $code = $LASTEXITCODE
  if ($code -eq 0 -and ($out -match "DATE_RANGE_OK")) {
    "  numpy==$nv -> date_range OK (exit 0)"
    # ALSO confirm the heavy stack still imports under this numpy
    $imp = python -c "import importlib; [importlib.import_module(m) for m in ['torch','torch_geometric','catboost','xgboost','lightgbm','sklearn','pandas','pyarrow']]; print('IMPORTS_OK')" 2>&1
    if ($imp -match "IMPORTS_OK") {
      "  heavy stack imports OK under numpy==$nv"
      $winner = $nv
      break
    } else {
      "  WARNING: numpy==$nv fixes date_range BUT breaks an import: $imp"
      "  (continuing to next candidate -- need both)"
    }
  } else {
    "  numpy==$nv -> date_range STILL crashes (exit $code); trying older"
  }
}

"=================================================="
if ($winner) {
  "WINNER: numpy==$winner fixes date_range AND keeps the heavy stack importable."
  "  -> This is the numpy to pin. Re-run the data-prep equivalence + full validation on it next."
  "  -> Currently INSTALLED: numpy==$winner (left in place for the equivalence re-check)."
} else {
  "NO WINNER among candidates. date_range could not be fixed by a numpy downgrade in range."
  "  -> Reinstalling original numpy==$cur so the environment is back to a known state."
  python -m pip install "numpy==$cur" --quiet 2>&1 | Select-Object -Last 1
  "  -> Recommend PATH 2 (roll back pandas to 2.3.3) since the numpy pairing fix did not work."
}
