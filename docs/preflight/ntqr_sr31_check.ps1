# docs/preflight/ntqr_sr31_check.ps1
# =====================================
# Standing Rule #31 smoke test for ntqr.
#
# STATUS (2026-06-03): ntqr 0.8 IS installable but has an INCOMPATIBLE API.
# ntqr 0.8 uses TrioVoteCounts (3-classifier ensemble evaluation) — it does
# NOT support single binary-classifier accuracy bounds.
# The Evaluator2 class we designed ntqr_evaluator.py for does not exist in
# any currently available ntqr version.
#
# ntqr_evaluator.py correctly runs in stub mode (all bounds = None).
# Do NOT add ntqr to requirements.txt until the correct API is identified
# and ntqr_evaluator.py is rewritten for it.
#
# This script will report INCOMPATIBLE_API (not PASS or FAIL) when ntqr 0.8
# is installed, to clearly distinguish API incompatibility from import failure.
#
# Run from repo root:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   & ".\docs\preflight\ntqr_sr31_check.ps1"

param([string]$PipArgs = "--break-system-packages")

Set-StrictMode -Version 1
$ErrorActionPreference = "Stop"

$pol = Get-ExecutionPolicy -Scope Process
if ($pol -eq "Restricted") {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
}

Write-Host "ntqr SR #31 smoke test" -ForegroundColor Cyan
Write-Host "=======================" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Pip dry-run
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Step 1: pip dry-run ..."
$dryOut = (pip install ntqr $PipArgs --dry-run 2>&1) -join " "
if ($dryOut -match "Would install|already satisfied") {
    Write-Host ("  pip: ntqr available on PyPI") -ForegroundColor Green
} else {
    Write-Host "  WARNING: ntqr not found on PyPI" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 2. Install
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Step 2: pip install ntqr ..."
pip install ntqr $PipArgs -q 2>&1 |
    Select-String "Successfully installed|already satisfied" |
    ForEach-Object { Write-Host ("  {0}" -f $_) -ForegroundColor Green }

# ---------------------------------------------------------------------------
# 3. Write Python API inspection to temp file
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Step 3: API inspection ..."

$tmpPy = [System.IO.Path]::GetTempFileName() + ".py"
$pyTest = @'
import sys

try:
    import ntqr
    version = ntqr.__version__
    print("ntqr version: " + str(version))
except ImportError as e:
    print("SR31 FAIL (ImportError): ntqr not installable -- " + str(e), file=sys.stderr)
    sys.exit(1)

# Check for the Evaluator2 API we originally designed for
try:
    from ntqr.r2 import Evaluator2
    print("Evaluator2 API: FOUND (original target API available)")
    ev = Evaluator2(n_0=100, n_1=50)
    bounds = ev.classifier_accuracy_bounds(q_00=90, q_01=10, q_10=5, q_11=45)
    assert 0 in bounds and 1 in bounds
    print("ntqr SR31 PASS  version: " + str(version))
    sys.exit(0)
except ImportError:
    pass

# Evaluator2 not found -- inspect actual API
print("Evaluator2 API: NOT FOUND in ntqr " + str(version))
print()
print("ntqr " + str(version) + " actual r2 API:")
try:
    import ntqr.r2.evaluators as ev_mod
    import inspect
    for name in sorted(dir(ev_mod)):
        obj = getattr(ev_mod, name)
        if inspect.isclass(obj) and not name.startswith("_"):
            try:
                sig = str(inspect.signature(obj.__init__))
            except Exception:
                sig = "(unknown)"
            print("  class " + name + sig)
except Exception as e:
    print("  (could not inspect: " + str(e) + ")")

print()
print("INCOMPATIBLE_API: ntqr " + str(version) + " does not have Evaluator2.")
print("ntqr_evaluator.py correctly runs in stub mode.")
print("Do not add ntqr to requirements.txt until a compatible version is found.")
print("See docs/incidents/INCIDENT_2026-06-03_ntqr-api-incompatibility.md")
sys.exit(3)
'@

[System.IO.File]::WriteAllText($tmpPy, $pyTest, [System.Text.UTF8Encoding]::new($false))

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "python"
$psi.Arguments = $tmpPy
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.UseShellExecute = $false

$proc = [System.Diagnostics.Process]::Start($psi)
$stdout = $proc.StandardOutput.ReadToEnd()
$stderr = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()
$exitCode = $proc.ExitCode

Remove-Item $tmpPy -ErrorAction SilentlyContinue

if ($stdout) { Write-Host $stdout.TrimEnd() }
if ($stderr) { Write-Host $stderr.TrimEnd() -ForegroundColor Red }

# ---------------------------------------------------------------------------
# 4. Verdict
# ---------------------------------------------------------------------------
Write-Host ""
if ($exitCode -eq 0 -and $stdout -match "SR31 PASS") {
    Write-Host "ntqr SR31 check: PASS -- safe to add ntqr to requirements.txt" -ForegroundColor Green
} elseif ($exitCode -eq 3 -or $stdout -match "INCOMPATIBLE_API") {
    Write-Host "ntqr SR31 check: INCOMPATIBLE_API" -ForegroundColor Yellow
    Write-Host "  ntqr is installable but its API does not match ntqr_evaluator.py." -ForegroundColor Yellow
    Write-Host "  ntqr_evaluator.py runs in stub mode (all bounds = None)." -ForegroundColor Yellow
    Write-Host "  ACTION: Research correct ntqr API version before adding to requirements.txt." -ForegroundColor Yellow
    Write-Host "  See: docs/incidents/INCIDENT_2026-06-03_ntqr-api-incompatibility.md" -ForegroundColor Yellow
} elseif ($exitCode -eq 1) {
    Write-Host "ntqr SR31 check: FAIL (import failure)" -ForegroundColor Red
    exit 1
} else {
    Write-Host ("ntqr SR31 check: UNKNOWN exit {0}" -f $exitCode) -ForegroundColor Yellow
}
