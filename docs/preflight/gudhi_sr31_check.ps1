# docs/preflight/gudhi_sr31_check.ps1
# =====================================
# Standing Rule #31 smoke test for gudhi.
#
# gudhi 3.12.0 PASSES this test (confirmed 2026-06-03).
# Safe to add to requirements.txt once this script reports PASS.
#
# FIX vs original: the original used (Get-Job).ChildJobs[0].State -eq "Completed"
# which is unreliable because PS 5.1 job State = "Completed" for any job that
# ran to completion, regardless of Python exit code. The fix uses text-based
# detection: check whether the Python output contains the PASS marker.
#
# Run from repo root:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   & ".\docs\preflight\gudhi_sr31_check.ps1"

param([string]$PipArgs = "--break-system-packages")

Set-StrictMode -Version 1
$ErrorActionPreference = "Stop"

$pol = Get-ExecutionPolicy -Scope Process
if ($pol -eq "Restricted") {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
}

Write-Host "gudhi SR #31 smoke test" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Pip dry-run: confirm package resolves
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Step 1: pip dry-run ..."
$dryOut = (pip install gudhi $PipArgs --dry-run 2>&1) -join " "
if ($dryOut -match "Would install|already satisfied") {
    Write-Host ("  pip: {0}" -f ($dryOut | Select-String "Would install|already satisfied" |
        ForEach-Object { $_.Matches[0].Value + " (gudhi)" })) -ForegroundColor Green
} else {
    Write-Host "  WARNING: could not confirm gudhi on PyPI" -ForegroundColor Yellow
    Write-Host ("  dry-run output: {0}" -f $dryOut[0..200])
}

# ---------------------------------------------------------------------------
# 2. Install
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Step 2: pip install gudhi ..."
pip install gudhi $PipArgs -q 2>&1 |
    Select-String "Successfully installed|already satisfied" |
    ForEach-Object { Write-Host ("  {0}" -f $_) -ForegroundColor Green }

# ---------------------------------------------------------------------------
# 3. Write Python test to temp file and run it
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Step 3: API smoke test ..."

$tmpPy = [System.IO.Path]::GetTempFileName() + ".py"
$pyTest = @'
import sys
try:
    import gudhi
    print("gudhi SR31 PASS  version: " + str(gudhi.__version__))
    st = gudhi.SimplexTree()
    st.insert([0], filtration=0.0)
    st.insert([1], filtration=0.0)
    st.insert([2], filtration=0.0)
    st.insert([0, 1], filtration=0.3)
    st.insert([1, 2], filtration=0.5)
    st.make_filtration_non_decreasing()
    st.compute_persistence()
    ph = st.persistence()
    assert isinstance(ph, list), "persistence() must return a list"
    assert len(ph) >= 1, "expected at least 1 persistence interval"
    print("SimplexTree API: PASS")
    print("  persistence intervals: " + str(ph[:3]))
    sys.exit(0)
except ImportError as e:
    print("SR31 FAIL (ImportError): " + str(e), file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print("SR31 FAIL (" + type(e).__name__ + "): " + str(e), file=sys.stderr)
    sys.exit(2)
'@

[System.IO.File]::WriteAllText($tmpPy, $pyTest, [System.Text.UTF8Encoding]::new($false))

# Capture stdout and stderr separately, and the exit code
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

# Print captured output
if ($stdout) { Write-Host $stdout.TrimEnd() }
if ($stderr) { Write-Host $stderr.TrimEnd() -ForegroundColor Red }

# ---------------------------------------------------------------------------
# 4. Verdict: text-based detection (not job state)
# ---------------------------------------------------------------------------
Write-Host ""
if ($exitCode -eq 0 -and $stdout -match "SR31 PASS") {
    Write-Host "gudhi SR31 check: PASS -- safe to add gudhi to requirements.txt" -ForegroundColor Green
    Write-Host ""
    Write-Host "To add gudhi: append 'gudhi' to requirements.txt and commit." -ForegroundColor Cyan
} else {
    Write-Host "gudhi SR31 check: FAIL -- do NOT add gudhi to requirements.txt" -ForegroundColor Red
    Write-Host ("  Python exit code: {0}" -f $exitCode)
    exit 1
}
