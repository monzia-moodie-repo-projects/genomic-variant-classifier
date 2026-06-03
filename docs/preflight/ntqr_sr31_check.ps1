# docs/preflight/ntqr_sr31_check.ps1
# ====================================
# Standing Rule #31 smoke test for ntqr.
#
# Run BEFORE adding ntqr to requirements.txt.
# Expected output (PASS):
#   ntqr SR31 PASS  version: X.Y.Z
#   Evaluator2 API: PASS
#
# Usage (from repo root):
#   .\docs\preflight\ntqr_sr31_check.ps1

$job = Start-Job -ScriptBlock {
    # 1. Pip dry-run: confirm the package resolves from PyPI.
    $dryRun = & pip install ntqr --break-system-packages --dry-run 2>&1 |
              Select-String "Would install|already satisfied"
    if ($dryRun) {
        Write-Host "pip dry-run: $dryRun" -ForegroundColor Cyan
    } else {
        Write-Host "WARNING: ntqr not found on PyPI -- check package name" -ForegroundColor Yellow
    }

    # 2. Actually install (if not already present).
    pip install ntqr --break-system-packages -q 2>&1 |
        Select-String "Successfully installed|already satisfied" | Write-Host

    # 3. Import test + API contract verification.
    $pyScript = @"
import sys
try:
    import ntqr
    from ntqr.r2 import Evaluator2
    print(f'ntqr SR31 PASS  version: {ntqr.__version__}')
    # Minimal API contract: instantiate and call classifier_accuracy_bounds
    ev = Evaluator2(n_0=100, n_1=50)
    bounds = ev.classifier_accuracy_bounds(q_00=90, q_01=10, q_10=5, q_11=45)
    assert 0 in bounds and 1 in bounds, 'bounds dict must have keys 0 and 1'
    lo0, hi0 = bounds[0]
    lo1, hi1 = bounds[1]
    assert 0.0 <= lo0 <= hi0 <= 1.0, f'benign bounds out of [0,1]: {lo0}, {hi0}'
    assert 0.0 <= lo1 <= hi1 <= 1.0, f'pathogen bounds out of [0,1]: {lo1}, {hi1}'
    print('Evaluator2 API: PASS')
    print(f'  benign accuracy    : [{lo0:.4f}, {hi0:.4f}]')
    print(f'  pathogen accuracy  : [{lo1:.4f}, {hi1:.4f}]')
    sys.exit(0)
except ImportError as e:
    print(f'SR31 FAIL (ImportError): {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'SR31 FAIL ({type(e).__name__}): {e}', file=sys.stderr)
    sys.exit(2)
"@
    $tmp = [System.IO.Path]::GetTempFileName() + ".py"
    [System.IO.File]::WriteAllText($tmp, $pyScript, [System.Text.UTF8Encoding]::new($false))
    python $tmp
    $exit = $LASTEXITCODE
    Remove-Item $tmp -ErrorAction SilentlyContinue
    exit $exit
}

Wait-Job $job -Timeout 90 | Out-Null
Receive-Job $job
$exitCode = (Get-Job -Id $job.Id).ChildJobs[0].State
Remove-Job $job

if ($exitCode -eq "Completed") {
    Write-Host "`nntqr SR31 check: PASS -- safe to add ntqr to requirements.txt" -ForegroundColor Green
} else {
    Write-Host "`nntqr SR31 check: FAIL -- do NOT add ntqr to requirements.txt" -ForegroundColor Red
    exit 1
}
