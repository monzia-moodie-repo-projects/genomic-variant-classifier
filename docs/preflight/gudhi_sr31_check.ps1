# docs/preflight/gudhi_sr31_check.ps1
# =====================================
# Standing Rule #31 smoke test for gudhi.
#
# Run BEFORE adding gudhi to requirements.txt.
# Expected output (PASS):
#   gudhi SR31 PASS  version: X.Y.Z
#   SimplexTree API: PASS
#
# Usage (from repo root):
#   .\docs\preflight\gudhi_sr31_check.ps1

$job = Start-Job -ScriptBlock {
    # 1. Pip dry-run.
    $dryRun = & pip install gudhi --break-system-packages --dry-run 2>&1 |
              Select-String "Would install|already satisfied"
    if ($dryRun) {
        Write-Host "pip dry-run: $dryRun" -ForegroundColor Cyan
    } else {
        Write-Host "WARNING: gudhi not found on PyPI -- check package name" -ForegroundColor Yellow
    }

    # 2. Install.
    pip install gudhi --break-system-packages -q 2>&1 |
        Select-String "Successfully installed|already satisfied" | Write-Host

    # 3. Import test + SimplexTree API contract.
    $pyScript = @"
import sys
try:
    import gudhi
    print(f'gudhi SR31 PASS  version: {gudhi.__version__}')
    # Minimal API contract: build a SimplexTree, insert edges, compute persistence.
    st = gudhi.SimplexTree()
    st.insert([0], filtration=0.0)
    st.insert([1], filtration=0.0)
    st.insert([2], filtration=0.0)
    st.insert([0, 1], filtration=0.3)
    st.insert([1, 2], filtration=0.5)
    st.make_filtration_non_decreasing()
    st.compute_persistence()
    ph = st.persistence()
    assert isinstance(ph, list), 'persistence() must return a list'
    print('SimplexTree API: PASS')
    print(f'  persistence intervals: {ph}')
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

Wait-Job $job -Timeout 120 | Out-Null
Receive-Job $job
$exitCode = (Get-Job -Id $job.Id).ChildJobs[0].State
Remove-Job $job

if ($exitCode -eq "Completed") {
    Write-Host "`ngudhi SR31 check: PASS -- safe to add gudhi to requirements.txt" -ForegroundColor Green
} else {
    Write-Host "`ngudhi SR31 check: FAIL -- do NOT add gudhi to requirements.txt" -ForegroundColor Red
    exit 1
}
