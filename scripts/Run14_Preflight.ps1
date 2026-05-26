# =============================================================================
# Run14_Preflight.ps1
# =============================================================================
# Purpose:  Exhaustive local pre-launch validation for Run 14 of
#           genomic-variant-classifier. Runs BEFORE any Vast.ai instance
#           is created. Designed to fail fast and loud if anything is off.
#
# Author:   Claude, for Monzia Moodie
# Created:  2026-05-26
# Target commit: bf2f665
#
# Exit codes:
#   0 = all green, safe to launch
#   1 = one or more checks failed, DO NOT launch
#
# Standing rule honored: Pre-flight before every run; never pay for GPU on
# uncertainty.
# =============================================================================

[CmdletBinding()]
param(
    [string]$RepoRoot   = "C:\Projects\genomic-variant-classifier",
    [string]$ExpectedHead = "bf2f665",
    [string]$VenvName   = ".venv312",
    [string]$SshKey     = "C:\Users\monzi\.ssh\id_lambda_run8",
    [switch]$SkipPytest,
    [switch]$SkipKanSmoke
)

$ErrorActionPreference = "Continue"
$script:Failed = @()
$script:Warned = @()
$script:Passed = @()

function Pass($msg) { $script:Passed += $msg; Write-Host "  PASS  $msg" -ForegroundColor Green }
function Fail($msg) { $script:Failed += $msg; Write-Host "  FAIL  $msg" -ForegroundColor Red   }
function Warn($msg) { $script:Warned += $msg; Write-Host "  WARN  $msg" -ForegroundColor Yellow }
function Section($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

Section "0. Working directory and venv"
if (-not (Test-Path $RepoRoot)) { Fail "Repo not found at $RepoRoot"; exit 1 }
Push-Location $RepoRoot
try {
    Pass "Repo present at $RepoRoot"
    $venvPython = Join-Path $RepoRoot "$VenvName\Scripts\python.exe"
    if (Test-Path $venvPython) { Pass "venv python found: $venvPython" }
    else { Fail "venv python not found at $venvPython" }

    Section "1. Git state vs expected commit ($ExpectedHead)"
    $head = (git rev-parse HEAD 2>$null).Substring(0,7)
    if ($head -eq $ExpectedHead) { Pass "HEAD = $head" }
    else { Fail "HEAD = $head (expected $ExpectedHead)" }

    $origin = (git rev-parse origin/main 2>$null).Substring(0,7)
    if ($origin -eq $ExpectedHead) { Pass "origin/main = $origin (matches HEAD)" }
    else { Fail "origin/main = $origin (expected $ExpectedHead). Push your local commits." }

    $status = git status --porcelain
    if (-not $status) { Pass "Working tree clean (no uncommitted changes)" }
    else {
        Warn "Working tree has uncommitted changes:"
        $status | ForEach-Object { Write-Host "        $_" -ForegroundColor Yellow }
    }

    Section "2. KAN attribute-injection fix in kan.py"
    $kanPath = "src\genomic_variant_classifier\models\kan.py"
    if (-not (Test-Path $kanPath)) { Fail "$kanPath missing" }
    else {
        $kanContent = Get-Content $kanPath -Raw
        $required = @(
            'self._imodelsx_model.test_size = 0.2',
            'self._imodelsx_model.random_state',
            'self._imodelsx_model.shuffle = True',
            'self._imodelsx_model.fit(X, y)'
        )
        foreach ($r in $required) {
            if ($kanContent.Contains($r)) { Pass "kan.py contains: $r" }
            else { Fail "kan.py missing required line: $r" }
        }
        # Verify attributes come BEFORE .fit() — order matters
        $idxAttr = $kanContent.IndexOf('self._imodelsx_model.test_size')
        $idxFit  = $kanContent.IndexOf('self._imodelsx_model.fit(X, y)')
        if ($idxAttr -gt 0 -and $idxFit -gt $idxAttr) { Pass "kan.py: attribute injection comes BEFORE .fit() (correct order)" }
        else { Fail "kan.py: attribute injection order is wrong or missing" }
    }

    Section "3. imodelsx package patch in launch script"
    $launchPath = "scripts\launch_run11_vm.sh"
    if (-not (Test-Path $launchPath)) { Fail "$launchPath missing" }
    else {
        $launchContent = Get-Content $launchPath -Raw
        $patchTokens = @(
            'imodelsx_patch',
            'test_size=self.test_size',
            'random_state=self.random_state',
            'shuffle=self.shuffle'
        )
        foreach ($t in $patchTokens) {
            if ($launchContent.Contains($t)) { Pass "launch script contains: $t" }
            else { Fail "launch script missing: $t" }
        }
        # Confirm patch runs AFTER pip install (it patches the installed package)
        $idxPip   = $launchContent.IndexOf('pip install')
        $idxPatch = $launchContent.IndexOf('imodelsx_patch')
        if ($idxPip -gt 0 -and $idxPatch -gt $idxPip) { Pass "imodelsx patch runs AFTER pip install (correct order)" }
        else { Warn "Could not verify imodelsx patch comes after pip install — inspect manually" }
    }

    Section "4. requirements.txt sanity"
    $reqPath = "requirements.txt"
    if (-not (Test-Path $reqPath)) { Fail "$reqPath missing" }
    else {
        $req = Get-Content $reqPath -Raw
        if ($req -match '(?m)^imodelsx')   { Pass "requirements.txt pins imodelsx" }      else { Fail "requirements.txt missing imodelsx" }
        if ($req -match '(?m)^pykan')      { Pass "requirements.txt pins pykan (fallback)" } else { Warn "requirements.txt missing pykan fallback" }
        if ($req -match '(?m)^fastkan')    { Fail "requirements.txt still references fastkan (NOT on PyPI). Remove." } else { Pass "fastkan correctly absent from requirements.txt" }
        if ($req -match '(?m)^lightgbm')   { Pass "requirements.txt pins lightgbm" }       else { Fail "requirements.txt missing lightgbm" }
        if ($req -match '(?m)^catboost')   { Pass "requirements.txt pins catboost" }       else { Fail "requirements.txt missing catboost" }
    }

    Section "5. Local KAN smoke test"
    if ($SkipKanSmoke) { Warn "KAN smoke test skipped by flag" }
    else {
        $smokeOut = & $venvPython -c @"
import numpy as np
from genomic_variant_classifier.models.kan import KANClassifier
k = KANClassifier()
X = np.random.randn(200, 10).astype(np.float32)
y = np.random.randint(0, 2, 200)
k.fit(X, y)
p = k.predict_proba(X)
backend = getattr(k, '_backend_used', 'unknown')
print(f'SMOKE_OK shape={p.shape} backend={backend}')
"@ 2>&1
        if ($LASTEXITCODE -eq 0 -and $smokeOut -match 'SMOKE_OK.*backend=imodelsx') {
            Pass "KAN smoke test: $smokeOut"
        } elseif ($LASTEXITCODE -eq 0 -and $smokeOut -match 'SMOKE_OK.*backend=pykan') {
            Warn "KAN smoke test passed via pykan FALLBACK (not imodelsx). Investigate before launch."
        } elseif ($LASTEXITCODE -eq 0 -and $smokeOut -match 'SMOKE_OK.*backend=mlp') {
            Fail "KAN smoke test fell back to MLP — both imodelsx AND pykan failed locally."
        } else {
            Fail "KAN smoke test FAILED: $smokeOut"
        }
    }

    Section "6. Pytest suite (501 expected green)"
    if ($SkipPytest) { Warn "pytest skipped by flag" }
    else {
        # Clear stale bytecode first (standing rule)
        Get-ChildItem -Recurse -Filter "__pycache__" -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        $pytestOut = & $venvPython -m pytest tests/ -q --no-header --tb=no 2>&1
        $lastLines = ($pytestOut | Select-Object -Last 5) -join "`n"
        if ($lastLines -match '(\d+) passed') {
            $passedCount = [int]$Matches[1]
            if ($passedCount -ge 501) { Pass "pytest: $passedCount passed (>= 501 expected)" }
            else { Fail "pytest: only $passedCount passed (< 501 expected). Tail:`n$lastLines" }
        } else {
            Fail "pytest output unparseable. Tail:`n$lastLines"
        }
    }

    Section "7. Training data files present locally"
    $expectedData = @(
        @{ Path = "data\processed\clinvar_grch38.parquet";                     MinMB = 100 },
        @{ Path = "data\processed\gnomad_v4_exomes.parquet";                   MinMB = 30  },
        @{ Path = "data\external\dbnsfp\dbnsfp_clinvar_index.parquet";         MinMB = 25  },
        @{ Path = "data\external\alphamissense\AlphaMissense_hg38.tsv.gz";     MinMB = 300 },
        @{ Path = "data\external\spliceai\spliceai_index.parquet";             MinMB = 300 },
        @{ Path = "data\external\string\9606.protein.links.detailed.v12.0.txt.gz"; MinMB = 100 },
        @{ Path = "data\external\gnomad\gnomad.v4.1.constraint_metrics.tsv";   MinMB = 50  }
    )
    foreach ($f in $expectedData) {
        if (Test-Path $f.Path) {
            $sizeMB = [math]::Round((Get-Item $f.Path).Length / 1MB, 1)
            if ($sizeMB -ge $f.MinMB) { Pass "$($f.Path) -> $sizeMB MB" }
            else { Fail "$($f.Path) -> $sizeMB MB (expected >= $($f.MinMB) MB)" }
        } else {
            Fail "$($f.Path) MISSING"
        }
    }

    Section "8. SSH key + Vast.ai CLI"
    if (Test-Path $SshKey) { Pass "SSH key present: $SshKey" }
    else { Fail "SSH key missing: $SshKey" }

    $vastVersion = & vastai --version 2>&1
    if ($LASTEXITCODE -eq 0) { Pass "vastai CLI: $vastVersion" }
    else { Fail "vastai CLI not on PATH or broken: $vastVersion" }

    # Verify API key works and 2FA is OFF for User Read
    $vastShow = & vastai show user 2>&1
    if ($LASTEXITCODE -eq 0) { Pass "vastai API key works (show user OK)" }
    else { Fail "vastai show user FAILED (likely 2FA on User Read). Disable all 2FA toggles. Output: $vastShow" }

    Section "9. Disk space (output dir for results download)"
    $drive = (Get-Item $RepoRoot).PSDrive
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    if ($freeGB -ge 20) { Pass "Free space on $($drive.Name): $freeGB GB" }
    else { Warn "Free space on $($drive.Name): $freeGB GB (< 20 GB recommended for run9_fresh-style output dump)" }

    Section "10. Pinned-package installability self-check (SR #31)"
    # Verify lightgbm + imodelsx + catboost actually import in the local venv
    $importOut = & $venvPython -c @"
import sys
errors = []
for pkg in ['lightgbm', 'xgboost', 'catboost', 'imodelsx', 'pykan', 'sklearn', 'torch', 'numpy', 'pandas']:
    try:
        __import__(pkg)
        print(f'IMPORT_OK {pkg}')
    except ImportError as e:
        print(f'IMPORT_FAIL {pkg} -> {e}')
        errors.append(pkg)
sys.exit(0 if not errors else 1)
"@ 2>&1
    foreach ($line in $importOut) {
        if ($line -match '^IMPORT_OK (.+)') { Pass "import $($Matches[1]) OK" }
        elseif ($line -match '^IMPORT_FAIL (.+)') { Fail "import $($Matches[1]) FAILED" }
    }

} finally {
    Pop-Location
}

Section "PRE-FLIGHT SUMMARY"
Write-Host ("Passed:  {0}" -f $script:Passed.Count) -ForegroundColor Green
Write-Host ("Warned:  {0}" -f $script:Warned.Count) -ForegroundColor Yellow
Write-Host ("Failed:  {0}" -f $script:Failed.Count) -ForegroundColor Red

if ($script:Failed.Count -gt 0) {
    Write-Host "`nDO NOT LAUNCH. Failures:" -ForegroundColor Red
    $script:Failed | ForEach-Object { Write-Host "  * $_" -ForegroundColor Red }
    exit 1
} else {
    Write-Host "`nAll critical checks passed. Safe to proceed to instance search." -ForegroundColor Green
    if ($script:Warned.Count -gt 0) {
        Write-Host "Address warnings if possible:" -ForegroundColor Yellow
        $script:Warned | ForEach-Object { Write-Host "  * $_" -ForegroundColor Yellow }
    }
    exit 0
}
