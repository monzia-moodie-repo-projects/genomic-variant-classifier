# =============================================================================
# Run_Preflight_Local.ps1  -  Charter v1.1 gate G1 (LOCAL pre-launch validation)
# Run 15, genomic-variant-classifier. Runs BEFORE any Vast.ai instance create.
# Adapted from Run14_Preflight.ps1 (PM13, 2026-05-27). Exit: 0 green / 1 fail.
# =============================================================================
[CmdletBinding()]
param(
    [string]$RepoRoot     = "C:\Projects\genomic-variant-classifier",
    [string]$VenvName     = ".venv312",
    [string]$SshKey       = "C:\Users\monzi\.ssh\id_lambda_run8",
    [string]$ExpectedHead = "",
    [int]$MinPytest       = 566,
    [switch]$SkipPytest,
    [switch]$SkipKanSmoke
)
$ErrorActionPreference = "Continue"
$script:Failed = @()
$script:Warned = @()
$script:Passed = @()
function Pass($msg) { $script:Passed += $msg; Write-Host "  PASS  $msg" -ForegroundColor Green }
function Fail($msg) { $script:Failed += $msg; Write-Host "  FAIL  $msg" -ForegroundColor Red }
function Warn($msg) { $script:Warned += $msg; Write-Host "  WARN  $msg" -ForegroundColor Yellow }
function Section($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

Section "0. Working directory and venv"
if (-not (Test-Path $RepoRoot)) { Fail "Repo not found at $RepoRoot"; exit 1 }
Push-Location $RepoRoot
try {
    Pass "Repo present at $RepoRoot"
    $venvPython = Join-Path $RepoRoot "$VenvName\Scripts\python.exe"
    if (Test-Path $venvPython) { Pass "venv python found: $venvPython" } else { Fail "venv python not found at $venvPython" }

    Section "1. Git state (tree clean + HEAD pushed to origin/main)"
    $h = (git rev-parse HEAD 2>$null).Trim()
    $o = (git rev-parse origin/main 2>$null).Trim()
    if ($h -eq $o) { Pass "HEAD == origin/main ($($h.Substring(0,7))) (pushed)" }
    else { Fail "HEAD ($($h.Substring(0,7))) != origin/main ($($o.Substring(0,7))) - push local commits" }
    if ($ExpectedHead -ne "") {
        if ($h.StartsWith($ExpectedHead)) { Pass "HEAD matches ExpectedHead $ExpectedHead" }
        else { Fail "HEAD $($h.Substring(0,7)) != ExpectedHead $ExpectedHead" }
    }
    $status = git status --porcelain
    if (-not $status) { Pass "Working tree clean" }
    else { Fail "Working tree has uncommitted changes:"; $status | ForEach-Object { Write-Host "        $_" -ForegroundColor Red } }

    Section "2. KAN attribute-injection fix in kan.py"
    $kanPath = "src\genomic_variant_classifier\models\kan.py"
    if (-not (Test-Path $kanPath)) { Fail "$kanPath missing" }
    else {
        $kc = Get-Content $kanPath -Raw
        $req = @('self._imodelsx_model.test_size = 0.2','self._imodelsx_model.random_state','self._imodelsx_model.shuffle = True','self._imodelsx_model.fit(X, y)')
        foreach ($r in $req) { if ($kc.Contains($r)) { Pass "kan.py contains: $r" } else { Fail "kan.py missing: $r" } }
        $ia = $kc.IndexOf('self._imodelsx_model.test_size'); $if = $kc.IndexOf('self._imodelsx_model.fit(X, y)')
        if ($ia -gt 0 -and $if -gt $ia) { Pass "kan.py: attribute injection BEFORE .fit()" } else { Fail "kan.py: injection order wrong/missing" }
    }

    Section "3. imodelsx patch in launch script"
    $launchPath = "scripts\launch_run11_vm.sh"
    if (-not (Test-Path $launchPath)) { Fail "$launchPath missing" }
    else {
        $lc = Get-Content $launchPath -Raw
        foreach ($t in @('imodelsx_patch','test_size=self.test_size','random_state=self.random_state','shuffle=self.shuffle')) {
            if ($lc.Contains($t)) { Pass "launch script contains: $t" } else { Fail "launch script missing: $t" }
        }
    }

    Section "4. requirements.txt sanity"
    if (-not (Test-Path "requirements.txt")) { Fail "requirements.txt missing" }
    else {
        $rq = Get-Content "requirements.txt" -Raw
        if ($rq -match '(?m)^imodelsx') { Pass "requirements pins imodelsx" } else { Fail "requirements missing imodelsx" }
        if ($rq -match '(?m)^pykan')    { Pass "requirements pins pykan (fallback)" } else { Warn "requirements missing pykan fallback" }
        if ($rq -match '(?m)^fastkan')  { Fail "requirements references fastkan (not on PyPI)" } else { Pass "fastkan absent (correct)" }
        if ($rq -match '(?m)^lightgbm') { Pass "requirements pins lightgbm" } else { Fail "requirements missing lightgbm" }
        if ($rq -match '(?m)^catboost') { Pass "requirements pins catboost" } else { Fail "requirements missing catboost" }
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
        if ($LASTEXITCODE -eq 0 -and $smokeOut -match 'SMOKE_OK.*backend=imodelsx') { Pass "KAN smoke: $smokeOut" }
        elseif ($LASTEXITCODE -eq 0 -and $smokeOut -match 'SMOKE_OK.*backend=pykan') { Warn "KAN smoke via pykan FALLBACK - investigate" }
        elseif ($LASTEXITCODE -eq 0 -and $smokeOut -match 'SMOKE_OK.*backend=mlp') { Fail "KAN smoke fell back to MLP (imodelsx+pykan failed)" }
        else { Fail "KAN smoke FAILED: $smokeOut" }
    }

    Section "6. Pytest suite (>= $MinPytest passed, 0 failed/errored)"
    if ($SkipPytest) { Warn "pytest skipped by flag" }
    else {
        Get-ChildItem -Recurse -Filter "__pycache__" -Directory -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch 'venv' } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        $po = & $venvPython -m pytest tests/ -q --no-header --tb=line 2>&1
        $tail = ($po | Select-Object -Last 6) -join "`n"
        $nFail = 0; $nPass = 0; $nSkip = 0
        if ($tail -match '(\d+) failed')  { $nFail = [int]$Matches[1] }
        if ($tail -match '(\d+) error')   { $nFail += [int]$Matches[1] }
        if ($tail -match '(\d+) passed')  { $nPass = [int]$Matches[1] }
        if ($tail -match '(\d+) skipped') { $nSkip = [int]$Matches[1] }
        $collected = $nPass + $nSkip
        $minPass = 560   # 566 collected minus 6 known-intentional skips (MC-dropout calibration TODOs pending Run 15; 1 coverage skip)
        if ($nFail -gt 0) { Fail "pytest: $nFail failed/errored ($nPass passed, $nSkip skipped). Tail:`n$tail" }
        elseif ($nPass -ge $minPass -and $collected -ge $MinPytest) { Pass "pytest: $nPass passed, $nSkip skipped, 0 failed (>= $minPass passed, collected $collected >= $MinPytest)" }
        else { Fail "pytest: $nPass passed / $nSkip skipped / collected $collected (expected >= $minPass passed and >= $MinPytest collected). Tail:`n$tail" }
    }

    Section "7. Run 15 prep-input data files (raw; SCP'd up for on-VM prep)"
    $reqData = @(
        @{ Path = "data\processed\clinvar_grch38.parquet";                       MinMB = 100 },
        @{ Path = "data\processed\gnomad_v4_exomes.parquet";                     MinMB = 30  },
        @{ Path = "data\external\dbnsfp\dbnsfp_clinvar_index.parquet";           MinMB = 25  },
        @{ Path = "data\external\alphamissense\AlphaMissense_hg38.tsv.gz";       MinMB = 300 },
        @{ Path = "data\external\spliceai\spliceai_index.parquet";               MinMB = 300 },
        @{ Path = "data\external\lovd\lovd_all_variants.parquet";                MinMB = 0.1   },
        @{ Path = "data\external\gnomad\gnomad.v4.1.constraint_metrics.tsv";     MinMB = 50  },
        @{ Path = "data\external\string\9606.protein.links.detailed.v12.0.txt.gz"; MinMB = 100 },
        @{ Path = "data\external\string\9606.protein.info.v12.0.txt.gz";         MinMB = 1   }
    )
    foreach ($f in $reqData) {
        if (Test-Path $f.Path) {
            $mb = [math]::Round((Get-Item $f.Path).Length / 1MB, 1)
            if ($mb -ge $f.MinMB) { Pass "$($f.Path) -> $mb MB" } else { Fail "$($f.Path) -> $mb MB (expected >= $($f.MinMB))" }
        } else { Fail "$($f.Path) MISSING (required for Run 15 prep)" }
    }
    foreach ($opt in @("data\external\finngen\finnge_R12_annotated_variants_v1.gz","data\external\1kg\1kg_phase3_af.parquet")) {
        if (Test-Path $opt) { Pass "(optional) $opt present" } else { Warn "(optional, deferred B.D1/B.D2) $opt absent - features 0" }
    }

    Section "8. SSH key + Vast.ai CLI"
    if (Test-Path $SshKey) { Pass "SSH key present: $SshKey" } else { Fail "SSH key missing: $SshKey" }
    $vv = & vastai --version 2>&1
    if ($LASTEXITCODE -eq 0) { Pass "vastai CLI: $vv" } else { Fail "vastai CLI not on PATH: $vv" }
    $vu = & vastai show user 2>&1
    if ($LASTEXITCODE -eq 0) { Pass "vastai API key works (show user OK)" } else { Fail "vastai show user FAILED (2FA on User Read?): $vu" }

    Section "9. Disk space"
    $drive = (Get-Item $RepoRoot).PSDrive
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    if ($freeGB -ge 20) { Pass "Free on $($drive.Name): $freeGB GB" } else { Warn "Free on $($drive.Name): $freeGB GB (< 20 GB recommended)" }

    Section "10. Pinned-package installability (SR #31)"
    $io = & $venvPython -c @"
import sys
errs = []
for pkg in ['lightgbm','xgboost','catboost','imodelsx','kan','sklearn','torch','numpy','pandas']:
    try:
        __import__(pkg); print(f'IMPORT_OK {pkg}')
    except Exception as e:
        print(f'IMPORT_FAIL {pkg} -> {e}'); errs.append(pkg)
sys.exit(0 if not errs else 1)
"@ 2>&1
    foreach ($ln in $io) {
        if ($ln -match '^IMPORT_OK (.+)') { Pass "import $($Matches[1]) OK" }
        elseif ($ln -match '^IMPORT_FAIL (.+)') { Fail "import $($Matches[1]) FAILED" }
    }

    Section "11. Run 15 code wiring (C.8 / C.9 / GNN)"
    $evalPath = "scripts\run_phase2_eval.py"
    if (Test-Path $evalPath) {
        $ec = Get-Content $evalPath -Raw
        if ($ec.Contains('"--unseen-gene-holdout"')) { Pass "eval: --unseen-gene-holdout flag (C.9)" } else { Fail "eval: --unseen-gene-holdout MISSING (C.9)" }
        if ($ec.Contains('unseen_gene_holdout_split')) { Pass "eval: unseen_gene_holdout_split wired (C.9)" } else { Fail "eval: unseen_gene_holdout_split MISSING (C.9)" }
        if ($ec.Contains('meta_train.parquet')) { Pass "eval: meta_train.parquet referenced (C.8)" } else { Fail "eval: meta_train.parquet NOT referenced (C.8)" }
    } else { Fail "$evalPath missing" }
    if (Test-Path $launchPath) {
        $lc2 = Get-Content $launchPath -Raw
        if ($lc2.Contains('--unseen-gene-holdout')) { Pass "launch: --unseen-gene-holdout wired (C.9)" } else { Fail "launch: --unseen-gene-holdout MISSING (C.9)" }
        if ($lc2.Contains('--string-db')) { Pass "launch: --string-db GNN wired" } else { Fail "launch: --string-db MISSING" }
    }
    if (Test-Path "tests\unit\test_patch_6b_meta_train.py") { Pass "C.8 regression test present" } else { Warn "C.8 regression test missing" }

    Section "12. Postflight / destroy gate infrastructure (plan L79-81)"
    $postPath = "scripts\Run15_Postflight.ps1"
    if (Test-Path $postPath) {
        $pc = Get-Content $postPath -Raw
        if ($pc -match '(?m)exit 1') { Pass "Run15_Postflight.ps1: exit 1 on FAIL (L80)" } else { Fail "Run15_Postflight.ps1: no exit 1 (L80)" }
        if ($pc.Contains('Test-ArtifactPresent')) { Pass "Run15_Postflight.ps1: Test-ArtifactPresent wired (L79)" } else { Fail "Run15_Postflight.ps1: Test-ArtifactPresent NOT wired (L79)" }
    } else { Fail "Run15_Postflight.ps1 missing (L80)" }
    if (Test-Path "scripts\Test-ArtifactPresent.ps1") { Pass "Test-ArtifactPresent.ps1 present (L79)" } else { Fail "Test-ArtifactPresent.ps1 missing (L79)" }
    if (Test-Path "scripts\Vastai_Destroy_Confirmed.ps1") { Pass "Vastai_Destroy_Confirmed.ps1 present (L81)" } else { Fail "Vastai_Destroy_Confirmed.ps1 missing (L81)" }

    Section "13. RUN_15_PLAN.md decision completeness (plan L77)"
    $planPath = "docs\runs\RUN_15_PLAN.md"
    if (Test-Path $planPath) {
        $plan = Get-Content $planPath -Raw
        $dc = ([regex]::Matches($plan, [regex]::Escape('<DECISION>'))).Count
        if ($dc -le 1) { Pass "RUN_15_PLAN.md: $dc literal <DECISION> (<=1 gate-mention, OK)" } else { Fail "RUN_15_PLAN.md: $dc <DECISION> tokens (unfilled)" }
    } else { Fail "RUN_15_PLAN.md missing" }

    Section "14. Correctness harness (stages 1-5; gates correctness before AUROC)"
    $harnessPy = @'
import json, re as _re
from genomic_variant_classifier.agent_layer.harness import (
    build_reference_slice, run_correctness_harness, KNOWN_ZERO_DEFAULT,
)
rep = run_correctness_harness(build_reference_slice())
non5 = [f for f in rep.failures if not f.startswith('[stage 5]')]
flagged = set()
for f in rep.failures:
    m = _re.search(r"feature '([^']+)'", f)
    if m:
        flagged.add(m.group(1))
unexpected = sorted(flagged - set(KNOWN_ZERO_DEFAULT))
print('HARNESS_JSON ' + json.dumps({
    'stages': list(rep.stages_run),
    'non_stage5': non5,
    'unexpected_zero': unexpected,
    'allowlist_hits': sorted(flagged & set(KNOWN_ZERO_DEFAULT)),
}))
'@
    $harnessTmp = Join-Path $env:TEMP ("g1_harness_{0}.py" -f $PID)
    [System.IO.File]::WriteAllText($harnessTmp, $harnessPy, (New-Object System.Text.UTF8Encoding($false)))
    try {
        $hraw = & $venvPython $harnessTmp 2>&1
    } finally {
        Remove-Item $harnessTmp -Force -ErrorAction SilentlyContinue
    }
    $hline = $hraw | Where-Object { $_ -match '^HARNESS_JSON ' } | Select-Object -First 1
    if (-not $hline) {
        Fail "Correctness harness did not emit a verdict. Output:`n$($hraw -join "`n")"
    } else {
        $hj = ($hline -replace '^HARNESS_JSON ', '') | ConvertFrom-Json
        if ($hj.stages.Count -ne 5) { Fail "Harness did not run all 5 stages (ran: $($hj.stages -join ','))" }
        else { Pass "Harness ran all 5 stages" }
        if ($hj.non_stage5.Count -gt 0) { Fail "Harness stages 1-4 FAILED: $($hj.non_stage5 -join '; ')" }
        else { Pass "Harness stages 1-4 (smoke/config/sanity/determinism) green" }
        if ($hj.unexpected_zero.Count -gt 0) { Fail "Stage 5 flagged NON-allowlist silent-zero(s): $($hj.unexpected_zero -join ', ')" }
        else { Pass "Stage 5: no silent-zeros outside known dead-connector allowlist" }
        if ($hj.allowlist_hits.Count -gt 0) { Warn "Stage 5: $($hj.allowlist_hits.Count) known dead-connector default(s) still zero (expected; 2026-04-30 audit)" }
    }

} finally { Pop-Location }

Section "PRE-FLIGHT SUMMARY (G1 local)"
Write-Host ("Passed:  {0}" -f $script:Passed.Count) -ForegroundColor Green
Write-Host ("Warned:  {0}" -f $script:Warned.Count) -ForegroundColor Yellow
Write-Host ("Failed:  {0}" -f $script:Failed.Count) -ForegroundColor Red
if ($script:Failed.Count -gt 0) {
    Write-Host "`nG1 FAIL - DO NOT LAUNCH. Failures:" -ForegroundColor Red
    $script:Failed | ForEach-Object { Write-Host "  * $_" -ForegroundColor Red }
    exit 1
} else {
    Write-Host "`nG1 PASS - local checks green. Proceed to instance search + G2 on VM." -ForegroundColor Green
    if ($script:Warned.Count -gt 0) { Write-Host "Warnings:" -ForegroundColor Yellow; $script:Warned | ForEach-Object { Write-Host "  * $_" -ForegroundColor Yellow } }
    exit 0
}