# =============================================================================
# Run_Preflight_Local.ps1  -  Charter v1.1 gate G1 (LOCAL pre-launch validation)
# Run 17, genomic-variant-classifier. Runs BEFORE any Vast.ai instance create.
# Adapted from Run14_Preflight.ps1 (PM13, 2026-05-27). Exit: 0 green / 1 fail.
# =============================================================================
[CmdletBinding()]
# [G1-RUN17-ADAPTED] Run-15 -> Run-17 (2026-06-27): launch path, kan.py imodelsx, test floor 1485/1480, FinnGen R12+R13 required, Run17 postflight+plan, agent liveness.
param(
    [string]$RepoRoot     = "C:\Projects\genomic-variant-classifier",
    [string]$VenvName     = ".venv312",
    [string]$SshKey       = "C:\Users\monzi\.ssh\id_lambda_run8",
    [string]$ExpectedHead = "",
    # COLLECTED floor -- refreshed 2026-07-12 (was 1496; the suite now collects 1,823).
    # Paired with $minPass (~line 136). A collected-floor that lags the suite lets tests
    # VANISH silently: at 1496, three hundred tests could stop being collected -- deleted,
    # mis-named, or lost to a collection error -- and this gate would still say PASS.
    # RAISE BOTH WHENEVER YOU ADD TESTS.
    [int]$MinPytest       = 1815,
    # -SkipPytest is an ESCAPE HATCH ON A GATE THAT PROTECTS PAID COMPUTE. On 2026-07-06 the
    # project shipped 24 red tests to a rented GPU. Use it only to debug this script itself,
    # never to get a run out the door -- the gate exists precisely for the moment you are
    # tempted to skip it.
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

    Section "3. imodelsx patch in kan.py (Run-17: patch moved from launch script)"
    $launchPath = "scripts\launch_run17_baseline.sh"
    if (-not (Test-Path $launchPath)) { Fail "$launchPath missing" }
    else { Pass "launch script present: $launchPath" }
    # Run-17: the imodelsx patch moved from the launch script into kan.py (FA2 2026-06-27).
    # Verify the integration lives in kan.py now, not the launcher.
    $kanPath = "src\genomic_variant_classifier\models\kan.py"
    if (-not (Test-Path $kanPath)) { Fail "$kanPath missing" }
    else {
        $kc2 = Get-Content $kanPath -Raw
        foreach ($t in @('self._imodelsx_model.test_size','self._imodelsx_model.fit(X, y)')) {
            if ($kc2.Contains($t)) { Pass "kan.py imodelsx integration: $t" } else { Fail "kan.py missing imodelsx integration: $t" }
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
        # -------------------------------------------------------------------------------
        # PASS FLOOR -- refreshed 2026-07-12. It was 1485 and had gone STALE BY ~330 TESTS.
        # -------------------------------------------------------------------------------
        # The old comment recorded "1498 collected / 1491 passed" (2026-06-28) and set the
        # floor at 1485. The suite now collects 1,823 and passes 1,815. A floor of 1485 would
        # therefore have accepted the SILENT LOSS OF 330 TESTS and still reported PASS --
        # which is precisely the class of rotted guard this project keeps finding. A floor
        # that drifts below reality is not a floor; it is a rubber stamp.
        #
        # Measured 2026-07-12 (outputs/fullsuite_2026-07-12d.log and 12e, two identical runs):
        #     1,823 collected = 1,815 passed + 8 skipped, 0 failed, 0 errors
        # The suite is now HERMETIC and IDEMPOTENT (88af150), so this number is reproducible
        # rather than a function of what happens to sit on the developer's disk.
        #
        # Headroom of 10 below 1,815 absorbs legitimate environment-dependent skips (e.g. the
        # POSIX-symlink test that cannot run on Windows). It is NOT headroom for regressions:
        # any failure or error fails this gate outright, regardless of the count.
        #
        # WHEN YOU ADD TESTS, RAISE THIS. A floor left behind by a growing suite silently
        # stops guarding. It has already happened once.
        $minPass = 1805
        if ($nFail -gt 0) { Fail "pytest: $nFail failed/errored ($nPass passed, $nSkip skipped). Tail:`n$tail" }
        elseif ($nPass -ge $minPass -and $collected -ge $MinPytest) { Pass "pytest: $nPass passed, $nSkip skipped, 0 failed (>= $minPass passed, collected $collected >= $MinPytest)" }
        else { Fail "pytest: $nPass passed / $nSkip skipped / collected $collected (expected >= $minPass passed and >= $MinPytest collected). Tail:`n$tail" }
    }

    Section "7. Run 17 prep-input data files (raw; SCP'd up for on-VM prep)"
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
        } else { Fail "$($f.Path) MISSING (required for Run 17 prep)" }
    }
    # Run-17: FinnGen is WIRED + REQUIRED (dual-release R12+R13). Hard-fail on absence to match
    # the launcher's exit-7 guard (launch_run17_baseline.sh L181/L183) -- no silent-zero annotation.
    foreach ($fg in @("data\external\finngen\finnge_R12_annotated_variants_v1.gz","data\external\finngen\finngen_R13_annotated_variants_v0.gz")) {
        if (Test-Path $fg) {
            $fgmb = [math]::Round((Get-Item $fg).Length / 1MB, 1)
            Pass "FinnGen (required, dual-release) $fg -> $fgmb MB"
        } else { Fail "FinnGen REQUIRED file MISSING: $fg (launcher exits 7; dual-release needs both R12+R13)" }
    }
    # 1KGP AF parquet stays optional (still deferred per RUN_15 B.D1).
    foreach ($opt in @("data\external\1kg\1kg_phase3_af.parquet")) {
        if (Test-Path $opt) { Pass "(optional) $opt present" } else { Warn "(optional, deferred B.D1) $opt absent - features 0" }
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
    $postPath = "scripts\Run17_Postflight.ps1"
    if (Test-Path $postPath) {
        $pc = Get-Content $postPath -Raw
        if ($pc -match '(?m)exit 1') { Pass "Run17_Postflight.ps1: exit 1 on FAIL" } else { Fail "Run17_Postflight.ps1: no exit 1" }
        if ($pc.Contains('Test-ArtifactPresent')) { Pass "Run17_Postflight.ps1: Test-ArtifactPresent wired" } else { Fail "Run17_Postflight.ps1: Test-ArtifactPresent NOT wired" }
    } else { Warn "Run17_Postflight.ps1 NOT YET BUILT -- required before launch (adapt from Run15_Postflight.ps1)" }
    if (Test-Path "scripts\Test-ArtifactPresent.ps1") { Pass "Test-ArtifactPresent.ps1 present (L79)" } else { Fail "Test-ArtifactPresent.ps1 missing (L79)" }
    if (Test-Path "scripts\Vastai_Destroy_Confirmed.ps1") { Pass "Vastai_Destroy_Confirmed.ps1 present (L81)" } else { Fail "Vastai_Destroy_Confirmed.ps1 missing (L81)" }

    Section "13. RUN_17_PLAN.md decision completeness"
    $planPath = "docs\runs\RUN_17_PLAN.md"
    if (Test-Path $planPath) {
        $plan = Get-Content $planPath -Raw
        $dc = ([regex]::Matches($plan, [regex]::Escape('<DECISION>'))).Count
        if ($dc -le 1) { Pass "RUN_17_PLAN.md: $dc literal <DECISION> (<=1 gate-mention, OK)" } else { Fail "RUN_17_PLAN.md: $dc <DECISION> tokens (unfilled)" }
    } else { Fail "RUN_17_PLAN.md missing" }

    # -----------------------------------------------------------------------------------
    Section "13c. RUN_17_PLAN.md feature contract matches the CODE (added 2026-07-12)"
    # -----------------------------------------------------------------------------------
    # WHY THIS EXISTS. Until today the plan's primary hypothesis read "the expanded 91-feature
    # contract (88 + 3 FinnGen R13 columns)". That was true when it was written (2026-06-27,
    # fbdcf4c). On 2026-07-06 (80eb9c8) KEGG, COSMIC and the Nucleotide Transformer took the
    # contract to 97. The RUNBOOK was corrected (61c2b04). The PLAN was not.
    #
    # G1 checked the plan for unfilled <DECISION> markers and passed it -- so the gate green-lit
    # a paid run against a document that misstated the very contract under test. Checking a
    # document for completeness while never checking it for TRUTH is how every stale number in
    # this project survived: KNOWN_ZERO_DEFAULT frozen at a count that had moved, a "65 features"
    # comment while the contract held 97, a pytest floor 330 tests below the suite.
    #
    # A number written down once and never re-derived becomes a lie on a schedule. So don't
    # write it down -- DERIVE it, and fail loud when the document and the code disagree.
    if (Test-Path $planPath) {
        $codeCount = (& $venvPython -c "from genomic_variant_classifier.models.variant_ensemble import EXPECTED_TABULAR_FEATURE_COUNT as n; print(n)" 2>&1).Trim()
        if ($codeCount -notmatch '^\d+$') {
            Fail "13c: could not read EXPECTED_TABULAR_FEATURE_COUNT from the package: $codeCount"
        } else {
            # Every "<N>-feature" claim the plan makes must agree with the code.
            #
            # Strip markdown emphasis/code marks FIRST. On the very first run of this check the
            # plan had been "corrected" to read `**97**-feature` -- and the guard could not see
            # the 97 at all, because the digit was followed by '*' rather than '-'. The only
            # machine-readable count left in the document was the STALE one, quoted inside the
            # note explaining the correction. The check failed, and it was right to.
            #
            # A number a human can read but a machine cannot verify is a number that will rot.
            # Do not let formatting hide a claim from its own guard.
            $planFlat = $plan -replace '[*_`~]', ''
            $claims = [regex]::Matches($planFlat, '(\d+)[- ]feature') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
            if (-not $claims) {
                Warn "13c: RUN_17_PLAN.md states no feature count to check (code says $codeCount)"
            } elseif ($claims -contains $codeCount -and $claims.Count -eq 1) {
                Pass "13c: plan feature contract ($claims) == code EXPECTED_TABULAR_FEATURE_COUNT ($codeCount)"
            } else {
                Fail "13c: RUN_17_PLAN.md claims feature count(s) [$($claims -join ', ')] but the code says $codeCount. The plan Run 17 is gated against misstates the contract under test. Fix the plan (or the code) before spending money."
            }
        }
    }

    Section "13b. Agent-layer liveness (check_agents_active.py)"
    $agentChk = "scripts\check_agents_active.py"
    if (-not (Test-Path $agentChk)) { Fail "$agentChk missing (agent-liveness gate)" }
    else {
        $ao = & $venvPython $agentChk 2>&1
        if ($LASTEXITCODE -eq 0) { Pass "agent liveness: all agents registered + scheduled" }
        else { Fail "agent liveness FAILED (exit $LASTEXITCODE): $(($ao | Select-Object -Last 3) -join '; ')" }
    }

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