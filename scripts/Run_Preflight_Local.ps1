# =============================================================================
# Run_Preflight_Local.ps1  -  Charter v1.1 gate G1 (LOCAL pre-launch validation)
# Run 17, genomic-variant-classifier. Runs BEFORE any Vast.ai instance create.
# Adapted from Run14_Preflight.ps1 (PM13, 2026-05-27). Exit: 0 green / 1 fail.
# =============================================================================
[CmdletBinding()]
# [G1-RUN17-ADAPTED] Run-15 -> Run-17 (2026-06-27): launch path, kan.py imodelsx, FinnGen R12+R13 required, Run17 postflight+plan, agent liveness.
# THERE ARE NO TEST FLOORS. The suite size lives in ONE file: tests\EXPECTED_SUITE_SIZE, and
# it is enforced BY PYTEST ITSELF (tests/conftest.py) under --assert-suite-size, which Section
# 6 below passes. Continuous Integration passes the same flag and reads the same file.
#
# This header has now been wrong TWICE about this, which is the point:
#   * It once carried a THIRD copy of the floor ("test floor 1485/1480") -- stale by ~370
#     tests and contradicting both live values. A number written down in three places is wrong
#     in at least two.
#   * It was then corrected on 2026-07-13 to say the floors "live in ONE place only --
#     $MinPytest and $minPass" -- and within the same day those two variables were DELETED and
#     this line became a lie again, describing parameters that no longer exist and pointing at
#     a line number that no longer holds them.
#
# That is root pattern (a) -- a claim written down once and never re-derived -- committed in
# the very change that abolished it. DO NOT RESTATE THE SUITE SIZE HERE, OR ANYWHERE ELSE.
# One number. One file. Enforced by a gate, not by a comment. See roadmap 6.14.
param(
    [string]$RepoRoot     = "C:\Projects\genomic-variant-classifier",
    [string]$VenvName     = ".venv312",
    [string]$SshKey       = "C:\Users\monzi\.ssh\id_lambda_run8",
    [string]$ExpectedHead = "",
    # NOTE: $MinPytest and $minPass ARE GONE (2026-07-13, roadmap 6.14).
    #
    # They were hard-coded pytest floors, and they ROTTED FIVE TIMES IN TWO DAYS:
    #     1485 -> 1805 -> 1842 -> 1850 -> 1853
    # Every single time, the number sat directly beneath an emphatic, all-capitals comment
    # ordering the next person to raise it -- written by the person who then failed to raise
    # it. At 1485 against a suite passing 1,815, THREE HUNDRED AND THIRTY tests could have
    # silently vanished and this gate would still have reported PASS.
    #
    # A COMMENT DOES NOT ENFORCE ITSELF. So the comment has been replaced by a GATE: the suite
    # size now lives in ONE file, `tests/EXPECTED_SUITE_SIZE`, and `pytest --assert-suite-size`
    # ABORTS if the collected count disagrees with it -- in EITHER direction. Adding a test
    # turns the suite RED until the number is bumped. Forgetting is no longer possible, because
    # forgetting FAILS. Same fail-loud pattern as EXPECTED_TABULAR_FEATURE_COUNT guarding
    # TABULAR_FEATURES. Section 6 below passes that flag; the number is not restated here, and
    # must never be.
    #
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

    Section "6. Pytest suite (full tree; SUITE-SIZE RATCHET enforced, 0 failed/errored)"
    if ($SkipPytest) { Warn "pytest skipped by flag" }
    else {
        Get-ChildItem -Recurse -Filter "__pycache__" -Directory -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch 'venv' } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

        # --assert-suite-size (2026-07-13, roadmap 6.14): the suite ABORTS if the number of
        # COLLECTED tests disagrees with tests/EXPECTED_SUITE_SIZE, in EITHER direction --
        # fewer means tests VANISHED, more means tests were added and the ratchet was not
        # bumped. The number lives in ONE file and is NOT restated here. Continuous Integration
        # passes the same flag and reads the same file.
        $po = & $venvPython -m pytest tests/ -q --no-header --tb=line --assert-suite-size 2>&1
        $rc = $LASTEXITCODE
        $tail = ($po | Select-Object -Last 10) -join "`n"
        $nFail = 0; $nPass = 0; $nSkip = 0; $nXfail = 0
        if ($tail -match '(\d+) failed')  { $nFail = [int]$Matches[1] }
        if ($tail -match '(\d+) error')   { $nFail += [int]$Matches[1] }
        if ($tail -match '(\d+) passed')  { $nPass = [int]$Matches[1] }
        if ($tail -match '(\d+) skipped') { $nSkip = [int]$Matches[1] }
        if ($tail -match '(\d+) xfailed') { $nXfail = [int]$Matches[1] }
        $collected = $nPass + $nSkip + $nXfail

        # Read the ratchet purely to REPORT it. It is ENFORCED inside pytest (conftest.py), not
        # here -- so this gate cannot drift away from Continuous Integration's view of it.
        $ratchetFile = Join-Path $RepoRoot "tests\EXPECTED_SUITE_SIZE"
        $expectedSize = "?"
        if (Test-Path $ratchetFile) {
            $m = Get-Content $ratchetFile | Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -First 1
            if ($m) { $expectedSize = $m.Trim() }
        } else {
            Fail "tests\EXPECTED_SUITE_SIZE is MISSING. The suite-size ratchet cannot run, and a missing guard must never degrade to a silent pass (roadmap 6.14)."
        }
        # -------------------------------------------------------------------------------
        # THE FLOORS ARE GONE. THE RATCHET REPLACES THEM. (2026-07-13, roadmap 6.14)
        # -------------------------------------------------------------------------------
        # This block used to hold TWO hand-maintained numbers -- $MinPytest (collected floor)
        # and $minPass (pass floor). They rotted FIVE TIMES IN TWO DAYS:
        #
        #     1485  ->  1805  ->  1842  ->  1850  ->  1853
        #
        # every single time beneath an emphatic, all-capitals comment ordering the next person
        # to raise them, written by the person who then failed to raise them. At 1485 against a
        # suite passing 1,815, THREE HUNDRED AND THIRTY tests could have silently vanished and
        # this gate would still have said PASS.
        #
        # There is now ONE number, in ONE file -- tests\EXPECTED_SUITE_SIZE -- and it is
        # enforced BY PYTEST ITSELF (tests/conftest.py) under --assert-suite-size, which this
        # gate passes above. If the collected count disagrees in EITHER direction, pytest
        # ABORTS and this section fails on the non-zero exit code. Adding a test turns the
        # suite RED until the number is bumped: forgetting is no longer possible, because
        # forgetting FAILS.
        #
        # Deliberately, the number is NOT restated here. Restating it is precisely how it came
        # to disagree with itself in four places. This gate READS it, to report it, and never
        # to decide with it.
        #
        # Note also: pass/skip counts are ENVIRONMENT-DEPENDENT (Windows 1863p/7s vs Linux CI
        # 1856p/13s/1xf), which is why the old $minPass could never have been correct in both.
        # The COLLECTED count is environment-independent -- 1,870 on both -- so that is what
        # the ratchet asserts.
        # -------------------------------------------------------------------------------
        if ($rc -ne 0 -and $nFail -eq 0) {
            # Non-zero exit with no test failures = the ratchet (or another usage error) fired.
            Fail "pytest exited $rc with 0 test failures -- THE SUITE-SIZE RATCHET LIKELY FIRED (expected $expectedSize collected, got $collected). Read the message below and DO NOT lower the number to make it pass.`n$tail"
        }
        elseif ($nFail -gt 0) {
            Fail "pytest: $nFail failed/errored ($nPass passed, $nSkip skipped). Tail:`n$tail"
        }
        else {
            Pass "pytest: $nPass passed, $nSkip skipped, 0 failed; suite-size ratchet OK (collected $collected == EXPECTED_SUITE_SIZE $expectedSize)"
        }
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
            # AUTHORITY: an explicit machine-readable assertion, NOT scraped prose.
            #
            # Scraping the narrative failed twice, in opposite directions:
            #   1. The hypothesis was written `**97**-feature`. The pattern wanted a digit
            #      followed by '-' or ' '; after 97 came '*'. The guard could not see the 97,
            #      and the only number it COULD see was the stale one quoted in a footnote.
            #   2. Stripping markdown to fix (1) then collapsed `finngen_r13_*` feature-importance
            #      into "finngenr13 feature-importance" -- and the guard read THIRTEEN features,
            #      failing a document that was correct.
            #
            # A contract must be ASSERTED, not INFERRED. The plan carries
            #     <!-- FEATURE_CONTRACT: 97 -->
            # and this compares that single number against the package. Prose is for humans;
            # this marker is for the gate. Neither can silently drift from the code again.
            $m = [regex]::Match($plan, '<!--\s*FEATURE_CONTRACT:\s*(\d+)\s*-->')
            if (-not $m.Success) {
                Fail "13c: RUN_17_PLAN.md has no '<!-- FEATURE_CONTRACT: N -->' marker. The plan Run 17 is gated against makes no checkable claim about the contract under test. Add it (code says $codeCount)."
            } elseif ($m.Groups[1].Value -ne $codeCount) {
                Fail "13c: RUN_17_PLAN.md asserts FEATURE_CONTRACT=$($m.Groups[1].Value) but the code says $codeCount (EXPECTED_TABULAR_FEATURE_COUNT). The plan misstates the contract under test. Fix the plan (or the code) BEFORE spending money."
            } else {
                Pass "13c: plan FEATURE_CONTRACT ($($m.Groups[1].Value)) == code EXPECTED_TABULAR_FEATURE_COUNT ($codeCount)"
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