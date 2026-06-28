#!/usr/bin/env python3
"""G1 preflight adaptation: Run_Preflight_Local.ps1 from Run-15 to Run-17.

Anchor-based, idempotent (marker sentinel), BOM-free, LF-preserving on the edited
regions. Validates each anchor occurs EXACTLY ONCE before applying. Writes a .bak.

Edits (all anchored against verbatim text read from the live file 2026-06-27):
  1. Header  '# Run 15, genomic-variant-classifier' -> '# Run 17, ...'
  2. Param   '[int]$MinPytest       = 566,'          -> '= 1485,'
  3. Sec 3    relabel + launch path -> launch_run17_baseline.sh + imodelsx check
             moves from launch-script tokens to kan.py (patch moved there, FA2).
  4. Sec 6    '$minPass = 560   # 566 collected...'   -> '$minPass = 1480  # 1483 passed...'
  5. Sec 7    relabel + FinnGen R12+R13 moved from optional-warn to REQUIRED hard-fail.
  6. Sec 12   Run15_Postflight.ps1 -> Run17_Postflight.ps1 (WARN not FAIL on absence, FA3).
  7. Sec 13   RUN_15_PLAN.md -> RUN_17_PLAN.md.
  8. ADD      agent-liveness section (check_agents_active.py) before Section 14.

Usage: python patch_g1_run17_preflight.py <path-to-Run_Preflight_Local.ps1>
"""
from __future__ import annotations
import sys
from pathlib import Path

MARKER = "# [G1-RUN17-ADAPTED]"

# Each entry: (label, old, new, expected_count). expected_count default 1.
EDITS = [
    # 1. Header line
    ("header",
     "# Run 15, genomic-variant-classifier. Runs BEFORE any Vast.ai instance create.",
     "# Run 17, genomic-variant-classifier. Runs BEFORE any Vast.ai instance create.",
     1),
    # 2. MinPytest param default
    ("min_pytest_param",
     "    [int]$MinPytest       = 566,",
     "    [int]$MinPytest       = 1485,",
     1),
    # 3. Section 3 -- relabel
    ("sec3_label",
     '    Section "3. imodelsx patch in launch script"',
     '    Section "3. imodelsx patch in kan.py (Run-17: patch moved from launch script)"',
     1),
    # 3b. Section 3 -- whole body: launch-script token check -> kan.py integration check.
    #     Anchor on the full block (launchPath assignment through the closing brace).
    ("sec3_body",
     '''    $launchPath = "scripts\\launch_run11_vm.sh"
    if (-not (Test-Path $launchPath)) { Fail "$launchPath missing" }
    else {
        $lc = Get-Content $launchPath -Raw
        foreach ($t in @('imodelsx_patch','test_size=self.test_size','random_state=self.random_state','shuffle=self.shuffle')) {
            if ($lc.Contains($t)) { Pass "launch script contains: $t" } else { Fail "launch script missing: $t" }
        }
    }''',
     '''    $launchPath = "scripts\\launch_run17_baseline.sh"
    if (-not (Test-Path $launchPath)) { Fail "$launchPath missing" }
    else { Pass "launch script present: $launchPath" }
    # Run-17: the imodelsx patch moved from the launch script into kan.py (FA2 2026-06-27).
    # Verify the integration lives in kan.py now, not the launcher.
    $kanPath = "src\\genomic_variant_classifier\\models\\kan.py"
    if (-not (Test-Path $kanPath)) { Fail "$kanPath missing" }
    else {
        $kc2 = Get-Content $kanPath -Raw
        foreach ($t in @('self._imodelsx_model.test_size','self._imodelsx_model.fit(X, y)')) {
            if ($kc2.Contains($t)) { Pass "kan.py imodelsx integration: $t" } else { Fail "kan.py missing imodelsx integration: $t" }
        }
    }''',
     1),
    # 4. Section 6 -- minPass floor
    ("sec6_minpass",
     "        $minPass = 560   # 566 collected minus 6 known-intentional skips (MC-dropout calibration TODOs pending Run 15; 1 coverage skip)",
     "        $minPass = 1480  # Run-17: 1483 passed / 2 skipped / 1485 collected (2026-06-27); 3-test headroom below 1483",
     1),
    # 5a. Section 7 -- relabel
    ("sec7_label",
     '    Section "7. Run 15 prep-input data files (raw; SCP\'d up for on-VM prep)"',
     '    Section "7. Run 17 prep-input data files (raw; SCP\'d up for on-VM prep)"',
     1),
    # 5b. Section 7 -- MISSING message relabel
    ("sec7_missing_msg",
     '        } else { Fail "$($f.Path) MISSING (required for Run 15 prep)" }',
     '        } else { Fail "$($f.Path) MISSING (required for Run 17 prep)" }',
     1),
    # 5c. Section 7 -- FinnGen R12+R13 from optional-warn to REQUIRED hard-fail.
    ("sec7_finngen_required",
     '''    foreach ($opt in @("data\\external\\finngen\\finnge_R12_annotated_variants_v1.gz","data\\external\\1kg\\1kg_phase3_af.parquet")) {
        if (Test-Path $opt) { Pass "(optional) $opt present" } else { Warn "(optional, deferred B.D1/B.D2) $opt absent - features 0" }
    }''',
     '''    # Run-17: FinnGen is WIRED + REQUIRED (dual-release R12+R13). Hard-fail on absence to match
    # the launcher's exit-7 guard (launch_run17_baseline.sh L181/L183) -- no silent-zero annotation.
    foreach ($fg in @("data\\external\\finngen\\finnge_R12_annotated_variants_v1.gz","data\\external\\finngen\\finngen_R13_annotated_variants_v0.gz")) {
        if (Test-Path $fg) {
            $fgmb = [math]::Round((Get-Item $fg).Length / 1MB, 1)
            Pass "FinnGen (required, dual-release) $fg -> $fgmb MB"
        } else { Fail "FinnGen REQUIRED file MISSING: $fg (launcher exits 7; dual-release needs both R12+R13)" }
    }
    # 1KGP AF parquet stays optional (still deferred per RUN_15 B.D1).
    foreach ($opt in @("data\\external\\1kg\\1kg_phase3_af.parquet")) {
        if (Test-Path $opt) { Pass "(optional) $opt present" } else { Warn "(optional, deferred B.D1) $opt absent - features 0" }
    }''',
     1),
    # 6. Section 12 -- postflight Run15 -> Run17, WARN (not FAIL) on absence since not yet built.
    ("sec12_postflight",
     '''    $postPath = "scripts\\Run15_Postflight.ps1"
    if (Test-Path $postPath) {
        $pc = Get-Content $postPath -Raw
        if ($pc -match '(?m)exit 1') { Pass "Run15_Postflight.ps1: exit 1 on FAIL (L80)" } else { Fail "Run15_Postflight.ps1: no exit 1 (L80)" }
        if ($pc.Contains('Test-ArtifactPresent')) { Pass "Run15_Postflight.ps1: Test-ArtifactPresent wired (L79)" } else { Fail "Run15_Postflight.ps1: Test-ArtifactPresent NOT wired (L79)" }
    } else { Fail "Run15_Postflight.ps1 missing (L80)" }''',
     '''    $postPath = "scripts\\Run17_Postflight.ps1"
    if (Test-Path $postPath) {
        $pc = Get-Content $postPath -Raw
        if ($pc -match '(?m)exit 1') { Pass "Run17_Postflight.ps1: exit 1 on FAIL" } else { Fail "Run17_Postflight.ps1: no exit 1" }
        if ($pc.Contains('Test-ArtifactPresent')) { Pass "Run17_Postflight.ps1: Test-ArtifactPresent wired" } else { Fail "Run17_Postflight.ps1: Test-ArtifactPresent NOT wired" }
    } else { Warn "Run17_Postflight.ps1 NOT YET BUILT -- required before launch (adapt from Run15_Postflight.ps1)" }''',
     1),
    # 7. Section 13 -- plan Run15 -> Run17 (label, path, and all message strings)
    ("sec13_label",
     '    Section "13. RUN_15_PLAN.md decision completeness (plan L77)"',
     '    Section "13. RUN_17_PLAN.md decision completeness"',
     1),
    ("sec13_planpath",
     '    $planPath = "docs\\runs\\RUN_15_PLAN.md"',
     '    $planPath = "docs\\runs\\RUN_17_PLAN.md"',
     1),
    ("sec13_pass_msg",
     '        if ($dc -le 1) { Pass "RUN_15_PLAN.md: $dc literal <DECISION> (<=1 gate-mention, OK)" } else { Fail "RUN_15_PLAN.md: $dc <DECISION> tokens (unfilled)" }',
     '        if ($dc -le 1) { Pass "RUN_17_PLAN.md: $dc literal <DECISION> (<=1 gate-mention, OK)" } else { Fail "RUN_17_PLAN.md: $dc <DECISION> tokens (unfilled)" }',
     1),
    ("sec13_missing_msg",
     '    } else { Fail "RUN_15_PLAN.md missing" }',
     '    } else { Fail "RUN_17_PLAN.md missing" }',
     1),
]

# 8. ADD agent-liveness section. Insert BEFORE Section 14. Anchor on the Section 14 label.
SEC14_ANCHOR = '    Section "14. Correctness harness (stages 1-5; gates correctness before AUROC)"'
AGENT_SECTION = '''    Section "13b. Agent-layer liveness (check_agents_active.py)"
    $agentChk = "scripts\\check_agents_active.py"
    if (-not (Test-Path $agentChk)) { Fail "$agentChk missing (agent-liveness gate)" }
    else {
        $ao = & $venvPython $agentChk 2>&1
        if ($LASTEXITCODE -eq 0) { Pass "agent liveness: all agents registered + scheduled" }
        else { Fail "agent liveness FAILED (exit $LASTEXITCODE): $(($ao | Select-Object -Last 3) -join '; ')" }
    }

'''


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python patch_g1_run17_preflight.py <path-to-Run_Preflight_Local.ps1>")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        return 2

    raw = path.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        print("ERROR: file has a UTF-8 BOM; expected BOM-free. Aborting (would need BOM-aware handling).")
        return 2
    text = raw.decode("utf-8")

    if MARKER in text:
        print(f"IDEMPOTENT: marker {MARKER} already present -- file already adapted. No-op.")
        return 0

    # --- VALIDATE all anchors occur EXACTLY their expected count BEFORE applying any ---
    problems = []
    for label, old, _new, count in EDITS:
        actual = text.count(old)
        if actual != count:
            problems.append(f"  [{label}] expected {count} occurrence(s), found {actual}")
    if text.count(SEC14_ANCHOR) != 1:
        problems.append(f"  [agent_section_anchor] expected 1 occurrence of Section 14 anchor, found {text.count(SEC14_ANCHOR)}")
    if problems:
        print("ANCHOR VALIDATION FAILED -- no changes written:")
        print("\n".join(problems))
        return 1

    # --- APPLY ---
    for label, old, new, _count in EDITS:
        text = text.replace(old, new, 1)
    # Insert agent section before Section 14
    text = text.replace(SEC14_ANCHOR, AGENT_SECTION + SEC14_ANCHOR, 1)

    # Stamp the marker after the header block (after line 3 region). Put it on its own line
    # right after the first '# ===' closing fence we can find, else prepend.
    stamp = f"\n{MARKER} Run-15 -> Run-17 (2026-06-27): launch path, kan.py imodelsx, test floor 1485/1480, FinnGen R12+R13 required, Run17 postflight+plan, agent liveness.\n"
    # place stamp right before the param( block for visibility
    if "param(" in text:
        text = text.replace("param(", stamp.strip() + "\nparam(", 1)
    else:
        text = stamp + text

    # --- WRITE .bak then file (BOM-free, preserve LF) ---
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_bytes(raw)
    path.write_text(text, encoding="utf-8", newline="")  # newline='' preserves existing \n / \r\n in text

    print(f"PATCHED: {path}")
    print(f"  backup: {bak}")
    print(f"  edits applied: {len(EDITS)} replacements + 1 agent-section insert + marker stamp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
