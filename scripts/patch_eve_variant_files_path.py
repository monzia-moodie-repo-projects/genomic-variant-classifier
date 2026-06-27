#!/usr/bin/env python3
r"""patch_eve_variant_files_path.py

Fix the LAST EVE silent-zero: the launch script wired --eve-path at the EVE
*bundle root* ($DATA/external/eve), which has 0 top-level CSVs (the 3,211 score
files live in EVE_all_data/variant_files). EVE's glob is non-recursive, so it
would find nothing and every variant would get 0.5 -- and the old `ls -A` guard
PASSES on the bundle root (it contains the EVE_all_data subdir), so it would not
even abort. This wires the correct directory and replaces `ls -A` with a hard
0-CSV abort. Staging is fixed to pull + tar ONLY variant_files (~10 GB, option B),
leaving the full 63 GB bundle preserved locally and on Drive.

Layout decision (B4, proven by tar round-trip): tar from $EXT\eve so the archive
root is EVE_all_data/variant_files; the VM extracts with `-C data/external/eve`,
yielding data/external/eve/EVE_all_data/variant_files -- IDENTICAL to the local
path, so launch EVE_DIR is the same string local and on the VM. bsdtar-safe
(tar -C <dir> <relative-subdir>), no --transform, no 10 GB copy.

Targets (anchors verified against live reads):
  scripts/launch_run17_baseline.sh
  scripts/Stage_Run17_EVE_ESM2.ps1

ANCHOR-BASED, IDEMPOTENT. launch .sh written LF-only (CRLF guard); Stage .ps1
written CRLF-free too (PS tolerates either; we keep it LF-clean for consistency).

  python scripts/patch_eve_variant_files_path.py            # apply
  python scripts/patch_eve_variant_files_path.py --check    # report only
"""
from __future__ import annotations

import argparse
from pathlib import Path

LAUNCH = Path("scripts/launch_run17_baseline.sh")
STAGE = Path("scripts/Stage_Run17_EVE_ESM2.ps1")

# ---------------------------------------------------------------- launch .sh
# Re-point EVE_DIR to variant_files + replace the weak `ls -A` guard with a hard
# 0-CSV abort (the bundle root passes `ls -A` but has 0 CSVs -> silent zero).
LAUNCH_ANCHOR = (
    "# EVE: directory of per-protein CSVs (gene_symbol + HGVSp-derived aa_change).\n"
    'EVE_DIR="$DATA/external/eve"\n'
    'if [ -d "$EVE_DIR" ] && [ -n "$(ls -A "$EVE_DIR" 2>/dev/null)" ]; then\n'
    '    ARGS="$ARGS --eve-path $EVE_DIR"; echo "==> EVE wired: $EVE_DIR ($(ls "$EVE_DIR" | wc -l) files)" | tee -a "$LOG"\n'
    "else\n"
    '    echo "==> ABORT: EVE dir missing/empty: $EVE_DIR (stage it to the VM)" | tee -a "$LOG"; exit 8\n'
    "fi\n"
)
LAUNCH_INSERT = (
    "# EVE: directory of per-protein score CSVs (gene_symbol + HGVSp-derived aa_change).\n"
    "# The 3,211 score CSVs live in EVE_all_data/variant_files (NOT the bundle root,\n"
    "# which has 0 top-level CSVs). EVE's glob is non-recursive, so point at the leaf\n"
    "# dir and ABORT on 0 CSVs -- the old `ls -A` check passed on the CSV-less bundle\n"
    "# root and would have silently scored every variant 0.5.\n"
    'EVE_DIR="$DATA/external/eve/EVE_all_data/variant_files"\n'
    '_EVE_CSVN=$(ls "$EVE_DIR"/*.csv 2>/dev/null | wc -l)\n'
    'if [ -d "$EVE_DIR" ] && [ "$_EVE_CSVN" -gt 0 ]; then\n'
    '    ARGS="$ARGS --eve-path $EVE_DIR"; echo "==> EVE wired: $EVE_DIR ($_EVE_CSVN CSVs)" | tee -a "$LOG"\n'
    "else\n"
    '    echo "==> ABORT: EVE variant_files missing or no CSVs: $EVE_DIR ($_EVE_CSVN found; expected ~3211). Stage variant_files to the VM." | tee -a "$LOG"; exit 8\n'
    "fi\n"
)

# ---------------------------------------------------------------- Stage .ps1
# (1) [1/6] banner + pull only variant_files (was: full 14,933-file bundle).
STAGE_PULL_ANCHOR = (
    'Write-Host "== [1/6] EVE (expect ~14933 CSVs) ==" -ForegroundColor Cyan\n'
    '$eveN = Pull-Source "eve" 14000\n'
)
STAGE_PULL_INSERT = (
    'Write-Host "== [1/6] EVE variant_files (expect ~3211 score CSVs; full 63GB bundle stays on Drive+local) ==" -ForegroundColor Cyan\n'
    '# Option B: stage ONLY the score CSVs the VM needs. MSAs/VCFs/plots remain preserved\n'
    '# locally (data\\external\\eve\\EVE_all_data) and on Drive for later phases.\n'
    '$eveN = Pull-Source "eve/EVE_all_data/variant_files" 3000\n'
)

# (2) tarball: tar ONLY variant_files, layout EVE_all_data/variant_files (B4).
STAGE_TAR_ANCHOR = (
    'Write-Host "`n== [6/6] Package EVE as a single tarball (fast SCP; extract on VM) ==" -ForegroundColor Cyan\n'
    '$EVE_TAR = "$STAGE\\eve.tar.gz"\n'
    "if (Test-Path $EVE_TAR) { Remove-Item $EVE_TAR -Force }\n"
    "Push-Location $EXT\n"
    "tar -czf $EVE_TAR eve\n"
    "Pop-Location\n"
    'if (-not (Test-Path $EVE_TAR)) { throw "EVE tarball not created." }\n'
    '"EVE tarball: $EVE_TAR ($("{0:N1}" -f ((Get-Item $EVE_TAR).Length/1MB)) MB)"\n'
)
STAGE_TAR_INSERT = (
    'Write-Host "`n== [6/6] Package EVE variant_files as a single tarball (~10GB; fast SCP; extract on VM) ==" -ForegroundColor Cyan\n'
    '$EVE_TAR = "$STAGE\\eve_variant_files.tar.gz"\n'
    "if (Test-Path $EVE_TAR) { Remove-Item $EVE_TAR -Force }\n"
    "# B4 layout: tar from $EXT\\eve so the archive root is EVE_all_data/variant_files.\n"
    "# VM extracts with `-C data/external/eve`, yielding\n"
    "#   data/external/eve/EVE_all_data/variant_files  == the local path (identical).\n"
    "# bsdtar-safe (tar -C <dir> <relative-subdir>); no --transform; no 10GB copy.\n"
    '$EVE_VF = "$EXT\\eve\\EVE_all_data\\variant_files"\n'
    "$vfCount = (Get-ChildItem $EVE_VF -Filter *.csv -File -ErrorAction SilentlyContinue | Measure-Object).Count\n"
    'if ($vfCount -lt 3000) { throw "EVE variant_files has $vfCount CSVs (expected ~3211) at $EVE_VF -- pull incomplete." }\n'
    "Push-Location \"$EXT\\eve\"\n"
    "tar -czf $EVE_TAR EVE_all_data/variant_files\n"
    "Pop-Location\n"
    'if (-not (Test-Path $EVE_TAR)) { throw "EVE tarball not created." }\n'
    "# Post-check: archive lists the expected leaf path + CSV count.\n"
    "$tarList = (tar -tzf $EVE_TAR)\n"
    '$tarCsv = ($tarList | Where-Object { $_ -match "EVE_all_data/variant_files/.+\\.csv$" } | Measure-Object).Count\n'
    'if ($tarCsv -lt 3000) { throw "EVE tarball lists only $tarCsv CSVs (expected ~3211) -- bad tar layout." }\n'
    '"EVE tarball: $EVE_TAR ($("{0:N1}" -f ((Get-Item $EVE_TAR).Length/1MB)) MB; $tarCsv CSVs under EVE_all_data/variant_files)"\n'
)

# (3) VM manifest line: extract under eve/ + expect 3211, --eve-path leaf.
STAGE_MANIFEST_ANCHOR = (
    '"  1. $EVE_TAR  -> /workspace/genomic-variant-classifier/data/external/  '
    'then: cd .../external && tar -xzf eve.tar.gz && ls eve | wc -l   # expect $eveN"\n'
)
STAGE_MANIFEST_INSERT = (
    '"  1. $EVE_TAR  -> /workspace/genomic-variant-classifier/data/external/eve/  '
    'then: cd .../external/eve && tar -xzf eve_variant_files.tar.gz && '
    'ls EVE_all_data/variant_files | wc -l   # expect $eveN (~3211)"\n'
    '"     launch --eve-path resolves to data/external/eve/EVE_all_data/variant_files (== local path)."\n'
)


def _apply(target: Path, edits, marker, check, suffix, lf_only):
    if not target.exists():
        print(f"FAIL: {target} not found.")
        return 2
    src = target.read_text(encoding="utf-8")
    if marker in src:
        print(f"OK (idempotent): {target.name} already patched.")
        return 0
    problems = []
    for name, anc, _new in edits:
        n = src.count(anc)
        if n != 1:
            problems.append(f"{target.name}/{name}: anchor occurs {n}x (need 1)")
    if problems:
        print("FAIL: cannot safely anchor:")
        for p in problems:
            print(f"  - {p}")
        return 3
    patched = src
    for _name, anc, new in edits:
        patched = patched.replace(anc, new, 1)
    if check:
        print(f"CHECK: {target.name} all {len(edits)} anchors found.")
        return 0
    backup = target.with_suffix(target.suffix + suffix)
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="\n")
        print(f"OK: backup -> {backup}")
    target.write_text(patched, encoding="utf-8", newline="\n")
    if lf_only and b"\r\n" in target.read_bytes():
        print(f"FAIL: CRLF in {target.name} (bash would break).")
        return 5
    print(f"OK: patched {target}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()

    rc1 = _apply(
        LAUNCH,
        [("EVE_DIR + 0-CSV guard", LAUNCH_ANCHOR, LAUNCH_INSERT)],
        marker='EVE_DIR="$DATA/external/eve/EVE_all_data/variant_files"',
        check=ns.check, suffix=".pre_variant_files.bak", lf_only=True,
    )
    rc2 = _apply(
        STAGE,
        [("pull variant_files", STAGE_PULL_ANCHOR, STAGE_PULL_INSERT),
         ("tar variant_files", STAGE_TAR_ANCHOR, STAGE_TAR_INSERT),
         ("VM manifest", STAGE_MANIFEST_ANCHOR, STAGE_MANIFEST_INSERT)],
        marker='Pull-Source "eve/EVE_all_data/variant_files"',
        check=ns.check, suffix=".pre_variant_files.bak", lf_only=False,
    )
    rc = rc1 or rc2
    if not ns.check and rc == 0:
        # Post-check: launch parses cleanly via py-side structural read (CRLF) is in installer;
        # here, confirm the new strings are present.
        lt = LAUNCH.read_text(encoding="utf-8")
        st = STAGE.read_text(encoding="utf-8")
        checks = [
            ('launch EVE_DIR -> variant_files', 'EVE_DIR="$DATA/external/eve/EVE_all_data/variant_files"' in lt),
            ('launch 0-CSV abort guard', '_EVE_CSVN=$(ls "$EVE_DIR"/*.csv 2>/dev/null | wc -l)' in lt),
            ('stage pulls variant_files only', 'Pull-Source "eve/EVE_all_data/variant_files"' in st),
            ('stage tars EVE_all_data/variant_files', 'tar -czf $EVE_TAR EVE_all_data/variant_files' in st),
            ('stage tar post-check', '$tarCsv -lt 3000' in st),
        ]
        ok = True
        for label, present in checks:
            print(f"  {'OK' if present else 'MISSING'}  {label}")
            ok &= present
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 4
    print("RESULT:", "PASS (check)" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
