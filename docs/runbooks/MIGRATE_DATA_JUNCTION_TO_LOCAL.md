# Runbook: migrate `data/` from a G-drive junction to local-canonical + mirror

**Author:** Monzia Moodie
**Goal:** move from option (a) -- `data/` is a junction into `G:\My Drive\...\data`
-- to option (b): `data/` is a real local directory, with Google Drive as a
selective rclone backup mirror. **Non-destructive: we COPY then verify then
switch; the G-drive original is never deleted until you choose to, after the
local copy is verified.** Every phase has a rollback.

> The single dangerous command in this whole procedure is removing the junction.
> Removing a junction the RIGHT way deletes only the link; the WRONG way
> (`rmdir /s`, `Remove-Item -Recurse`, `del`) recurses **through** it and destroys
> your G-drive data. Phase 4 spells out the safe command. Read it before acting.

## Phase 0 -- snapshot & confirm (no changes)
```powershell
cd C:\Projects\genomic-variant-classifier
git status                      # commit/stash any work first
git rev-parse HEAD              # record where you are
Get-Item .\data -Force | Format-List FullName,LinkType,Target   # confirm Junction + its Target
```
Record the Target (expected `G:\My Drive\genomic-variant-classifier\data`). If
`LinkType` is not `Junction`, `data/` is already a real dir -- skip to Phase 6.

## Phase 1 -- audit the current (junctioned) tree
```powershell
python scripts\maintenance\audit_data_tree.py --json data_audit_before.json
```
Save the rollup (must-back-up vs regenerable) and the orphan/alias list -- this is
your before-picture and your cleanup checklist.

## Phase 2 -- make the G-drive source complete
In Google Drive for Desktop, set `G:\My Drive\genomic-variant-classifier\data` to
**Available offline** and wait for full download (not placeholders). A partial/
streamed source would copy incompletely. Confirm G: is mounted and idle.

## Phase 3 -- COPY G-drive -> a local staging dir (outside the repo)
```powershell
$Target = "G:\My Drive\genomic-variant-classifier\data"
$Stage  = "C:\gvc_data_stage"
robocopy "$Target" "$Stage" /E /COPY:DAT /DCOPY:DAT /R:2 /W:5 /NFL /NDL /TEE /LOG:robocopy_data.log
# robocopy exit codes 0-7 are SUCCESS (>=8 is an error). Check:
$LASTEXITCODE; if ($LASTEXITCODE -ge 8) { "ROBOCOPY FAILED -- stop and inspect robocopy_data.log" }
```
Staging is a separate path, so nothing in the repo or on G: is altered yet.

## Phase 4 -- verify the copy byte-for-byte (rclone check if available)
```powershell
# if rclone is installed, verify content hashes match (authoritative):
rclone check "$Target" "$Stage" --one-way        # expect "0 differences found"
# always sanity-check sizes/counts both ways:
"{0:N0} files, {1:N0} bytes (G)" -f (Get-ChildItem $Target -Recurse -File).Count, ((Get-ChildItem $Target -Recurse -File) | Measure-Object Length -Sum).Sum
"{0:N0} files, {1:N0} bytes (stage)" -f (Get-ChildItem $Stage -Recurse -File).Count, ((Get-ChildItem $Stage -Recurse -File) | Measure-Object Length -Sum).Sum
```
**Do not proceed unless counts/bytes match (and `rclone check` shows 0 diffs).**
Rollback so far: just `Remove-Item $Stage -Recurse -Force` (the staging copy is
disposable; G: untouched).

## Phase 5 -- remove the junction (THE careful step)
```powershell
# SAFE: rmdir WITHOUT /s removes only the reparse point (the link), not the target.
cmd /c rmdir "C:\Projects\genomic-variant-classifier\data"
# Verify the link is gone and the G-drive target is INTACT:
Test-Path "C:\Projects\genomic-variant-classifier\data"   # expect False
Test-Path "$Target\external"                              # expect True (G: data still there)
```
NEVER use `rmdir /s`, `Remove-Item -Recurse`, or `del` on the junction -- those
follow the link and delete your G-drive data. If anything looks wrong, ROLLBACK
by recreating the junction and stop:
```powershell
cmd /c mklink /J "C:\Projects\genomic-variant-classifier\data" "$Target"
```

## Phase 6 -- put the local copy in place
```powershell
Move-Item "C:\gvc_data_stage" "C:\Projects\genomic-variant-classifier\data"
python scripts\maintenance\preflight_data_guard.py data      # expect: OK ... (real dir)
python scripts\maintenance\audit_data_tree.py --json data_audit_after.json
```
Confirm the auditor now reports `data/ status: REAL_DIR`. Compare
`data_audit_after.json` to `_before` -- same sources, same sizes.

## Phase 7 -- prove nothing broke
```powershell
python -m pytest -q          # expect the full suite green (your baseline ~1100 passed)
```
If green, the migration is functionally complete. Rollback (if red and data-
related): delete the local `data/`, recreate the junction (Phase 5 rollback), and
re-run the suite to confirm you are back to the prior state.

## Phase 8 -- establish Google Drive as the selective mirror
```powershell
rclone config            # one-time: create remote "genvarcla" (type: drive)
python scripts\maintenance\setup_data_tree.py                 # (re)generate configs/rclone_data_filter.txt
python scripts\maintenance\sync_data_to_gdrive.py             # DRY-RUN preview (controlled gate enforced)
python scripts\maintenance\sync_data_to_gdrive.py --execute   # push the synced subset only
```
Only `sync: true`, non-controlled sources are mirrored. Controlled/licensed data
(HGMD, OMIM, COSMIC, TCGA, TOPMed) is intentionally excluded -- back those up
encrypted/offline, never to personal cloud.

## Phase 9 -- cleanup (optional, only after Phases 6-8 verified)
The old `G:\My Drive\...\data` is still your pre-migration copy. Once the new
mirror is verified you may retire it (move it aside, or let rclone manage
`genvarcla:genomic-variant-classifier/data` going forward). Keep at least one
verified backup of every `irreplaceable` and `offline-only` source before
deleting anything.

## Make it durable
- Call `assert_data_usable()` (from `preflight_data_guard.py`) at the top of your
  run/smoke scripts so a future dangling/shadow `data/` fails loud and early.
- Run `audit_data_tree.py` at session start and before every run.
- After acquiring/regenerating any source, update `configs/data_manifest.yaml`
  (version, class, sync) and re-run `setup_data_tree.py`.
