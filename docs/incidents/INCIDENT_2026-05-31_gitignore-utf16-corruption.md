# INCIDENT 2026-05-31 -- .gitignore corrupted by PowerShell >> (UTF-16), broke git add

**Status:** RESOLVED
**Severity:** LOW (no data loss; blocked staging until fixed)
**Cause:** Claude-suggested command using PowerShell redirection.

## Symptom
After `echo "*.bak_*" >> .gitignore`, `git add docs/... scripts/... src/... tests/...`
reported "The following paths are ignored by one of your .gitignore files: docs scripts src tests".
Effectively everything top-level appeared ignored; nothing could be staged via that command.

## Root cause
Windows PowerShell 5.1 `>>` (and `>`) write **UTF-16LE**, not UTF-8. Appending UTF-16LE bytes to
the UTF-8 `.gitignore` produced a mixed-encoding file with embedded null bytes. Git parsed the
garbled trailing content as a broad ignore rule, masking docs/scripts/src/tests.

## Fix
1. Restore the pristine tracked file: `git checkout HEAD -- .gitignore`.
2. Append the intended pattern as UTF-8 no-BOM via `[System.IO.File]::AppendAllText(path, "`n*.bak_*`n", UTF8Encoding($false))`.
3. Verify with `git check-ignore -v <legit files>` (no output) and on a `.bak_` file (matches).

## Lesson (reinforces PS hygiene memory #21)
NEVER use PowerShell `>`/`>>` to write or append text files in this project -- they emit UTF-16.
Use `[System.IO.File]::AppendAllText/WriteAllText` with `UTF8Encoding($false)`. The Install_*.ps1
generators already follow this; the ad-hoc `echo >>` did not. Claude will not suggest `>>`/`>`
for text files again.
