# audit_docs_state.ps1 -- READ-ONLY. Current state of the canonical docs before
# authoring the CNN + RNA-delta session close (CHANGELOG / ROADMAP / session log).
$ErrorActionPreference = "Stop"
function Hdr($t) {
    ""
    "############################################################"
    "# $t"
    "############################################################"
}

Hdr "(1) docs/CHANGELOG.md -- last 60 lines (format + append point)"
$cl = "docs\CHANGELOG.md"
if (Test-Path $cl) { Get-Content $cl | Select-Object -Last 60 } else { "MISSING: $cl" }

Hdr "(2) docs/ROADMAP.md -- heading structure (# / ## / ###)"
$rm = "docs\ROADMAP.md"
if (Test-Path $rm) {
    Get-Content $rm | Select-String "^#{1,3} " |
      ForEach-Object { "{0,5}: {1}" -f $_.LineNumber, $_.Line }
} else { "MISSING: $rm" }

Hdr "(2b) docs/ROADMAP.md -- lines touching RNA/CNN/sequence/feature-count/launch-contract"
if (Test-Path $rm) {
    Get-Content $rm |
      Select-String "maxentscan|RNA|CNN|fasta_seq|launch contract|clinvar_grch38_clean_seq|esm2_llr|Run 1[56]|feature count|preflight|EXPECTED_TABULAR" |
      ForEach-Object { "{0,5}: {1}" -f $_.LineNumber, ($_.Line.Trim()) }
} else { "MISSING: $rm" }

Hdr "(3) docs/sessions/ -- existing session docs (naming + same-day suffix convention)"
if (Test-Path "docs\sessions") {
    Get-ChildItem "docs\sessions" -Filter *.md | Sort-Object Name | ForEach-Object { $_.Name }
} else { "MISSING: docs\sessions" }

Hdr "(4) docx regen tool + current ROADMAP.docx"
if (Test-Path "scripts\make_roadmap_docx.py") { "scripts\make_roadmap_docx.py: PRESENT" } else { "scripts\make_roadmap_docx.py: MISSING" }
if (Test-Path "docs\ROADMAP.docx") { "docs\ROADMAP.docx: PRESENT ($((Get-Item 'docs\ROADMAP.docx').Length) bytes)" } else { "docs\ROADMAP.docx: MISSING" }

Hdr "(5) untracked helper scripts still in the tree (for-the-record decision)"
git status --short | Select-String "^\?\?"

""
"audit complete (read-only)."
