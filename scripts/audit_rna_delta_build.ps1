# audit_rna_delta_build.ps1 -- READ-ONLY. Pull the exact regions an option-b
# maxentscan_delta build must edit, plus loader column-preservation and any
# schema-column gate that a +1 feature would trip.
$ErrorActionPreference = "Stop"
$src = "src\genomic_variant_classifier"
$rdp = "$src\data\real_data_prep.py"
$ve  = "$src\models\variant_ensemble.py"
$rna = "$src\pipelines\rna_pipeline.py"

function Show($path, $a, $b, $label) {
    ""
    "===== $label  ($path)  lines $a..$b ====="
    $L = Get-Content $path
    $hi = [Math]::Min($b, $L.Count)
    for ($i = $a; $i -le $hi; $i++) { "{0,5}: {1}" -f $i, $L[$i-1] }
}
function Hdr($t) {
    ""
    "############################################################"
    "# $t"
    "############################################################"
}

Hdr "(1) variant_ensemble.py -- feature-name list (where maxentscan_delta is added)"
Show $ve 222 245 "TABULAR feature list (maxentscan_score in context)"

Hdr "(2) variant_ensemble.py -- _engineer_features RNA block (duplicate #1)"
Show $ve 500 530 "variant_ensemble RNA feature assignment"

Hdr "(3) real_data_prep.py -- _engineer_features RNA block (duplicate #2)"
Show $rdp 1158 1192 "real_data_prep RNA feature assignment"

Hdr "(4) real_data_prep.py -- post-RNA default-fill tuple (add maxentscan_delta default)"
Show $rdp 846 866 "default-fill tuple"

Hdr "(5) rna_pipeline.py -- splice scorers (reused to score the alt window)"
Show $rna 100 165 "_score_donor / _score_acceptor bodies"

Hdr "(6) real_data_prep.py -- _load_and_label: does it preserve all input columns?"
$L = Get-Content $rdp
$s = ($L | Select-String -SimpleMatch "def _load_and_label" | Select-Object -First 1).LineNumber
if ($s) { Show $rdp $s ($s + 45) "_load_and_label body" } else { "def _load_and_label NOT FOUND" }

Hdr "(7) schema / feature-count gate references across src/ (will +1 col trip it?)"
Get-ChildItem -Path $src -Recurse -Filter *.py |
  Select-String -Pattern "schema_baseline|expected_columns|n_features|EXPECTED_FEATURE|feature_count|schema_ref|validate.*schema|column.*count" |
  ForEach-Object { "{0}:{1}: {2}" -f $_.Filename, $_.LineNumber, ($_.Line.Trim()) }

""
"audit complete (read-only)."
