# audit_rna_wiring.ps1 -- READ-ONLY. Scope the RNA MaxEntScan activation:
#   (1) is the RNA pipeline wired into the prep flow at all?
#   (2) does the df it receives carry fasta_seq_ref/alt?
#   (3) are its outputs (maxentscan_score, ...) registered features?
#   (4) scorer signatures for an option-b ref/alt delta build.
$ErrorActionPreference = "Stop"
$src = "src\genomic_variant_classifier"
$rdp = "$src\data\real_data_prep.py"
$rna = "$src\pipelines\rna_pipeline.py"

function Hdr($t) {
    ""
    "############################################################"
    "# $t"
    "############################################################"
}

Hdr "(1) Every RNA-pipeline reference across src/ (is it wired in at all?)"
Get-ChildItem -Path $src -Recurse -Filter *.py |
  Select-String -Pattern "RNASpliceIsoform|annotate_dataframe|maxentscan|rna_pipeline|RNASplice" |
  ForEach-Object { "{0}:{1}: {2}" -f $_.Filename, $_.LineNumber, ($_.Line.Trim()) }

Hdr "(2) real_data_prep.py -- RNA call site / import / _annotate_scores in context"
Get-Content $rdp |
  Select-String -Pattern "RNASpliceIsoform|\.annotate_dataframe\(|import.*[Rr]na|def _annotate_scores" -Context 3,4 |
  ForEach-Object { $_ }

Hdr "(2b) real_data_prep.py -- where fasta_seq / fasta_seq_ref / fasta_seq_alt appear"
Get-Content $rdp |
  Select-String -Pattern "fasta_seq_ref|fasta_seq_alt|fasta_seq" |
  ForEach-Object { "{0,5}: {1}" -f $_.LineNumber, ($_.Line.Trim()) }

Hdr "(3) Are maxentscan/splice outputs REGISTERED features anywhere in src/?"
Get-ChildItem -Path $src -Recurse -Filter *.py |
  Select-String -Pattern "maxentscan_score|dist_to_splice_site|is_canonical_splice|maxentscan_delta" |
  ForEach-Object { "{0}:{1}: {2}" -f $_.Filename, $_.LineNumber, ($_.Line.Trim()) }

Hdr "(3b) real_data_prep _engineer_features -- feature selection near splice/maxent"
Get-Content $rdp |
  Select-String -Pattern "def _engineer_features|maxentscan|splice|exon_number|TABULAR|feature_cols|FEATURE" |
  ForEach-Object { "{0,5}: {1}" -f $_.LineNumber, ($_.Line.Trim()) }

Hdr "(4) rna_pipeline.py -- defaults, class, init, scorer signatures (for option-b)"
Get-Content $rna |
  Select-String -Pattern "DEFAULT_MAXENTSCAN|DEFAULT_DIST|DEFAULT_EXON|DEFAULT_IS_CANON|^class |def __init__|def _score_donor|def _score_acceptor|def annotate_dataframe" |
  ForEach-Object { "{0,5}: {1}" -f $_.LineNumber, ($_.Line.Trim()) }

""
"audit complete (read-only)."
