from __future__ import annotations

from pathlib import Path

DESIGN_DIR = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\designs")
DESIGN_DIR.mkdir(parents=True, exist_ok=True)

HEADER = [
    "sample_id", "condition", "batch", "study_id", "tissue",
    "assay_platform", "design_family", "contrast_id",
    "source_matrix", "leakage_guard", "notes",
]

DESIGNS = {
    "option_A_external_case_control_TEMPLATE.tsv": [
        ["REPLACE_SAMPLE_001", "case", "REPLACE_BATCH", "REPLACE_STUDY", "REPLACE_TISSUE", "RNA-seq", "option_A_external_case_control", "REPLACE_CASE_vs_CONTROL", "REPLACE_MATRIX_PATH", "condition_not_derived_from_variant_or_clinvar_labels", "Replace with real sample metadata"],
        ["REPLACE_SAMPLE_002", "control", "REPLACE_BATCH", "REPLACE_STUDY", "REPLACE_TISSUE", "RNA-seq", "option_A_external_case_control", "REPLACE_CASE_vs_CONTROL", "REPLACE_MATRIX_PATH", "condition_not_derived_from_variant_or_clinvar_labels", "Replace with real sample metadata"],
        ["REPLACE_SAMPLE_003", "case", "REPLACE_BATCH", "REPLACE_STUDY", "REPLACE_TISSUE", "RNA-seq", "option_A_external_case_control", "REPLACE_CASE_vs_CONTROL", "REPLACE_MATRIX_PATH", "condition_not_derived_from_variant_or_clinvar_labels", "Minimum 3 samples per condition"],
        ["REPLACE_SAMPLE_004", "control", "REPLACE_BATCH", "REPLACE_STUDY", "REPLACE_TISSUE", "RNA-seq", "option_A_external_case_control", "REPLACE_CASE_vs_CONTROL", "REPLACE_MATRIX_PATH", "condition_not_derived_from_variant_or_clinvar_labels", "Minimum 3 samples per condition"],
        ["REPLACE_SAMPLE_005", "case", "REPLACE_BATCH", "REPLACE_STUDY", "REPLACE_TISSUE", "RNA-seq", "option_A_external_case_control", "REPLACE_CASE_vs_CONTROL", "REPLACE_MATRIX_PATH", "condition_not_derived_from_variant_or_clinvar_labels", "Add more samples before DE"],
        ["REPLACE_SAMPLE_006", "control", "REPLACE_BATCH", "REPLACE_STUDY", "REPLACE_TISSUE", "RNA-seq", "option_A_external_case_control", "REPLACE_CASE_vs_CONTROL", "REPLACE_MATRIX_PATH", "condition_not_derived_from_variant_or_clinvar_labels", "Add more samples before DE"],
    ],
    "option_C_gtex_Brain_Cortex_vs_Whole_Blood_TEMPLATE.tsv": [
        ["REPLACE_GTEX_SAMPLE_001", "Brain_Cortex", "unknown", "GTEx_v11", "Brain_Cortex", "RNA-seq", "option_C_gtex_tissue_contrast", "Brain_Cortex_vs_Whole_Blood", "GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_tpm.gct.gz", "normal_tissue_reference_not_variant_label", "Replace with GTEx Brain Cortex sample ID"],
        ["REPLACE_GTEX_SAMPLE_002", "Brain_Cortex", "unknown", "GTEx_v11", "Brain_Cortex", "RNA-seq", "option_C_gtex_tissue_contrast", "Brain_Cortex_vs_Whole_Blood", "GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_tpm.gct.gz", "normal_tissue_reference_not_variant_label", "Replace with GTEx Brain Cortex sample ID"],
        ["REPLACE_GTEX_SAMPLE_003", "Brain_Cortex", "unknown", "GTEx_v11", "Brain_Cortex", "RNA-seq", "option_C_gtex_tissue_contrast", "Brain_Cortex_vs_Whole_Blood", "GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_tpm.gct.gz", "normal_tissue_reference_not_variant_label", "Minimum 3 per condition"],
        ["REPLACE_GTEX_SAMPLE_004", "Whole_Blood", "unknown", "GTEx_v11", "Whole_Blood", "RNA-seq", "option_C_gtex_tissue_contrast", "Brain_Cortex_vs_Whole_Blood", "GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_tpm.gct.gz", "normal_tissue_reference_not_variant_label", "Replace with GTEx Whole Blood sample ID"],
        ["REPLACE_GTEX_SAMPLE_005", "Whole_Blood", "unknown", "GTEx_v11", "Whole_Blood", "RNA-seq", "option_C_gtex_tissue_contrast", "Brain_Cortex_vs_Whole_Blood", "GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_tpm.gct.gz", "normal_tissue_reference_not_variant_label", "Replace with GTEx Whole Blood sample ID"],
        ["REPLACE_GTEX_SAMPLE_006", "Whole_Blood", "unknown", "GTEx_v11", "Whole_Blood", "RNA-seq", "option_C_gtex_tissue_contrast", "Brain_Cortex_vs_Whole_Blood", "GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_tpm.gct.gz", "normal_tissue_reference_not_variant_label", "Minimum 3 per condition"],
    ],
    "option_D_TCGA_BRCA_Tumor_vs_Normal_TEMPLATE.tsv": [
        ["REPLACE_TCGA_SAMPLE_001", "Primary_Tumor", "REPLACE_BATCH", "TCGA-BRCA", "breast", "RNA-seq", "option_D_tcga_tumor_normal", "TCGA_BRCA_Tumor_vs_Normal", "GDC_STAR_COUNTS_gene_expression_matrix", "condition_from_tcga_sample_type_not_clinvar_or_variant_labels", "Replace with real TCGA tumor sample ID"],
        ["REPLACE_TCGA_SAMPLE_002", "Primary_Tumor", "REPLACE_BATCH", "TCGA-BRCA", "breast", "RNA-seq", "option_D_tcga_tumor_normal", "TCGA_BRCA_Tumor_vs_Normal", "GDC_STAR_COUNTS_gene_expression_matrix", "condition_from_tcga_sample_type_not_clinvar_or_variant_labels", "Replace with real TCGA tumor sample ID"],
        ["REPLACE_TCGA_SAMPLE_003", "Primary_Tumor", "REPLACE_BATCH", "TCGA-BRCA", "breast", "RNA-seq", "option_D_tcga_tumor_normal", "TCGA_BRCA_Tumor_vs_Normal", "GDC_STAR_COUNTS_gene_expression_matrix", "condition_from_tcga_sample_type_not_clinvar_or_variant_labels", "Minimum 3 per condition"],
        ["REPLACE_TCGA_SAMPLE_004", "Solid_Tissue_Normal", "REPLACE_BATCH", "TCGA-BRCA", "breast", "RNA-seq", "option_D_tcga_tumor_normal", "TCGA_BRCA_Tumor_vs_Normal", "GDC_STAR_COUNTS_gene_expression_matrix", "condition_from_tcga_sample_type_not_clinvar_or_variant_labels", "Replace with real TCGA normal sample ID"],
        ["REPLACE_TCGA_SAMPLE_005", "Solid_Tissue_Normal", "REPLACE_BATCH", "TCGA-BRCA", "breast", "RNA-seq", "option_D_tcga_tumor_normal", "TCGA_BRCA_Tumor_vs_Normal", "GDC_STAR_COUNTS_gene_expression_matrix", "condition_from_tcga_sample_type_not_clinvar_or_variant_labels", "Replace with real TCGA normal sample ID"],
        ["REPLACE_TCGA_SAMPLE_006", "Solid_Tissue_Normal", "REPLACE_BATCH", "TCGA-BRCA", "breast", "RNA-seq", "option_D_tcga_tumor_normal", "TCGA_BRCA_Tumor_vs_Normal", "GDC_STAR_COUNTS_gene_expression_matrix", "condition_from_tcga_sample_type_not_clinvar_or_variant_labels", "Minimum 3 per condition"],
    ],
}

for filename, rows in DESIGNS.items():
    path = DESIGN_DIR / filename
    text = "\t".join(HEADER) + "\n"
    text += "\n".join("\t".join(row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {path} bytes={path.stat().st_size}")
