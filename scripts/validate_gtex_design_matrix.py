from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


MATRIX = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\gtex_v11_Brain_Cortex_vs_Whole_Blood.gene_tpm.tsv")
DESIGN = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\designs\gtex_v11_Brain_Cortex_vs_Whole_Blood.tsv")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    if not MATRIX.exists():
        fail(f"missing matrix: {MATRIX}")
    if not DESIGN.exists():
        fail(f"missing design: {DESIGN}")

    design = pd.read_csv(DESIGN, sep="\t", dtype=str)
    df = pd.read_csv(MATRIX, sep="\t", dtype=str)

    expected_samples = design["sample_id"].tolist()
    matrix_samples = df.columns[2:].tolist()

    if matrix_samples != expected_samples:
        fail("matrix sample columns do not exactly match design sample order")

    if df.shape[0] <= 0:
        fail("matrix has no genes")

    if df.shape[1] != len(expected_samples) + 2:
        fail(f"unexpected matrix width: {df.shape[1]}")

    numeric = df[matrix_samples].apply(pd.to_numeric, errors="coerce")

    if numeric.isna().any().any():
        fail("NaN/non-numeric expression values detected")

    if (numeric < 0).any().any():
        fail("negative TPM values detected")

    print("OK RNA-seq design matrix")
    print(f"genes={df.shape[0]}")
    print(f"samples={len(matrix_samples)}")
    print(f"conditions={design['condition'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
