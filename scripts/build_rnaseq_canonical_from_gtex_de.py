from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


DE = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\gtex_v11_Brain_Cortex_vs_Whole_Blood.de_features.tsv")
OUT = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\gtex_v11_Brain_Cortex_vs_Whole_Blood.rnaseq_gene_expression.parquet")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    if not DE.exists():
        fail(f"missing DE table: {DE}")

    df = pd.read_csv(DE, sep="\t", dtype={"gene_id": str, "gene_symbol": str}).fillna("")

    required = {
        "gene_id",
        "gene_symbol",
        "rnaseq_log2fc",
        "rnaseq_de_neglog10p",
        "mean_tpm_Brain_Cortex",
        "mean_tpm_Whole_Blood",
    }
    missing = required - set(df.columns)
    if missing:
        fail(f"missing DE columns: {sorted(missing)}")

    if df["gene_id"].duplicated().any():
        fail("duplicate gene_id values in DE table")

    before = len(df)
    df = df[df["gene_symbol"].astype(str).str.len() > 0].copy()
    df = df[~df["gene_symbol"].astype(str).str.startswith("ENSG")].copy()

    if len(df) <= 0:
        fail("no non-Ensembl gene symbols remain after filtering")

    if not (df["gene_symbol"] == "TP53").any():
        fail("TP53 missing after symbol filtering")
    if not (df["gene_symbol"] == "HBB").any():
        fail("HBB missing after symbol filtering")

    # Approximate summary features from the validated two-condition means.
    # DE fields remain sourced from the validated full DE artifact.
    mean_tpm = (pd.to_numeric(df["mean_tpm_Brain_Cortex"], errors="coerce") +
                pd.to_numeric(df["mean_tpm_Whole_Blood"], errors="coerce")) / 2.0

    df = df.assign(
        rnaseq_mean_log_tpm=np.log1p(mean_tpm),
        rnaseq_detection_rate=(
            (pd.to_numeric(df["mean_tpm_Brain_Cortex"], errors="coerce") > 0) |
            (pd.to_numeric(df["mean_tpm_Whole_Blood"], errors="coerce") > 0)
        ).astype(float),
        rnaseq_log2_cv=0.0,
    )

    numeric = [
        "rnaseq_mean_log_tpm",
        "rnaseq_detection_rate",
        "rnaseq_log2_cv",
        "rnaseq_log2fc",
        "rnaseq_de_neglog10p",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric].isna().any().any():
        fail("NaN values detected before canonical output")

    # Deterministic symbol dedupe for current production connector.
    # Prefer strongest DE evidence, then larger absolute effect.
    df["_abs_fc"] = df["rnaseq_log2fc"].abs()
    out = (
        df.sort_values(
            ["gene_symbol", "rnaseq_de_neglog10p", "_abs_fc", "gene_id"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("gene_symbol", keep="first")
        [
            [
                "gene_symbol",
                "rnaseq_mean_log_tpm",
                "rnaseq_detection_rate",
                "rnaseq_log2_cv",
                "rnaseq_log2fc",
                "rnaseq_de_neglog10p",
            ]
        ]
        .copy()
    )

    if out["gene_symbol"].duplicated().any():
        fail("duplicate gene_symbol values remain")

    ens_frac = out["gene_symbol"].astype(str).str.startswith("ENSG").mean()
    if ens_frac != 0.0:
        fail(f"Ensembl-like symbols remain: fraction={ens_frac}")

    if not (out["gene_symbol"] == "TP53").any():
        fail("TP53 missing from canonical Parquet")
    if not (out["gene_symbol"] == "HBB").any():
        fail("HBB missing from canonical Parquet")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    print(f"OK wrote: {OUT}")
    print(f"input_rows={before}")
    print(f"symbol_rows_after_filter={len(df)}")
    print(f"canonical_rows={len(out)}")
    print(f"TP53 rows={(out['gene_symbol'] == 'TP53').sum()}")
    print(f"HBB rows={(out['gene_symbol'] == 'HBB').sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
