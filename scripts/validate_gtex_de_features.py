from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


DE = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\gtex_v11_Brain_Cortex_vs_Whole_Blood.de_features.tsv")
MANIFEST = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\gtex_v11_Brain_Cortex_vs_Whole_Blood.de_features.manifest.json")


REQUIRED_COLUMNS = {
    "gene_id",
    "gene_symbol",
    "rnaseq_log2fc",
    "rnaseq_de_pvalue",
    "rnaseq_de_qvalue",
    "rnaseq_de_neglog10p",
    "mean_tpm_Brain_Cortex",
    "mean_tpm_Whole_Blood",
    "n_Brain_Cortex",
    "n_Whole_Blood",
    "contrast_id",
    "leakage_guard",
}


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    if not DE.exists():
        fail(f"missing DE artifact: {DE}")
    if DE.stat().st_size <= 0:
        fail(f"zero-byte DE artifact: {DE}")
    if not MANIFEST.exists():
        fail(f"missing manifest: {MANIFEST}")

    df = pd.read_csv(DE, sep="\t")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        fail(f"missing columns: {sorted(missing)}")

    if len(df) != 74628:
        fail(f"unexpected row count: {len(df)}")

    if df["gene_id"].duplicated().any():
        fail("duplicate gene_id values detected")

    numeric_cols = [
        "rnaseq_log2fc",
        "rnaseq_de_pvalue",
        "rnaseq_de_qvalue",
        "rnaseq_de_neglog10p",
        "mean_tpm_Brain_Cortex",
        "mean_tpm_Whole_Blood",
    ]

    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            fail(f"NaN/non-numeric values in {col}")
        if not values.map(math.isfinite).all():
            fail(f"non-finite values in {col}")

    if not df["rnaseq_de_pvalue"].between(0, 1).all():
        fail("p-values outside [0, 1]")

    if not df["rnaseq_de_qvalue"].between(0, 1).all():
        fail("q-values outside [0, 1]")

    if (df["rnaseq_de_neglog10p"] < 0).any():
        fail("negative rnaseq_de_neglog10p values")

    if set(df["contrast_id"]) != {"Brain_Cortex_vs_Whole_Blood"}:
        fail("unexpected contrast_id values")

    if set(df["leakage_guard"]) != {"normal_tissue_reference_not_variant_label"}:
        fail("unexpected leakage_guard values")

    if manifest.get("rows") != len(df):
        fail("manifest row count mismatch")

    print("OK GTEx DE features")
    print(f"rows={len(df)}")
    print(f"columns={df.shape[1]}")
    print("top absolute log2FC genes:")
    top = df.reindex(df["rnaseq_log2fc"].abs().sort_values(ascending=False).index).head(10)
    print(top[["gene_symbol", "rnaseq_log2fc", "rnaseq_de_neglog10p", "mean_tpm_Brain_Cortex", "mean_tpm_Whole_Blood"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
