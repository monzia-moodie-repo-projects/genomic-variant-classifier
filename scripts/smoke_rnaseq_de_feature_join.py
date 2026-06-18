from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


DE = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\gtex_v11_Brain_Cortex_vs_Whole_Blood.de_features.tsv")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    if not DE.exists():
        fail(f"missing DE features: {DE}")

    de = pd.read_csv(
        DE,
        sep="\t",
        usecols=[
            "gene_id",
            "gene_symbol",
            "rnaseq_log2fc",
            "rnaseq_de_neglog10p",
            "rnaseq_de_test_status",
            "leakage_guard",
        ],
        dtype={"gene_id": str, "gene_symbol": str},
    )

    if de["gene_id"].duplicated().any():
        dupes = de.loc[de["gene_id"].duplicated(), "gene_id"].head(10).tolist()
        fail(f"duplicate gene_id values: {dupes}")

    variants = pd.DataFrame(
        {
            "variant_id": [
                "test:17:43071077:G:T",
                "test:11:5227002:T:A",
                "test:1:1:A:T",
            ],
            "gene_symbol": ["TP53", "HBB", "NOT_A_REAL_GENE_XYZ"],
        }
    )

    # Deterministic symbol fallback for smoke test only:
    # use the row with strongest DE evidence when symbols are duplicated.
    symbol_de = (
        de.sort_values(
            ["gene_symbol", "rnaseq_de_neglog10p", "rnaseq_log2fc"],
            ascending=[True, False, False],
        )
        .drop_duplicates("gene_symbol", keep="first")
    )

    out = variants.merge(symbol_de, on="gene_symbol", how="left")

    feature_cols = ["rnaseq_log2fc", "rnaseq_de_neglog10p"]
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["rnaseq_de_test_status"] = out["rnaseq_de_test_status"].fillna("missing_gene_default_zero")
    out["leakage_guard"] = out["leakage_guard"].fillna("missing_gene_default_zero")

    if len(out) != len(variants):
        fail("row count changed after join")

    if out[feature_cols].isna().any().any():
        fail("RNA-seq feature columns contain NaN after defaulting")

    unknown = out.loc[out["gene_symbol"] == "NOT_A_REAL_GENE_XYZ"].iloc[0]
    if float(unknown["rnaseq_log2fc"]) != 0.0:
        fail("unknown gene rnaseq_log2fc did not default to zero")
    if float(unknown["rnaseq_de_neglog10p"]) != 0.0:
        fail("unknown gene rnaseq_de_neglog10p did not default to zero")

    allowed_guards = {
        "normal_tissue_reference_not_variant_label",
        "missing_gene_default_zero",
    }
    if not set(out["leakage_guard"]).issubset(allowed_guards):
        fail(f"unexpected leakage guards: {sorted(set(out['leakage_guard']))}")

    print("RNA-seq DE feature integration smoke test passed.")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
