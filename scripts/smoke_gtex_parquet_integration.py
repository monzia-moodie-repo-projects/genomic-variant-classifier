from __future__ import annotations

from pathlib import Path
import pandas as pd

from genomic_variant_classifier.data.gtex import annotate_gtex_expression_from_parquet

PARQUET_PATH = Path("data/processed/gtex/gtex_v11_gene_expression.parquet")

def main() -> int:
    if not PARQUET_PATH.exists():
        raise SystemExit(f"ERROR: missing GTEx parquet artifact: {PARQUET_PATH}")

    variants = pd.DataFrame({
        "variant_id": ["test:17:43071077:G:T", "test:17:43071078:A:C", "test:1:1:A:T"],
        "gene_symbol": ["TP53", "BRCA1", "NOT_A_REAL_GENE_XYZ"],
    })

    out = annotate_gtex_expression_from_parquet(variants, PARQUET_PATH)

    required = ["gtex_max_tpm", "gtex_n_tissues_expressed", "gtex_tissue_specificity"]

    if len(out) != len(variants):
        raise SystemExit("ERROR: row count changed")

    for col in required:
        if col not in out.columns:
            raise SystemExit(f"ERROR: missing column: {col}")
        if out[col].isna().any():
            raise SystemExit(f"ERROR: null values in {col}")

    unknown = out.loc[out["gene_symbol"] == "NOT_A_REAL_GENE_XYZ"].iloc[0]
    assert float(unknown["gtex_max_tpm"]) == 0.0
    assert int(unknown["gtex_n_tissues_expressed"]) == 0
    assert float(unknown["gtex_tissue_specificity"]) == 0.0

    print("GTEx integration smoke test passed.")
    print(out[["gene_symbol"] + required])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
